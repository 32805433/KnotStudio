"""Failure-only style normalization with independent-mask agreement.

Rasterized pen strokes can have jagged boundaries; photographed marker/chalk
can have broad low-contrast halos.  These are pixel hypotheses, never a lookup
of a source identity or a desired knot.  Existing successful recognition must
take precedence.  The caller applies orientations from the untouched source.
"""
from copy import deepcopy
import hashlib
import time

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def preserves_ink_components(source, candidate, *, radius, minimum_area):
    """Require local support for every substantial source ink component.

    A mask is not allowed to make a label, faint component, or large isolated
    mark disappear just to obtain a closed diagram.  Subpixel antialiasing
    fragments below the stated area floor are not treated as separate marks.
    """
    source, candidate = np.asarray(source, bool), np.asarray(candidate, bool)
    if source.shape != candidate.shape or not source.any() or not candidate.any():
        return False
    _, labels, stats, _ = cv2.connectedComponentsWithStats(source.astype(np.uint8), 8)
    nearby = ndimage.distance_transform_edt(~candidate) <= radius
    kept = np.bincount(labels[nearby & source], minlength=len(stats))
    substantial = stats[:, 4] >= minimum_area
    substantial[0] = False
    return bool(np.all(kept[substantial] >= .8*stats[substantial, 4]))


def diagram_supported_by_ink(diagram, source, width):
    """Reject a valid graph traced from background instead of the drawing.

    A permissive absolute threshold can turn a gray paper background into a
    single closed frame.  Agreement across smoothing scales would repeat that
    same error.  Require both source-to-curve and curve-to-source coverage;
    allow local crossing gaps and short orientation wings, not distant frames.
    """
    source = np.asarray(source, bool)
    curves = np.zeros(source.shape, np.uint8)
    for edge in diagram.get('edges', []):
        points = np.asarray(edge.get('points', []), float)
        if len(points) < 2 or not np.isfinite(points).all():
            continue
        points = np.rint(points).astype(np.int32)
        cv2.polylines(curves, [points], False, 1, 1)
    if not curves.any() or not source.any():
        return False
    if not preserves_ink_components(source, curves, radius=max(2., 1.5*width),
                                    minimum_area=max(6., .3*width*width)):
        return False
    distance = ndimage.distance_transform_edt(~source)
    return bool(np.mean(distance[curves.astype(bool)] <= max(3., 3*width)) >= .7)


def _flat_masks(rgb):
    """Smooth at fractions of the measured pen width, without rescaling."""
    dark = rgb.min(2)
    threshold = min(205., float(np.median(dark))-20.)
    source = dark < threshold
    if not .0001 < source.mean() < .25:
        return
    radii = ndimage.distance_transform_edt(source)[skeletonize(source)]
    if not len(radii):
        return
    width = max(1.5, 2*float(np.median(radii)))
    sigmas = sorted(set(round(float(np.clip(width*f, .55, 2.)), 5)
                        for f in (.2, .3, .4)))
    # Clipping two proposed scales to the same value is one observation.
    if len(sigmas) < 2:
        return
    for sigma in sigmas:
        pixels = cv2.GaussianBlur(rgb, (0, 0), sigma)
        candidate = pixels.min(2) < threshold
        if not preserves_ink_components(source, candidate, radius=max(1., 2*sigma),
                                        minimum_area=max(6., .3*width*width)):
            continue
        yield pixels, dict(method='pen_boundary_smoothing', sigma_pixels=sigma,
                           measured_stroke_width=width,
                           component_support_radius=max(1., 2*sigma))


def _recognize_flat(rgb):
    from .pipeline import _recognize_array
    from .geometry import check_embedding
    from .motion import _same_projection

    complete, attempts = [], []
    dark = rgb.min(2)
    source = dark < min(205., float(np.median(dark))-20.)
    for pixels, preparation in _flat_masks(rgb):
        result = _recognize_array(pixels, options={'board_photo': 'off'})
        diagram = result.get('diagram')
        record = {**preparation, 'status': result['status']}
        attempts.append(record)
        if diagram is None:
            continue
        if not check_embedding(diagram, check_strokes=False)['valid']:
            record['invalid_embedding'] = True
            continue
        if not diagram_supported_by_ink(diagram, source, preparation['measured_stroke_width']):
            record['insufficient_source_ink_support'] = True
            continue
        complete.append((result, preparation))
    if len(complete) < 2:
        return None
    result, preparation = complete[0]
    if any(not _same_projection(result['diagram'], other['diagram'])
           for other, _ in complete[1:]):
        return None
    result = deepcopy(result)
    result['diagnostics']['style_recovery'] = dict(
        method='pen_boundary_smoothing', successful_candidates=len(complete),
        attempts=attempts, minimum_agreement=2,
        semantic_label_removal=False, source_dimensions_preserved=True)
    result['diagnostics']['preprocessing'].update(style_normalization=preparation)
    # The smoothing kernel must not change the displayed pen thickness on a
    # later recognize / convert-to-drawing round trip.
    result['diagnostics']['stroke_width'] = preparation['measured_stroke_width']
    result['diagram'].setdefault('recognition', {})['stroke_width'] = preparation['measured_stroke_width']
    warning = ('Pen boundaries were smoothed at several small scales which '
               'agreed on the diagram. Inspect narrow gaps and crossings.')
    result['warnings'].insert(0, warning)
    result['diagram']['recognition']['warnings'] = result['warnings']
    result['status'] = 'needs_review'
    return result


def _photo_contrast(rgb, mode):
    from .board_photo import BoardContrast

    class StyleContrast(BoardContrast):
        def candidates(self):
            # A broad envelope can join chalk erasure residue to real ink.
            # Sweep stronger boundaries while deriving seeds from the same
            # image's contrast distribution, for either background polarity.
            peak = float(np.percentile(self.morph, 99.5))
            source = self.morph > max(20., .5*peak)
            seen = set()
            for fraction in (.15, .25, .35, .5, .65):
                weak = max(6., peak*fraction)
                strong = max(20., peak*max(.5, fraction))
                keep, removed = self._seeded(self.morph > weak, self.morph > strong)
                if not preserves_ink_components(source, keep, radius=1.5,
                                                minimum_area=25.):
                    continue
                signature = hashlib.sha256(keep.tobytes()).digest()
                if signature in seen:
                    continue
                seen.add(signature)
                pixels = np.full_like(self.rgb, 255)
                pixels[keep] = 0
                yield pixels, dict(method='style_contrast_envelope', strong=strong,
                                   weak=weak, foreground_fraction=float(keep.mean()),
                                   discarded_regions=removed)

    return StyleContrast(rgb, mode)


def style_photo_mask(rgb, preparation):
    """Recreate the accepted pre-repair photo mask for source orientations."""
    contrast = _photo_contrast(rgb, preparation.get('polarity', 'board'))
    for pixels, record in contrast.candidates():
        if all(record.get(key) == preparation.get(key)
               for key in ('method', 'strong', 'weak')):
            return pixels, contrast.scale
    return None


def _recognize_photo(rgb, mode):
    from .board_photo import recognize_board
    contrast = _photo_contrast(rgb, mode)
    result = recognize_board(rgb, mode, _contrast=contrast,
                             _minimum_agreement=2, _require_embedding=True)
    if result.get('diagram') is None:
        return None
    preparation = result['diagnostics']['preprocessing']
    accepted = style_photo_mask(rgb, preparation)
    if accepted is None:
        return None
    pixels, scale = accepted
    source_frame = {'edges': [{'points': (np.asarray(edge['points'])*scale).tolist()}
                              for edge in result['diagram']['edges']]}
    if not diagram_supported_by_ink(source_frame, pixels.min(2) < 205,
                                   result['diagnostics']['stroke_width']*scale):
        return None
    result['diagnostics']['style_recovery'] = dict(
        method='photo_contrast_envelope', minimum_agreement=2,
        successful_candidates=result['diagnostics']['board_photo']['successful_candidates'],
        semantic_label_removal=False,
        component_support_radius=1.5, source_dimensions_preserved=True)
    warning = ('Stronger local contrast masks agreed on the photographed '
               'diagram. Inspect faint strokes and discarded low-contrast marks.')
    result['warnings'].insert(0, warning)
    result['diagram']['recognition']['warnings'] = result['warnings']
    result['status'] = 'needs_review'
    return result


def recognize_style_recovery(rgb, *, mode='auto'):
    """Return a guarded recovery or None, for an otherwise failed diagram.

    Each family needs at least two agreeing complete, valid planar embeddings.
    This function does not detect/remove labels or boxes.  Public callers must
    retain separate box handling and the user's explicit tracing options.
    """
    from .board_photo import looks_photographic
    started = time.monotonic()
    rgb = np.asarray(rgb, dtype=np.uint8)
    photo = looks_photographic(rgb)[0]
    if photo and mode == 'off':
        return None
    if photo or mode in ('board', 'light', 'dark'):
        result = _recognize_photo(rgb, 'board' if mode == 'auto' else mode)
    else:
        result = _recognize_flat(rgb)
    if result is not None:
        result['diagnostics']['style_recovery']['elapsed_seconds'] = time.monotonic()-started
    return result
