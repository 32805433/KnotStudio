"""Repair narrow enclosed voids inside a continuing scanned ink band.

Only enclosed background changes. Neither foreground component connectivity nor
open strand gaps are changed. The two solid continuations distinguish an ink
sliver from an isolated thin oval or the interior of a curl.
"""
from copy import deepcopy
import time

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def fill_stroke_holes(mask):
    """Return a copy and review records; retain ambiguous holes untouched."""
    from .bundled_crossings import stable_tangent
    from .knotfolio_backend import _adjacency, _trace, _prune_spurs

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 3:
        raise ValueError('Expected a two-dimensional foreground mask.')
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    if not np.any(holes):
        return mask.copy(), []
    skeleton = skeletonize(mask)
    radii = ndimage.distance_transform_edt(mask)[skeleton]
    width = max(1.5, float(np.median(radii)*2)) if len(radii) else 1.5
    labels, count = ndimage.label(holes)
    if count > 512:
        return mask.copy(), []
    candidates = []
    provisional = mask.copy()
    for label, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        local = labels[region] == label
        if int(local.sum()) < 3:
            continue
        # Padding is necessary for a hole whose crop touches every boundary.
        radius = float(ndimage.distance_transform_edt(np.pad(local, 1))[1:-1, 1:-1].max())
        if radius > .4*width:
            continue
        graph = _adjacency(skeletonize(np.pad(local, 1)))
        # A raster notch in the void creates a tiny side branch in its medial
        # axis. Analyze the principal void curve, without changing any ink.
        _prune_spurs(graph, 2*radius)
        chains = _trace(graph)
        if not chains:
            continue
        chain = max(chains, key=len)
        if (len(chain) < .9*len(graph)
                or len(graph[chain[0]]) != 1 or len(graph[chain[-1]]) != 1):
            continue
        points = np.asarray(chain, float)[:, ::-1]
        points += np.array([region[1].start-1, region[0].start-1])
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if length < max(2., 2*radius):
            continue
        candidates.append((label, region, local, points, radius, length))
        provisional[region] |= local
    if not candidates:
        return mask.copy(), []
    # A short neighboring sliver should not interrupt continuation evidence.
    # Only enclosed, narrow, elongated candidates enter this auxiliary mask.
    distance = ndimage.distance_transform_edt(provisional)
    reach = max(1, int(np.floor(.35*width)))
    yy, xx = np.ogrid[-reach:reach+1, -reach:reach+1]
    nearby_radius = ndimage.maximum_filter(distance, footprint=xx*xx+yy*yy <= reach*reach)
    result, evidence = mask.copy(), []
    for label, region, local, points, radius, length in candidates:
        continuing = True
        for end in (0, 1):
            tip = points[0 if end == 0 else -1]
            tangent = stable_tangent(points, end, 2*width)
            samples = tip + np.linspace(width, 3*width, 10)[:, None]*tangent
            pixels = np.rint(samples).astype(int)
            if (np.any(pixels < 0) or np.any(pixels[:, 0] >= mask.shape[1])
                    or np.any(pixels[:, 1] >= mask.shape[0])
                    or np.any(nearby_radius[pixels[:, 1], pixels[:, 0]] < .3*width)):
                continuing = False
                break
        if not continuing:
            continue
        result[region] |= local
        evidence.append({'bbox': [region[1].start, region[0].start,
                                  region[1].stop, region[0].stop],
                         'ends': [points[0].tolist(), points[-1].tolist()],
                         'filled_pixels': int(local.sum()), 'inradius': radius,
                         'centerline_length': length, 'stroke_width': width})
    return result, evidence


def recognize_stroke_holes(rgb):
    """Conservative last resort; require two valid agreeing threshold masks."""
    from .board_photo import looks_photographic
    from .diagram import pd_code, validate
    from .geometry import assemble, beautify, check_embedding
    from .knotfolio_backend import extract, _color_layers
    from .motion import _same_projection
    from .preprocess import prepare

    started = time.monotonic()
    if looks_photographic(rgb)[0]:
        return None
    pixels, preprocessing = prepare(rgb)
    if _color_layers(pixels.astype(float), pixels.min(2) < 205)[1] > 1:
        return None
    complete, attempts = [], []
    for threshold in (150., 180., 205., 230., 240.):
        mask = pixels.min(2) < threshold
        if not .0001 < mask.mean() < .3:
            continue
        repaired, evidence = fill_stroke_holes(mask)
        attempts.append({'threshold': threshold, 'filled_holes': len(evidence)})
        if not evidence:
            continue
        rendered = np.repeat(np.where(repaired, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
        trace = extract(rendered, {'arrow_wings': True, 'micro_seams': True,
                                'stop_at_branches': True, 'max_endpoints': 180})
        if not trace['complete']:
            continue
        try:
            diagram = beautify(assemble(trace, rgb.shape[1], rgb.shape[0]))
            verdict = validate(diagram)
            if not verdict['valid'] or not check_embedding(diagram, check_strokes=False)['valid']:
                continue
            code = pd_code(diagram)
            complete.append((diagram, trace, verdict, code, evidence))
        except (ValueError, RuntimeError, KeyError, IndexError):
            continue
    if len(complete) < 2:
        return None
    diagram, trace, verdict, code, evidence = complete[0]
    if any(not _same_projection(diagram, candidate[0]) for candidate in complete[1:]):
        return None
    warnings = ['Thin enclosed slivers inside continuing strokes were filled; '
                'inspect the marked source regions.']
    diagnostic = deepcopy(trace['diagnostics'])
    diagnostic.update(validation=verdict, stroke_hole_repairs=evidence,
                      stroke_hole_attempts=attempts, stroke_hole_agreement=len(complete),
                      preprocessing={**preprocessing, 'stroke_holes_filled': True},
                      elapsed_seconds=time.monotonic()-started)
    diagram['recognition'] = {'stroke_width': diagnostic['stroke_width'], 'warnings': warnings}
    return {'status': 'needs_review', 'diagram': diagram, 'pd_code': code,
            'unlinked_unknot_components': verdict['crossing_free_components'],
            'diagnostics': diagnostic, 'warnings': warnings}
