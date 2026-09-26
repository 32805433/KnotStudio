"""Pixels-to-compact-box reconstruction, independent of dataset sidecars."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import cv2
import numpy as np

from .twist_boxes import make_twist_box, box_corners, bundle_ports, twist_label


def checked_detection(rgb, detection):
    if not isinstance(detection, dict) or not isinstance(detection.get('boxes'), list):
        raise ValueError('Twist-box review must contain box proposals and the source pixel hash.')
    digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    if detection.get('source_sha256', detection.get('image_sha256')) != digest:
        raise ValueError('Twist-box markings are stale. Mark twist boxes again before recognizing.')
    if detection.get('image_size') != [rgb.shape[1], rgb.shape[0]]:
        raise ValueError('Twist-box markings have different image dimensions.')
    return deepcopy(detection)


def _observed_boundary_ports(rgb, mask, box, stroke, dark):
    """Locate distinct source strands just outside the erased box boundary.

    Projecting a detected interior-frame port straight outward can miss a
    curved strand. Move the temporary port to observed ink at the *expanded*
    boundary instead. Require exactly the expected, ordered strand runs on
    both sides; do not invent a bridge when ink or correspondence is missing.
    """
    center = np.asarray(box['point'], float)
    axis = np.asarray(box['axis'], float)
    right = np.array([axis[1], -axis[0]])
    extent = np.asarray(box['size'], float)/2
    level = rgb.mean(axis=2)
    ink = (level > 160 if dark else level < 100) & (mask == 0)
    y, x = np.nonzero(ink)
    points = np.column_stack([x, y]).astype(float)
    across, along = (points-center)@right, (points-center)@axis
    replacements = {}
    for sign, ports in ((-1, bundle_ports(box)[0]), (1, bundle_ports(box)[1])):
        expected = np.asarray([(np.asarray(box['port_points'][i])-center)@right
                               for i in ports])
        # A warped frame can leave an isolated corner beyond the strand
        # bundle. It is not an extra attachment; restrict the boundary scan
        # to the same bounded displacement allowed for the observed ports.
        band = ((sign*along > extent[1]+.5) &
                (sign*along <= extent[1]+max(2., .5*stroke)) &
                (np.abs(across) < extent[0]) &
                (across >= expected.min()-4*stroke) &
                (across <= expected.max()+4*stroke))
        positions = np.sort(across[band])
        if not len(positions):
            return False
        groups = np.split(positions, np.flatnonzero(np.diff(positions) > 1.8)+1)
        if len(groups) != len(ports):
            return False
        observed = [float(np.median(g)) for g in groups]
        ordered = sorted(ports, key=lambda i: (np.asarray(box['port_points'][i])-center)@right)
        for port, value in zip(ordered, observed):
            old = float((np.asarray(box['port_points'][port])-center)@right)
            if abs(value-old) > 4*stroke:
                return False
            replacements[port] = (center+value*right+sign*extent[1]*axis).tolist()
    for port, point in replacements.items():
        box['port_points'][port] = point
    return True



def _identity_stroke_color(rgb, dark):
    """Do not introduce a new color layer into a single-color drawing.

    Color-aware tracing would separate a black temporary strand from a green
    source strand at their overlap. That artificial seam can become a pair of
    fake crossings. Keep a single chromatic ink's measured core color; mixed
    colors, grayscale drawings and photographs retain the established proxy.
    """
    default = (255, 255, 255) if dark else (0, 0, 0)
    if dark or np.median(rgb.min(axis=2)) < 245:
        return default
    mask = rgb.min(axis=2) < 205
    values = rgb[mask]
    if not len(values):
        return default
    median = np.median(values, axis=0)
    if np.ptp(median) < 30:
        return default
    from .knotfolio_backend import _color_layers
    if _color_layers(rgb.astype(float), mask)[1] != 1:
        return default
    darkest = values.min(axis=1)
    core = values[darkest <= np.median(darkest)]
    return tuple(int(round(v)) for v in np.median(core, axis=0))


def identity_box_image(rgb, proposals, *, observed_ports=False):
    """Substitute simple identity bundles for tracing the *exterior* graph.

    The numeric twist is not rasterized, even temporarily. After the exterior
    attachments are recovered, their exact discrete permutation is restored.
    """
    output = rgb.copy()
    dark = float(np.median(rgb.mean(axis=2))) < 115.
    color = _identity_stroke_color(rgb, dark)
    boxes = []
    for index, proposal in enumerate(proposals):
        if proposal.get('half_twists') is None:
            raise ValueError('A twist coefficient is unreadable. Click its box and enter the value.')
        if any(proposal.get(k) is None for k in ('strand_count', 'point', 'size', 'axis', 'port_points', 'bundle_ports')):
            raise ValueError('A twist box has ambiguous strand attachments. Unmark the selected box, then use Mark a missed twist box to replace its rectangle.')
        count = proposal['half_twists']
        strands = proposal['strand_count']
        if (isinstance(count, bool) or not isinstance(count, int)
                or isinstance(strands, bool) or not isinstance(strands, int) or strands < 2):
            raise ValueError('Twist coefficients and strand counts must use exact integers.')
        axis = np.asarray(proposal['axis'], float)
        right = np.array([axis[1], -axis[0]])
        center = np.asarray(proposal['point'], float)
        quad = np.asarray(proposal['outline'], float)
        measured = [p.get('stroke_width', 1.) for p in proposal.get('ports', [])]
        stroke = max(1., float(np.median(measured))) if measured else 2.
        # The detector sees the interior border. Include its stroke and the
        # slight nonrectangularity of hand-drawn frames in the replacement.
        local = np.column_stack([(quad-center)@right, (quad-center)@axis])
        extent = np.max(np.abs(local), axis=0)+max(2., proposal.get('diagnostics', {}).get('stroke_width', stroke)*1.5)
        box = make_twist_box(index, center.tolist(), (2*extent).tolist(), '0', axis,
                             strand_count=strands)
        box['bundle_ports'] = deepcopy(proposal['bundle_ports'])
        box['port_points'] = deepcopy(proposal['port_points'])
        top, bottom = bundle_ports(box)
        for sign, ports in ((-1, top), (1, bottom)):
            for port in ports:
                point = np.asarray(box['port_points'][port], float)
                across = float((point-center)@right)
                box['port_points'][port] = (center+across*right+sign*extent[1]*axis).tolist()
        corners = np.asarray(box_corners(box), float)
        left, top_y = np.maximum(0, np.floor(corners.min(axis=0)-8*stroke)).astype(int)
        right_x, bottom_y = np.minimum([rgb.shape[1], rgb.shape[0]], np.ceil(corners.max(axis=0)+8*stroke)).astype(int)
        patch = rgb[top_y:bottom_y, left:right_x]
        level = patch.mean(axis=2)
        background = np.median(patch[level < np.quantile(level, .3)] if dark else patch[level >= np.quantile(level, .7)], axis=0)
        if background.shape != (3,) or not np.isfinite(background).all():
            background = np.array([0, 0, 0] if dark else [255, 255, 255])
        mask = np.zeros(rgb.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 1)
        if observed_ports:
            _observed_boundary_ports(rgb, mask, box, stroke, dark)
        output[mask.astype(bool)] = background.astype(np.uint8)
        for a, b in zip(top, bottom):
            pa = np.asarray(box['port_points'][a]); pb = np.asarray(box['port_points'][b])
            # A short overlap joins source ink at the outside of the erased
            # border; it supplies no hidden crossing or component identity.
            line = [pa-axis*stroke, pa, pb, pb+axis*stroke]
            cv2.polylines(output, [np.rint(line).astype(np.int32)], False, color,
                          max(1, round(stroke)), cv2.LINE_AA)
        box['source_half_twists'] = count
        box['source_box_id'] = proposal.get('id', f'twist-box-{index+1}')
        boxes.append(box)
    return output, boxes


def recognize_boxes(rgb, detection, options=None, *, long_gap_retry=True):
    from .pipeline import _recognize_array
    from .box_geometry import collapse_boxes
    from .diagram import assign_components, crossing_free_count, pd_code, validate
    detection = checked_detection(rgb, detection)
    proposals = detection['boxes']
    diagnostics = {'twist_boxes': detection}
    try:
        pixels, boxes = identity_box_image(rgb, proposals)
    except (ValueError, KeyError, TypeError) as exc:
        return dict(status='needs_review', diagram=None, pd_code=None,
                    warnings=[str(exc)], diagnostics=diagnostics, preview_paths=[], unlinked_unknot_components=0)
    opts = dict(options or {})
    opts.pop('twist_boxes', None)
    result = _recognize_array(pixels, options=opts)
    if result.get('diagram') is None:
        # Preserve existing successful interpretations. This retry changes only
        # temporary box ports, using visible source ink outside the frame.
        observed_pixels, observed_boxes = identity_box_image(rgb, proposals, observed_ports=True)
        if not np.array_equal(observed_pixels, pixels):
            trial = _recognize_array(observed_pixels, options=opts)
            if trial.get('diagram') is not None:
                result, pixels, boxes = trial, observed_pixels, observed_boxes
                result['diagnostics']['box_boundary_ports'] = 'observed exterior ink'
    default_tracing = (not (set(opts)-{'board_photo', 'remove_labels', 'endpoint_hook_retry'})
                       and not opts.get('remove_labels', False))
    if (result.get('diagram') is None and long_gap_retry and default_tracing
            and opts.get('board_photo', 'auto') in ('auto', 'off')):
        # Box diagrams previously bypassed the public long-gap fallback. Apply
        # the same multi-threshold, visible-overstrand checks to the exterior.
        from .long_gaps import recognize_long_underpasses
        recovered = recognize_long_underpasses(pixels)
        if recovered is not None:
            result = recovered
    # Some illustrations use faint colored strands alongside black frames.
    # Losing the colors to luminance thresholding can erase entire arcs. Try
    # their ink union only after ordinary tracing fails, on light, clean
    # backgrounds. Require two independent thresholds to agree on the whole
    # exterior projection before adopting it; source pixels remain untouched.
    if (result.get('diagram') is None and np.median(pixels.min(axis=2)) >= 245
            and np.count_nonzero(pixels.max(axis=2).astype(int)-pixels.min(axis=2) > 30) >= 20):
        from .motion import _same_projection
        alternatives, attempts = [], []
        for threshold in (200, 225, 240):
            mask = pixels.min(axis=2) < threshold
            binary = np.repeat(np.where(mask, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
            trial = _recognize_array(binary, options={**opts, 'board_photo': 'off'})
            attempts.append({'threshold': threshold, 'complete': trial.get('diagram') is not None})
            if trial.get('diagram') is not None:
                alternatives.append(trial)
        if len(alternatives) >= 2 and all(_same_projection(alternatives[0]['diagram'], a['diagram']) for a in alternatives[1:]):
            result = alternatives[0]
            result['warnings'].insert(0, 'Faint colored strands were traced as an ink union; inspect their crossing order.')
            result['diagnostics']['box_color_union'] = {'attempts': attempts, 'projection_agreement': True}
        else:
            result.setdefault('diagnostics', {})['box_color_union'] = {'attempts': attempts, 'projection_agreement': False}
    if result.get('diagram') is None and default_tracing:
        from .box_exterior import exterior_marks, recognize_exterior_retry, same_box_exterior
        observed_pixels, observed_boxes = identity_box_image(rgb, proposals, observed_ports=True)
        alternatives = [(pixels, boxes)]
        if not np.array_equal(observed_pixels, pixels):
            alternatives.append((observed_pixels, observed_boxes))
        completed = []
        for candidate_pixels, candidate_boxes in alternatives:
            clean_pixels, marks = exterior_marks(rgb, candidate_pixels, proposals)
            trial = recognize_exterior_retry(clean_pixels, candidate_boxes, long_gaps=long_gap_retry)
            if trial is not None:
                trial['diagnostics']['box_exterior_marks'] = marks
                completed.append((trial, clean_pixels, candidate_boxes))
        if completed and all(same_box_exterior(
                collapse_boxes(completed[0][0]['diagram'], completed[0][2]),
                collapse_boxes(c[0]['diagram'], c[2])) for c in completed[1:]):
            result, pixels, boxes = completed[0]
            if result['diagnostics']['box_exterior_marks']:
                result['warnings'].append('Isolated direction arrows or detached box-frame remnants were omitted from the temporary strand trace; the source image is unchanged.')
    result.setdefault('diagnostics', {}).update(diagnostics)
    if result.get('diagram') is None:
        result['warnings'].insert(0, 'The boxes were located, but the surrounding strands still need repair.')
        return result
    try:
        diagram = collapse_boxes(result['diagram'], boxes)
        for box in diagram['crossings']:
            if box.get('kind') == 'twist_box':
                box['half_twists'] = box.pop('source_half_twists')
                box['label'] = twist_label(box)
        # Source arrows are applied once by the public recognition boundary,
        # after the real box permutations have established the components.
        diagram['component_orientations'] = {}
        assign_components(diagram)
        result['diagram'] = diagram
        result['diagnostics']['validation'] = validate(diagram)
        try:
            result['pd_code'] = pd_code(diagram)
        except ValueError as exc:
            result['pd_code'] = None
            result['warnings'].append(str(exc))
        result['status'] = 'needs_review'
        result['warnings'].insert(0, 'Twist boxes were reconstructed as discrete bundle tangles. Check the coefficients, strand attachments and handedness; papers may use different sign conventions.')
        result['unlinked_unknot_components'] = crossing_free_count(diagram, result['pd_code'])
        return result
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return dict(status='needs_review', diagram=None, pd_code=None,
                    warnings=[f'Twist-box attachments could not be certified: {exc}'],
                    diagnostics=result['diagnostics'], preview_paths=[], unlinked_unknot_components=0)
