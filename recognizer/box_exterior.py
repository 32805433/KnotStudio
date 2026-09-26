"""Conservative last-resort tracing outside recognized twist-box frames.

Only temporary tracing pixels are changed.  Original images and explicit box
coefficients are untouched, and all completed threshold interpretations must
agree before the exterior can be adopted.
"""
from copy import deepcopy
import time
import hashlib

import cv2
import numpy as np
from scipy import ndimage


def exterior_marks(source, pixels, proposals):
    """Remove isolated diagram arrows and small, proven box-frame remnants.

    A corner remnant must have been attached to the erased source frame, remain
    close to its original corner, have no hole, and avoid every strand port.
    Detached letters, loops, and ordinary open tangle ends are not removed.
    """
    from .label_geometry import arrow_component
    if np.median(pixels.min(axis=2)) < 235:
        return pixels, []
    ink = pixels.min(axis=2) < 205
    source_ink = source.min(axis=2) < 205
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    _, original_labels = cv2.connectedComponents(source_ink.astype(np.uint8), 8)
    remove = np.zeros(ink.shape, bool)
    records = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        local = labels[y:y+h, x:x+w] == i
        reason = None
        # The arrow shape classifier was originally non-destructive label
        # protection. An arrow-shaped CLOSED curve is a real unknot.
        if np.any(ndimage.binary_fill_holes(local) & ~local):
            continue
        if arrow_component(local):
            reason = 'isolated_direction_arrow'
        else:
            yy, xx = np.nonzero(local)
            points = np.column_stack((xx+x, yy+y)).astype(float)
            for box in proposals:
                stroke = max(1., box.get('diagnostics', {}).get('stroke_width', 1.))
                if area > 8*stroke*stroke or np.hypot(w, h) > 6*stroke:
                    continue
                # A little detached circle is still a legitimate link component.
                if np.any(ndimage.binary_fill_holes(local) & ~local):
                    continue
                corners = np.asarray(box['outline'], float)
                distances = np.linalg.norm(points[:, None]-corners, axis=2)
                if np.min(np.max(distances, axis=0)) > 4*stroke:
                    continue
                ports = np.asarray(box['port_points'], float)
                if np.min(np.linalg.norm(points[:, None]-ports, axis=2)) < 3*stroke:
                    continue
                source_ids = set(original_labels[points[:, 1].astype(int), points[:, 0].astype(int)])-{0}
                if not source_ids:
                    continue
                # Positive proof that this exact source component touches the
                # frame, rather than assuming every mark near a box is debris.
                frame = np.zeros(ink.shape, np.uint8)
                cv2.polylines(frame, [np.rint(corners).astype(np.int32)], True, 1,
                              max(1, round(2*stroke)))
                frame_ids = set(original_labels[(frame > 0) & source_ink])-{0}
                if not source_ids <= frame_ids:
                    continue
                reason = 'detached_box_corner_remnant'
                break
        if reason:
            remove[y:y+h, x:x+w] |= local
            records.append({'reason': reason, 'bbox': [int(x), int(y), int(x+w), int(y+h)],
                            'pixels': int(area)})
    if not records:
        return pixels, []
    # Include antialiasing immediately around the identified object without
    # touching another foreground component.
    halo = ndimage.binary_dilation(remove, iterations=1) & ~ink
    result = pixels.copy()
    result[remove | halo] = 255
    return result, records


def same_box_exterior(first, second):
    """Compare compact projections with every source box/port held fixed.

    Closing the boxes with identity strands before comparing PD codes loses
    the distinction between boundary attachments. Preserve those anchors and
    each ordinary crossing's cyclic order and over/under pair instead.
    """
    import networkx as nx
    from .twist_boxes import bundle_ports

    def graph(diagram):
        result = nx.MultiDiGraph()
        box_ids = set()
        for vertex in diagram['crossings']:
            ident = vertex['id']
            is_box = vertex.get('kind') == 'twist_box'
            if is_box:
                anchor = vertex.get('source_box_id', ident)
                if anchor in box_ids:
                    raise ValueError('Source box identities must be unique.')
                box_ids.add(anchor)
                if vertex.get('passages') or vertex.get('passage_crossings'):
                    raise ValueError('The exterior fallback does not infer passages through boxes.')
                top, bottom = bundle_ports(vertex)
                roles = {port: (side, lane) for side, ports in (('top', top), ('bottom', bottom))
                         for lane, port in enumerate(ports)}
            count = len(vertex['ports'])
            for port in range(count):
                color = ('box', anchor, roles[port]) if is_box else ('crossing', port % 2 == vertex['over'])
                result.add_node((ident, port), color=color)
                if not is_box:
                    result.add_edge((ident, port), (ident, (port+1) % count), kind='rotate')
        for edge in diagram['edges']:
            if edge.get('start') is None and edge.get('end') is None and edge.get('closed'):
                result.add_node(('closed', edge['id']), color=('closed',))
                continue
            a, b = tuple(edge['start']), tuple(edge['end'])
            result.add_edge(a, b, kind='arc')
            result.add_edge(b, a, kind='arc')
        return result

    try:
        a, b = graph(first), graph(second)
    except (ValueError, KeyError, TypeError):
        return False
    return nx.is_isomorphic(a, b,
        node_match=nx.algorithms.isomorphism.categorical_node_match('color', None),
        edge_match=nx.algorithms.isomorphism.categorical_multiedge_match('kind', None))


def recognize_exterior_retry(pixels, boxes=(), *, long_gaps=True):
    """Try moderately wide crossing gaps with complete projection agreement.

    Thin printed strokes can have gaps just beyond the ordinary 18-width cap.
    Longer matches still need well-aligned tips and explicit over-strand ink;
    arbitrary breaks, hidden/hidden intersections, and incomplete graphs fail.
    """
    from .board_photo import looks_photographic
    from .diagram import pd_code, validate
    from .geometry import assemble, beautify, check_embedding
    from .knotfolio_backend import extract
    from .box_geometry import collapse_boxes
    from .style_recovery import diagram_supported_by_ink
    from .bundled_crossings import overlaps_visible_ink

    started = time.monotonic()
    if looks_photographic(pixels)[0]:
        return None
    complete, attempts, seen_masks = [], [], set()
    for threshold in (150., 205., 230.):
        # Match extract's effective foreground, including tiny-object removal
        # and the image boundary. Identical masks are only one observation.
        mask = pixels.min(axis=2) < threshold
        labels, _ = ndimage.label(mask)
        sizes = np.bincount(labels.ravel())
        mask &= sizes[labels] >= 6
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
        digest = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
        if digest in seen_masks:
            attempts.append({'threshold': threshold, 'duplicate_mask': True})
            continue
        seen_masks.add(digest)
        trace = extract(pixels, {'threshold': threshold, 'max_gap_widths': 26. if long_gaps else 18.,
                                'min_tangent_dot': .6, 'stop_at_branches': True,
                                'max_endpoints': 180, 'micro_seams': True,
                                'arrow_wings': True})
        record = {'threshold': threshold, 'complete': trace['complete']}
        attempts.append(record)
        if not trace['complete']:
            continue
        width = trace['diagnostics']['stroke_width']
        paths = [np.asarray(p['points'], float) for p in trace['paths']]
        starts = np.concatenate([p[:-1] for p in paths])
        ends = np.concatenate([p[1:] for p in paths])
        supported = True
        extended = []
        for match in trace['matches']:
            if match['distance'] <= 18*width:
                continue
            a, b = [paths[match[key][0]][0 if match[key][1] == 0 else -1] for key in ('a', 'b')]
            hits = match['crossings']
            if (not hits or min(match['alignment']) < .6
                    or min(h['sine_angle'] for h in hits) < .25
                    or overlaps_visible_ink(a, b, starts, ends, width)):
                supported = False
                break
            extended.append({'ends': [a.tolist(), b.tolist()], 'distance_widths': match['distance']/width,
                             'alignment': match['alignment'], 'crossings': [h['point'] for h in hits]})
        if not supported:
            record['unsupported_extended_gap'] = True
            continue
        try:
            diagram = beautify(assemble(trace, pixels.shape[1], pixels.shape[0]))
            validation = validate(diagram)
            if not validation['valid'] or not check_embedding(diagram, check_strokes=False)['valid']:
                record['invalid_embedding'] = True
                continue
            if not diagram_supported_by_ink(diagram, pixels.min(axis=2) < threshold, width):
                record['insufficient_ink_support'] = True
                continue
            code = pd_code(diagram)
            compact = collapse_boxes(diagram, boxes)
            complete.append((diagram, trace, code, validation, extended, compact))
        except (ValueError, RuntimeError, KeyError, IndexError):
            record['invalid_assembly'] = True
    if len(complete) < 2 or any(not same_box_exterior(complete[0][5], c[5]) for c in complete[1:]):
        return None
    diagram, trace, code, validation, extended, compact = complete[0]
    warnings = ['The exterior was recovered using agreement across contrast thresholds. Inspect crossing gaps and box attachments.']
    diagnostics = deepcopy(trace['diagnostics'])
    diagnostics.update(validation=validation, box_exterior_retry={
        'attempts': attempts, 'agreement': len(complete), 'distinct_masks': len(seen_masks),
        'box_port_agreement': True, 'extended_gaps': extended},
        elapsed_seconds=time.monotonic()-started)
    diagram['recognition'] = {'stroke_width': diagnostics['stroke_width'], 'warnings': warnings}
    return {'status': 'needs_review', 'diagram': diagram, 'pd_code': code,
            'unlinked_unknot_components': validation['crossing_free_components'],
            'diagnostics': diagnostics, 'warnings': warnings}
