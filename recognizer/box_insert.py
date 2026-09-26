"""Insert a discrete whole-bundle twist along a transverse user-drawn slice.

Only a crossing-free neighborhood of the slice is contracted. No source curve
outside that rectangle is displaced to create room; ambiguous or crowded cuts
are rejected before the immutable input diagram can be changed.
"""
from __future__ import annotations

from copy import deepcopy
import math
import cv2
import numpy as np

from .box_geometry import _boundary_cuts, _inside, collapse_box, _physical_directions
from .diagram import assign_components, component_walks, validate
from .geometry import _section, intersection
from .render import edge_points, line_width
from .twist_boxes import box_corners, bundle_ports, is_twist_box, make_twist_box, parse_twist_label, twist_label


def _slice_hits(diagram, segment, width):
    a, b = segment
    right = (b-a)/np.linalg.norm(b-a)
    axis = np.array([-right[1], right[0]])
    hits = []
    for edge in diagram['edges']:
        points = np.asarray(edge['points'], float)
        if (points[:, 0].max() < min(a[0], b[0]) or points[:, 0].min() > max(a[0], b[0])
                or points[:, 1].max() < min(a[1], b[1]) or points[:, 1].min() > max(a[1], b[1])):
            continue
        distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        for index, (p, q) in enumerate(zip(points, points[1:])):
            hit = intersection(a, b, p, q)
            if hit is None:
                continue
            along, local, location = hit
            if min(along, 1-along)*np.linalg.norm(b-a) < 2*width:
                raise ValueError('Extend the slicing segment beyond the outer strands; its ends must lie in clear space.')
            if any(old['edge'] == edge['id'] and np.linalg.norm(location-old['point']) < 1e-5 for old in hits):
                continue
            distance = distances[index]+local*(distances[index+1]-distances[index])
            window = max(5., 3*width)
            ends = np.clip([distance-window, distance+window], 0, distances[-1])
            tangent_points = np.column_stack([np.interp(ends, distances, points[:, k]) for k in (0, 1)])
            tangent = tangent_points[1]-tangent_points[0]
            length = np.linalg.norm(tangent)
            if length < 1e-6 or abs(float(tangent@axis))/length < .8:
                raise ValueError('The slice must cut the strands transversely; choose a more nearly perpendicular segment.')
            # A touch at a polygon vertex has two branches on the same side.
            sample = (tangent_points-location)@axis
            if sample[0]*sample[1] >= -1e-8:
                raise ValueError('The slice touches a strand instead of crossing it. Move the segment slightly.')
            hits.append({'edge': edge['id'], 'point': location, 'parameter': index+local,
                         'tangent': tangent/length, 'across': float(location@right)})
    hits.sort(key=lambda h: h['across'])
    if len(hits) < 2:
        raise ValueError('Cut through at least two nearby parallel strands to add a twist box.')
    if any(abs(float(a['tangent']@b['tangent'])) < .86 for a in hits for b in hits):
        raise ValueError('The cut strands are not locally parallel. Choose a straighter part of the bundle.')
    if min(b['across']-a['across'] for a, b in zip(hits, hits[1:])) < max(5., 2.5*width):
        raise ValueError('The strands are too close for distinct box attachments. Spread the bundle slightly first.')
    return right, axis, hits


def _rectangle_ports(diagram, box, width):
    """Measure the actual boundary cuts rather than extrapolating tangents."""
    polygon = np.asarray(box_corners(box), float)
    center = np.asarray(box['point']); axis = np.asarray(box['axis'])
    right = np.array([axis[1], -axis[0]])
    ends = {-1: [], 1: []}
    for edge in diagram['edges']:
        points = np.asarray(edge['points'], float)
        cuts = _boundary_cuts(points, polygon)
        if edge.get('closed') and abs(cv2.pointPolygonTest(polygon.astype(np.float32), tuple(points[0]), True)) < 1e-6:
            cuts.append(0.)
        for cut in cuts:
            point = np.asarray(_section(points, cut, cut)[0])
            across, along = float((point-center)@right), float((point-center)@axis)
            if abs(abs(along)-box['size'][1]/2) > 1e-5:
                raise ValueError('A strand enters the side of the proposed box; use a shorter slice or a straighter bundle.')
            if box['size'][0]/2-abs(across) < max(2., width):
                raise ValueError('A strand is too near a box corner. Leave more clear room around the slice.')
            ends[-1 if along < 0 else 1].append((across, point.tolist()))
    m = box['strand_count']
    if len(ends[-1]) != m or len(ends[1]) != m:
        raise ValueError('There is not a clear rectangular neighborhood containing exactly the cut strands.')
    points = [None]*(2*m)
    for sign, ports in ((-1, bundle_ports(box)[0]), (1, bundle_ports(box)[1])):
        for port, (_, point) in zip(ports, sorted(ends[sign])):
            points[port] = point
    box['port_points'] = points


def prepare_twist_insertion(diagram, segment):
    """Return an identity-box candidate and normalized diagram for a clear cut.

    This is also suitable for a UI preview. It does not change connectivity or
    input coordinates, and performs no numeric-twist expansion.
    """
    if not validate(diagram)['valid']:
        raise ValueError('The existing diagram must validate before adding twists.')
    segment = np.asarray(segment, float)
    if segment.shape != (2, 2) or not np.isfinite(segment).all() or np.linalg.norm(segment[1]-segment[0]) < 8:
        raise ValueError('Draw a finite slicing segment across the bundle.')
    source = deepcopy(diagram)
    vertices = {v['id']: v for v in source['crossings']}
    for edge in source['edges']:
        edge['points'] = [list(p) for p in edge_points(edge, vertices)]
    width = line_width(source)
    right, axis, hits = _slice_hits(source, segment, width)
    center = (hits[0]['point']+hits[-1]['point'])/2
    margin = max(9., 3.5*width)
    transverse = hits[-1]['across']-hits[0]['across']+2*margin
    preferred = max(40., 14*width, min(90., transverse*.55))
    minimum = max(24., 8*width)
    errors = []
    for length in sorted({preferred, max(minimum, preferred*.75), minimum}, reverse=True):
        box = make_twist_box(max(vertices, default=-1)+1, center.tolist(), [transverse, length], '0', axis,
                             strand_count=len(hits))
        polygon = np.asarray(box_corners(box), float)
        try:
            for vertex in vertices.values():
                if is_twist_box(vertex):
                    area, _ = cv2.intersectConvexConvex(polygon.astype(np.float32), np.asarray(box_corners(vertex), np.float32))
                    if area > 1e-5:
                        raise ValueError('The slicing neighborhood overlaps an existing twist box. Choose clear strands outside it.')
                elif cv2.pointPolygonTest(polygon.astype(np.float32), tuple(map(float, vertex['point'])), True) >= -2*width:
                    raise ValueError('A crossing is too close to the slice. Choose a crossing-free part of the bundle.')
            _rectangle_ports(source, box, width)
            # Incidence validation alone cannot detect an unrecorded geometric
            # crossing in a saved polyline. Certify the actual local projection
            # as an identity braid before replacing it by a numeric tangle.
            from .box_compression import certify_selected_bundle
            compact = collapse_box(source, box, attachment_tolerance=max(1e-4, width*.1),
                                   bundle_certificate=certify_selected_bundle)
            inserted = next(v for v in compact['crossings'] if v['id'] == box['id'])
            if inserted.get('passages'):
                raise ValueError('An unselected strand crosses the proposed box. Choose a clearer slicing neighborhood.')
            return {'diagram': compact, 'box_id': inserted['id'], 'segment': segment.tolist(),
                    'strand_count': len(hits), 'box': deepcopy(inserted)}
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError('There is no clear parallel neighborhood around this slice. '+
                     (errors[-1] if errors else 'Leave more room for the twist box.'))


def add_twists(diagram, segment, label):
    """Insert an integer/half-integer twist; this operation can change the link."""
    count = parse_twist_label(label)
    prepared = prepare_twist_insertion(diagram, segment)
    result = prepared['diagram']
    old_directions = _physical_directions(result)
    box = next(v for v in result['crossings'] if v['id'] == prepared['box_id'])
    box.update(half_twists=count, label=twist_label(count))
    result['component_orientations'] = {}
    reassigned = 0
    for ci, walk in enumerate(component_walks(result)):
        eid, direction = walk[0]
        reverse = old_directions[eid] ^ direction
        if any(old_directions[e] ^ d != reverse for e, d in walk):
            reassigned += 1
        result['component_orientations'][str(ci)] = -1 if reverse else 1
    assign_components(result)
    verdict = validate(result)
    if not verdict['valid']:
        raise ValueError('; '.join(verdict['errors']))
    result['twist_box_edit'] = dict(operation='add_twists', box_id=box['id'],
        half_twists=count, strand_count=prepared['strand_count'], segment=prepared['segment'],
        orientation_reassigned_components=reassigned,
        warning=('The half twist reconnects differently oriented arcs; affected component orientations were reassigned.' if reassigned else None))
    from .box_alignment import align_created_box
    return align_created_box(result, box['id'])
