"""Local arithmetic and explicit expansion of compact bundle twist boxes.

All public operations are transactional: input dictionaries remain unchanged.
Labels are measured in full twists, and arithmetic uses exact half-twist
integers. Ordinary geometry outside the affected rectangles is retained.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .diagram import assign_components, component_walks, validate
from .twist_boxes import (box_corners, box_port_points, bundle_ports, expand_boxes,
                          is_twist_box, make_twist_box, parse_twist_label,
                          twist_label)


def _box(diagram, box_id):
    result = next((v for v in diagram['crossings'] if v['id'] == box_id), None)
    if result is None or not is_twist_box(result):
        raise ValueError('Select a twist box first.')
    return result


def _clean(diagram):
    for key in ('spatial_curves', 'spatial_fingerprint', 'twist_box_expansion',
                'twist_box_edge_lineage'):
        diagram.pop(key, None)
    return diagram


def _finish(result, original=None):
    _clean(result)
    if original is not None:
        orientations = original.get('component_orientations', {})
        old = {eid: direction ^ (orientations.get(str(ci), orientations.get(ci, 1)) == -1)
               for ci, walk in enumerate(component_walks(original)) for eid, direction in walk}
        result['component_orientations'] = {}
        for ci, walk in enumerate(component_walks(result)):
            witnesses = {old[eid] ^ direction for eid, direction in walk if eid in old}
            if len(witnesses) != 1:
                raise ValueError('The operation could not preserve the boundary orientations.')
            if witnesses.pop():
                result['component_orientations'][str(ci)] = -1
    assign_components(result)
    checked = validate(result)
    if not checked['valid']:
        raise ValueError('The proposed box operation does not form a valid planar link: '
                         + '; '.join(checked.get('errors', [])))
    return result


def _rebuild_ports(diagram):
    vertices = {v['id']: v for v in diagram['crossings']}
    for v in vertices.values():
        v['ports'] = [None] * len(v['ports'])
    for edge in diagram['edges']:
        for end, key in enumerate(('start', 'end')):
            if edge.get(key) is not None:
                vid, port = edge[key]
                if vertices[vid]['ports'][port] is not None:
                    raise ValueError('Two strands claim the same box attachment.')
                vertices[vid]['ports'][port] = {'edge': edge['id'], 'end': end}


def _readable(box, count, *, length=None, stroke_width=3.):
    """Avoid producing a geometrically unreadable explicit braid."""
    if not count:
        return
    m = box.get('strand_count', 2)
    steps = abs(count)*m*(m-1)//2
    axial = box['size'][1] if length is None else length
    collar = min(6., axial*.1)
    minimum = max(4., float(stroke_width)*2.)
    points = box_port_points(box)
    top, bottom = bundle_ports(box)
    spacing = min(box['size'][0]*.6/(m-1),
                  *(math.dist(points[a], points[b]) for ports in (top, bottom)
                    for a, b in zip(ports, ports[1:])))
    if spacing < minimum:
        raise ValueError('There is too little transverse room between the box lanes for readable twists. '
                         'Use a wider box region before expanding it.')
    if (axial-4*collar)/steps < minimum-1e-8:
        raise ValueError('There is too little room in this box for readable explicit twists. '
                         'Expand a smaller amount, or use a larger box region.')


def delete_zero_box(diagram, box_id):
    """Remove a zero box and splice its parallel strands, retaining passages."""
    box = _box(diagram, box_id)
    if box['half_twists'] != 0:
        raise ValueError('Only a box labeled 0 can be deleted without changing the link.')
    return _finish(expand_boxes(diagram, box_ids=[box_id]))


def expand_twist_box(diagram, box_id, amount=None, side='bottom'):
    """Expand all or a same-sign part of a box into actual diagram crossings.

    ``amount`` is the printed number of full twists, e.g. '-1/2' or '-1'.
    The residual box stays on the side opposite ``side``. The box and clear
    adjoining parallel collars supply the room without displacing exterior
    ink. If that region is occupied, the operation is rejected. For whole-box
    expansion, ``side='both'`` grows equally at both ends, keeping the center.
    """
    box = _box(diagram, box_id)
    if side not in ('top', 'bottom', 'both'):
        raise ValueError('Choose the top, bottom, or both sides for expansion.')
    count = box['half_twists']
    extracted = count if amount is None else parse_twist_label(amount)
    if side == 'both' and extracted != count:
        raise ValueError('Both sides is available only when expanding the entire box.')
    if not count and not extracted:
        return delete_zero_box(diagram, box_id)
    if not extracted or extracted*count <= 0 or abs(extracted) > abs(count):
        raise ValueError('Expand a nonzero amount with the same sign, no larger than the box label.')
    from .render import line_width
    m = box.get('strand_count', 2)
    steps = abs(extracted)*m*(m-1)//2
    total = steps+sum(not is_twist_box(v) for v in diagram['crossings'])
    if total > 10000:
        raise ValueError(f'Twist-box expansion needs {total} crossings; the limit is 10000.')
    # Check the transverse lanes independently before looking for more axial
    # room. Room growth reuses clear existing collars; it cannot widen lanes.
    minimum_step = max(4., line_width(diagram)*2.)
    residual_min, residual_max, gap = 28., 44., 6.
    required = minimum_step*steps+24
    if extracted != count:
        # Solve H = braid + gap + clamp(H/4, 28, 44), using the same
        # residual-box layout as below. A fixed 50-unit allowance needlessly
        # reached neighboring crossings/corners when a smaller box would fit.
        braid_and_gap = required+gap
        required = braid_and_gap+max(residual_min, min(residual_max, braid_and_gap/3.))
    _readable(box, extracted, length=max(box['size'][1], required), stroke_width=line_width(diagram))
    if extracted != count and box.get('passages'):
        raise ValueError('Partially expanding a box with passing strands is not yet supported; '
                         'expand the whole box first.')
    if box['size'][1] < required-1e-8:
        from .box_room import grow_twist_box_room
        grown = grow_twist_box_room(diagram, box_id, required, side=side)
        return expand_twist_box(grown, box_id, amount, side)
    if extracted == count:
        return _finish(expand_boxes(diagram, box_ids=[box_id]))

    result = deepcopy(diagram)
    old_points = box_port_points(box)
    top, bottom = bundle_ports(box)
    m = box.get('strand_count', 2)
    width, height = box['size']
    axis = np.asarray(box['axis'], float)
    center = np.asarray(box['point'], float)
    # A readable label stays in the residual rectangle. The rest of the
    # existing disc accommodates the explicit braid and a short clear join.
    residual_height = max(residual_min, min(residual_max, height*.25))
    braid_height = height-residual_height-gap
    if braid_height <= 0:
        raise ValueError('There is too little room for both a residual box and explicit twists.')
    _readable(box, extracted, length=braid_height, stroke_width=line_width(diagram))
    sign = 1 if side == 'bottom' else -1
    residual_center = center-sign*axis*(height-residual_height)/2
    braid_center = center+sign*axis*(height-braid_height)/2
    temporary_id = max(v['id'] for v in result['crossings'])+1
    residual = make_twist_box(box_id, residual_center.tolist(), [width, residual_height],
                             twist_label(count-extracted), axis, strand_count=m)
    braid = make_twist_box(temporary_id, braid_center.tolist(), [width, braid_height],
                          twist_label(extracted), axis, strand_count=m)
    # Preserve measured, uneven external endpoints; interpolate inner lanes
    # without moving the exterior drawing or changing their left/right order.
    rpoints, bpoints = box_port_points(residual), box_port_points(braid)
    rt, rb = bundle_ports(residual)
    bt, bb = bundle_ports(braid)
    for lane in range(m):
        a, b = np.asarray(old_points[top[lane]]), np.asarray(old_points[bottom[lane]])
        for candidate, points, ups, downs in ((residual, rpoints, rt, rb), (braid, bpoints, bt, bb)):
            for indices, along in ((ups, -candidate['size'][1]/2), (downs, candidate['size'][1]/2)):
                t = (float((np.asarray(candidate['point'])-center)@axis)+along+height/2)/height
                points[indices[lane]] = (a*(1-t)+b*t).tolist()
    residual['port_points'], braid['port_points'] = rpoints, bpoints
    outer_top, outer_bottom = (residual, braid) if side == 'bottom' else (braid, residual)
    ot, _ = bundle_ports(outer_top)
    _, ob = bundle_ports(outer_bottom)
    mapping = {(box_id, p): [outer_top['id'], ot[lane]] for lane, p in enumerate(top)}
    mapping.update({(box_id, p): [outer_bottom['id'], ob[lane]] for lane, p in enumerate(bottom)})
    for edge in result['edges']:
        for key in ('start', 'end'):
            if edge.get(key) is not None and tuple(edge[key]) in mapping:
                edge[key] = mapping[tuple(edge[key])]
    result['crossings'] = [v for v in result['crossings'] if v['id'] != box_id]+[residual, braid]
    up, down = (residual, braid) if side == 'bottom' else (braid, residual)
    _, up_bottom = bundle_ports(up)
    down_top, _ = bundle_ports(down)
    upp, downp = box_port_points(up), box_port_points(down)
    next_edge = max((e['id'] for e in result['edges']), default=-1)+1
    for lane, (a, b) in enumerate(zip(up_bottom, down_top)):
        result['edges'].append({'id': next_edge+lane, 'start': [up['id'], a],
                                'end': [down['id'], b], 'points': [upp[a], downp[b]], 'closed': False})
    _rebuild_ports(result)
    _finish(result, diagram)
    return _finish(expand_boxes(result, box_ids=[temporary_id]))


def positive_half_twist(diagram, box_id, *, side='bottom'):
    """Draw one positive whole-bundle half twist from the selected box."""
    return expand_twist_box(diagram, box_id, '1/2', side)


def negative_half_twist(diagram, box_id, *, side='bottom'):
    return expand_twist_box(diagram, box_id, '-1/2', side)


def positive_full_twist(diagram, box_id, *, side='bottom'):
    return expand_twist_box(diagram, box_id, '1', side)


def negative_full_twist(diagram, box_id, *, side='bottom'):
    return expand_twist_box(diagram, box_id, '-1', side)


def _segment_inside(a, b, bounds, tolerance=1e-6):
    """Whether a segment has positive length in an open axis-aligned box."""
    low, high = np.asarray(bounds[0])+tolerance, np.asarray(bounds[1])-tolerance
    a, b = np.asarray(a, float), np.asarray(b, float)
    t0, t1 = 0., 1.
    for k in (0, 1):
        d = b[k]-a[k]
        if abs(d) < 1e-12:
            if not low[k] < a[k] < high[k]:
                return False
        else:
            x, y = sorted(((low[k]-a[k])/d, (high[k]-a[k])/d))
            t0, t1 = max(t0, x), min(t1, y)
            if t1 <= t0:
                return False
    return t1-t0 > 1e-9


def _paths_intersect(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return False
    if np.any(a.max(axis=0) < b.min(axis=0)-1e-7) or np.any(b.max(axis=0) < a.min(axis=0)-1e-7):
        return False
    low, high = np.minimum(b[:-1], b[1:]), np.maximum(b[:-1], b[1:])
    work = 0
    for p, q in zip(a, a[1:]):
        u = q-p
        candidates = np.flatnonzero(np.all(high >= np.minimum(p, q)-1e-7, axis=1)
                                    & np.all(low <= np.maximum(p, q)+1e-7, axis=1))
        work += len(candidates)
        if work > 2000000:
            raise ValueError('The connecting strands are too densely sampled or overlapping to certify; simplify their geometry first.')
        for j in candidates:
            r, s = b[j], b[j+1]
            v, w = s-r, r-p
            det = u[0]*v[1]-u[1]*v[0]
            if abs(det) < 1e-9:
                # Parallel touching/overlap is also an unsafe corridor.
                if abs(u[0]*w[1]-u[1]*w[0]) < 1e-7:
                    if np.all(np.maximum(np.minimum(p, q), np.minimum(r, s)) <=
                              np.minimum(np.maximum(p, q), np.maximum(r, s))+1e-7):
                        return True
                continue
            t, z = (w[0]*v[1]-w[1]*v[0])/det, (w[0]*u[1]-w[1]*u[0])/det
            if -1e-8 <= t <= 1+1e-8 and -1e-8 <= z <= 1+1e-8:
                return True
    return False


def _self_intersects(path):
    """Reject unencoded own crossings and backtracking in a direct connector."""
    if len(path) > 12000:
        raise ValueError('This connector has too many samples to certify safely; simplify its geometry first.')
    for i, (p, q) in enumerate(zip(path, path[1:])):
        if i+2 < len(path):
            u, v = q-p, path[i+2]-q
            if abs(u[0]*v[1]-u[1]*v[0]) < 1e-8 and float(u@v) < -1e-8:
                return True
        rest = path[i+2:]
        if len(rest) < 2:
            continue
        low, high = np.minimum(rest[:-1], rest[1:]), np.maximum(rest[:-1], rest[1:])
        candidates = np.flatnonzero(np.all(high >= np.minimum(p, q)-1e-7, axis=1)
                                    & np.all(low <= np.maximum(p, q)+1e-7, axis=1))
        if any(_paths_intersect(np.array([p, q]), rest[j:j+2]) for j in candidates):
            return True
    return False


def _certify_combination_rectangle(proxy, box, allowed, clearance, budget, source_boxes):
    """Certify the entire enclosing rectangle, without fitting a smaller crop.

    An identity parity proxy has no crossing landmarks. A generic tangle fit
    can otherwise choose an unrelated small strip of its selection, losing the
    boundary frame in which the omitted full twists must be restored.
    """
    from .box_compression import _trace_region, _word_from_trails, certify_twist_word, _geometry_work
    from .box_geometry import _boundary_cuts
    rectangle = np.asarray(box_corners(box), float)
    boundary, trails, selected = _trace_region(proxy, rectangle,
                                               crossing_clearance=clearance, budget=budget)
    m = box['strand_count']
    if len(trails) != m or len(boundary) != 2*m or selected != allowed:
        raise ValueError('The common region includes extra strands or crossings; try another box direction.')
    center, axis = np.asarray(box['point']), np.asarray(box['axis'])
    right = np.array([axis[1], -axis[0]])
    local = (np.asarray(boundary)-center)@np.column_stack([right, axis])
    binding, points = {}, [None]*(2*m)
    for sign, ports in zip((-1, 1), bundle_ports(box)):
        ordered = sorted((point[0], i) for i, point in enumerate(local)
                         if abs(point[1]-sign*box['size'][1]/2) < 1e-5)
        if len(ordered) != m:
            raise ValueError('The outer strands do not meet two clear opposite ends of the common region.')
        if any(b[0]-a[0] < 1e-5 for a, b in zip(ordered, ordered[1:])):
            raise ValueError('Two strands meet the same boundary point of the common region.')
        for port, (_, bid) in zip(ports, ordered):
            binding[bid], points[port] = port, boundary[bid]
    if len(binding) != len(boundary):
        raise ValueError('A strand enters the side of the common region.')
    box['port_points'] = points
    # Merely finding m identity strands is insufficient when both parity
    # proxies are zero. Every selected material strand must actually traverse
    # BOTH source boxes once, in the same order and from the same end bundle.
    # Otherwise an unrelated arc could replace a source strand in the count.
    common_traversal = None
    for trail in trails:
        path = np.asarray(trail['points'], float)
        if binding[trail['ends'][0]] not in bundle_ports(box)[0]:
            path = path[::-1]
        traversal = []
        for source in source_boxes:
            _geometry_work(budget, 4*len(path))
            cuts = _boundary_cuts(path, np.asarray(box_corners(source), float))
            if len(cuts) != 2:
                raise ValueError('Every selected strand must traverse both original boxes exactly once.')
            ports = []
            coordinates = np.asarray(box_port_points(source), float)
            for parameter in cuts:
                index = min(int(parameter), len(path)-2)
                point = path[index]+(parameter-index)*(path[index+1]-path[index])
                distances = np.linalg.norm(coordinates-point, axis=1)
                port = int(np.argmin(distances))
                if distances[port] > 1e-4:
                    raise ValueError('A source box is crossed away from its bundle attachments.')
                ports.append(port)
            top, bottom = bundle_ports(source)
            if ports[0] in top and ports[1] in bottom:
                direction = 1
            elif ports[0] in bottom and ports[1] in top:
                direction = -1
            else:
                raise ValueError('A selected strand turns around inside an original box.')
            traversal.append((cuts[0], source['id'], direction))
        signature = [(cid, direction) for _, cid, direction in sorted(traversal)]
        if common_traversal is not None and signature != common_traversal:
            raise ValueError('The original boxes are not traversed in one common bundle frame.')
        common_traversal = signature
    vertices = {vertex['id']: vertex for vertex in proxy['crossings']}
    word = _word_from_trails(box, trails, binding, vertices, budget=budget)
    certificate = certify_twist_word(m, word)
    if certificate['half_twists'] != box['half_twists']:
        raise ValueError('The connecting strands carry extra twists, not an adjacent parallel bundle.')


def _combine_certified(diagram, first_id, second_id):
    """Certify a clear common braid frame for neighboring tilted boxes.

    Each discarded even power is central in the bundle braid group. It is
    therefore sufficient to certify that the two signed parity proxies form
    Delta^(p+q) in one common boundary frame before restoring the exact sum.
    This never expands geometry proportional to the stored coefficients.
    """
    from .box_compression import certify_selected_bundle, TwistCertificateLimit
    from .box_geometry import collapse_box
    from .render import line_width
    import cv2

    a, b = _box(diagram, first_id), _box(diagram, second_id)
    m = a.get('strand_count', 2)
    # Require a complete direct bundle connection. Certification below decides
    # its lane order, rather than guessing from two different local frames.
    edges = {e['id']: e for e in diagram['edges']}
    connector_bundles = []
    for group in bundle_ports(a):
        destinations = []
        for port in group:
            ref = a['ports'][port]
            other = edges[ref['edge']]['end' if ref['end'] == 0 else 'start']
            if other is None or other[0] != second_id:
                break
            destinations.append(other[1])
        if len(destinations) == m and any(set(destinations) == set(group_b) for group_b in bundle_ports(b)):
            connector_bundles.append([edges[a['ports'][port]['edge']]['points'] for port in group])
    if not connector_bundles:
        raise ValueError('Adjacent boxes need a complete direct bundle of connecting strands.')

    axis = np.asarray(a['axis'], float)
    other = np.asarray(b['axis'], float)
    if float(axis@other) < 0:
        other = -other
    directions = [axis+other, axis, other, np.asarray(b['point'])-a['point']]
    candidates = []
    for vector in directions:
        if np.linalg.norm(vector) < 1e-8:
            continue
        vector = vector/np.linalg.norm(vector)
        if float(vector@axis) < 0:
            vector = -vector
        if not any(abs(float(vector@old)) > 1-1e-8 for old in candidates):
            candidates.append(vector)
    width = line_width(diagram)
    proxy = expand_boxes(diagram, proxy=True, box_ids=[first_id, second_id])
    allowed = {cid for entry in proxy.get('twist_box_expansion', []) for cid in entry['crossing_ids']}
    parity = sum((1 if box['half_twists'] >= 0 else -1)*(abs(box['half_twists']) % 2) for box in (a, b))
    errors = []
    budget = dict(work=0, limit=2000000)
    # A tall source box can make the average direction sweep across unrelated
    # neighboring ink. Test its own direction and the center line too. Always
    # enclose both original rectangles AND every connecting strand.
    frames = []
    for connector_paths in connector_bundles:
        # A closed braid can connect the boxes through BOTH ends: the short
        # interior corridor and the long exterior closure. Do not let click
        # order select the latter and force the whole component into the box.
        occupied_points = np.asarray(box_corners(a)+box_corners(b)+
                                     [point for path in connector_paths for point in path], float)
        for candidate in candidates:
            basis = np.array([[candidate[1], -candidate[0]], candidate])
            bounds = occupied_points@basis.T
            for padding in (max(1.5, width), max(3., width*2), max(6., width*3)):
                frames.append((candidate, basis, bounds.min(axis=0)-padding, bounds.max(axis=0)+padding))
    frames.sort(key=lambda frame: float(np.prod(frame[3]-frame[2])))
    for axis, basis, low, high in frames:
        polygon = np.array([[low[0], low[1]], [high[0], low[1]],
                            [high[0], high[1]], [low[0], high[1]]])@basis
        p32 = polygon.astype(np.float32)
        occupied = False
        for vertex in diagram['crossings']:
            if vertex['id'] in (first_id, second_id):
                continue
            if is_twist_box(vertex):
                area, _ = cv2.intersectConvexConvex(p32, np.asarray(box_corners(vertex), np.float32))
                hit = area > 1e-5
            else:
                hit = cv2.pointPolygonTest(p32, tuple(map(float, vertex['point'])), True) >= -width
            if hit:
                occupied = True
                break
        if occupied:
            errors.append('Another crossing or box occupies the proposed common region.')
            continue
        try:
            proposal = make_twist_box(-1, (((low+high)/2)@basis).tolist(), (high-low).tolist(),
                                      twist_label(parity), axis, strand_count=m)
            _certify_combination_rectangle(proxy, proposal, allowed, max(1., width*.5), budget, (a, b))
            result = collapse_box(proxy, proposal, bundle_certificate=certify_selected_bundle,
                                  attachment_tolerance=1e-4)
            previous = {v['id'] for v in diagram['crossings'] if v['id'] not in (first_id, second_id)}
            merged = next(v for v in result['crossings'] if is_twist_box(v) and v['id'] not in previous)
            if merged.get('passages'):
                raise ValueError('A passing strand occupies the common box region.')
            old_id = merged['id']
            merged['id'] = first_id
            merged['half_twists'] = a['half_twists']+b['half_twists']
            merged['label'] = twist_label(merged['half_twists'])
            for edge in result['edges']:
                for key in ('start', 'end'):
                    if edge.get(key) is not None and edge[key][0] == old_id:
                        edge[key][0] = first_id
            result['twist_box_combination'] = {'box_id': first_id, 'method': 'certified_common_frame',
                                                'source_ids': [first_id, second_id],
                                                'proxy_half_twists': parity}
            from .box_alignment import align_created_box
            return align_created_box(_finish(result), first_id)
        except TwistCertificateLimit:
            raise
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError('These boxes do not have a clear common parallel region: '+errors[0]
                     +' Straighten their connecting strands or move nearby ink clear.')


def combine_twist_boxes(diagram, first_id, second_id):
    """Add adjacent equal-bundle boxes through a certified empty corridor.

    A 180-degree frame reversal preserves the sign and reverses both lane
    orders. Tilted frames require an exact braid certificate on small parity
    proxies. Braided connectors and passing strands are rejected.
    """
    a, b = _box(diagram, first_id), _box(diagram, second_id)
    if first_id == second_id:
        raise ValueError('Select two different twist boxes.')
    if a.get('strand_count', 2) != b.get('strand_count', 2):
        raise ValueError('The boxes must act on the same number of strands.')
    if a.get('passages') or b.get('passages'):
        raise ValueError('Boxes with passing strands cannot be combined yet; move those strands clear first.')
    axis = np.asarray(a['axis'], float)
    if abs(float(axis@np.asarray(b['axis'])))-1 < -1e-7:
        return _combine_certified(diagram, first_id, second_id)
    right = np.array([axis[1], -axis[0]])
    basis = np.array([right, axis])
    def local(points):
        return np.asarray(points, float)@basis.T
    def framed(box):
        top, bottom = bundle_ports(box)
        return (top, bottom) if float(axis@np.asarray(box['axis'])) > 0 else (bottom[::-1], top[::-1])
    if float(np.asarray(a['point'])@axis) > float(np.asarray(b['point'])@axis):
        up, down = b, a
    else:
        up, down = a, b
    up_top, up_bottom = framed(up)
    down_top, down_bottom = framed(down)
    upp, downp = box_port_points(up), box_port_points(down)
    if float(np.asarray(upp[up_bottom[0]])@axis) >= float(np.asarray(downp[down_top[0]])@axis)-1e-6:
        raise ValueError('The boxes overlap or have no clear corridor between their ends.')
    removed, connector_paths = set(), []
    edge_map = {e['id']: e for e in diagram['edges']}
    for x, y in zip(up_bottom, down_top):
        ref = up['ports'][x]
        edge = edge_map[ref['edge']]
        far = edge['end' if ref['end'] == 0 else 'start']
        if far != [down['id'], y]:
            raise ValueError('Adjacent boxes need direct, ordered parallel connections on the whole bundle.')
        path = deepcopy(edge['points'])
        if ref['end']:
            path.reverse()
        path = [upp[x]]+path+[downp[y]]
        path = [p for j, p in enumerate(path) if j == 0 or math.dist(p, path[j-1]) > 1e-9]
        connector_paths.append(local(path))
        removed.add(edge['id'])
    corners = local(box_corners(a)+box_corners(b))
    low, high = corners.min(axis=0), corners.max(axis=0)
    for path in connector_paths:
        if np.any(path < low-1e-6) or np.any(path > high+1e-6):
            raise ValueError('A connecting strand leaves the proposed merged box region.')
        start_level = float(np.asarray(upp[up_bottom[0]])@axis)
        end_level = float(np.asarray(downp[down_top[0]])@axis)
        if np.any(path[:, 1] < start_level-1e-6) or np.any(path[:, 1] > end_level+1e-6):
            raise ValueError('A connecting strand enters a source box outside its attachment; move it into the clear corridor first.')
        if _self_intersects(path):
            raise ValueError('A connecting strand crosses or overlaps itself; separate it before combining boxes.')
    for i, path in enumerate(connector_paths):
        # Non-monotone geometry is accepted only if the strands do not meet.
        if any(_paths_intersect(path, other) for other in connector_paths[:i]):
            raise ValueError('The connecting strands cross or touch; their twist cannot be discarded.')
    # Any exterior strand entering this new disc would be hidden by the box.
    from .render import edge_points
    vertices = {v['id']: v for v in diagram['crossings']}
    for edge in diagram['edges']:
        if edge['id'] in removed:
            continue
        pts = local(edge_points(edge, vertices))
        if any(_segment_inside(p, q, (low, high)) for p, q in zip(pts, pts[1:])):
            raise ValueError('Another strand enters the region between these boxes; clear that region first.')
    for vertex in diagram['crossings']:
        if vertex['id'] in (first_id, second_id):
            continue
        pts = local(box_corners(vertex) if is_twist_box(vertex) else [vertex['point']])
        if len(pts) == 1:
            overlaps = np.all(pts[0] > low) and np.all(pts[0] < high)
        else:
            # Parallel or rotated foreign box: any overlap of its bounding
            # extent is conservatively treated as occupied room.
            overlaps = np.all(pts.max(axis=0) > low) and np.all(pts.min(axis=0) < high)
        if overlaps:
            raise ValueError('Another crossing or box occupies the proposed merged region.')
    center = ((low+high)/2)@basis
    merged = make_twist_box(first_id, center.tolist(), (high-low).tolist(),
                            twist_label(a['half_twists']+b['half_twists']), axis,
                            strand_count=a.get('strand_count', 2))
    mt, mb = bundle_ports(merged)
    points = box_port_points(merged)
    mapping = {}
    for old_box, old_ports, new_ports, old_points in ((up, up_top, mt, upp), (down, down_bottom, mb, downp)):
        for old_port, new_port in zip(old_ports, new_ports):
            mapping[old_box['id'], old_port] = [first_id, new_port]
            points[new_port] = deepcopy(old_points[old_port])
    merged['port_points'] = points
    result = deepcopy(diagram)
    result['crossings'] = [v for v in result['crossings'] if v['id'] not in (first_id, second_id)]+[merged]
    result['edges'] = [e for e in result['edges'] if e['id'] not in removed]
    for edge in result['edges']:
        for key in ('start', 'end'):
            if edge.get(key) is not None and tuple(edge[key]) in mapping:
                edge[key] = mapping[tuple(edge[key])]
    _rebuild_ports(result)
    from .box_alignment import align_created_box
    return align_created_box(_finish(result, diagram), first_id)
