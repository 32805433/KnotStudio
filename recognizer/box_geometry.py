"""Boundary-preserving geometry for compact twist tangles.

Boxes are discs with explicitly ordered attachment ports. Geometry outside a
box remains ordinary link geometry; an extra arc across its disc is recorded
as a passage above or below the entire tangle, not as a change to its label.
"""
from __future__ import annotations

from copy import deepcopy
import math
import cv2
import numpy as np

from .geometry import intersection, _section


def _inside(point, polygon, tolerance=1e-7):
    return cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(map(float, point)), True) > tolerance


def _boundary_cuts(points, polygon):
    points = np.asarray(points, float)
    polygon = np.asarray(polygon, float)
    cuts = []
    # Local edits must not perform four Python intersection tests for every
    # distant sample. The bounding filter is conservative at its boundary.
    lower, upper = polygon.min(axis=0)-1e-6, polygon.max(axis=0)+1e-6
    indexes = np.flatnonzero(np.all(np.maximum(points[:-1], points[1:]) >= lower, axis=1)
                            & np.all(np.minimum(points[:-1], points[1:]) <= upper, axis=1))
    # Test the handful of boundary edges against nearby segments together.
    # This uses precisely intersection()'s denominator and endpoint tolerances;
    # parallel/overlapping segments still produce no transverse boundary cut.
    if len(indexes):
        a = points[indexes, None, :]
        r = points[indexes+1, None, :]-a
        c = polygon[None, :, :]
        s = (np.roll(polygon, -1, axis=0)-polygon)[None, :, :]
        q = c-a
        den = r[..., 0]*s[..., 1]-r[..., 1]*s[..., 0]
        valid = np.abs(den) >= 1e-9
        safe_den = np.where(valid, den, 1.)
        t = (q[..., 0]*s[..., 1]-q[..., 1]*s[..., 0])/safe_den
        u = (q[..., 0]*r[..., 1]-q[..., 1]*r[..., 0])/safe_den
        valid &= (t >= -1e-7) & (t <= 1+1e-7) & (u >= -1e-7) & (u <= 1+1e-7)
        values = (np.clip(t, 0., 1.)+indexes[:, None])[valid]
        cuts = values[(values > 1e-7) & (values < len(points)-1-1e-7)].tolist()
    unique = []
    for value in sorted(cuts):
        if not unique or value-unique[-1] > 1e-6:
            unique.append(value)
    return unique


def _array_section(points, left, right):
    """The same polyline cut as _section, retaining numeric arrays internally."""
    points = np.asarray(points, float)
    if left == 0. and right == len(points)-1:
        return points
    def point(t):
        i = min(int(t), len(points)-2)
        return points[i]+(t-i)*(points[i+1]-points[i])
    return np.vstack((point(left), points[int(left)+1:math.ceil(right)], point(right)))


def _graph_from_segments(template, vertices, segments, *, _array_geometry=False):
    """Contract degree-two joins; preserve explicit crossing/box endpoints."""
    vertices = deepcopy(vertices)
    lookup = {v['id']: v for v in vertices}
    adjacency = {}
    for i, seg in enumerate(segments):
        for end, node in enumerate(seg['nodes']):
            adjacency.setdefault(tuple(node), []).append((i, end))
    used, edges = set(), []
    for seed, segment in enumerate(segments):
        if seed in used:
            continue
        # Find an explicit port, walking backwards through temporary joins.
        start = (seed, 0)
        seen = set()
        while start not in seen:
            seen.add(start)
            i, end = start
            node = tuple(segments[i]['nodes'][end])
            if node[0] == 'port':
                break
            options = [ref for ref in adjacency[node] if ref != start]
            if len(options) != 1:
                raise ValueError('A box boundary left an unresolved strand endpoint.')
            j, other = options[0]
            start = (j, 1-other)
        first_node = tuple(segments[start[0]]['nodes'][start[1]])
        cursor, points, lineage = start, [], []
        while cursor[0] not in used:
            i, end = cursor; used.add(i)
            seg = segments[i]
            path = seg['points'][::1 if end == 0 else -1]
            if _array_geometry:
                points.append(np.asarray(path if not points else path[1:], float))
            else:
                points.extend(path if not points else path[1:])
            lineage.extend([[eid, direction ^ end] for eid, direction in seg.get('lineage', [])])
            node = tuple(seg['nodes'][1-end])
            if node[0] == 'port':
                break
            options = [ref for ref in adjacency[node] if ref != (i, 1-end)]
            if len(options) != 1:
                raise ValueError('A box boundary has an ambiguous strand join.')
            cursor = options[0]
        if _array_geometry:
            points = np.concatenate(points)
        closed = first_node[0] != 'port'
        if closed and math.dist(points[0], points[-1]) > 1e-7:
            points = np.vstack((points, points[0])) if _array_geometry else [*points, points[0]]
        edge = {'id': len(edges), 'points': points, 'start': None if closed else list(first_node[1:]),
                'end': None if closed else list(node[1:]), 'closed': closed,
                'box_expansion_lineage': lineage}
        edges.append(edge)
        if not closed:
            for end, loc in enumerate((edge['start'], edge['end'])):
                lookup[loc[0]]['ports'][loc[1]] = {'edge': edge['id'], 'end': end}
    result = {k: deepcopy(v) for k, v in template.items()
              if k not in ('edges', 'crossings', 'spatial_curves', 'spatial_fingerprint', 'twist_box_expansion')}
    result.update(crossings=vertices, edges=edges)
    return result


def _physical_directions(diagram):
    from .diagram import component_walks
    orientations = diagram.get('component_orientations', {})
    return {eid: direction ^ (orientations.get(str(ci), orientations.get(ci, 1)) == -1)
            for ci, walk in enumerate(component_walks(diagram)) for eid, direction in walk}


def _restore_directions(result, original):
    from .diagram import component_walks, assign_components
    directions = _physical_directions(original)
    result['component_orientations'] = {}
    edges = {e['id']: e for e in result['edges']}
    for ci, walk in enumerate(component_walks(result)):
        for eid, traversal in walk:
            witnesses = edges[eid].get('box_expansion_lineage', [])
            if witnesses:
                old, relative = witnesses[0]
                result['component_orientations'][str(ci)] = -1 if directions[old] ^ relative ^ traversal else 1
                break
    assign_components(result)


def collapse_box(diagram, box, *, default_over=True, attachment_tolerance=None,
                 bundle_certificate=None, _array_geometry=False):
    """Replace a recognized/proxy tangle disc by one compact box vertex.

    The interior must be the box's signed parity proxy (identity for even
    half-twists, one whole-bundle half-twist for odd). The boundary-rooted
    crossing sequences are certified before discarding that proxy geometry.
    Inconsistent or mixed over/under passages are rejected. A supplied
    ``bundle_certificate(box, trails, binding, vertices)`` may certify a more
    general braid; it must raise ValueError unless the entire bundle is equal
    relative to its boundary to the proposed box. The default remains the
    strict parity-proxy certificate used by dragging and raster tracing.
    The input is immutable, so a rejected drag can retain its previous frame.
    """
    from .twist_boxes import box_corners, box_port_points, transition_port, bundle_ports, certify_proxy_bundle
    from .diagram import validate
    polygon = np.asarray(box_corners(box), float)
    section = _array_section if _array_geometry else _section
    old_vertices = {v['id']: v for v in diagram['crossings']}
    inside_vertices = {i for i, v in old_vertices.items() if _inside(v['point'], polygon)}
    if any(v.get('kind') == 'twist_box' for i, v in old_vertices.items() if i in inside_vertices):
        raise ValueError('Twist boxes cannot overlap or contain another box.')
    segments, boundary = [], []
    for edge in diagram['edges']:
        points = np.asarray(edge['points'], float)
        cuts = [0., *_boundary_cuts(points, polygon), float(len(points)-1)]
        seam = None
        if edge.get('closed') and edge.get('start') is None and abs(cv2.pointPolygonTest(
                polygon.astype(np.float32), tuple(map(float, points[0])), True)) < 1e-6:
            seam = ('boundary', len(boundary))
            boundary.append(points[0].tolist())
        nodes = []
        for j, t in enumerate(cuts):
            if j in (0, len(cuts)-1):
                end = int(j != 0)
                loc = edge['end' if end else 'start']
                nodes.append(('port', *loc) if loc is not None else seam or ('join', edge['id']))
            else:
                node = ('boundary', len(boundary))
                point = section(points, t, t)[0]
                if isinstance(point, np.ndarray):
                    point = point.tolist()
                boundary.append(point); nodes.append(node)
        for j, (a, b) in enumerate(zip(cuts, cuts[1:])):
            path = section(points, a, b)
            middle = section(points, (a+b)/2, (a+b)/2)[0]
            segments.append({'nodes': [nodes[j], nodes[j+1]], 'points': path,
                             'inside': _inside(middle, polygon), 'lineage': [[edge['id'], 0]]})
    if not boundary:
        raise ValueError('No strands attach to this twist box.')
    adjacent = {}
    for i, s in enumerate(segments):
        if s['inside']:
            for end, node in enumerate(s['nodes']):
                adjacent.setdefault(tuple(node), []).append((i, end))
    trails, visited = [], set()
    for bid in range(len(boundary)):
        node = ('boundary', bid)
        refs = adjacent.get(node, [])
        if len(refs) != 1:
            raise ValueError('A strand is tangent to the box boundary; move it clear of the corner.')
        if refs[0][0] in visited:
            continue
        cursor, trace, crossings, crossing_parameters = refs[0], [], [], []
        while cursor[0] not in visited:
            i, end = cursor; visited.add(i)
            s = segments[i]; path = s['points'][::1 if end == 0 else -1]
            if isinstance(path, np.ndarray):
                path = path.tolist()
            trace.extend(path if not trace else path[1:])
            dest = tuple(s['nodes'][1-end])
            if dest[0] == 'boundary':
                break
            if dest[0] == 'port':
                cid, p = dest[1:]
                if cid not in inside_vertices:
                    raise ValueError('A crossing lies on the box boundary.')
                crossings.append((cid, p))
                crossing_parameters.append(len(trace)-1)
                target = ('port', cid, (p+2)%4)
                options = adjacent.get(target, [])
            else:
                options = [r for r in adjacent.get(dest, []) if r != (i, 1-end)]
            if len(options) != 1:
                raise ValueError('The strands inside this box cannot be connected consistently.')
            cursor = options[0]
        else:
            raise ValueError('A closed strand lies entirely inside the box.')
        trails.append({'ends': [bid, dest[1]], 'points': trace, 'crossings': crossings,
                       'crossing_parameters': crossing_parameters})
    if len(visited) != sum(s['inside'] for s in segments):
        raise ValueError('The box contains an unattached closed component.')
    expected = box_port_points(box)
    top, bottom = bundle_ports(box)
    bundle = {'top': top, 'bottom': bottom}
    tolerance = attachment_tolerance or max(3., min(box['size'])*.12)
    binding, bound = {}, set()
    for p in bundle['top']+bundle['bottom']:
        distances = sorted((math.dist(expected[p], q), i) for i, q in enumerate(boundary) if i not in bound)
        if not distances or distances[0][0] > tolerance:
            raise ValueError('A bundle attachment moved away from the box. Choose a smaller move radius.')
        bid = distances[0][1]; binding[bid] = p; bound.add(bid)
    bundle_trails, passage_trails = [], []
    for trail in trails:
        a, b = trail['ends']
        if a in binding or b in binding:
            if a not in binding or b not in binding or transition_port(box, binding[a]) != binding[b]:
                raise ValueError('The detected endpoint order disagrees with the twist label.')
            bundle_trails.append(trail)
        else:
            passage_trails.append(trail)
    if len(bundle_trails) != box['strand_count']:
        raise ValueError('The box does not have the specified number of bundle strands.')
    (bundle_certificate or certify_proxy_bundle)(box, bundle_trails, binding, old_vertices)
    bundle_crossings = {cid for t in bundle_trails for cid, _ in t['crossings']}
    for t in passage_trails:
        evidence = {p % 2 == old_vertices[cid]['over'] for cid, p in t['crossings'] if cid in bundle_crossings}
        if len(evidence) > 1:
            raise ValueError('A strand must pass wholly above or wholly below a twist box.')
        t['over'] = evidence.pop() if evidence else bool(default_over)
    center = np.asarray(box['point'])
    ordered = sorted(range(len(boundary)), key=lambda i: math.atan2(-(boundary[i][1]-center[1]), boundary[i][0]-center[0]))
    port_index = {bid: p for p, bid in enumerate(ordered)}
    new_box = deepcopy(box)
    new_box['id'] = max(old_vertices, default=-1)+1
    new_box['ports'] = [None]*len(boundary)
    new_box['port_points'] = [boundary[i] for i in ordered]
    inverse_binding = {p: bid for bid, p in binding.items()}
    new_box['bundle_ports'] = {side: [port_index[inverse_binding[p]] for p in indexes] for side, indexes in bundle.items()}
    new_box['passages'] = [{'ports': [port_index[e] for e in t['ends']], 'points': t['points'], 'over': t['over']}
                           for t in passage_trails]
    # Preserve mutual passage crossings too; their order is independent of
    # whether each passage is above or below the underlying bundle.
    new_box['passage_crossings'] = []
    for i, a in enumerate(passage_trails):
        for cid in {cid for cid, _ in a['crossings']}:
            visits = [j for j, (cc, _) in enumerate(a['crossings']) if cc == cid]
            if len(visits) == 2:
                parameters = [a['crossing_parameters'][j]/max(1, len(a['points'])-1) for j in visits]
                over_visit = next(j for j in visits if a['crossings'][j][1] % 2 == old_vertices[cid]['over'])
                new_box['passage_crossings'].append({'point': old_vertices[cid]['point'],
                    'pair': [i, i], 'parameters': parameters,
                    'over_parameter': a['crossing_parameters'][over_visit]/max(1, len(a['points'])-1)})
        for j, b in enumerate(passage_trails[:i]):
            shared = {cid for cid, _ in a['crossings']} & {cid for cid, _ in b['crossings']}
            for cid in shared:
                p = next(p for cc, p in a['crossings'] if cc == cid)
                new_box['passage_crossings'].append({'point': old_vertices[cid]['point'],
                    'pair': [j, i], 'over': i if p % 2 == old_vertices[cid]['over'] else j})
    outside = []
    for s in segments:
        if s['inside']:
            continue
        # _section created these point rows specifically for this collapse.
        # Only replace the node list here; recursively copying every exterior
        # point again made each local drag pay for all distant ink.
        s = dict(s)
        s['nodes'] = [('port', new_box['id'], port_index[n[1]]) if n[0] == 'boundary' else n for n in s['nodes']]
        outside.append(s)
    result = _graph_from_segments(diagram, [v for i, v in old_vertices.items() if i not in inside_vertices]+[new_box], outside, _array_geometry=_array_geometry)
    _restore_directions(result, diagram)
    for edge in result['edges']:
        edge.pop('box_expansion_lineage', None)
    verdict = validate(result)
    if not verdict['valid']:
        raise ValueError('; '.join(verdict['errors']))
    return result


def collapse_boxes(diagram, boxes, **kwargs):
    result = diagram
    for box in boxes:
        result = collapse_box(result, box, **kwargs)
    return result


def expand_passages(diagram, *, original_diagram, provenance, max_crossings=10000):
    """Insert above/below passage arcs into an explicitly expanded tangle.

    Existing braid geometry is used only during explicit expansion. Passage
    crossings are determined by the stored layer, never by the label's sign.
    Outside-edge lineage survives all splits for orientation reconstruction.
    """
    from .twist_boxes import box_port_points
    pieces, overrides = [], {}
    for record in provenance:
        box = record['box']
        coordinates = box_port_points(box)
        for index, passage in enumerate(box.get('passages', [])):
            nodes = []
            for end, port in enumerate(passage['ports']):
                old_ref = box['ports'][port]
                node = ('join', 'passage', box['id'], index, end)
                found = []
                for edge in diagram['edges']:
                    for side, key in enumerate(('start', 'end')):
                        if edge.get(key) != [box['id'], port]:
                            continue
                        if any(old == old_ref['edge'] and side == (old_ref['end'] ^ relative)
                               for old, relative in edge.get('box_expansion_lineage', [])):
                            found.append((edge['id'], side))
                if len(found) != 1:
                    raise ValueError('A passage lost its original box attachment during expansion.')
                overrides[found[0]] = node; nodes.append(node)
            path = deepcopy(passage.get('points') or [coordinates[p] for p in passage['ports']])
            path[0], path[-1] = coordinates[passage['ports'][0]], coordinates[passage['ports'][1]]
            pieces.append({'nodes': nodes, 'points': path, 'lineage': [],
                           'passage': (box['id'], index), 'over': passage['over'], 'box': box})
    passage_count = len(pieces)
    for edge in diagram['edges']:
        nodes = []
        for end, key in enumerate(('start', 'end')):
            loc = edge.get(key)
            nodes.append(overrides.get((edge['id'], end), ('port', *loc) if loc is not None else ('join', 'closed', edge['id'])))
        pieces.append({'nodes': nodes, 'points': deepcopy(edge['points']),
                       'lineage': deepcopy(edge.get('box_expansion_lineage', []))})
    # A passage can meet the *projection* of a braid crossing. Shift its
    # interior very slightly to obtain a generic projection, keeping endpoints
    # and its over/under layer fixed. This does not change the represented link.
    vertices = deepcopy(diagram['crossings'])
    ordinary_points = np.asarray([v['point'] for v in vertices], float).reshape(-1, 2)
    for k in range(passage_count):
        piece = pieces[k]
        pts = np.asarray(piece['points'], float)
        if len(pts) == 2:
            pts = np.array([pts[0], (pts[0]+pts[1])/2, pts[1]])
        for attempt in range(6):
            contact = False
            for a, b in zip(pts, pts[1:]):
                v = b-a; norm = float(v@v)
                t = np.clip((ordinary_points-a)@v/max(norm, 1e-20), 0., 1.)
                if np.any(np.linalg.norm(ordinary_points-(a+t[:, None]*v), axis=1) < 1e-5):
                    contact = True
                if contact:
                    break
            if not contact:
                break
            axis = np.asarray(piece['box']['axis'])
            fraction = np.linspace(0, 1, len(pts))
            pts += np.sin(np.pi*fraction)[:, None]*axis*(.017*(attempt+1))
        else:
            raise ValueError('The passage projection coincides with a crossing; move it slightly.')
        piece['points'] = pts.tolist()
    events = [[] for _ in pieces]
    next_id = max((v['id'] for v in vertices), default=-1)+1
    crossing_over = {}
    arrays = [np.asarray(piece['points'], float) for piece in pieces]
    bounds = []
    for points in arrays:
        # intersection() accepts tiny endpoint extrapolations. Include their
        # coordinate-sized tolerance so the broad phase stays conservative.
        pad = np.abs(np.diff(points, axis=0))*1e-7+1e-12
        bounds.append((np.minimum(points[:-1], points[1:])-pad,
                       np.maximum(points[:-1], points[1:])+pad))
    for i in range(passage_count):
        a_points = arrays[i]
        arc_lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(a_points, axis=0), axis=1))]
        for j in range(i, len(pieces)):
            b_points = arrays[j]
            low, high = bounds[j]
            hits = []
            a_low, a_high = bounds[i]
            # Most explicit braid arcs occupy a tiny part of the panel.
            if np.any(high.max(axis=0) < a_low.min(axis=0)) or np.any(low.min(axis=0) > a_high.max(axis=0)):
                continue
            # Bound memory while doing the broad phase in numeric batches,
            # rather than one Python/numpy dispatch per passage sample.
            batch = max(1, 1000000//max(1, len(low)))
            for offset in range(0, len(a_low), batch):
                lower, upper = a_low[offset:offset+batch], a_high[offset:offset+batch]
                nearby = ((high[None, :, 0] >= lower[:, None, 0]) &
                          (high[None, :, 1] >= lower[:, None, 1]) &
                          (low[None, :, 0] <= upper[:, None, 0]) &
                          (low[None, :, 1] <= upper[:, None, 1]))
                ai_ids, bi_ids = np.nonzero(nearby)
                for ai, bi in zip(ai_ids+offset, bi_ids):
                    if i == j and bi <= ai+1:
                        continue
                    a, b = a_points[ai:ai+2]
                    c, d = b_points[bi:bi+2]
                    hit = intersection(a, b, c, d)
                    if hit is None:
                        continue
                    t, u, point = hit
                    pa, pb = ai+t, bi+u
                    if i == j and (np.interp(pb, np.arange(len(a_points)), arc_lengths)
                                   - np.interp(pa, np.arange(len(a_points)), arc_lengths)) < 1e-6:
                        continue  # Duplicate join samples do not create a crossing.
                    if min(pa, len(a_points)-1-pa, pb, len(b_points)-1-pb) < 1e-7:
                        # The joined boundary endpoints are ordinary joins.
                        continue
                    if any(np.linalg.norm(point-old[2]) < 1e-6 for old in hits):
                        continue
                    hits.append((pa, pb, point))
            for pa, pb, point in hits:
                over = i if pieces[i]['over'] else j
                over_parameter = None
                if j < passage_count:
                    a_key, b_key = pieces[i]['passage'], pieces[j]['passage']
                    if a_key[0] != b_key[0]:
                        raise ValueError('Overlapping twist boxes have ambiguous passage layers.')
                    pair = sorted((a_key[1], b_key[1]))
                    records = [r for r in pieces[i]['box'].get('passage_crossings', []) if sorted(r['pair']) == pair]
                    if not records:
                        raise ValueError('A crossing between box passages has no stored over/under order.')
                    record = min(records, key=lambda r: math.dist(r['point'], point))
                    if i == j:
                        parameters = [pa/(len(a_points)-1), pb/(len(a_points)-1)]
                        branch = min(range(2), key=lambda k: abs(parameters[k]-record['over_parameter']))
                        over_parameter = (pa, pb)[branch]
                        over = i
                    else:
                        over = i if record['over'] == a_key[1] else j
                if max_crossings is not None and len(vertices) >= max_crossings:
                    raise ValueError(f'Twist-box expansion exceeds the crossing limit ({max_crossings}), including passage crossings.')
                cid = next_id; next_id += 1
                vertices.append({'id': cid, 'point': point.tolist(), 'ports': [None]*4, 'over': 0})
                crossing_over[cid] = (over, over_parameter)
                events[i].append((pa, cid)); events[j].append((pb, cid))
    segments = []
    for i, piece in enumerate(pieces):
        points = np.asarray(piece['points'], float)
        cuts = [(0., piece['nodes'][0])]+[(t, ('cross', cid)) for t, cid in sorted(events[i])]+[(len(points)-1., piece['nodes'][1])]
        for (a, start), (b, end) in zip(cuts, cuts[1:]):
            if b-a < 1e-8:
                raise ValueError('The passage forms a triple crossing; move it slightly.')
            segments.append({'nodes': [start, end], 'points': _section(points, a, b),
                             'lineage': piece['lineage'], 'owner': i, 'parameters': (a, b)})
    rays = {}
    for i, segment in enumerate(segments):
        for end, node in enumerate(segment['nodes']):
            if node[0] != 'cross':
                continue
            path = np.asarray(segment['points'][::1 if end == 0 else -1], float)
            vector = next((p-path[0] for p in path[1:] if np.linalg.norm(p-path[0]) > 1e-7), None)
            if vector is None:
                raise ValueError('A passage has zero length at a crossing.')
            rays.setdefault(node[1], []).append((math.atan2(-vector[1], vector[0]), i, end))
    vertex_map = {v['id']: v for v in vertices}
    for cid, endpoints in rays.items():
        if len(endpoints) != 4:
            raise ValueError('A passage crossing must have exactly four branches.')
        over_ports = []
        for port, (_, i, end) in enumerate(sorted(endpoints)):
            segments[i]['nodes'][end] = ('port', cid, port)
            owner, parameter = crossing_over[cid]
            if (segments[i]['owner'] == owner and (parameter is None
                    or abs(segments[i]['parameters'][end]-parameter) < 1e-5)):
                over_ports.append(port)
        if len(over_ports) != 2 or (over_ports[1]-over_ports[0]) % 4 != 2:
            raise ValueError('The box passage is tangent to a strand.')
        vertex_map[cid]['over'] = over_ports[0] % 2
    return _graph_from_segments(diagram, vertices, segments)
