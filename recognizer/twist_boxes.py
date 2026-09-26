"""Compact, oriented bundle twist boxes and explicit PD expansion.

A box is a vertex tagged ``kind='twist_box'``.  Its signed integer
``half_twists`` is twice the number printed in the box.  A label of 1/2 means
one whole-bundle half-twist: every pair of strands crosses once, for
m(m-1)/2 crossings.  Thus a two-strand 1/2 box contains one crossing.
No sampled internal strands are stored.  ``axis`` points from the top pair of
ports toward the bottom pair; ``size`` is (transverse width, axial length).
For two strands ports 0,1,2,3 are top-left, bottom-left, bottom-right, top-right.
For m strands they follow the same counterclockwise boundary order: top-left,
all bottom ports from left to right, remaining top ports from right to left.

Positive twists use the existing PD convention: with both strands directed
along the axis, the top-right strand goes over the top-left strand.  Reversing
one component changes its oriented crossing signs, not the box's handedness.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math
import re


def is_twist_box(vertex):
    return vertex.get('kind') == 'twist_box'


def parse_twist_label(value):
    """Return exact signed half-twist count; reject non-half-integral values."""
    if isinstance(value, bool):
        raise ValueError('A twist label must be an integer or half-integer.')
    text = str(value).strip().replace('\N{MINUS SIGN}', '-').replace(' ', '')
    if text in ('½', '+½', '-½'):
        text = ('-' if text.startswith('-') else '') + '1/2'
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/\d+)?', text):
        raise ValueError('A twist label must be an integer or half-integer.')
    try:
        amount = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError('A twist label must be an integer or half-integer.') from exc
    if amount.denominator not in (1, 2):
        raise ValueError('A twist label must be an integer or half-integer.')
    return int(2 * amount)


def twist_label(box_or_count):
    """Canonical full-twist label from a box or integer half-twist count."""
    count = box_or_count['half_twists'] if isinstance(box_or_count, dict) else box_or_count
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError('half_twists must be an integer.')
    return str(count // 2) if count % 2 == 0 else f'{count}/2'


def make_twist_box(id, point, size, label, axis=(0., 1.), ports=None, *, strand_count=2):
    if isinstance(strand_count, bool) or not isinstance(strand_count, int) or strand_count < 2:
        raise ValueError('A twist box must have an integer strand_count of at least two.')
    axis = [float(v) for v in axis]
    norm = math.hypot(*axis) if len(axis) == 2 else 0
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('A box axis must have a finite nonzero direction.')
    count = parse_twist_label(label)
    result = dict(id=int(id), kind='twist_box', point=[float(v) for v in point],
                  size=[float(v) for v in size], axis=[v / norm for v in axis],
                  half_twists=count, label=twist_label(count), strand_count=strand_count,
                  ports=deepcopy(ports) if ports is not None else [None] * (2 * strand_count))
    validate_box(result, require_ports=ports is not None)
    return result


def validate_box(box, *, require_ports=True):
    """Validate the constant-size representation without expanding any twists."""
    if not is_twist_box(box):
        raise ValueError('Expected a twist-box vertex.')
    for key in ('point', 'size', 'axis'):
        values = box.get(key, [])
        if len(values) != 2 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f'A twist box requires finite {key} coordinates.')
    if min(box['size']) <= 0:
        raise ValueError('Twist-box dimensions must be positive.')
    if abs(math.hypot(*box['axis']) - 1) > 1e-6:
        raise ValueError('A twist-box axis must be a unit vector.')
    count = box.get('half_twists')
    twist_label(count)
    if 'label' in box and parse_twist_label(box['label']) != count:
        raise ValueError('The printed twist label disagrees with half_twists.')
    m = box.get('strand_count', 2)
    if isinstance(m, bool) or not isinstance(m, int) or m < 2:
        raise ValueError('A twist box must have an integer strand_count of at least two.')
    top, bottom = bundle_ports(box)
    if len(top) != m or len(bottom) != m:
        raise ValueError('Bundle ports must describe both ends of every strand.')
    used = top + bottom
    for passage in box.get('passages', []):
        pair = passage.get('ports', [])
        if len(pair) != 2 or not isinstance(passage.get('over'), bool):
            raise ValueError('A box passage requires two ports and an over/under flag.')
        used += pair
    if (len(set(used)) != len(used) or sorted(used) != list(range(len(box.get('ports', []))))):
        raise ValueError('Every box port must belong to exactly one bundle strand or passage.')
    if 'port_points' in box:
        pts = box['port_points']
        if len(pts) != len(used) or any(len(p) != 2 or not all(math.isfinite(float(v)) for v in p) for p in pts):
            raise ValueError('Box attachment geometry must supply one finite point per port.')
    pts = box_port_points(box)
    tolerance = max(1e-5, max(box['size'])*1e-5)
    ux, uy = box['axis']
    def local(point):
        x, y = point[0]-box['point'][0], point[1]-box['point'][1]
        return x*uy-y*ux, x*ux+y*uy
    local_ports = [local(p) for p in pts]
    width, height = box['size']
    for across, along in local_ports:
        inside = abs(across) <= width/2+tolerance and abs(along) <= height/2+tolerance
        boundary = abs(abs(across)-width/2) <= tolerance or abs(abs(along)-height/2) <= tolerance
        if not inside or not boundary:
            raise ValueError('A box attachment must lie on the rectangle boundary.')
    for ports, sign in ((top, -1), (bottom, 1)):
        if any(abs(local_ports[p][1]-sign*height/2) > tolerance for p in ports):
            raise ValueError('Bundle endpoints must lie on opposite axial ends of the box.')
        if any(local_ports[b][0]-local_ports[a][0] <= tolerance for a, b in zip(ports, ports[1:])):
            raise ValueError('Bundle endpoints must be ordered from left to right.')
    passages = box.get('passages', [])
    for passage in passages:
        path = passage.get('points')
        if (not isinstance(path, (list, tuple)) or len(path) < 2
                or any(not isinstance(p, (list, tuple)) or len(p) != 2
                       or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p) for p in path)):
            raise ValueError('A box passage must have at least two finite planar points.')
        for end, port in enumerate(passage['ports']):
            if math.dist(path[0 if end == 0 else -1], pts[port]) > tolerance:
                raise ValueError('A passage endpoint disagrees with its boundary port.')
        if any(abs(local(p)[0]) > width/2+tolerance or abs(local(p)[1]) > height/2+tolerance for p in path):
            raise ValueError('A stored box passage must remain inside its rectangle.')
    for record in box.get('passage_crossings', []):
        pair = record.get('pair', [])
        point = record.get('point', [])
        if (len(pair) != 2 or any(isinstance(i, bool) or not isinstance(i, int)
                                 or not 0 <= i < len(passages) for i in pair)):
            raise ValueError('A passage crossing must refer to existing passages.')
        if len(point) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
            raise ValueError('A passage crossing requires a finite planar point.')
        if abs(local(point)[0]) > width/2+tolerance or abs(local(point)[1]) > height/2+tolerance:
            raise ValueError('A passage crossing must lie inside the box.')
        if pair[0] != pair[1]:
            if isinstance(record.get('over'), bool) or record.get('over') not in pair:
                raise ValueError('The overpassing strand must belong to the recorded crossing pair.')
        else:
            parameters = record.get('parameters', [])
            over = record.get('over_parameter')
            if (len(parameters) != 2 or any(not isinstance(t, (int, float)) or not math.isfinite(t)
                                           or not 0 <= t <= 1 for t in parameters)
                    or abs(parameters[0]-parameters[1]) < 1e-10
                    or not isinstance(over, (int, float)) or not math.isfinite(over)
                    or min(abs(over-t) for t in parameters) > 1e-9):
                raise ValueError('A self-crossing needs two distinct branch parameters and the overpassing branch.')
            path = passages[pair[0]]['points']
            for parameter in parameters:
                coordinate = parameter*(len(path)-1)
                index = min(int(coordinate), len(path)-2)
                fraction = coordinate-index
                location = [path[index][k]*(1-fraction)+path[index+1][k]*fraction for k in (0, 1)]
                if math.dist(location, point) > tolerance:
                    raise ValueError('A self-crossing parameter disagrees with its recorded point.')
        for index in set(pair):
            nearest = math.inf
            for a, b in zip(passages[index]['points'], passages[index]['points'][1:]):
                dx, dy = b[0]-a[0], b[1]-a[1]
                length2 = dx*dx+dy*dy
                fraction = 0 if length2 == 0 else max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length2))
                nearest = min(nearest, math.dist(point, [a[0]+fraction*dx, a[1]+fraction*dy]))
            if nearest > tolerance:
                raise ValueError('A crossing point does not lie on its recorded passages.')
    if require_ports:
        for port in box['ports']:
            if not isinstance(port, dict) or 'edge' not in port or port.get('end') not in (0, 1):
                raise ValueError('Every twist-box port must reference an edge end.')
    return box


def transition_port(vertex, port):
    """Continue a strand through either an ordinary crossing or a compact box."""
    if not is_twist_box(vertex):
        if port not in (0, 1, 2, 3):
            raise ValueError('A classical crossing port must be 0, 1, 2 or 3.')
        return (port + 2) % 4
    for passage in vertex.get('passages', []):
        a, b = passage['ports']
        if port == a:
            return b
        if port == b:
            return a
    top, bottom = bundle_ports(vertex)
    for source, target in ((top, bottom), (bottom, top)):
        if port in source:
            lane = source.index(port)
            return target[len(source)-1-lane if vertex['half_twists'] % 2 else lane]
    raise ValueError('Port is not part of the twist box.')


def bundle_ports(box):
    """Return top/bottom port indices in transverse left-to-right lane order."""
    explicit = box.get('bundle_ports')
    if explicit is not None:
        return list(explicit['top']), list(explicit['bottom'])
    m = box.get('strand_count', 2)
    return [0] + list(range(2*m-1, m, -1)), list(range(1, m+1))


def certify_proxy_bundle(box, trails, binding, vertices):
    """Certify the fixed signed parity braid before discarding proxy geometry.

    ``trails`` and ``binding`` are the boundary traces produced by
    box_geometry.collapse_box.  Orienting each trace from its top port labels
    actual material strands unambiguously.  Crossing sequences are compared
    to the exact signed Delta_m word used by expand_boxes(proxy=True).
    Intersections with passing strands are excluded from this braid signature.
    The work depends on strand count and proxy size, never the stored integer.
    """
    top, _ = bundle_ports(box)
    records, visits = {}, {}
    for trail in trails:
        a, b = (binding[end] for end in trail['ends'])
        if a in top:
            lane = top.index(a)
            crossings = list(trail['crossings'])
        else:
            lane = top.index(b)
            crossings = [(cid, (port+2) % 4) for cid, port in reversed(trail['crossings'])]
        if lane in records:
            raise ValueError('The bundle repeats a top attachment.')
        records[lane] = crossings
        for cid, port in crossings:
            visits.setdefault(cid, []).append((lane, port))
    if len(records) != len(top):
        raise ValueError('The bundle is missing a top attachment.')
    braid_crossings = {cid: pair for cid, pair in visits.items() if len(pair) >= 2}
    m = len(top)
    parity = abs(box['half_twists']) % 2
    if len(braid_crossings) != parity*m*(m-1)//2:
        raise ValueError('The bundle inside this box no longer matches its fixed twist proxy.')
    actual = [[] for _ in range(m)]
    for lane, sequence in records.items():
        for cid, port in sequence:
            if cid not in braid_crossings:
                continue
            pair = braid_crossings[cid]
            if len(pair) != 2 or pair[0][0] == pair[1][0]:
                raise ValueError('A proxy bundle strand acquired an internal self-crossing.')
            other, other_port = next(item for item in pair if item[0] != lane)
            over = port % 2 == vertices[cid]['over']
            incoming_over, incoming_under = (port, other_port) if over else (other_port, port)
            sign = 1 if (incoming_over-incoming_under) % 4 == 3 else -1
            actual[lane].append((other, over, sign))
    expected = [[] for _ in range(m)]
    if parity:
        sign = 1 if box['half_twists'] > 0 else -1
        word = [lane for stop in range(m-1, 0, -1) for lane in range(stop)]
        if sign < 0:
            word.reverse()
        material = list(range(m))
        for lane in word:
            a, b = material[lane:lane+2]
            expected[a].append((b, sign < 0, sign))
            expected[b].append((a, sign > 0, sign))
            material[lane], material[lane+1] = b, a
    if actual != expected:
        raise ValueError('The bundle crossing order or handedness differs from its fixed twist proxy.')
    return True


def _position(box, across, along):
    x, y = box['point']
    ux, uy = box['axis']
    return [x + across * uy + along * ux, y - across * ux + along * uy]


def box_corners(box):
    """Rectangle corners in port order; edges connect at its axial ends."""
    w, h = box['size']
    return [_position(box, sx * w / 2, sy * h / 2)
            for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))]


def box_port_points(box):
    """Attachment points, inset from corners, in the vertex's boundary order."""
    if 'port_points' in box:
        return deepcopy(box['port_points'])
    w, h = box['size']
    m = box.get('strand_count', 2)
    top, bottom = bundle_ports(box)
    points = [None] * len(box['ports'])
    for lane, (a, b) in enumerate(zip(top, bottom)):
        across = w * .6 * (lane / (m-1) - .5)
        points[a], points[b] = _position(box, across, -h/2), _position(box, across, h/2)
    if any(p is None for p in points):
        raise ValueError('A box with extra passages must provide explicit port_points.')
    return points


def compact_component_walks(diagram):
    """Directed edge walks with correct even/odd box connectivity, O(E+V)."""
    vertices = {int(v['id']): v for v in diagram['crossings']}
    edges = {int(e['id']): e for e in diagram['edges']}
    used, walks = set(), []
    for eid, edge in edges.items():
        if eid in used:
            continue
        if edge.get('closed') and edge.get('start') is None:
            used.add(eid)
            walks.append([(eid, 0)])
            continue
        first = dart = (eid, 0)
        walk = []
        for _ in range(len(edges) + 1):
            current, direction = dart
            if current in used:
                if dart != first:
                    raise ValueError('A strand traversal does not close consistently.')
                break
            used.add(current)
            walk.append(dart)
            vid, port = edges[current]['end' if direction == 0 else 'start']
            vertex = vertices[vid]
            target = vertex['ports'][transition_port(vertex, port)]
            dart = (int(target['edge']), int(target['end']))
        else:
            raise ValueError('Component traversal exceeded the number of edges.')
        walks.append(walk)
    return walks


def _clean_points(points):
    result = []
    for point in points:
        p = [float(v) for v in point[:2]]
        if not result or math.dist(p, result[-1]) > 1e-10:
            result.append(p)
    return result if len(result) > 1 else result * 2


def _splice_zero_boxes(vertices, edges, zero, lineage):
    """Splice even-parity identity tangles without leaving artificial vertices."""
    vertex_map = {v['id']: v for v in vertices}
    edge_map = {e['id']: e for e in edges}
    def is_zero_bundle(loc):
        if loc[0] not in zero:
            return False
        top, bottom = bundle_ports(vertex_map[loc[0]])
        return loc[1] in top + bottom

    used, result, result_lineage = set(), [], {}
    for eid, edge in edge_map.items():
        if eid in used:
            continue
        if edge.get('closed') and edge.get('start') is None:
            used.add(eid)
            result.append(edge)
            result_lineage[eid] = lineage[eid]
            continue
        first = (eid, 0)
        # Walk backwards until an ordinary vertex, or around a zero-only circle.
        dart, visited = first, set()
        while dart not in visited:
            visited.add(dart)
            current, end = dart
            loc = edge_map[current]['start' if end == 0 else 'end']
            if not is_zero_bundle(loc):
                first = dart
                break
            target = vertex_map[loc[0]]['ports'][transition_port(vertex_map[loc[0]], loc[1])]
            dart = (target['edge'], 1 - target['end'])
        current, direction = first
        start = edge_map[current]['start' if direction == 0 else 'end']
        if is_zero_bundle(start):
            start = None
        points, witnesses, finish = [], [], None
        while current not in used:
            used.add(current)
            current_edge = edge_map[current]
            path = current_edge['points'] if direction == 0 else list(reversed(current_edge['points']))
            points.extend(path)
            witnesses.extend((old, relative ^ direction) for old, relative in lineage[current])
            loc = current_edge['end' if direction == 0 else 'start']
            if not is_zero_bundle(loc):
                finish = loc
                break
            box = vertex_map[loc[0]]
            point_ports = box_port_points(box)
            opposite = transition_port(box, loc[1])
            points.extend([point_ports[loc[1]], point_ports[opposite]])
            target = box['ports'][opposite]
            current, direction = target['edge'], target['end']
        new = deepcopy(edge_map[first[0]])
        new.update(id=first[0], start=start, end=finish, closed=start is None,
                   points=_clean_points(points))
        if start is None and math.dist(new['points'][0], new['points'][-1]) > 1e-10:
            new['points'].append(new['points'][0].copy())
        result.append(new)
        result_lineage[new['id']] = witnesses
    return [v for v in vertices if v['id'] not in zero], result, result_lineage


def expand_boxes(diagram, *, proxy=False, max_crossings=10000, keep_lineage=False,
                 box_ids=None):
    """Return an ordinary diagram for explicit PD export or a parity proxy.

    ``proxy=True`` replaces each box with its signed parity (zero or one
    bundle half-twist), preserving boundary connectivity but NOT its link type. It is
    intended only for box-aware geometric editing.  Never export that proxy's
    PD as the original link.  Full expansion allocates one crossing per pair of
    strands per half-twist, so the budget is checked before allocation.

    ``twist_box_expansion`` records each original box, boundary attachments and
    emitted IDs.  Geometry rebuilders must preserve/recover those boundary
    anchors before recompressing; IDs alone cannot identify a moved tangle.

    ``box_ids`` optionally selects a subset to expand. Other compact boxes and
    their possibly enormous coefficients remain compact.
    """
    boxes = [v for v in diagram['crossings'] if is_twist_box(v)]
    if box_ids is not None:
        selected = set(box_ids)
        if selected - {b['id'] for b in boxes}:
            raise ValueError('The selected twist box does not exist.')
        boxes = [b for b in boxes if b['id'] in selected]
    if not boxes:
        return deepcopy(diagram)
    for box in boxes:
        validate_box(box)
    counts = {b['id']: ((abs(b['half_twists']) % 2) * (1 if b['half_twists'] >= 0 else -1)
                        if proxy else b['half_twists']) for b in boxes}
    total = sum(not is_twist_box(v) for v in diagram['crossings']) + sum(
        abs(counts[b['id']]) * b.get('strand_count', 2) * (b.get('strand_count', 2)-1) // 2
        for b in boxes)
    if max_crossings is not None and total > max_crossings:
        raise ValueError(f'Twist-box expansion needs {total} crossings; the limit is {max_crossings}.')
    orientations = diagram.get('component_orientations', {})
    old_directions = {eid: direction ^ (orientations.get(str(ci), orientations.get(ci, 1)) == -1)
                      for ci, walk in enumerate(compact_component_walks(diagram)) for eid, direction in walk}
    out = deepcopy(diagram)
    out.pop('spatial_curves', None)
    out.pop('spatial_fingerprint', None)
    vertices = [v for v in out['crossings'] if v['id'] not in counts or counts[v['id']] == 0]
    edges = out['edges']
    lineage = {e['id']: [(e['id'], 0)] for e in edges}
    next_vertex = max((v['id'] for v in diagram['crossings']), default=-1) + 1
    next_edge = max((e['id'] for e in edges), default=-1) + 1
    substitutions, anchors, provenance = {}, {}, []
    for box in boxes:
        count, port_points = counts[box['id']], box_port_points(box)
        entry = dict(box=deepcopy(box), proxy=bool(proxy), expanded_half_twists=count,
                     crossing_ids=[], internal_edge_ids=[], boundary=[
                         dict(port=p, point=port_points[p], edge=ref['edge'], end=ref['end'])
                         for p, ref in enumerate(box['ports'])])
        provenance.append(entry)
        if count == 0:
            continue
        m = box.get('strand_count', 2)
        half_word = [lane for stop in range(m-1, 0, -1) for lane in range(stop)]
        if count < 0:
            half_word.reverse()
        word = half_word * abs(count)
        crossing_ids = [box['id']] + list(range(next_vertex, next_vertex + len(word) - 1))
        next_vertex += len(word) - 1
        entry['crossing_ids'] = crossing_ids
        length = box['size'][1]
        # A straight collar keeps endpoint tangents independent of the last
        # generator. Without it a cubic editor fit rounds a diagonal exactly
        # at the box boundary and slowly moves a supposedly fixed attachment.
        collar = min(6., length*.1)
        # All measured lanes normalize at one common axial plane. Routing
        # each uneven port directly to its own first generator creates long
        # diagonals through neighboring strands (and unrecorded crossings).
        # An additional collar strip interpolates the ordered boundary lanes
        # to canonical lane coordinates before any braid generator starts.
        braid_length = length-4*collar
        braid_start = -length/2+2*collar
        lane_x = [box['size'][0] * .6 * (lane / (m-1) - .5) for lane in range(m)]
        positions = [_position(box, (lane_x[lane] + lane_x[lane+1])/2,
                               braid_start + braid_length * (j + .5) / len(word))
                     for j, lane in enumerate(word)]
        for j, vid in enumerate(crossing_ids):
            vertices.append(dict(id=vid, point=positions[j], ports=[None] * 4,
                                 over=1 if count > 0 else 0,
                                 twist_box_origin=box['id'], twist_box_index=j))
        top, bottom = bundle_ports(box)
        tails = [None] * m
        tail_steps = [None] * m
        step = braid_length / len(word)
        for j, lane in enumerate(word):
            for incoming_lane, upper, lower in ((lane, 0, 1), (lane+1, 3, 2)):
                vid = crossing_ids[j]
                if tails[incoming_lane] is None:
                    p = top[incoming_lane]
                    substitutions[box['id'], p] = [vid, upper]
                    # Each untouched lane travels vertically before entering
                    # its first crossing; otherwise a long diagonal could
                    # intersect unrelated braid generators.
                    entry_point = _position(box, lane_x[incoming_lane],
                                            braid_start + step*j)
                    collar_point = [port_points[p][k] + collar*box['axis'][k] for k in (0, 1)]
                    normalized_point = _position(box, lane_x[incoming_lane], braid_start)
                    anchors[box['id'], p] = [port_points[p], collar_point, normalized_point,
                                            entry_point, positions[j]]
                else:
                    previous_j = tail_steps[incoming_lane]
                    exit_point = _position(box, lane_x[incoming_lane],
                                           braid_start + step*(previous_j+1))
                    entry_point = _position(box, lane_x[incoming_lane], braid_start + step*j)
                    edges.append(dict(id=next_edge, start=tails[incoming_lane],
                                      end=[vid, upper], closed=False,
                                      points=_clean_points([positions[previous_j], exit_point,
                                                            entry_point, positions[j]]),
                                      twist_box_origin=box['id']))
                    lineage[next_edge] = []
                    entry['internal_edge_ids'].append(next_edge)
                    next_edge += 1
                tails[incoming_lane] = [vid, lower]
                tail_steps[incoming_lane] = j
        for lane, p in enumerate(bottom):
            substitutions[box['id'], p] = tails[lane]
            exit_point = _position(box, lane_x[lane], braid_start + step*(tail_steps[lane]+1))
            collar_point = [port_points[p][k] - collar*box['axis'][k] for k in (0, 1)]
            normalized_point = _position(box, lane_x[lane], braid_start+braid_length)
            anchors[box['id'], p] = [port_points[p], collar_point, normalized_point,
                                    exit_point, positions[tail_steps[lane]]]
    # Only original external edges refer to compact boxes. Internal generated
    # vertices may reuse the first box ID and must not be substituted again.
    for edge in edges[:len(diagram['edges'])]:
        points = [list(point) for point in edge['points']]
        for end, key in enumerate(('start', 'end')):
            loc = edge.get(key)
            if loc is None or tuple(loc) not in substitutions:
                continue
            route = anchors[tuple(loc)]
            points = list(reversed(route)) + points if end == 0 else points + route
            edge[key] = substitutions[tuple(loc)]
        edge['points'] = _clean_points(points)
    zero = {bid for bid, count in counts.items() if count == 0}
    if zero:
        vertices, edges, lineage = _splice_zero_boxes(vertices, edges, zero, lineage)
    out.update(crossings=vertices, edges=edges)
    if any(b.get('passages') for b in boxes):
        from .box_geometry import expand_passages
        for edge in edges:
            edge['box_expansion_lineage'] = lineage[edge['id']]
        changed = expand_passages(out, original_diagram=diagram, provenance=provenance,
                                  max_crossings=max_crossings)
        if changed is not None:
            out = changed
        vertices, edges = out['crossings'], out['edges']
        lineage = {e['id']: e.pop('box_expansion_lineage', []) for e in edges}
    vertex_map = {v['id']: v for v in vertices}
    for edge in edges:
        for end, key in enumerate(('start', 'end')):
            if edge.get(key) is not None:
                vid, p = edge[key]
                vertex_map[vid]['ports'][p] = dict(edge=edge['id'], end=end)
    out.update(crossings=vertices, edges=edges, component_orientations={},
               twist_box_expansion=provenance)
    edge_map = {e['id']: e for e in edges}
    for ci, walk in enumerate(compact_component_walks(out)):
        selected = None
        for eid, direction in walk:
            if lineage[eid]:
                old, relative = lineage[eid][0]
                selected = old_directions[old] ^ relative ^ direction
                break
        if selected is None:
            raise ValueError('An expanded component has no original boundary-edge witness.')
        if selected:
            out['component_orientations'][str(ci)] = -1
        for eid, _ in walk:
            edge_map[eid]['component'] = ci
    if keep_lineage:
        out['twist_box_edge_lineage'] = {str(eid): witnesses for eid, witnesses in lineage.items()}
    return out
