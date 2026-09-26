"""Oriented classical link diagrams, independent of image recognition.

Ports at each crossing are in counterclockwise order in mathematical coordinates
(the display's y axis points down). Opposite ports are connected through the
crossing. Each graph edge is an arc between crossings; closed crossing-free
components are represented explicitly and do not contribute PD tuples.
"""
from copy import deepcopy
from collections import Counter
import math
import numpy as np


def copy_diagram(diagram, memo=None):
    """Independent value-geometry copy without recursive scalar bookkeeping.

    The optional deepcopy memo preserves sharing between a diagram and its
    enclosing recognition result/undo record. Point rows are numeric value
    records; all lists remain independent from the source.
    """
    if memo is None:
        memo = {}
    for item in diagram.get('edges', []) + diagram.get('spatial_curves', []):
        points = item.get('points')
        if isinstance(points, list) and id(points) not in memo:
            memo[id(points)] = [point.copy() if isinstance(point, list) else deepcopy(point, memo)
                                for point in points]
    return deepcopy(diagram, memo)


def _maps(diagram):
    return ({int(c['id']): c for c in diagram['crossings']},
            {int(e['id']): e for e in diagram['edges']})


def component_walks(diagram):
    """Return one directed traversal per component, as (edge_id, from_end)."""
    crossings, edges = _maps(diagram)
    used, walks = set(), []
    for eid, edge in edges.items():
        if eid in used:
            continue
        if edge.get('closed') and edge.get('start') is None:
            walks.append([(eid, 0)])
            used.add(eid)
            continue
        walk, dart = [], (eid, 0)
        first = dart
        for _ in range(len(edges) + 1):
            ei, direction = dart
            if ei in used:
                if dart != first:
                    raise ValueError('A strand traversal does not close consistently.')
                break
            used.add(ei)
            walk.append(dart)
            e = edges[ei]
            cid, port = e['end' if direction == 0 else 'start']
            vertex = crossings[cid]
            if vertex.get('kind') == 'twist_box':
                from .twist_boxes import transition_port
                next_port = transition_port(vertex, port)
            else:
                next_port = (port + 2) % 4
            opposite = vertex['ports'][next_port]
            dart = (int(opposite['edge']), int(opposite['end']))
        else:
            raise ValueError('Component traversal exceeded the number of edges.')
        walks.append(walk)
    return walks


def components(diagram):
    walks = component_walks(diagram)
    states = _component_crossing_states(diagram, walks)
    return [{'id': i, 'edges': [eid for eid, _ in walk], 'crossing_free': states[i]}
            for i, walk in enumerate(walks)]


def _component_crossing_states(diagram, walks):
    """True/False, or None when a compact passage needs its expanded PD.

    Nonzero bundles contain crossings on every strand.  Zero bundles do not.
    A passage whose boundary endpoints separate a bundle's endpoints must
    intersect that bundle, regardless of the integer coefficient. Remaining
    passage geometry can introduce optional pairs of crossings, so it must
    not be guessed from an arbitrary parity representative.
    """
    edge_component = {eid: ci for ci, walk in enumerate(walks) for eid, _ in walk}
    crossed, uncertain = set(), set()
    from .twist_boxes import bundle_ports, box_port_points, transition_port
    for vertex in diagram['crossings']:
        if vertex.get('kind') != 'twist_box':
            crossed.update(edge_component[port['edge']] for port in vertex['ports'])
            continue
        top, bottom = bundle_ports(vertex)
        if vertex['half_twists']:
            crossed.update(edge_component[vertex['ports'][p]['edge']] for p in top+bottom)
        passages = vertex.get('passages', [])
        if not passages:
            continue
        uncertain.update(edge_component[port['edge']] for port in vertex['ports'])
        for crossing in vertex.get('passage_crossings', []):
            crossed.update(edge_component[vertex['ports'][passages[index]['ports'][0]]['edge']]
                           for index in crossing['pair'])
        center = vertex['point']
        coordinates = box_port_points(vertex)
        order = sorted(range(len(coordinates)), key=lambda p: math.atan2(
            -(coordinates[p][1]-center[1]), coordinates[p][0]-center[0]))
        ranks = {p: rank for rank, p in enumerate(order)}
        n = len(order)
        for passage in passages:
            a, b = (ranks[p] for p in passage['ports'])
            def between(p):
                return 0 < (ranks[p]-a) % n < (b-a) % n
            for p in top:
                if between(p) != between(transition_port(vertex, p)):
                    crossed.add(edge_component[vertex['ports'][p]['edge']])
                    crossed.add(edge_component[vertex['ports'][passage['ports'][0]]['edge']])
    return [False if ci in crossed else None if ci in uncertain else True for ci in range(len(walks))]


def crossing_free_count(diagram, pd=None):
    """Number of components omitted by PD, without implicitly expanding boxes.

    Pass an already-computed PD to obtain the exact count for any compact
    diagram. Without PD, zero-box components are counted directly; an
    ambiguous passage returns None rather than an incorrect integer. This
    keeps validation and dragging independent of a large twist coefficient.
    """
    walks = component_walks(diagram)
    if pd is None:
        states = _component_crossing_states(diagram, walks)
        return None if None in states else sum(states)
    parent, occurrences = {}, Counter()
    def find(label):
        parent.setdefault(label, label)
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label
    for row in pd:
        if len(row) != 4:
            raise ValueError('PD crossings must contain four arc labels.')
        occurrences.update(row)
        for a, b in ((row[0], row[2]), (row[1], row[3])):
            parent[find(a)] = find(b)
    if any(count != 2 for count in occurrences.values()):
        raise ValueError('Each PD arc label must occur exactly twice.')
    represented = len({find(label) for label in parent})
    omitted = len(walks)-represented
    if omitted < 0:
        raise ValueError('The PD code has more components than the diagram.')
    return omitted


def assign_components(diagram):
    _, edges = _maps(diagram)
    for i, walk in enumerate(component_walks(diagram)):
        for eid, _ in walk:
            edges[eid]['component'] = i
    return diagram


def pd_code(diagram):
    """SnapPy convention: incoming understrand, then counterclockwise.

    Labels start at 1 and increase while following each oriented component. Reversing any
    component updates both label order and the incoming-under choice. Component
    orientation is arbitrary on initial unoriented image import.
    """
    if any(c.get('kind') == 'twist_box' for c in diagram['crossings']):
        from .twist_boxes import expand_boxes
        return pd_code(expand_boxes(diagram))
    crossings, edges = _maps(diagram)
    walks = component_walks(diagram)
    labels, directions = {}, {}
    next_label = 1
    orientations = diagram.get('component_orientations', {})
    for ci, walk in enumerate(walks):
        reverse = orientations.get(str(ci), orientations.get(ci, 1)) == -1
        if reverse:
            walk = [(eid, 1 - d) for eid, d in reversed(walk)]
        for eid, direction in walk:
            directions[eid] = direction
            if edges[eid].get('closed') and edges[eid].get('start') is None:
                continue
            labels[eid] = next_label
            next_label += 1
    result = []
    for c in diagram['crossings']:
        ports = c['ports']
        under_parity = 1 - int(c.get('over', 1))
        candidates = [p for p in (under_parity, under_parity + 2)
                      if directions[int(ports[p]['edge'])] != int(ports[p]['end'])]
        if len(candidates) != 1:
            raise ValueError(f"Crossing {c['id']} has inconsistent understrand orientation.")
        start = candidates[0]
        result.append([labels[int(ports[(start + j) % 4]['edge'])] for j in range(4)])
    return result


def validate(diagram):
    """Validate incidence, traversal, PD multiplicity and planar rotation system."""
    errors = []
    try:
        for name in ('width', 'height'):
            value = diagram.get(name)
            if name in diagram and (not math.isfinite(value) or value <= 0):
                raise ValueError('Diagram width and height must be finite positive numbers.')
        crossings, edges = _maps(diagram)
        if len(crossings) != len(diagram['crossings']) or len(edges) != len(diagram['edges']):
            raise ValueError('Repeated edge or crossing identifiers.')
        refs = Counter()
        boxes = [c for c in crossings.values() if c.get('kind') == 'twist_box']
        for c in crossings.values():
            point = c.get('point', [])
            if len(point) != 2 or not all(math.isfinite(v) for v in point):
                raise ValueError('Crossing geometry must contain a finite planar point.')
            if c.get('kind') == 'twist_box':
                from .twist_boxes import validate_box
                validate_box(c)
            elif len(c['ports']) != 4 or c.get('over', 1) not in (0, 1):
                raise ValueError('Every classical crossing requires four ports and an over pair.')
            for p, port in enumerate(c['ports']):
                eid, end = int(port['edge']), int(port['end'])
                if end not in (0, 1) or eid not in edges:
                    raise ValueError('Invalid crossing port.')
                if edges[eid]['start' if end == 0 else 'end'] != [c['id'], p]:
                    raise ValueError('Crossing/edge incidence is not reciprocal.')
                refs[eid, end] += 1
        for e in edges.values():
            points=e['points']
            # A finite list can still contain scalar, empty, or XYZ rows.
            # Reject those before a JSON import reaches drawing/motion code
            # that requires exactly two numeric coordinates per point.
            finite=(points.ndim == 2 and points.shape[1] == 2
                    and points.dtype.kind in 'iuf' and np.isfinite(points).all()
                    if isinstance(points,np.ndarray) else all(
                        len(p) == 2 and all(math.isfinite(v) for v in p)
                        for p in points))
            if len(points) < 2 or not finite:
                raise ValueError('Edge geometry must contain at least two finite planar points.')
            if e.get('closed') and e.get('start') is None:
                if e.get('end') is not None:
                    raise ValueError('Invalid crossing-free component.')
                continue
            if refs[e['id'], 0] != 1 or refs[e['id'], 1] != 1:
                raise ValueError('Each arc must have exactly two incident ports.')
        walks = component_walks(diagram)
        if not boxes:
            pd = pd_code(diagram)
            multiplicities = Counter(a for x in pd for a in x)
            if any(n != 2 for n in multiplicities.values()) or len(multiplicities) != 2*len(pd):
                raise ValueError('Each PD arc label must occur twice.')
        # Euler characteristic of each connected crossing graph, using its
        # prescribed rotation system. This catches nonplanar endpoint matches.
        unseen = set(crossings)
        while unseen:
            seed = min(unseen)
            todo, vertices, component_edges = [seed], set(), set()
            while todo:
                cid = todo.pop()
                if cid in vertices:
                    continue
                vertices.add(cid)
                for port in crossings[cid]['ports']:
                    e = edges[int(port['edge'])]
                    component_edges.add(e['id'])
                    todo.extend([e['start'][0], e['end'][0]])
            unseen -= vertices
            darts = {(cid, p) for cid in vertices for p in range(len(crossings[cid]['ports']))}
            maximum_darts = len(darts)
            faces = 0
            while darts:
                first = dart = min(darts)
                for _ in range(maximum_darts+1):
                    if dart not in darts:
                        if dart != first:
                            raise ValueError('Invalid face traversal.')
                        break
                    darts.remove(dart)
                    cid, p = dart
                    port = crossings[cid]['ports'][p]
                    e = edges[int(port['edge'])]
                    next_crossing, next_port = e['end' if port['end'] == 0 else 'start']
                    dart = (next_crossing, (next_port + 1) % len(crossings[next_crossing]['ports']))
                faces += 1
            if len(vertices) - len(component_edges) + faces != 2:
                raise ValueError('Reconstructed crossing rotation system is not planar.')
        if boxes:
            crossing_states = _component_crossing_states(diagram, walks)
            return {'valid': True, 'errors': [], 'crossings': len(crossings)-len(boxes),
                    'twist_boxes': len(boxes), 'components': len(walks),
                    'crossing_free_components': None if None in crossing_states else sum(crossing_states),
                    'implicit_crossings': sum(abs(b['half_twists'])*b.get('strand_count', 2)*(b.get('strand_count', 2)-1)//2 for b in boxes)}
        return {'valid': True, 'errors': [], 'crossings': len(crossings),
                'components': len(walks), 'crossing_free_components': sum(
                    e.get('closed', False) and e.get('start') is None for e in edges.values())}
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(str(exc))
    return {'valid': False, 'errors': errors}


def switch_crossing(diagram, crossing_id):
    result = copy_diagram(diagram)
    crossings, _ = _maps(result)
    vertex = crossings[int(crossing_id)]
    if vertex.get('kind') == 'twist_box':
        from .twist_boxes import twist_label
        vertex['half_twists'] = -vertex['half_twists']
        vertex['label'] = twist_label(vertex)
    else:
        vertex['over'] = 1 - vertex.get('over', 1)
    return result


def reverse_component(diagram, component_id):
    result = copy_diagram(diagram)
    orientations = result.setdefault('component_orientations', {})
    key = str(component_id)
    previous = orientations.get(key, orientations.get(int(component_id), 1))
    orientations.pop(int(component_id), None)
    orientations[key] = -previous
    return result


def delete_component(diagram, component_id):
    """Remove a strand and splice the surviving arcs through mixed crossings."""
    result = copy_diagram(diagram)
    assign_components(result)
    if any(c.get('kind') == 'twist_box' for c in result['crossings']):
        raise ValueError('Expand twist boxes before deleting a component. Box strand attachments must be preserved.')
    crossings, edges = _maps(result)
    orientations = result.get('component_orientations', {})
    # Remember physical directions on the old geometry. Splicing can reverse
    # an arc's storage order, and renumbering can change a component's seed,
    # so merely copying its old +/- flag would not preserve its orientation.
    old_directions = {
        eid: direction ^ (orientations.get(str(ci), orientations.get(ci, 1)) == -1)
        for ci, walk in enumerate(component_walks(result))
        for eid, direction in walk
    }
    removed_edges = {eid for eid, e in edges.items() if e['component'] == int(component_id)}
    if not removed_edges:
        return result
    removed_crossings = {c['id'] for c in crossings.values()
                         if any(p['edge'] in removed_edges for p in c['ports'])}
    # Follow surviving edges through crossings that have lost their other strand.
    used, new_edges, new_directions = set(), [], []
    for eid, edge in edges.items():
        if eid in removed_edges or eid in used:
            continue
        if edge.get('closed'):
            used.add(eid)
            new_edges.append(deepcopy(edge))
            new_directions.append(old_directions[eid])
            continue
        start_dart = (eid, 0)
        # Prefer a retained endpoint for an open chain of spliced graph edges.
        for end in (0, 1):
            loc = edge['start' if end == 0 else 'end']
            if loc[0] not in removed_crossings:
                start_dart = (eid, end)
                break
        else:
            # Walk backwards to find a retained crossing, if there is one.
            dart, visited = (eid, 0), set()
            while dart not in visited:
                visited.add(dart)
                ee, end = dart
                loc = edges[ee]['start' if end == 0 else 'end']
                if loc[0] not in removed_crossings:
                    start_dart = dart
                    break
                pp = crossings[loc[0]]['ports'][(loc[1]+2)%4]
                dart = (pp['edge'], 1-pp['end'])
        dart = start_dart
        points = []
        first_edge, first_end = dart
        start = edges[first_edge]['start' if first_end == 0 else 'end']
        if start[0] in removed_crossings:
            start = None
        finish = None
        while dart[0] not in used:
            ee, direction = dart
            used.add(ee)
            e = edges[ee]
            pts = e['points'] if direction == 0 else list(reversed(e['points']))
            points.extend(pts if not points else pts[1:])
            loc = e['end' if direction == 0 else 'start']
            if loc[0] not in removed_crossings:
                finish = loc
                break
            pp = crossings[loc[0]]['ports'][(loc[1]+2)%4]
            dart = (pp['edge'], pp['end'])
        new_edges.append(dict(id=len(new_edges), points=points, start=start, end=finish,
                              closed=start is None, component=0))
        new_directions.append(old_directions[first_edge] ^ first_end)
    kept = [deepcopy(c) for c in result['crossings'] if c['id'] not in removed_crossings]
    cmap = {c['id']: c for c in kept}
    for i, e in enumerate(new_edges):
        e['id'] = i
        for end, key in enumerate(('start','end')):
            if e[key] is not None:
                cid, port = e[key]
                cmap[cid]['ports'][port] = {'edge': i, 'end': end}
    result.update(crossings=kept, edges=new_edges, component_orientations={})
    assign_components(result)
    for ci, walk in enumerate(component_walks(result)):
        eid, direction = walk[0]
        if new_directions[eid] != direction:
            result['component_orientations'][str(ci)] = -1
    return result
