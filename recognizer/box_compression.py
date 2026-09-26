"""Certify and compress a circled braid tangle into a whole-bundle twist box.

The geometric front end accepts ordinary strands with two ordered boundary
bundles and a planar braid sweep. Curved strands need not be pointwise monotone.
It never guesses from crossing count or endpoint permutation. The algebraic
certificate uses Artin's faithful action on a free group, so cancellation,
braid relations and distant commutations do not require special templates.
See https://arxiv.org/abs/math/0203167 for a proof of faithfulness.

All geometry outside the fitted rectangle is retained. The rectangle must
lie wholly inside the user's lasso, and its boundary must avoid crossings.
Non-braid tangles, closed interior components, ambiguous geometry and work
budget exhaustion are rejected without modifying the source diagram.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .box_geometry import _boundary_cuts, _inside
from .geometry import _section, intersection
from .twist_boxes import box_corners, bundle_ports, make_twist_box, twist_label


class TwistCertificateLimit(ValueError):
    """A conservative computational limit; this does not mean non-equivalence."""


def _inverse(word):
    return [-letter for letter in reversed(word)]


def _reduce(parts, budget):
    result = []
    for part in parts:
        for letter in part:
            budget['work'] += 1
            if budget['work'] > budget['max_work']:
                raise TwistCertificateLimit('The exact twist certificate exceeded its work budget; select a smaller tangle.')
            if result and result[-1] == -letter:
                result.pop()
            else:
                result.append(letter)
            if len(result) > budget['max_letters']:
                raise TwistCertificateLimit('The exact twist certificate grew too large; select a smaller tangle.')
    return result


def artin_action(strands, word, *, max_letters=50000, max_work=2000000, max_word=4096):
    """Return the exact freely reduced images of x_1,...,x_m under a braid.

    Letters are signed one-based adjacent generator indices. Accumulating
    composition by updating only two images makes ordinary twist words cheap.
    This is a faithful representation, not a polynomial or matrix invariant.
    """
    if isinstance(strands, bool) or not isinstance(strands, int) or not 2 <= strands <= 64:
        raise ValueError('The certificate supports between 2 and 64 strands.')
    word = list(word)
    if len(word) > max_word:
        raise TwistCertificateLimit('The selected tangle has too many crossings for the exact certificate.')
    if any(isinstance(g, bool) or not isinstance(g, int) or not 1 <= abs(g) < strands for g in word):
        raise ValueError('A braid word contains an invalid adjacent generator.')
    budget = dict(work=0, max_work=max_work, max_letters=max_letters)
    images = [[i+1] for i in range(strands)]
    size = strands
    for generator in word:
        index = abs(generator)-1
        left, right = images[index], images[index+1]
        if generator > 0:
            a, b = _reduce((left, right, _inverse(left)), budget), left
        else:
            a, b = right, _reduce((_inverse(right), left, right), budget)
        size += len(a)+len(b)-len(left)-len(right)
        if size > max_letters:
            raise TwistCertificateLimit('The exact twist certificate grew too large; select a smaller tangle.')
        images[index], images[index+1] = a, b
    return tuple(tuple(image) for image in images)


def half_twist_word(strands, half_twists=1, *, max_word=4096):
    """Canonical signed Delta_m power in the convention used by twist_boxes."""
    if isinstance(half_twists, bool) or not isinstance(half_twists, int):
        raise ValueError('half_twists must be an integer.')
    if isinstance(strands, bool) or not isinstance(strands, int) or not 2 <= strands <= 64:
        raise ValueError('The certificate supports between 2 and 64 strands.')
    if abs(half_twists)*strands*(strands-1)//2 > max_word:
        raise TwistCertificateLimit('The selected tangle has too many crossings for the exact certificate.')
    word = [lane+1 for stop in range(strands-1, 0, -1) for lane in range(stop)]
    if half_twists < 0:
        word = [-generator for generator in reversed(word)]
    return word*abs(half_twists)


def certify_twist_word(strands, word, **limits):
    """Identify Delta_m^k exactly, or raise rather than replacing another braid."""
    word = list(word)
    # Validate letters even if the inexpensive exponent-sum filter rejects.
    if isinstance(strands, bool) or not isinstance(strands, int) or not 2 <= strands <= 64:
        raise ValueError('The certificate supports between 2 and 64 strands.')
    if any(isinstance(g, bool) or not isinstance(g, int) or not 1 <= abs(g) < strands for g in word):
        raise ValueError('A braid word contains an invalid adjacent generator.')
    if len(word) > limits.get('max_word', 4096):
        raise TwistCertificateLimit('The selected tangle has too many crossings for the exact certificate.')
    exponent = sum(1 if g > 0 else -1 for g in word)
    pairs = strands*(strands-1)//2
    if exponent % pairs:
        raise ValueError('The selected braid is not a whole-bundle half/full twist: its signed crossing sum disagrees.')
    half = exponent//pairs
    actual = artin_action(strands, word, **limits)
    expected = artin_action(strands, half_twist_word(strands, half, max_word=limits.get('max_word', 4096)), **limits)
    if actual != expected:
        raise ValueError('The selected braid is not equivalent to a whole-bundle half/full twist.')
    return dict(method='faithful_artin_action', half_twists=half, label=twist_label(half),
                strand_count=strands, braid_word=word, signed_crossing_sum=exponent,
                reduced_action_letters=sum(map(len, actual)))


def _polygon(polygon):
    points = np.asarray(polygon, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('Circle a finite, simple region of the diagram.')
    keep = [0]
    for index in range(1, len(points)):
        if np.linalg.norm(points[index]-points[keep[-1]]) > 1e-7:
            keep.append(index)
    points = points[keep]
    if len(points) > 1 and np.linalg.norm(points[0]-points[-1]) < 1e-7:
        points = points[:-1]
    if len(points) < 3 or abs(cv2.contourArea(points.astype(np.float32))) < 4:
        raise ValueError('Circle a region with enough room for a twist box.')
    if len(points) > 1024:
        raise ValueError('The selection outline is too intricate; draw a simpler circle.')
    for i, (a, b) in enumerate(zip(points, np.roll(points, -1, axis=0))):
        for j in range(i+2, len(points)):
            if i == 0 and j == len(points)-1:
                continue
            hit = intersection(a, b, points[j], points[(j+1) % len(points)])
            if hit is not None:
                raise ValueError('The selection outline crosses itself; draw a simple circle.')
            direction = b-a
            cross = lambda delta: direction[0]*delta[1]-direction[1]*delta[0]
            if (abs(cross(points[j]-a)) < 1e-7
                    and abs(cross(points[(j+1) % len(points)]-a)) < 1e-7):
                direction = direction/np.linalg.norm(direction)
                projected = [(point-a)@direction for point in (points[j], points[(j+1) % len(points)])]
                if min(np.linalg.norm(b-a), max(projected))-max(0., min(projected)) >= -1e-7:
                    raise ValueError('The selection outline overlaps itself; draw a simple circle.')
    return points


def _geometry_work(budget, amount=1):
    if budget is None:
        return
    budget['work'] += amount
    if budget['work'] > budget['limit']:
        raise TwistCertificateLimit('The selected geometry exceeded the local inspection budget; select a smaller tangle.')


def _trace_region(diagram, polygon, *, crossing_clearance=1e-5, budget=None):
    """Trace every interior arc between actual boundary cuts, without mutation."""
    vertices = {v['id']: v for v in diagram['crossings']}
    distances = {i: cv2.pointPolygonTest(polygon.astype(np.float32), tuple(map(float, v['point'])), True)
                 for i, v in vertices.items()}
    if any(abs(distance) <= crossing_clearance for distance in distances.values()):
        raise ValueError('A crossing is too close to the selection boundary; leave a wider collar.')
    inside = {i for i, distance in distances.items() if distance > 0}
    if any(vertices[i].get('kind') == 'twist_box' for i in inside):
        raise ValueError('Select ordinary strands; expand or combine existing boxes separately.')
    for vertex in vertices.values():
        if vertex.get('kind') != 'twist_box':
            continue
        other = np.asarray(box_corners(vertex), float)
        if (any(_inside(point, polygon, tolerance=-1e-6) for point in other)
                or any(_inside(point, other, tolerance=-1e-6) for point in polygon)
                or _boundary_cuts(np.vstack([other, other[0]]), polygon)):
            raise ValueError('The selection touches an existing box; select ordinary strands clear of that box.')
    segments, boundary = [], []
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    for edge in diagram['edges']:
        points = np.asarray(edge['points'], float)
        if np.any(points.max(axis=0) < low) or np.any(points.min(axis=0) > high):
            continue
        # Limit circle/segment intersections to samples near the selection;
        # a distant high-resolution part of a component need not be revisited.
        nearby = np.flatnonzero(np.all(np.maximum(points[:-1], points[1:]) >= low, axis=1)
                                & np.all(np.minimum(points[:-1], points[1:]) <= high, axis=1))
        _geometry_work(budget, len(nearby)*len(polygon))
        parameters = []
        for index in nearby:
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                hit = intersection(points[index], points[index+1], a, b)
                if hit is not None:
                    parameter = index+min(1., max(0., hit[0]))
                    if 1e-7 < parameter < len(points)-1-1e-7:
                        parameters.append(parameter)
        unique = []
        for value in sorted(parameters):
            if not unique or value-unique[-1] > 1e-6:
                unique.append(value)
        cuts = [0., *unique, float(len(points)-1)]
        seam = None
        if edge.get('closed') and edge.get('start') is None and abs(cv2.pointPolygonTest(
                polygon.astype(np.float32), tuple(map(float, points[0])), True)) < 1e-6:
            seam = ('boundary', len(boundary)); boundary.append(points[0].tolist())
        nodes = []
        for j, parameter in enumerate(cuts):
            if j in (0, len(cuts)-1):
                location = edge['start' if j == 0 else 'end']
                nodes.append(('port', *location) if location is not None else seam or ('join', edge['id']))
            else:
                nodes.append(('boundary', len(boundary)))
                boundary.append(_section(points, parameter, parameter)[0])
        for j, (a, b) in enumerate(zip(cuts, cuts[1:])):
            if b-a < 1e-8:
                continue
            if _inside(_section(points, (a+b)/2, (a+b)/2)[0], polygon):
                segments.append(dict(nodes=[nodes[j], nodes[j+1]], points=_section(points, a, b)))
    if not boundary:
        raise ValueError('The selection has no boundary strands; a twist box must be a tangle, not a closed component.')
    adjacent = {}
    for index, segment in enumerate(segments):
        for end, node in enumerate(segment['nodes']):
            adjacent.setdefault(tuple(node), []).append((index, end))
    trails, visited = [], set()
    for bid in range(len(boundary)):
        refs = adjacent.get(('boundary', bid), [])
        if len(refs) != 1:
            raise ValueError('A strand is tangent to the selection boundary; move the outline slightly.')
        if refs[0][0] in visited:
            continue
        cursor, path, crossings = refs[0], [], []
        while cursor[0] not in visited:
            index, end = cursor; visited.add(index)
            segment = segments[index]
            part = segment['points'][::1 if end == 0 else -1]
            path.extend(part if not path else part[1:])
            destination = tuple(segment['nodes'][1-end])
            if destination[0] == 'boundary':
                break
            if destination[0] == 'port':
                cid, port = destination[1:]
                if cid not in inside:
                    raise ValueError('A crossing touches the selection boundary.')
                crossings.append((cid, port))
                options = adjacent.get(('port', cid, (port+2) % 4), [])
            else:
                options = [ref for ref in adjacent.get(destination, []) if ref != (index, 1-end)]
            if len(options) != 1:
                raise ValueError('The selected strands do not form an unambiguous tangle.')
            cursor = options[0]
        else:
            raise ValueError('A closed strand lies entirely inside the selection.')
        trails.append(dict(ends=[bid, destination[1]], points=path, crossings=crossings))
    if len(visited) != len(segments):
        raise ValueError('The selection contains a closed internal component.')
    if not 2 <= len(trails) <= 64:
        raise ValueError('Select a bundle of between 2 and 64 strands.')
    return boundary, trails, inside


def _candidate_axes(polygon, trails, vertices, axis=None):
    if axis is not None:
        candidates = [axis]
    else:
        candidates = [[0., 1.], [1., 0.]]
        for data in (polygon, np.asarray([v['point'] for v in vertices])):
            if len(data) >= 2:
                _, _, vectors = np.linalg.svd(data-data.mean(axis=0), full_matrices=False)
                candidates.extend(vectors)
        for trail in trails:
            candidates.append(np.asarray(trail['points'][-1])-trail['points'][0])
        candidates.extend([math.sin(angle), math.cos(angle)] for angle in np.arange(0., math.pi, math.pi/12))
    result = []
    for candidate in candidates:
        vector = np.array(candidate, dtype=float, copy=True)
        if vector.shape != (2,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-7:
            if axis is not None:
                raise ValueError('The braid axis must be a finite nonzero planar vector.')
            continue
        vector /= np.linalg.norm(vector)
        if vector[1] < -1e-9 or (abs(vector[1]) < 1e-9 and vector[0] < 0):
            vector = -vector
        if not any(abs(float(vector@old)) > 1-1e-8 for old in result):
            result.append(vector)
    return result


def _level_parameter(points, heights, level):
    hits = []
    for index, (a, b) in enumerate(zip(heights, heights[1:])):
        if abs(b-a) < 1e-9:
            if abs(level-a) < 1e-7 and np.linalg.norm(points[index+1]-points[index]) > 1e-6:
                raise ValueError('The box boundary would be tangent to a strand.')
            continue
        fraction = (level-a)/(b-a)
        if -1e-8 <= fraction <= 1+1e-8:
            parameter = index+max(0., min(1., fraction))
            if not hits or abs(parameter-hits[-1]) > 1e-6:
                hits.append(parameter)
    if len(hits) != 1:
        raise ValueError('The selected strands turn back across a proposed box boundary; straighten them or select a smaller region.')
    return hits[0]


def _rectangle_inside(rectangle, polygon):
    contour = polygon.astype(np.float32)
    if any(cv2.pointPolygonTest(contour, tuple(map(float, point)), True) <= 1e-5 for point in rectangle):
        return False
    return not _boundary_cuts(np.vstack([rectangle, rectangle[0]]), polygon)


def _fit_rectangle(diagram, polygon, source_trails, selected, axis, clearance, budget=None):
    right = np.array([axis[1], -axis[0]])
    paths = [np.asarray(trail['points'], float) for trail in source_trails]
    paths = [points if points[0]@axis < points[-1]@axis else points[::-1] for points in paths]
    lower = max(float(points[0]@axis) for points in paths)
    upper = min(float(points[-1]@axis) for points in paths)
    if upper-lower < 2*clearance:
        raise ValueError('The region does not have two opposite bundles of boundary strands.')
    if selected:
        levels = [np.asarray(vertex['point'])@axis for vertex in diagram['crossings'] if vertex['id'] in selected]
        first, last = min(levels), max(levels)
        if first-lower <= clearance or upper-last <= clearance:
            raise ValueError('Leave more room above and below the selected crossings for a box collar.')
        collar = max(clearance, min(20., .12*max(upper-lower, last-first)))
        bottom = last+min(collar, (upper-last)*.75)
        top = first-min(collar, (first-lower)*.75)
    else:
        top, bottom = lower+(upper-lower)*.25, upper-(upper-lower)*.25
    # A curved strand can cross a plane near the first/last crossing several
    # times. Move that plane outward into the collar, keeping every selected
    # crossing inside and cutting every material strand exactly once.
    def cuts_at(level, outer):
        last_error = None
        for fraction in (0., .35, .7, .9):
            value = level+(outer-level)*fraction
            try:
                cuts = [_level_parameter(p, p@axis, value) for p in paths]
                return value, cuts
            except ValueError as exc:
                last_error = exc
        raise last_error
    top, starts = cuts_at(top, lower+clearance*.5)
    bottom, ends = cuts_at(bottom, upper-clearance*.5)
    crops = []
    for points, a, b in zip(paths, starts, ends):
        if b <= a:
            raise ValueError('The selected strands are not in braid position along this axis.')
        crops.append(np.asarray(_section(points, a, b), float))
    left = min(float((crop@right).min()) for crop in crops)-clearance
    right_bound = max(float((crop@right).max()) for crop in crops)+clearance
    center = axis*((top+bottom)/2)+right*((left+right_bound)/2)
    box = make_twist_box(-1, center.tolist(), [right_bound-left, bottom-top], '0',
                         axis=axis.tolist(), strand_count=len(crops))
    rectangle = np.asarray(box_corners(box), float)
    if not _rectangle_inside(rectangle, polygon):
        raise ValueError('The selected region is too tight for an inscribed twist box; circle a wider area.')
    boundary, trails, actual = _trace_region(diagram, rectangle, crossing_clearance=clearance*.5, budget=budget)
    if actual != selected or len(trails) != len(source_trails) or len(boundary) != 2*len(crops):
        raise ValueError('The fitted box would include different strands or crossings; adjust the selection.')
    m = len(crops)
    top_ports, bottom_ports = bundle_ports(box)
    endpoints = {'top': [], 'bottom': []}
    for bid, point in enumerate(boundary):
        height = float(np.asarray(point)@axis)
        side = 'top' if abs(height-top) < 1e-5 else 'bottom' if abs(height-bottom) < 1e-5 else None
        if side is None:
            raise ValueError('A strand meets a side of the proposed box; choose a more parallel region.')
        endpoints[side].append((float(np.asarray(point)@right), bid, point))
    if any(len(endpoints[side]) != m for side in endpoints):
        raise ValueError('The proposed box does not have equal top and bottom bundles.')
    port_points, binding = [None]*(2*m), {}
    for side, ports in (('top', top_ports), ('bottom', bottom_ports)):
        ordered = sorted(endpoints[side])
        if any(b[0]-a[0] < 1e-5 for a, b in zip(ordered, ordered[1:])):
            raise ValueError('Two bundle endpoints coincide; move the selection boundary.')
        for port, (_, bid, point) in zip(ports, ordered):
            port_points[port] = list(point)
            binding[bid] = port
    box['port_points'] = port_points
    return box, trails, binding


def _check_projection(paths, visits, vertices, axis, budget=None):
    """A local monotone sweep checks that all geometric intersections are encoded."""
    expected = {}
    for cid, pair in visits.items():
        expected.setdefault(tuple(sorted(lane for lane, _ in pair)), []).append(cid)
    tolerance = 1e-4
    for i, a in enumerate(paths):
        ah = a@axis
        for j in range(i):
            b, found = paths[j], set()
            bh = b@axis
            ai = bi = 0
            allowed = expected.get((j, i), [])
            sites = {}
            for cid in allowed:
                point = vertices[cid]['point']
                key = tuple(round(coordinate/tolerance) for coordinate in point)
                sites.setdefault(key, []).append(cid)
            while ai < len(a)-1 and bi < len(b)-1:
                _geometry_work(budget)
                if min(ah[ai+1], bh[bi+1])+1e-7 >= max(ah[ai], bh[bi]):
                    hit = intersection(a[ai], a[ai+1], b[bi], b[bi+1])
                    if hit is not None:
                        x, y = (round(coordinate/tolerance) for coordinate in hit[2])
                        nearby = [cid for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                  for cid in sites.get((x+dx, y+dy), [])]
                        near = [cid for cid in nearby if math.dist(hit[2], vertices[cid]['point']) < tolerance]
                        if len(near) != 1:
                            raise ValueError('The selected geometry has an unrecorded or ambiguous strand intersection.')
                        found.add(near[0])
                    else:
                        # Parallel overlapping segments have no transverse
                        # intersection result and must never be erased silently.
                        av, bv = a[ai+1]-a[ai], b[bi+1]-b[bi]
                        if np.linalg.norm(av) > 1e-7 and np.linalg.norm(bv) > 1e-7:
                            cross = lambda u, v: float(u[0]*v[1]-u[1]*v[0])
                            if abs(cross(av, bv)) < 1e-7 and abs(cross(av, b[bi]-a[ai])) < 1e-7:
                                direction = av/np.linalg.norm(av)
                                endpoints = [(point-a[ai])@direction for point in (b[bi], b[bi+1])]
                                overlap = min(np.linalg.norm(av), max(endpoints))-max(0., min(endpoints))
                                if overlap > 1e-6:
                                    raise ValueError('Selected strands overlap in the drawing; separate them before compression.')
                if ah[ai+1] < bh[bi+1]-1e-8:
                    ai += 1
                elif bh[bi+1] < ah[ai+1]-1e-8:
                    bi += 1
                else:
                    ai += 1; bi += 1
            if set(allowed) != found:
                raise ValueError('A recorded crossing disagrees with the selected strand geometry.')


def _check_curved_projection(paths, sequences, visits, vertices, budget=None):
    """Verify the planar embedding, including self contacts and crossing rays.

    A bounded x sweep inspects overlapping segment boxes, rather than assuming
    that pixel samples progress along the proposed braid axis. The rotation of
    all four rays must agree with the stored counterclockwise port order.
    """
    tolerance = 1e-4
    rays, sites, found = {}, {}, set()
    for lane, crossings in sequences.items():
        points = paths[lane]
        for cid, incoming in crossings:
            _geometry_work(budget, len(points))
            indices = np.flatnonzero(np.linalg.norm(points-np.asarray(vertices[cid]['point']), axis=1) < 1e-7)
            if len(indices) != 1 or indices[0] in (0, len(points)-1):
                raise ValueError('A recorded crossing disagrees with the selected strand geometry.')
            index = int(indices[0])
            for port, delta in ((incoming, points[index-1]-points[index]),
                                ((incoming+2) % 4, points[index+1]-points[index])):
                rays.setdefault(cid, []).append((math.atan2(-delta[1], delta[0]), port))
    for cid, items in rays.items():
        ordered = sorted(items)
        if len(ordered) != 4 or any((b[1]-a[1]) % 4 != 1
                                   for a, b in zip(ordered, ordered[1:]+ordered[:1])):
            raise ValueError('A crossing has inconsistent strand geometry or port order.')
        angles = [item[0] for item in ordered]
        if min(np.diff(angles+[angles[0]+2*math.pi])) < 1e-7:
            raise ValueError('Selected strands are tangent at a crossing; separate them before compression.')
        lanes = tuple(sorted(lane for lane, _ in visits[cid]))
        cell = tuple(round(value/tolerance) for value in vertices[cid]['point'])
        sites.setdefault((lanes, cell), []).append(cid)
    segments = []
    for lane, points in enumerate(paths):
        for index, (a, b) in enumerate(zip(points, points[1:])):
            segments.append((np.minimum(a, b), np.maximum(a, b), lane, index, a, b))
    segments.sort(key=lambda s: s[0][0])
    active = []
    for item in segments:
        low, high, lane, index, a, b = item
        _geometry_work(budget, len(active)+1)
        active = [old for old in active if old[1][0]+tolerance >= low[0]]
        for other in active:
            olow, ohigh, olane, oindex, c, d = other
            if ohigh[1]+tolerance < low[1] or high[1]+tolerance < olow[1]:
                continue
            adjacent = lane == olane and abs(index-oindex) == 1
            hit = intersection(a, b, c, d)
            if hit is not None:
                if adjacent:
                    continue  # Ordinary polyline joint; two straight segments meet once.
                if lane == olane:
                    raise ValueError('The selected strand has an unrecorded self-intersection.')
                x, y = (round(value/tolerance) for value in hit[2])
                pair = tuple(sorted((lane, olane)))
                nearby = [cid for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                          for cid in sites.get((pair, (x+dx, y+dy)), [])
                          if math.dist(hit[2], vertices[cid]['point']) < tolerance]
                if len(nearby) != 1:
                    raise ValueError('The selected geometry has an unrecorded or ambiguous strand intersection.')
                found.add(nearby[0])
            else:
                av, bv = b-a, d-c
                cross = lambda u, v: float(u[0]*v[1]-u[1]*v[0])
                if abs(cross(av, bv)) < 1e-7 and abs(cross(av, c-a)) < 1e-7:
                    direction = av/np.linalg.norm(av)
                    ends = [(point-a)@direction for point in (c, d)]
                    overlap = min(np.linalg.norm(av), max(ends))-max(0., min(ends))
                    if overlap > 1e-6 or (overlap >= -1e-7 and not adjacent):
                        raise ValueError('Selected strands overlap or touch in the drawing; separate them before compression.')
        active.append(item)
    if set(visits) != found:
        raise ValueError('A recorded crossing disagrees with the selected strand geometry.')


def _braid_sweep(sequences, visits, endings, vertices, budget=None):
    """Read a planar braid from material order, independent of crossing heights.

    Only adjacent current lanes may cross. Their incoming ports must have the
    rotation of a braid crossing. Together with checked planar geometry and the
    fixed boundary order this retains the embedding, not just a Gauss word.
    Simultaneously available crossings involve disjoint strands and commute.
    """
    successors = {cid: set() for cid in visits}
    degree = dict.fromkeys(visits, 0)
    for crossings in sequences.values():
        for (a, _), (b, _) in zip(crossings, crossings[1:]):
            if b not in successors[a]:
                successors[a].add(b)
                degree[b] += 1
    ready = {cid for cid in visits if not degree[cid]}
    lanes, word = list(range(len(sequences))), []
    while ready:
        chosen = None
        positions = {lane: index for index, lane in enumerate(lanes)}
        for cid in sorted(ready):
            _geometry_work(budget)
            places = sorted((positions[lane], port) for lane, port in visits[cid])
            if places[1][0]-places[0][0] == 1 and (places[0][1]-places[1][1]) % 4 == 1:
                chosen = cid
                break
        if chosen is None:
            raise ValueError('The selected crossing order cannot be swept as a planar braid; straighten the bundle.')
        ready.remove(chosen)
        left = places[0][0]
        right_over = places[1][1] % 2 == vertices[chosen]['over']
        word.append((left+1)*(1 if right_over else -1))
        lanes[left], lanes[left+1] = lanes[left+1], lanes[left]
        for cid in successors[chosen]:
            degree[cid] -= 1
            if not degree[cid]:
                ready.add(cid)
    if len(word) != len(visits):
        raise ValueError('The selected tangle has cyclic crossing order, not a bundle braid.')
    if any(endings[material] != lane for lane, material in enumerate(lanes)):
        raise ValueError('The braid crossing sequence disagrees with its boundary permutation.')
    return word


def _word_from_trails(box, trails, binding, vertices, budget=None):
    top, bottom = bundle_ports(box)
    paths, sequences, endings = [None]*len(top), {}, {}
    visits, monotone = {}, True
    axis = np.asarray(box['axis'], float)
    for trail in trails:
        a, b = (binding[end] for end in trail['ends'])
        if a in top and b in bottom:
            lane, end = top.index(a), bottom.index(b)
            points, crossings = np.asarray(trail['points'], float), list(trail['crossings'])
        elif b in top and a in bottom:
            lane, end = top.index(b), bottom.index(a)
            points = np.asarray(trail['points'][::-1], float)
            crossings = [(cid, (port+2) % 4) for cid, port in reversed(trail['crossings'])]
        else:
            raise ValueError('Each selected strand must connect the top bundle to the bottom bundle.')
        if paths[lane] is not None:
            raise ValueError('The selected tangle repeats a top attachment.')
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
        points = points[keep]
        if np.any(np.diff(points@axis) < -1e-6):
            monotone = False
        right = np.array([axis[1], -axis[0]])
        previous_horizontal = 0.
        for delta in np.diff(points, axis=0):
            if abs(delta@axis) < 1e-7:
                across = float(delta@right)
                if across*previous_horizontal < -1e-10:
                    raise ValueError('A selected strand doubles back along itself; straighten it before compression.')
                previous_horizontal = across
            else:
                previous_horizontal = 0.
        paths[lane], sequences[lane], endings[lane] = points, crossings, end
        for cid, port in crossings:
            visits.setdefault(cid, []).append((lane, port))
    if any(path is None for path in paths):
        raise ValueError('The selected tangle is missing a bundle strand.')
    if any(len(pair) != 2 or pair[0][0] == pair[1][0] for pair in visits.values()):
        raise ValueError('The selected tangle contains a self-crossing or an extra passing strand, not a bundle braid.')
    if monotone:
        _check_projection(paths, visits, vertices, axis, budget=budget)
    else:
        _check_curved_projection(paths, sequences, visits, vertices, budget=budget)
    return _braid_sweep(sequences, visits, endings, vertices, budget=budget)


def certify_selected_bundle(box, trails, binding, vertices):
    """collapse_box callback: independently certify the actual region being removed."""
    budget = dict(work=0, limit=2000000)
    certificate = certify_twist_word(box['strand_count'], _word_from_trails(box, trails, binding, vertices, budget=budget))
    if certificate['half_twists'] != box['half_twists']:
        raise ValueError('The selected braid no longer agrees with the proposed twist label.')
    return certificate


def analyze_twist_region(diagram, polygon, *, axis=None, clearance=None, max_crossings=4096,
                         max_geometry_work=2000000):
    """Fit and certify a twist box inside a user-drawn lasso, without modifying ink.

    The returned proposal can be displayed for review. Failed attempts raise
    ValueError with a concrete reason. Geometric work is local to the lasso;
    algebraic work has explicit bounds independent of any existing box label.
    """
    polygon = _polygon(polygon)
    if clearance is None:
        from .render import line_width
        clearance = max(1.5, float(line_width(diagram)))
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError('Box boundary clearance must be positive and finite.')
    budget = dict(work=0, limit=max_geometry_work)
    _, source_trails, selected = _trace_region(diagram, polygon, crossing_clearance=clearance*.5, budget=budget)
    if len(selected) > max_crossings:
        raise TwistCertificateLimit('The selected tangle has too many crossings for the exact certificate.')
    vertices = {vertex['id']: vertex for vertex in diagram['crossings']}
    errors, algebra_error = [], None
    for candidate in _candidate_axes(polygon, source_trails, [vertices[cid] for cid in selected], axis):
        try:
            box, trails, binding = _fit_rectangle(diagram, polygon, source_trails, selected, candidate, clearance, budget=budget)
            word = _word_from_trails(box, trails, binding, vertices, budget=budget)
            if len(word) != len(selected):
                raise ValueError('Some selected crossings are not attached to the traced bundle.')
            try:
                certificate = certify_twist_word(box['strand_count'], word, max_word=max_crossings)
            except ValueError as exc:
                algebra_error = exc
                raise
            box['half_twists'], box['label'] = certificate['half_twists'], certificate['label']
            return dict(box=box, rectangle=box_corners(box), half_twists=box['half_twists'], label=box['label'],
                        strand_count=box['strand_count'], crossing_ids=sorted(selected), certificate=certificate,
                        geometry_work=budget['work'])
        except TwistCertificateLimit:
            raise
        except ValueError as exc:
            errors.append(str(exc))
    if algebra_error is not None:
        raise algebra_error
    reason = errors[0] if errors else 'No consistent braid direction was found.'
    raise ValueError(f'{reason} The selection needs two end bundles and a planar braid sweep; try a wider selection or straighten the bundle.')


def compress_twist_region(diagram, polygon, **options):
    """Replace a certified circled Delta_m power by a compact labeled box."""
    from .box_geometry import collapse_box
    proposal = analyze_twist_region(diagram, polygon, **options)
    result = collapse_box(diagram, proposal['box'], bundle_certificate=certify_selected_bundle,
                          attachment_tolerance=1e-4)
    result.pop('twist_box_edge_lineage', None)
    from .box_alignment import align_created_box
    old_ids = {vertex['id'] for vertex in diagram['crossings']}
    box_id = next(vertex['id'] for vertex in result['crossings']
                  if vertex.get('kind') == 'twist_box' and vertex['id'] not in old_ids)
    return align_created_box(result, box_id)
