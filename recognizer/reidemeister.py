"""Explicit Reidemeister I curls and geometrically empty monogon regions.

A curl is a smooth immersed strand with one transverse crossing, not a literal
nonregular cusp. All operations return a copy and preserve physical component
orientations. Crossings use the same right-handed sign convention as PD export.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .diagram import assign_components, component_walks, validate, copy_diagram
from .geometry import check_embedding, intersection
from .render import edge_points, line_width
from .box_exterior_embedding import check_exterior_embedding
from .twist_boxes import box_corners, is_twist_box


def _directions(diagram):
    orientations = diagram.get('component_orientations', {})
    return {eid: direction ^ (orientations.get(str(ci), orientations.get(ci, 1)) == -1)
            for ci, walk in enumerate(component_walks(diagram)) for eid, direction in walk}


def _finish(diagram, directions):
    cmap = {c['id']: c for c in diagram['crossings']}
    for edge in diagram['edges']:
        for end, key in enumerate(('start', 'end')):
            loc = edge.get(key)
            if loc is not None:
                cmap[loc[0]]['ports'][loc[1]] = {'edge': edge['id'], 'end': end}
    assign_components(diagram)
    diagram['component_orientations'] = {}
    for ci, walk in enumerate(component_walks(diagram)):
        eid, direction = walk[0]
        if directions[eid] != direction:
            diagram['component_orientations'][str(ci)] = -1
    verdict = validate(diagram)
    if not verdict['valid']:
        raise ValueError('; '.join(verdict['errors']))
    checker = check_exterior_embedding if any(is_twist_box(c) for c in diagram['crossings']) else check_embedding
    verdict = checker(diagram)
    if not verdict['valid']:
        raise ValueError('There is not enough clear space for this curl: ' + '; '.join(verdict['errors']))
    return diagram


def crossing_sign(diagram, crossing_id):
    """Return +1/-1 using the physical orientations of over and under strands."""
    crossing = next(c for c in diagram['crossings'] if c['id'] == int(crossing_id))
    edges = {e['id']: e for e in diagram['edges']}
    directions = _directions(diagram)
    return _crossing_sign(crossing, edges, directions)


def _crossing_sign(crossing, edges, directions):
    """Use an existing discovery traversal rather than walking every component."""
    tangents = []
    center = np.asarray(crossing['point'], dtype=float)
    for parity in (int(crossing['over']), 1-int(crossing['over'])):
        port = crossing['ports'][parity]
        points = edges[port['edge']]['points']
        if port['end']:
            points = list(reversed(points))
        tangent = next(np.asarray(p)-center for p in points[1:] if np.linalg.norm(np.asarray(p)-center) > 1e-7)
        if directions[port['edge']] != port['end']:
            tangent = -tangent
        tangents.append(tangent)
    a, b = tangents
    determinant = a[0]*b[1]-a[1]*b[0]
    if abs(determinant) < 1e-10:
        raise ValueError('Crossing strands are not transverse.')
    # SnapPy's positive crossing has incoming-under port 0 and incoming-over
    # port 3, with ports counterclockwise in mathematical coordinates.
    return 1 if determinant < 0 else -1


def _clean_points(points):
    points = np.asarray(points, dtype=float)
    # Most editable paths contain no duplicates. Avoid a Python/numpy call per
    # raster sample; retain the last-kept-point rule for the exceptional case
    # where several sub-tolerance steps could accumulate into a real segment.
    if len(points) > 1 and np.any(np.sum(np.diff(points, axis=0)**2, axis=1) <= 1e-16):
        kept = [0]
        for index in range(1, len(points)):
            delta = points[index]-points[kept[-1]]
            if float(delta@delta) > 1e-16:
                kept.append(index)
        points = points[kept]
    if len(points) < 2:
        raise ValueError('The selected strand has no usable length.')
    if len(points)>2:
        left,right=np.diff(points,axis=0)[:-1],np.diff(points,axis=0)[1:]
        cross=left[:,0]*right[:,1]-left[:,1]*right[:,0]
        straight=(np.abs(cross)<=1e-10*np.maximum(1.,np.linalg.norm(left+right,axis=1))) & (np.sum(left*right,axis=1)>0)
        points=points[np.r_[True,~straight,True]]
    return points


class _PathGeometry:
    """Operation-scoped paths and bounds; no cache survives a diagram edit.

    A face test or surgery asks only for paths whose bounds meet its region.
    Point cleaning is lazy, so unrelated strands are never resampled. Bounds
    reject only impossible contacts; all nearby paths use the exact predicates.
    """
    def __init__(self, diagram):
        self.cmap = {c['id']: c for c in diagram['crossings']}
        self.emap = {e['id']: e for e in diagram['edges']}
        self.paths = {eid: np.asarray(edge_points(e, self.cmap), dtype=float)
                      for eid, e in self.emap.items()}
        self.ids = list(self.paths)
        self.lower = np.array([p.min(axis=0) for p in self.paths.values()]).reshape(-1, 2)
        self.upper = np.array([p.max(axis=0) for p in self.paths.values()]).reshape(-1, 2)
        self.cleaned = {}
        # A compact box is a fixed hole, including when its coefficient is
        # zero. Its boundary participates in every local empty-disk test.
        self.box_boundaries = [np.asarray(box_corners(b), float) for b in diagram['crossings']
                               if is_twist_box(b)]

    def path(self, eid):
        if eid not in self.cleaned:
            self.cleaned[eid] = _clean_points(self.paths[eid])
        return self.cleaned[eid]

    def near(self, points, exclude=()):
        points = np.asarray(points)
        low, high = points.min(axis=0)-1e-7, points.max(axis=0)+1e-7
        mask = np.all(self.lower <= high, axis=1) & np.all(self.upper >= low, axis=1)
        paths = [self.paths[self.ids[i]] for i in np.flatnonzero(mask) if self.ids[i] not in exclude]
        for boundary in self.box_boundaries:
            if np.all(boundary.min(axis=0) <= high) and np.all(boundary.max(axis=0) >= low):
                paths.append(np.vstack([boundary, boundary[:1]]))
        return paths

    def box_in_region(self, polygon):
        """Exclude a face containing any box, even with no exposed interior ink."""
        if not self.box_boundaries:
            return False
        import cv2
        polygon = np.asarray(polygon, float)
        low, high = polygon.min(axis=0), polygon.max(axis=0)
        for boundary in self.box_boundaries:
            if not (np.all(boundary.min(axis=0) <= high) and np.all(boundary.max(axis=0) >= low)):
                continue
            if (_any_inside(boundary, polygon)
                    or cv2.pointPolygonTest(boundary.astype(np.float32), tuple(polygon[0]), False) >= 0):
                return True
            # Intersections rule out a box which partly overlaps a face while
            # neither polygon has a vertex strictly inside the other.
            ends = np.roll(boundary, -1, axis=0)
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                for c, d in zip(boundary, ends):
                    if intersection(a, b, c, d) is not None:
                        return True
        return False


def _measure(points):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def _at(points, cumulative, s):
    s = max(0., min(float(s), cumulative[-1]))
    i = min(len(points)-2, max(0, int(np.searchsorted(cumulative, s, side='right'))-1))
    tangent = points[i+1]-points[i]
    tangent /= np.linalg.norm(tangent)
    return points[i]+tangent*(s-cumulative[i]), tangent


def _slice(points, cumulative, left, right):
    a, _ = _at(points, cumulative, left)
    b, _ = _at(points, cumulative, right)
    middle = points[(cumulative > left+1e-8) & (cumulative < right-1e-8)]
    return np.vstack([a, middle, b])


def _project(points, cumulative, point):
    q = np.asarray(point, dtype=float)
    delta = np.diff(points, axis=0)
    t = np.clip(np.sum((q-points[:-1])*delta, axis=1)/np.sum(delta*delta, axis=1), 0., 1.)
    closest = points[:-1]+delta*t[:, None]
    i = int(np.argmin(np.linalg.norm(closest-q, axis=1)))
    return float(cumulative[i]+t[i]*(cumulative[i+1]-cumulative[i]))


def _bezier(a, b, c, d, count=36):
    t = np.linspace(0., 1., count)[:, None]
    return (1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d


def _join(*pieces):
    return _clean_points(np.vstack(pieces))


def _clear_local_disk(swept_points, other_paths, allowed_contacts):
    """Conservative empty-disk certificate for the local strand replacement.

    An embedding test alone cannot rule out jumping across a disjoint strand.
    Requiring the convex hull of both old and new local geometry to be empty
    makes the replacement local in the topological sense as well.
    """
    points = np.asarray(swept_points, dtype=float)
    try:
        hull = ConvexHull(points)
    except QhullError:
        return
    polygon = points[hull.vertices]
    low, high = polygon.min(axis=0)-1e-7, polygon.max(axis=0)+1e-7
    allowed = [np.asarray(p) for p in allowed_contacts]
    ends = np.roll(polygon, -1, axis=0)
    boundary_low, boundary_high = np.minimum(polygon, ends), np.maximum(polygon, ends)
    boundary_length = np.linalg.norm(ends-polygon, axis=1).max()
    for path in other_paths:
        path = np.asarray(path, dtype=float)
        if _any_inside(np.vstack([path, (path[:-1]+path[1:])/2]), polygon):
            raise ValueError('Another strand lies in the local region needed for this curl.')
        nearby = np.all(np.minimum(path[:-1], path[1:]) <= high, axis=1) & np.all(
            np.maximum(path[:-1], path[1:]) >= low, axis=1)
        for index in np.flatnonzero(nearby):
            a, b = path[index:index+2]
            # Preserve intersection()'s endpoint tolerance while excluding
            # hull edges that cannot meet this segment. Dense curved faces
            # otherwise trigger millions of irrelevant segment comparisons.
            padding = 1e-7*(1+boundary_length+np.linalg.norm(b-a))
            possible = np.all(boundary_low <= np.maximum(a, b)+padding, axis=1) & np.all(
                boundary_high >= np.minimum(a, b)-padding, axis=1)
            for j in np.flatnonzero(possible):
                c, d = polygon[j], ends[j]
                hit = intersection(a, b, c, d)
                if hit is not None and not any(np.linalg.norm(hit[2]-p) < 1e-6 for p in allowed):
                    raise ValueError('Another strand borders the local region needed for this curl.')


def _place_crossing(diagram, crossing):
    center = np.asarray(crossing['point'])
    refs = []
    for edge in diagram['edges']:
        for end, key in enumerate(('start', 'end')):
            loc = edge.get(key)
            if loc is None or loc[0] != crossing['id']:
                continue
            pts = edge['points'] if end == 0 else edge['points'][::-1]
            delta = next(np.asarray(p)-center for p in pts[1:] if np.linalg.norm(np.asarray(p)-center) > 1e-7)
            refs.append((math.atan2(-delta[1], delta[0]), edge, end))
    if len(refs) != 4:
        raise ValueError('A curl must have four crossing ports.')
    crossing['ports'] = []
    for index, (_, edge, end) in enumerate(sorted(refs, key=lambda x: x[0])):
        crossing['ports'].append({'edge': edge['id'], 'end': end})
        edge['start' if end == 0 else 'end'] = [crossing['id'], index]


def add_curl(diagram, edge_id, point, sign=1, side_point=None, size=None):
    """Insert a positive/negative smooth curl near a strand point on a chosen side.

    ``side_point`` is a second canvas click selecting one of the strand's local
    sides. ``size`` is the curl scale in image coordinates. Space and crossing
    clearance are checked; an impossible placement raises ValueError unchanged.
    """
    if sign not in (-1, 1):
        raise ValueError('Choose crossing sign +1 or -1.')
    for box in diagram['crossings']:
        if is_twist_box(box):
            axis = np.asarray(box['axis'], float)
            local = (np.asarray(point)-box['point'])@np.array([[axis[1], -axis[0]], axis]).T
            if np.all(np.abs(local) <= np.asarray(box['size'])/2+1e-7):
                raise ValueError('Choose a strand point outside the twist boxes to add a curl.')
    original = next((e for e in diagram['edges'] if e['id'] == int(edge_id)), None)
    if original is None:
        raise ValueError('Select a strand before adding a curl.')
    cmap = {c['id']: c for c in diagram['crossings']}
    points = _clean_points(edge_points(original, cmap))
    cumulative = _measure(points)
    selected = _project(points, cumulative, point)
    center, tangent = _at(points, cumulative, selected)
    normal = np.array([-tangent[1], tangent[0]])
    if side_point is not None:
        dot = float((np.asarray(side_point)-center)@normal)
        if abs(dot) < line_width(diagram)*.2:
            raise ValueError('Click clearly to one side of the strand.')
        if dot < 0:
            normal = -normal
    closed = bool(original.get('closed') and original.get('start') is None)
    stroke = line_width(diagram)
    preferred = float(size) if size is not None else max(16., min(38., min(diagram.get('width', 800), diagram.get('height', 600))/18.))
    if not math.isfinite(preferred) or preferred <= 0:
        raise ValueError('Curl size must be positive and finite.')
    minimum = max(6., stroke*3.5)
    if not closed:
        preferred = min(preferred, (selected-stroke*3)/2, (cumulative[-1]-selected-stroke*3)/2)
    else:
        preferred = min(preferred, cumulative[-1]/9)
    if preferred < minimum:
        raise ValueError('Choose a point farther from crossings, with more free strand on both sides.')
    directions = _directions(diagram)
    geometry = _PathGeometry(diagram)
    # A smaller loop is often appropriate near another strand, but never shrink
    # until a nominal crossing becomes an unreadable stroke-width collision.
    last_error = None
    scales = [preferred] if size is not None else [preferred, preferred*.8, preferred*.65]
    for radius in scales:
        if radius < minimum:
            continue
        try:
            pts = points.copy()
            arc = cumulative.copy()
            s = selected
            if closed:
                # Rotate the seam well away from the insertion interval.
                seam = (selected+arc[-1]/2) % arc[-1]
                pts = _join(_slice(pts, arc, seam, arc[-1]), _slice(pts, arc, 0, seam))
                arc = _measure(pts)
                s = arc[-1]/2
            left, right = s-2*radius, s+2*radius
            a, ta = _at(pts, arc, left)
            b, tb = _at(pts, arc, right)
            c = center+normal*radius*.5
            lead = _bezier(a, a+ta*radius*.9, c-(tangent+normal)*radius*.65, c)
            loop = _bezier(c, c+(tangent+normal)*radius*1.7, c+(-tangent+normal)*radius*1.7, c, 73)
            tail = _bezier(c, c+(tangent-normal)*radius*.65, b-tb*radius*.9, b)
            inserted = np.vstack([lead, loop, tail])
            if (np.any(inserted < stroke) or np.any(inserted[:, 0] > diagram.get('width', 800)-stroke)
                    or np.any(inserted[:, 1] > diagram.get('height', 600)-stroke)):
                raise ValueError('The curl would leave the drawing canvas; choose the other side or more room.')
            before = _slice(pts, arc, 0, left)
            after = _slice(pts, arc, right, arc[-1])
            swept = np.vstack([inserted, _slice(pts, arc, left, right)])
            others = geometry.near(swept, {original['id']})
            _clear_local_disk(swept,
                              others+[before, after], [a, b])
            out = copy_diagram({key: value for key, value in diagram.items()
                                if key not in ('spatial_curves', 'spatial_fingerprint')})
            old = next(e for e in out['edges'] if e['id'] == int(edge_id))
            cid = max((x['id'] for x in out['crossings']), default=-1)+1
            eid = max((x['id'] for x in out['edges']), default=-1)+1
            curl_edge = dict(id=eid, start=[cid, -1], end=[cid, -1], points=loop.tolist(), closed=False)
            new_directions = dict(directions)
            new_directions[eid] = directions[old['id']]
            if closed:
                outside = _join(tail, after, before, lead)
                old.update(points=outside.tolist(), start=[cid, -1], end=[cid, -1], closed=False)
                out['edges'].append(curl_edge)
            else:
                finish = old['end']
                old.update(points=_join(before, lead).tolist(), end=[cid, -1], closed=False)
                tail_edge = dict(id=eid+1, start=[cid, -1], end=finish, points=_join(tail, after).tolist(), closed=False)
                out['edges'].extend([curl_edge, tail_edge])
                new_directions[eid+1] = directions[old['id']]
            crossing = dict(id=cid, point=c.tolist(), ports=[], over=0)
            out['crossings'].append(crossing)
            _place_crossing(out, crossing)
            out = _finish(out, new_directions)
            if crossing_sign(out, cid) != sign:
                crossing['over'] = 1-crossing['over']
            if not removable_monogons(out, edge_id=eid):
                raise ValueError('The chosen side does not contain an empty region for the curl.')
            return out
        except ValueError as exc:
            last_error = exc
    raise ValueError(str(last_error or 'There is not enough room for a curl here.'))


def _inside(point, polygon):
    """Strict interior test, excluding numerical boundary contacts."""
    p = np.asarray(point, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    a, b = poly, np.roll(poly, -1, axis=0)
    delta = b-a
    length2 = np.sum(delta*delta, axis=1)
    t = np.clip(np.sum((p-a)*delta, axis=1)/np.maximum(length2, 1e-24), 0., 1.)
    if np.any(np.linalg.norm(a+delta*t[:, None]-p, axis=1) < 1e-7):
        return False
    crossing = (a[:, 1] > p[1]) != (b[:, 1] > p[1])
    relevant = np.flatnonzero(crossing)
    x = delta[relevant, 0]*(p[1]-a[relevant, 1])/delta[relevant, 1]+a[relevant, 0]
    return bool(np.count_nonzero(p[0] < x) % 2)


def _simple_polygon(points):
    if len(points) < 4 or np.linalg.norm(points[0]-points[-1]) > 1e-5:
        return False
    lower = np.minimum(points[:-1], points[1:])
    upper = np.maximum(points[:-1], points[1:])
    # Query the segment boxes together instead of scanning all N boxes for
    # each of N boundary segments. Large smooth faces have few nearby pairs.
    # The broad phase is deliberately a superset; retain the former endpoint
    # tolerance and exact intersection predicate for every surviving pair.
    from .motion import _box_pairs
    pairs = _box_pairs(lower-1e-8, upper+1e-8)
    if len(pairs):
        i, j = pairs.T
        keep = ((j > i+1) & ~((i == 0) & (j == len(points)-2)) &
                np.all(lower[j] <= upper[i]+1e-8, axis=1) &
                np.all(upper[j] >= lower[i]-1e-8, axis=1))
        for i, j in pairs[keep]:
            if intersection(points[i], points[i+1], points[j], points[j+1]) is not None:
                return False
    area = abs(float(np.sum(points[:-1, 0]*points[1:, 1]-points[1:, 0]*points[:-1, 1])))/2
    return area > 1.


def _any_inside(candidates,polygon):
    """Strict point-in-polygon checks, with shared bounds and batched edges."""
    a=np.asarray(polygon);delta=np.roll(a,-1,axis=0)-a
    low,high=a.min(axis=0),a.max(axis=0)
    candidates=candidates[np.all((candidates>=low)&(candidates<=high),axis=1)]
    length2=np.maximum(np.sum(delta*delta,axis=1),1e-24)
    for start in range(0,len(candidates),128):
        p=candidates[start:start+128,None,:]
        t=np.clip(np.sum((p-a)*delta,axis=2)/length2,0.,1.)
        boundary=np.any(np.sum((a+t[:,:,None]*delta-p)**2,axis=2)<1e-14,axis=1)
        crossing=(a[:,1]>p[:,:,1])!=((a+delta)[:,1]>p[:,:,1])
        x=a[:,0]+np.divide(delta[:,0]*(p[:,:,1]-a[:,1]),delta[:,1],
                          out=np.zeros_like(crossing,dtype=float),where=delta[:,1]!=0)
        inside=np.count_nonzero(crossing & (p[:,:,0]<x),axis=1)%2==1
        if np.any(inside & ~boundary):return True
    return False


def removable_monogons(diagram, *, edge_id=None, crossing_id=None, _geometry=None):
    """List only empty bounded one-edge faces, including imported diagrams.

    The complementary region must contain no other strand, crossing, or free
    circle. Two monogons at a one-crossing unknot have different ``id`` values.
    """
    records = []
    cmap = {c['id']: c for c in diagram['crossings']}
    geometry = _geometry
    directions = None
    signs = {}
    for edge in diagram['edges']:
        if edge_id is not None and edge['id']!=int(edge_id):continue
        start, end = edge.get('start'), edge.get('end')
        if start is None or end is None or start[0] != end[0] or (start[1]-end[1]) % 4 not in (1, 3):
            continue
        if is_twist_box(cmap[start[0]]):
            continue
        if crossing_id is not None and start[0]!=int(crossing_id):continue
        if geometry is None:
            geometry = _PathGeometry(diagram)
        points = geometry.path(edge['id'])
        if not _simple_polygon(points):
            continue
        polygon = points[:-1]
        if geometry.box_in_region(polygon):
            continue
        empty = True
        for path in geometry.near(polygon, {edge['id']}):
            candidates = np.vstack([path, (path[:-1]+path[1:])/2])
            if _any_inside(candidates, polygon):
                empty = False
                break
        if empty:
            cid = start[0]
            if cid not in signs:
                if directions is None:
                    directions = _directions(diagram)
                signs[cid] = _crossing_sign(cmap[cid], geometry.emap, directions)
            records.append(dict(id=f"{start[0]}:{edge['id']}", crossing_id=start[0], edge_id=edge['id'],
                                polygon=polygon.tolist(), sign=signs[cid]))
    return records


def _rounded_join(left, right, distance, other_paths):
    al, ar = _measure(left), _measure(right)
    distance = min(distance, al[-1]*.35, ar[-1]*.35)
    if distance < 1e-5:
        raise ValueError('The incident arcs are too short to remove this curl smoothly.')
    a, ta = _at(left, al, al[-1]-distance)
    b, tb = _at(right, ar, distance)
    bridge = _bezier(a, a+ta*distance*.6, b-tb*distance*.6, b)
    before, after = _slice(left, al, 0, al[-1]-distance), _slice(right, ar, distance, ar[-1])
    removed = np.vstack([_slice(left, al, al[-1]-distance, al[-1]), _slice(right, ar, 0, distance)])
    _clear_local_disk(np.vstack([bridge, removed]), list(other_paths)+[before, after], [a, b])
    return _join(before, bridge, after)


def remove_curl(diagram, crossing_id, edge_id=None):
    """Remove the selected empty monogon and smoothly splice its outer arms."""
    geometry = _PathGeometry(diagram)
    records = removable_monogons(diagram,crossing_id=crossing_id,edge_id=edge_id,_geometry=geometry)
    if not records:
        raise ValueError('Select a shaded empty monogon region to remove its curl.')
    record = min(records, key=lambda r: len(r['polygon']))
    cmap = {c['id']: c for c in diagram['crossings']}
    emap = {e['id']: e for e in diagram['edges']}
    loop = emap[record['edge_id']]
    crossing = cmap[int(crossing_id)]
    p, q = loop['start'][1], loop['end'][1]
    first = crossing['ports'][(p+2) % 4]
    second = crossing['ports'][(q+2) % 4]
    a, b = emap[first['edge']], emap[second['edge']]
    directions = _directions(diagram)
    loop_length = _measure(np.asarray(edge_points(loop, cmap)))[-1]
    local_bounds = np.vstack([geometry.paths[eid] for eid in {loop['id'], a['id'], b['id']}])
    others = geometry.near(local_bounds, {loop['id'], a['id'], b['id']})
    last_error = None
    for factor in (.12, .07, .04, .02):
        try:
            new_directions = dict(directions)
            if a['id'] == b['id']:
                pts = _clean_points(edge_points(a, cmap))
                arc = _measure(pts)
                # Put the closed seam at an existing vertex, avoiding an arbitrarily
                # tiny first segment beside a long last segment.
                lengths = np.diff(arc)
                seam_index = 1+int(np.argmax(np.minimum(lengths[:-1], lengths[1:])))
                middle = arc[seam_index]
                left = _slice(pts, arc, middle, arc[-1])
                right = _slice(pts, arc, 0, middle)
                path = _rounded_join(left, right, loop_length*factor, others)
                merged = deepcopy(a)
                merged.update(points=path.tolist(), start=None, end=None, closed=True)
            else:
                first_direction, second_direction = 1-first['end'], second['end']
                left = _clean_points(edge_points(a, cmap))[::1 if first_direction == 0 else -1]
                right = _clean_points(edge_points(b, cmap))[::1 if second_direction == 0 else -1]
                path = _rounded_join(left, right, loop_length*factor, others)
                merged = deepcopy(a)
                merged.update(points=path.tolist(), start=a['start' if first_direction == 0 else 'end'],
                              end=b['end' if second_direction == 0 else 'start'], closed=False)
                new_directions[a['id']] = directions[a['id']] ^ first_direction
            removed = {loop['id'], a['id'], b['id']}
            out = copy_diagram({key: value for key, value in diagram.items()
                                if key not in ('spatial_curves', 'spatial_fingerprint')})
            out['edges'] = [merged if e['id'] == a['id'] else e for e in out['edges']
                            if e['id'] not in removed or e['id'] == a['id']]
            out['crossings'] = [c for c in out['crossings'] if c['id'] != int(crossing_id)]
            return _finish(out, new_directions)
        except ValueError as exc:
            last_error = exc
    raise ValueError(str(last_error or 'This monogon cannot be removed without touching another strand.'))
