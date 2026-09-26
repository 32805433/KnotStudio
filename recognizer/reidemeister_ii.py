"""Empty bigon discovery and certified Reidemeister II cancellation.

A boundary strand must stay over (or stay under) at both vertices.
Cancellation preserves the original opposite-port strand connections. Geometry
is changed in an empty swept disk; an embedding check additionally verifies the
result. No crossing choices or unrelated component orientations are inferred.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import combinations
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .diagram import assign_components, component_walks, validate, copy_diagram
from .geometry import check_embedding, embedding_stroke_contacts, intersection
from .render import edge_points, line_width
from .box_exterior_embedding import check_exterior_embedding, exterior_stroke_contacts
from .twist_boxes import is_twist_box
from .reidemeister import (_any_inside, _at, _bezier, _clean_points, _directions,
                          _clear_local_disk, _join, _measure, _simple_polygon, _slice, _PathGeometry)


def _from_crossing(edge, crossing_id, cmap):
    points = _clean_points(edge_points(edge, cmap))
    return points if edge['start'][0] == crossing_id else points[::-1].copy()


def _port_at(edge, crossing_id):
    return edge['start'][1] if edge['start'][0] == crossing_id else edge['end'][1]


def removable_bigons(diagram, *, bigon_id=None, _geometry=None):
    """Return empty bounded two-edge faces with a cancellable crossing pair.

    Each record has a stable ``id``, ``crossing_ids``, ``edge_ids``, and a
    ``polygon`` suitable for hit testing and shading. Geometry is taken from
    the actual diagram, including imported diagrams without editing history.
    Occupied lenses, the unbounded face, and Hopf-link crossing pairs are omitted.
    """
    cmap = {c['id']: c for c in diagram['crossings']}
    geometry = _geometry
    grouped = defaultdict(list)
    for edge in diagram['edges']:
        start, end = edge.get('start'), edge.get('end')
        if start is not None and end is not None and start[0] != end[0]:
            if is_twist_box(cmap[start[0]]) or is_twist_box(cmap[end[0]]):
                continue
            grouped[tuple(sorted((start[0], end[0])))].append(edge)
    records = []
    for (c1, c2), edges in sorted(grouped.items()):
        for first, second in combinations(sorted(edges, key=lambda e: e['id']), 2):
            identifier = f"{c1}:{c2}:{first['id']}:{second['id']}"
            if bigon_id is not None and identifier != str(bigon_id):
                continue
            if any((_port_at(first, cid)-_port_at(second, cid)) % 4 not in (1, 3)
                   for cid in (c1, c2)):
                continue
            # A given boundary strand must stay over (or stay under) at both
            # ends. Opposite writhe signs alone do not certify a Reidemeister II.
            if ((_port_at(first, c1) % 2 == cmap[c1]['over']) !=
                    (_port_at(first, c2) % 2 == cmap[c2]['over'])):
                continue
            if geometry is None:
                geometry = _PathGeometry(diagram)
            a = geometry.path(first['id'])[::1 if first['start'][0] == c1 else -1]
            b = geometry.path(second['id'])[::1 if second['start'][0] == c1 else -1]
            boundary = _join(a, b[::-1])
            if not _simple_polygon(boundary):
                continue
            polygon = boundary[:-1]
            if geometry.box_in_region(polygon):
                continue
            if any(_any_inside(np.vstack([p, (p[:-1]+p[1:])/2]), polygon)
                   for p in geometry.near(polygon, {first['id'], second['id']})):
                continue
            records.append(dict(id=identifier,
                                crossing_ids=[c1, c2], edge_ids=[first['id'], second['id']],
                                polygon=polygon.tolist()))
    return records


class _BigonGeometry(_PathGeometry):
    def __init__(self, diagram):
        super().__init__(diagram)
        self.diagram = diagram

    def select(self, record):
        self.local_ids = set(record['edge_ids'])
        for cid in record['crossing_ids']:
            self.local_ids.update(port['edge'] for port in self.cmap[cid]['ports'])
        # Every proposal scores exactly this union. The energy of all other
        # paths is the same additive constant and cannot affect its ordering.
        self.local_paths = {eid: self.path(eid) for eid in self.local_ids}
        self._stroke_contacts = None
        self._collar_lengths = None
        self._boundary_samples = None

    def stroke_contacts(self):
        # Compute once per operation, only after a candidate's swept disk has
        # been certified. A distant pre-existing tight gap must not reject
        # every local proposal or trigger dozens of redundant global scans.
        if self._stroke_contacts is None:
            collect = (exterior_stroke_contacts if self.box_boundaries else embedding_stroke_contacts)
            self._stroke_contacts = collect(self.diagram)
        return self._stroke_contacts

    def collar_lengths(self, record):
        """Escape acute crossing tips before reconnecting the exterior arms.

        The visible gap and the distance needed along almost parallel arms
        are different scales. Choose the first adequately separated pair of
        attachment points; final embedding and swept-disk checks still decide
        whether the proposed reconnection is safe.
        """
        if self._collar_lengths is None:
            self._collar_lengths = {}
            stroke = line_width(self.diagram)
            for cid in record['crossing_ids']:
                arms = []
                for eid in record['edge_ids']:
                    ref = self.cmap[cid]['ports'][(_port_at(self.emap[eid], cid)+2) % 4]
                    p = self.path(ref['edge'])[::1 if ref['end'] == 0 else -1]
                    arms.append((p, _measure(p)))
                limit = min(arc[-1] for _, arc in arms)*.45
                count = max(2, min(257, int(math.ceil(limit/(stroke*.25)))+1))
                lengths = np.linspace(0., limit, count)
                points = [np.column_stack([np.interp(lengths, arc, p[:, k]) for k in (0, 1)])
                          for p, arc in arms]
                clear = np.flatnonzero(np.linalg.norm(points[0]-points[1], axis=1) >= stroke*1.4)
                self._collar_lengths[cid] = lengths[clear[0]] if len(clear) else limit
        return self._collar_lengths

    def sampled_boundaries(self, boundaries):
        # All joint candidates reuse the same two source boundaries. Changing
        # the blend or desired gap does not require measuring/sampling again.
        if self._boundary_samples is None:
            arcs = [_measure(p) for p in boundaries]
            count = max(33, min(181, int(max(a[-1] for a in arcs)/2)+1))
            self._boundary_samples = [
                np.column_stack([np.interp(np.linspace(0., arc[-1], count), arc, p[:, k])
                                 for k in (0, 1)])
                for p, arc in zip(boundaries, arcs)]
        return self._boundary_samples

    def obstacles(self, paths, swept, excluded):
        return self.near(swept, set(paths) | set(excluded)) + [
            p for eid, p in paths.items() if eid not in excluded]


class _BigonProposal:
    """Keep scored local geometry for later certification instead of rebuilding."""
    def __init__(self, diagram, record, paths, certificate, context):
        self.diagram, self.record, self.paths, self.certificate = diagram, record, paths, certificate
        self.context = context
        self.energy = _shape_energy(paths.values(), 8*line_width(diagram))

    def commit(self):
        self.certificate()
        return _splice(self.diagram, self.paths, set(self.record['crossing_ids']),
                       inherited_stroke_contacts=self.context.stroke_contacts(),
                       _source_checked=True)


def _offset(points, distance):
    """Mitered parallel polyline; distances are signed toward its left side."""
    tangent = np.diff(points, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    middle_dot = np.sum(tangent[:-1]*tangent[1:], axis=1)
    if np.any(middle_dot < -.95):
        raise ValueError('The bigon boundary turns too sharply for a smooth cancellation.')
    middle = (normals[:-1]+normals[1:])/(1+middle_dot[:, None])
    return points+distance*np.vstack([normals[:1], middle, normals[-1:]])


def _smooth_offset_centerline(points, distance):
    """Remove raster-scale bends before constructing a wide parallel pair.

    A raw miter offset folds into little loops wherever its distance exceeds
    the radius of a pixel-scale bend. Smooth only the proposed replacement's
    centerline, at a scale relative to the desired gap. The original diagram
    stays intact, and the common empty-disk certificate still covers every
    resulting point, bridge, and deleted piece of the original tangle.
    """
    arc = _measure(points)
    samples = np.linspace(0., arc[-1], max(100, len(points)))
    sampled = np.column_stack([np.interp(samples, arc, points[:, k]) for k in (0, 1)])
    sigma = abs(distance)*.5/(samples[1]-samples[0])
    smooth = gaussian_filter1d(sampled, sigma, axis=0, mode='nearest')
    smooth += np.linspace(points[0]-smooth[0], points[-1]-smooth[-1], len(smooth))
    return _clean_points(smooth)


def _empty_sweep(old_path, new_path, other_paths, contacts):
    """Certify the disk swept by the moving boundary contains no third strand.

    The other boundary of the selected bigon is exempted by the caller; in a
    joint replacement it receives its own disk certificate. Other arcs of the
    same physical component remain obstacles. The outside collar therefore
    cannot jump over a disconnected component that would be invisible to a
    final-embedding-only test.
    """
    boundary = _join(old_path, new_path[::-1])
    if not _simple_polygon(boundary):
        raise ValueError('There is not enough clear space beside this bigon.')
    polygon = boundary[:-1]
    low, high = polygon.min(axis=0)-1e-7, polygon.max(axis=0)+1e-7
    contacts = np.asarray(contacts)
    for path in other_paths:
        if _any_inside(np.vstack([path, (path[:-1]+path[1:])/2]), polygon):
            raise ValueError('Another strand lies in the region needed to remove this bigon.')
        nearby = np.all(np.minimum(path[:-1], path[1:]) <= high, axis=1) & np.all(
            np.maximum(path[:-1], path[1:]) >= low, axis=1)
        boundary_low = np.minimum(boundary[:-1], boundary[1:])
        boundary_high = np.maximum(boundary[:-1], boundary[1:])
        boundary_length = np.max(np.linalg.norm(np.diff(boundary, axis=0), axis=1))
        for index in np.flatnonzero(nearby):
            a, b = path[index:index+2]
            # intersection() accepts endpoint parameters up to 1e-7 outside
            # [0,1]. Pad by segment length so this broad phase cannot discard
            # one of those accepted near-endpoint contacts.
            padding = 1e-7*(1+boundary_length+np.linalg.norm(b-a))
            possible = np.all(boundary_low <= np.maximum(a, b)+padding, axis=1) & np.all(
                boundary_high >= np.minimum(a, b)-padding, axis=1)
            for j in np.flatnonzero(possible):
                c, d = boundary[j:j+2]
                hit = intersection(a, b, c, d)
                if hit is not None and not np.any(np.linalg.norm(contacts-hit[2], axis=1) < 1e-6):
                    raise ValueError('Another strand crosses the region needed to remove this bigon.')


def _splice(diagram, paths, eliminated, *, inherited_stroke_contacts=(), _source_checked=False):
    """Contract removed vertices along opposite ports, including closed chains."""
    cmap = {c['id']: c for c in diagram['crossings']}
    emap = {e['id']: e for e in diagram['edges']}
    component_ids = {eid: ci for ci, walk in enumerate(component_walks(diagram)) for eid, _ in walk}
    directions = _directions(diagram)
    seeds = [(e['id'], end) for e in diagram['edges'] for end, key in enumerate(('start', 'end'))
             if e.get(key) is not None and e[key][0] not in eliminated]
    seeds += [(e['id'], 0) for e in diagram['edges']]
    used, merged, new_directions, changed_ids = set(), [], {}, set()
    for seed in seeds:
        if seed[0] in used:
            continue
        original = emap[seed[0]]
        if not any(loc is not None and loc[0] in eliminated
                   for loc in (original.get('start'), original.get('end'))):
            # This whole edge is untouched. In particular, distant raster
            # paths need neither cleaning, joining, nor conversion to arrays.
            merged.append(original)
            new_directions[original['id']] = directions[original['id']]
            used.add(original['id'])
            continue
        loc = original['start' if seed[1] == 0 else 'end']
        start = deepcopy(loc) if loc is not None and loc[0] not in eliminated else None
        dart, pieces, finish = seed, [], None
        for _ in range(len(emap)+1):
            eid, direction = dart
            if eid in used:
                if dart != seed or start is not None:
                    raise ValueError('The selected bigon does not have consistent strand connections.')
                break
            used.add(eid)
            edge = emap[eid]
            pieces.append(paths[eid] if not direction else paths[eid][::-1])
            target = edge['end' if direction == 0 else 'start']
            if target is None:
                break
            if target[0] not in eliminated:
                finish = deepcopy(target)
                break
            port = cmap[target[0]]['ports'][(target[1]+2) % 4]
            dart = (port['edge'], port['end'])
        else:
            raise ValueError('Strand contraction did not terminate.')
        points = _join(*pieces)
        closed = start is None and finish is None
        if closed and np.linalg.norm(points[0]-points[-1]) > 1e-7:
            raise ValueError('Cancelling this bigon would open a component.')
        if closed:
            # An eliminated crossing often splits a regular polyline sample
            # into one tiny and one long segment. Put the free-circle seam at
            # a well-separated existing vertex, as remove_curl does, so the
            # display-clearance validator sees ordinary cyclic neighbors.
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            seam = int(np.argmax(np.minimum(lengths, np.roll(lengths, 1))))
            points = np.vstack([points[seam:-1], points[:seam+1]])
        edge = {key: value for key, value in original.items() if key != 'points'}
        edge.update(start=start, end=finish, points=points.tolist(), closed=closed)
        changed_ids.add(edge['id'])
        merged.append(edge)
        new_directions[edge['id']] = directions[seed[0]] ^ seed[1]
    # Keep component ordering (and therefore display colors) stable even when
    # a formerly crossed component becomes free while another retains nodes.
    merged.sort(key=lambda e: component_ids[e['id']])
    out = {k: v for k, v in diagram.items()
           if k not in ('edges', 'crossings', 'spatial_curves', 'spatial_fingerprint')}
    out['crossings'] = [c for c in diagram['crossings'] if c['id'] not in eliminated]
    out['edges'] = merged
    out = copy_diagram(out)
    merged = out['edges']
    # Cached lifts and drag branch parameters refer to the pre-cancellation
    # graph and must not survive the change in topology of its projection.
    for key in ('spatial_curves', 'spatial_fingerprint'):
        out.pop(key, None)
    for c in out['crossings']:
        c.pop('motion_branches', None)
    retained = {c['id']: c for c in out['crossings']}
    for edge in merged:
        for end, key in enumerate(('start', 'end')):
            loc = edge[key]
            if loc is not None:
                retained[loc[0]]['ports'][loc[1]] = dict(edge=edge['id'], end=end)
    assign_components(out)
    out['component_orientations'] = {}
    for ci, walk in enumerate(component_walks(out)):
        eid, direction = walk[0]
        if new_directions[eid] != direction:
            out['component_orientations'][str(ci)] = -1
    verdict = validate(out)
    if not verdict['valid']:
        raise ValueError('; '.join(verdict['errors']))
    if len(component_walks(out)) != len(component_walks(diagram)):
        raise ValueError('Cancelling this bigon would change the number of components.')
    # The source was checked once when its inheritable contacts were recorded.
    # Every arc incident to a removed crossing is contracted and marked here,
    # including its retained tails and former crossing-clearance halos. All
    # other arcs keep exact points, endpoints, closure, and material spacing;
    # pairs between those unchanged arcs need not be tested for every proposal.
    checker = check_exterior_embedding if any(is_twist_box(c) for c in out['crossings']) else check_embedding
    verdict = checker(out, inherited_stroke_contacts=inherited_stroke_contacts,
                              active_edge_ids=changed_ids if _source_checked else None)
    if not verdict['valid']:
        raise ValueError('There is not enough clear space for this bigon: '+'; '.join(verdict['errors']))
    return out


def _shape_energy(paths, unit):
    """Length plus bending energy for comparing local reconnections.

    E = length/unit + unit * integral(curvature**2 ds). Uniform arclength
    samples prevent dense raster vertices from getting extra weight. Unlike
    the relaxation solver's landmark energy, this score compares geometric
    paths without depending on how the removed crossing vertices were meshed.
    Outside paths are unchanged and contribute the same constant to candidates.
    """
    energy = 0.
    for points in paths:
        points = _clean_points(points)
        arc = _measure(points)
        length = arc[-1]
        if length < 1e-8:
            continue
        count = max(8, int(math.ceil(length/(unit/3))))
        samples = np.linspace(0., length, count+1)
        p = np.column_stack([np.interp(samples, arc, points[:, k]) for k in (0, 1)])
        tangent = np.diff(p, axis=0)
        size = np.linalg.norm(tangent, axis=1)
        if np.any(size < 1e-9):
            return math.inf
        tangent /= size[:, None]
        cross = tangent[:-1, 0]*tangent[1:, 1]-tangent[:-1, 1]*tangent[1:, 0]
        angles = np.arctan2(cross, np.sum(tangent[:-1]*tangent[1:], axis=1))
        energy += length/unit + unit*np.sum(angles**2/((size[:-1]+size[1:])/2))
    return float(energy)


def _attempt(diagram, record, moving_id, distance, trim_factor, *, _score_only=False,
             _context=None, _prepare=False):
    context = _context or _BigonGeometry(diagram)
    if _context is None:
        context.select(record)
    cmap, emap = context.cmap, context.emap
    c1, c2 = record['crossing_ids']
    fixed_id = next(eid for eid in record['edge_ids'] if eid != moving_id)
    moving, fixed = emap[moving_id], emap[fixed_id]
    old = context.path(moving_id)[::1 if moving['start'][0] == c1 else -1]
    stationary = context.path(fixed_id)[::1 if fixed['start'][0] == c1 else -1]
    boundary = _join(old, stationary[::-1])
    area = np.sum(boundary[:-1, 0]*boundary[1:, 1]-boundary[1:, 0]*boundary[:-1, 1])
    offset = _offset(stationary, math.copysign(distance, area))
    offset_arc = _measure(offset)
    # Stop the parallel curve before the former crossing corners. Continuing
    # all the way to their normal offsets forces the endpoint bridges to turn
    # backwards through the old arms at acute-angle lens vertices.
    corner = min(distance*trim_factor, offset_arc[-1]*.22)
    offset = _slice(offset, offset_arc, corner, offset_arc[-1]-corner)
    paths = context.local_paths.copy()
    cuts, arms = {}, []
    for cid in (c1, c2):
        ref = cmap[cid]['ports'][(_port_at(moving, cid)+2) % 4]
        eid, end = ref['edge'], ref['end']
        path = paths[eid] if end == 0 else paths[eid][::-1]
        arc = _measure(path)
        length = min(distance*trim_factor, arc[-1]*.45)
        if length < distance*.3:
            raise ValueError('The arcs beside this bigon are too short for cancellation.')
        cut, outward = _at(path, arc, length)
        arms.append((cut, outward, _slice(path, arc, 0, length)))
        cuts[eid, end] = length
    for eid in {eid for eid, _ in cuts}:
        arc = _measure(paths[eid])
        paths[eid] = _slice(paths[eid], arc, cuts.get((eid, 0), 0.),
                            arc[-1]-cuts.get((eid, 1), 0.))
    a, ta, arm_a = arms[0]
    b, tb, arm_b = arms[1]
    ua, ub = offset[1]-offset[0], offset[-1]-offset[-2]
    ua, ub = ua/np.linalg.norm(ua), ub/np.linalg.norm(ub)
    da = min(np.linalg.norm(a-offset[0])*.5, distance*trim_factor*.65)
    db = min(np.linalg.norm(b-offset[-1])*.5, distance*trim_factor*.65)
    lead = _bezier(a, a-ta*da, offset[0]-ua*da, offset[0], 18)
    tail = _bezier(offset[-1], offset[-1]+ub*db, b-tb*db, b, 18)
    replacement = _join(lead, offset, tail)
    old_path = _join(arm_a[::-1], old, arm_b)
    paths[moving_id] = replacement if moving['start'][0] == c1 else replacement[::-1]
    if _score_only:
        return _shape_energy(paths.values(), 8*line_width(diagram))
    def certify():
        others = context.obstacles(paths, np.vstack([old_path, replacement]), {moving_id, fixed_id})
        _empty_sweep(old_path, replacement, others, [a, b, old[0], old[-1]])
    proposal = _BigonProposal(diagram, record, paths, certify, context)
    return proposal if _prepare else proposal.commit()


def _joint_attempt(diagram, record, distance, trim_factor, blend, *, _score_only=False,
                   _context=None, _prepare=False, _direct=None):
    """Use both sides of the empty face when an exterior parallel is too tight.

    The two strands are rerouted through the face, using parallel curves or
    direct tangent-matched bridges. Their four external attachment points and
    opposite-port connections stay fixed. A common empty convex disk or two
    simple empty swept disks certify the replacement relative to other strands.
    """
    context = _context or _BigonGeometry(diagram)
    if _context is None:
        context.select(record)
    cmap, emap = context.cmap, context.emap
    c1, c2 = record['crossing_ids']
    ids = record['edge_ids']
    boundaries = [context.path(eid)[::1 if emap[eid]['start'][0] == c1 else -1] for eid in ids]
    if _direct is None:
        # Uniform material samples remove subpixel raster wiggles in this new
        # local replacement only. Its topology is certified by the empty disk.
        sampled = context.sampled_boundaries(boundaries)
        middle = _clean_points(sampled[0]*blend+sampled[1]*(1-blend))
        middle = _smooth_offset_centerline(middle, distance)
        boundary = _join(boundaries[0], boundaries[1][::-1])
        area = np.sum(boundary[:-1, 0]*boundary[1:, 1]-boundary[1:, 0]*boundary[:-1, 1])
    paths = context.local_paths.copy()
    arms, cuts = {}, {}
    for eid in ids:
        for cid in (c1, c2):
            ref = cmap[cid]['ports'][(_port_at(emap[eid], cid)+2) % 4]
            exterior_id, end = ref['edge'], ref['end']
            path = paths[exterior_id] if end == 0 else paths[exterior_id][::-1]
            arc = _measure(path)
            length = min(max(distance*trim_factor, context.collar_lengths(record)[cid]), arc[-1]*.45)
            if length < .3:
                raise ValueError('The neighboring crossings leave too little room for this cancellation.')
            cut, outward = _at(path, arc, length)
            arms[eid, cid] = cut, outward, _slice(path, arc, 0, length)
            cuts[exterior_id, end] = length
    for eid in {eid for eid, _ in cuts}:
        arc = _measure(paths[eid])
        paths[eid] = _slice(paths[eid], arc, cuts.get((eid, 0), 0.),
                            arc[-1]-cuts.get((eid, 1), 0.))
    swept = list(boundaries)+[arm[2] for arm in arms.values()]
    sweeps = []
    contacts = [arm[0] for arm in arms.values()]
    for index, eid in enumerate(ids):
        a, ta, _ = arms[eid, c1]
        b, tb, _ = arms[eid, c2]
        if _direct is not None:
            handle = np.linalg.norm(b-a)*_direct
            replacement = _bezier(a, a-ta*handle, b-tb*handle, b, 65)
        else:
            parallel = _offset(middle, math.copysign(distance, area)*(1 if index == 0 else -1))
            arc = _measure(parallel)
            corner = min(distance*trim_factor, arc[-1]*.2)
            parallel = _slice(parallel, arc, corner, arc[-1]-corner)
            ua, ub = parallel[1]-parallel[0], parallel[-1]-parallel[-2]
            ua, ub = ua/np.linalg.norm(ua), ub/np.linalg.norm(ub)
            da, db = np.linalg.norm(a-parallel[0])*.4, np.linalg.norm(b-parallel[-1])*.4
            lead = _bezier(a, a-ta*da, parallel[0]-ua*da, parallel[0], 18)
            tail = _bezier(parallel[-1], parallel[-1]+ub*db, b-tb*db, b, 18)
            replacement = _join(lead, parallel, tail)
        sweeps.append((_join(arms[eid, c1][2][::-1], boundaries[index], arms[eid, c2][2]), replacement))
        swept.append(replacement)
        paths[eid] = replacement if emap[eid]['start'][0] == c1 else replacement[::-1]
    if _direct is not None:
        # Shortcuts compete at the same requested visible gap as parallel
        # reroutes. Their final intersection check remains in _splice.
        def clearance(points, path):
            delta = np.diff(path, axis=0)
            relative = points[:, None, :]-path[:-1]
            t = np.clip(np.sum(relative*delta, axis=2)/np.sum(delta*delta, axis=1), 0., 1.)
            return np.linalg.norm(relative-t[:, :, None]*delta, axis=2).min()
        a, b = (sweep[1] for sweep in sweeps)
        if min(clearance(a, b), clearance(b, a)) < 2*distance:
            raise ValueError('The shortcut leaves insufficient separation between the two strands.')
    if _score_only:
        return _shape_energy(paths.values(), 8*line_width(diagram))
    def certify():
        swept_points = np.vstack(swept)
        others = context.obstacles(paths, swept_points, ids)
        try:
            _clear_local_disk(swept_points, others, contacts)
        except ValueError:
            # A concave empty face need not have an empty convex hull. Each
            # boundary may instead sweep its own simple empty disk. The two
            # moving arcs keep their common over/under order and can pass
            # each other at separate heights; all retained arcs, including
            # the trimmed outside arms, remain obstacles for both sweeps.
            for old, new in sweeps:
                _empty_sweep(old, new, others, contacts)
    proposal = _BigonProposal(diagram, record, paths, certify, context)
    return proposal if _prepare else proposal.commit()


def _shortcut_attempt(diagram, record, distance, trim_factor, handle, **options):
    """Try short tangent-matched bridges, rather than retaining a long detour."""
    return _joint_attempt(diagram, record, distance, trim_factor, .5, _direct=handle, **options)


def remove_bigon(diagram, bigon_id):
    """Cancel the selected shaded RII pair, leaving the input unchanged.

    At each preferred gap, rank single-arc and joint replacements by length
    and bending energy. Certify candidates in that order; scoring never grants
    permission to cross another strand. Insufficient display-width clearance
    raises an error without changing the input.
    """
    context = _BigonGeometry(diagram)
    records = removable_bigons(diagram, bigon_id=bigon_id, _geometry=context)
    record = records[0] if records else None
    if record is None:
        raise ValueError('Select a shaded empty bigon with the same strand over at both crossings.')
    context.select(record)
    last_error = None
    stroke = line_width(diagram)
    # A successful RII should leave two easily distinguished strands. Search
    # by desired *visible separation*, trying both geometric constructions at
    # a broad spacing before resorting to a tighter local cancellation. For a
    # joint reroute each curve moves half of the requested total separation.
    for separation in (8., 6., 4., 2.5, 1.7, 1.3):
        distance = stroke*separation
        trials = [(_attempt, (moving_id, distance, trim))
                  for moving_id in record['edge_ids'] for trim in (4., 2.)]
        # Include paths near either boundary, not just a middle compromise:
        # when one side is nearly straight, the curvier side can move toward it.
        trials += [(_joint_attempt, (distance/2, trim, blend))
                   for trim in (3., 1.5) for blend in (0., .1, .3, .5, .7, .9, 1.)]
        trials += [(_shortcut_attempt, (distance/2, trim, handle))
                   for trim in (6., 4.5, 3., 1.5) for handle in (.2, .4, .65)]
        ranked = []
        # Scoring builds geometry only. A proposal must subsequently pass the
        # ordinary certificates; an attractive energy never bypasses them.
        for index, (build, args) in enumerate(trials):
            try:
                proposal = build(diagram, record, *args, _context=context, _prepare=True)
                if math.isfinite(proposal.energy):
                    ranked.append((proposal.energy, index, proposal))
            except ValueError as exc:
                last_error = exc
        for _, _, proposal in sorted(ranked):
            try:
                return proposal.commit()
            except ValueError as exc:
                last_error = exc
    raise ValueError(str(last_error or 'This bigon cannot be removed without touching another strand.'))
