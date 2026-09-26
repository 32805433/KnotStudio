"""KnotJob-inspired planar energy flow, relative to a lasso boundary.

An independently implemented quartic spring / inverse-square repulsion energy
acts on arc landmarks, with a finer arc-bending penalty. A renewable
triangulation extends their motion to an ambient piecewise-affine isotopy.
Twist-box rectangles are fixed holes, with an exterior attachment-angle term.
All triangles touching the exterior are fixed, and a
line search certifies positive triangle area for the *whole* time step. Thus
the graph's crossings, their order, and the exterior tangle cannot change.
"""
from __future__ import annotations

from collections import defaultdict
from copy import copy, deepcopy
import math

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay, QhullError, cKDTree

from .diagram import copy_diagram, validate
from .motion import _box_pairs
from .render import edge_points
from .twist_boxes import box_corners, box_port_points, is_twist_box


class RelaxationCancelled(Exception):
    pass


def _check_stop(stop):
    if stop is not None and stop.is_set():
        raise RelaxationCancelled()


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _inside(points, polygon):
    points = np.atleast_2d(points)
    answer = np.zeros(len(points), dtype=bool)
    x, y = points.T
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if a[1] == b[1]:
            continue
        answer ^= ((a[1] > y) != (b[1] > y)) & (x < a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]))
    return answer


def _boundary_distance(points, polygon):
    p = np.atleast_2d(points)
    a, b = polygon, np.roll(polygon, -1, axis=0)
    v = b-a
    q = p[:, None, :]-a
    t = np.clip(np.sum(q*v, axis=2)/np.sum(v*v, axis=1), 0, 1)
    return np.sqrt(np.min(np.sum((q-t[..., None]*v)**2, axis=2), axis=1))


def _segment_hits(a, b, c, d):
    """Intersect two sets of segments using exact bounding-box candidates."""
    n = len(a)
    start, end = np.vstack([a, c]), np.vstack([b, d])
    pairs = _box_pairs(np.minimum(start, end)-1e-9, np.maximum(start, end)+1e-9)
    if not len(pairs):
        return np.empty(0, int), np.empty(0, int), np.empty(0), np.empty(0)
    i, j = pairs.T
    keep = (i < n) & (j >= n)
    i, j = i[keep], j[keep]-n
    v, w, q = b[i]-a[i], d[j]-c[j], c[j]-a[i]
    det = _cross(v, w)
    safe = np.where(abs(det) > 1e-12, det, 1.)
    t, u = _cross(q, w)/safe, _cross(q, v)/safe
    keep = (abs(det) > 1e-12) & (t >= -1e-9) & (t <= 1+1e-9) & (u >= -1e-9) & (u <= 1+1e-9)
    return i[keep], j[keep], np.clip(t[keep], 0, 1), np.clip(u[keep], 0, 1)


def validate_region(polygon):
    p = np.asarray(polygon, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError('Draw a closed region with at least three distinct points.')
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-7]]
    if len(p) > 1 and np.linalg.norm(p[0]-p[-1]) < 1e-7:
        p = p[:-1]
    if len(p) < 3 or len(p) > 600:
        raise ValueError('Draw a simpler closed region.')
    a, b = p, np.roll(p, -1, axis=0)
    pairs = _box_pairs(np.minimum(a, b)-1e-8, np.maximum(a, b)+1e-8)
    from .motion import _segment_distance
    for i, j in pairs:
        if (i-j) % len(p) in (1, len(p)-1):
            continue
        if _segment_distance(np.r_[a[i], 0], np.r_[b[i], 0], np.r_[a[j], 0], np.r_[b[j], 0]) < 1e-7:
            raise ValueError('The selection crosses or touches itself. Draw one simple closed region.')
    centered = p-p[0]
    if abs(np.sum(_cross(centered, np.roll(centered, -1, axis=0)))) < 50:
        raise ValueError('The selected region is too small. Circle a larger part of the diagram.')
    return p


def energy_gradient(points, adjacency, stop=None):
    """E=sum_edges r^4/4 + sum_nonedge_pairs 1/(2r^2), in unit coordinates."""
    n = len(points)
    energy = 0.
    gradient = np.zeros_like(points)
    for left in range(0, n, 128):
        _check_stop(stop)
        right = min(left+128, n)
        delta = points[left:right, None, :]-points[None, :, :]
        r2 = np.sum(delta*delta, axis=2)
        diagonal = np.arange(left, right)[:, None] == np.arange(n)
        neighbors = adjacency[left:right]
        if np.any((r2 < 1e-18) & ~diagonal):
            return math.inf, gradient
        safe = np.where(diagonal, 1., np.maximum(r2, 1e-18))
        weights = np.where(neighbors, safe, -1/safe**2)
        weights[diagonal] = 0
        gradient[left:right] = np.sum(weights[..., None]*delta, axis=1)
        terms = np.where(neighbors, safe**2/4, .5/safe)
        terms[diagonal] = 0
        energy += float(terms.sum())/2
    return energy, gradient


def _fair_samples(points, lengths, count, unit, stop=None):
    """Retain bends hidden between uniform fairing landmarks.

    Landmark-only energy cannot see a narrow bend that a whole affine mesh
    triangle carries between two landmarks. Add material vertices needed to
    approximate the actual path to a small geometric tolerance. This adds
    freedom at those bends without using every raster pixel as a mesh node.
    """
    uniform = np.linspace(0., lengths[-1], count+1)
    keep = {0, len(points)-1}
    stack = [(0, len(points)-1)]
    tolerance = unit*.01
    while stack:
        _check_stop(stop)
        a, b = stack.pop()
        if b-a < 2:
            continue
        delta = points[b]-points[a]
        q = points[a+1:b]-points[a]
        t = np.clip(q@delta/max(float(delta@delta), 1e-20), 0., 1.)
        error = np.linalg.norm(q-t[:, None]*delta, axis=1)
        i = int(error.argmax())
        if error[i] > tolerance:
            middle = a+i+1
            keep.add(middle)
            stack.extend(((a, middle), (middle, b)))
    features = lengths[sorted(keep)]
    # A feature already within a tiny arclength of a sample is represented
    # there. Avoid near-coincident mesh nodes and ill-conditioned bend rows.
    near = np.searchsorted(uniform, features).clip(1, len(uniform)-1)
    gap = np.minimum(abs(features-uniform[near]), abs(features-uniform[near-1]))
    return np.sort(np.r_[uniform, features[gap > unit*.001]])


class RelativeRelaxation:
    # Refresh only between accepted steps. The minimum interval amortizes
    # rebuilding the sparse solve; the maximum also catches bends developing
    # between landmarks without appreciable endpoint-spacing distortion.
    resample_min_interval = 60
    resample_interval = 120

    def __init__(self, diagram, polygon, stop=None, *, dynamic_sampling=True, _unit=None,
                 _coarse_counts=None, _center_target=None, _retained_anchors=None):
        self.stop = stop
        self.dynamic_sampling = dynamic_sampling
        _check_stop(stop)
        if not validate(diagram)['valid']:
            raise ValueError('Load a valid diagram before minimizing energy.')
        self.original = copy_diagram(diagram)
        self.region = validate_region(polygon)
        self.region_bounds = self.region.min(axis=0), self.region.max(axis=0)
        crossing_map = {c['id']: c for c in diagram['crossings']}
        self.boxes = [c for c in self.original['crossings'] if is_twist_box(c)]
        ordinary = {cid: c for cid, c in crossing_map.items() if not is_twist_box(c)}
        self.box_boundaries = [np.asarray(box_corners(box), float) for box in self.boxes]
        self.edge_geometry = [np.asarray(edge_points(e, crossing_map), float) for e in diagram['edges']]
        total = sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in self.edge_geometry)
        self.unit = (_unit if _unit is not None else
                     max(8., math.sqrt(diagram.get('width', 800)*diagram.get('height', 600))/35, total/1100))
        if ordinary and np.min(_boundary_distance([c['point'] for c in ordinary.values()], self.region)) < self.unit*.15:
            raise ValueError('The boundary is too close to a crossing. Circle it with the crossing clearly inside or outside.')
        nodes, lookup, links = [], {}, set()

        def node(point):
            key = tuple(np.round(point, 8))
            if key not in lookup:
                lookup[key] = len(nodes)
                nodes.append(np.asarray(point, float))
            return lookup[key]

        self.crossing_nodes = {cid: node(c['point']) for cid, c in ordinary.items()}
        self.coarse_counts = {}
        self.coarse_ids = {}
        for edge, points in zip(diagram['edges'], self.edge_geometry):
            _check_stop(stop)
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            if s[-1] < 1e-8:
                raise ValueError('A strand has collapsed; repair it before minimizing energy.')
            closed = edge.get('closed', False)
            count = (_coarse_counts[edge['id']] if _coarse_counts is not None else
                     max(12 if closed else 3, int(math.ceil(s[-1]/self.unit))))
            self.coarse_counts[edge['id']] = count
            samples = np.linspace(0, s[-1], count+1)
            sampled = np.column_stack([np.interp(samples, s, points[:, k]) for k in (0, 1)])
            if _retained_anchors is not None and edge['id'] in _retained_anchors:
                sampled = _retained_anchors[edge['id']][0]
            ids = [node(p) for p in sampled]
            self.coarse_ids[edge['id']] = np.asarray(ids, int)
            links.update(tuple(sorted((a, b))) for a, b in zip(ids, ids[1:]) if a != b)
        self.n_graph = len(nodes)
        if not links:
            raise ValueError('Circle a nonempty part of a link diagram.')
        # Additional landmarks actively fair the arcs, including narrow bends
        # that fall between the uniform samples. They
        # are not added to the all-pairs repulsion graph: that would make
        # sampling density change the desired knot size and cost quadratically.
        bends, coefficients, sampling_pairs = [], [], []
        fair_ends = {}
        self.fair_ids = {}
        self.fair_distances = {}
        for edge, points in zip(diagram['edges'], self.edge_geometry):
            _check_stop(stop)
            lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            coarse_count = self.coarse_counts[edge['id']]
            count = 4*coarse_count
            if _retained_anchors is not None and edge['id'] in _retained_anchors:
                _, samples, sampled = _retained_anchors[edge['id']]
            else:
                samples = _fair_samples(points, lengths, count, self.unit, self.stop)
                sampled = np.column_stack([np.interp(samples, lengths, points[:, k]) for k in (0, 1)])
            ids = [node(p) for p in sampled]
            self.fair_ids[edge['id']] = np.asarray(ids, int)
            self.fair_distances[edge['id']] = np.asarray(samples)/self.unit
            sampling_pairs.extend(zip(ids[:-1], ids[1:]))
            bends.extend(zip(ids[:-2], ids[1:-1], ids[2:]))
            spacing = np.diff(samples)/self.unit
            h = lengths[-1]/(count*self.unit)
            def row(left, right):
                # h³ times the squared change in material tangent / dual
                # length: exactly [1, -2, 1] on the former uniform grid, but
                # zero on straight paths even when samples are unequal.
                scale = np.sqrt(h**3/((left+right)/2))
                return np.column_stack([scale/left, -scale*(1/left+1/right), scale/right])
            coefficients.extend(row(spacing[:-1], spacing[1:]))
            fair_ends[edge['id']] = ((ids[1], spacing[0], h),
                                     (ids[-2], spacing[-1], h))
            if edge.get('closed'):
                bends.append((ids[-2], ids[0], ids[1]))
                coefficients.extend(row(spacing[-1:], spacing[:1]))
        # A crossing splits a physical strand into graph edges. Fair through
        # its opposite ports as well: otherwise the crossing is an unpenalized
        # hinge, which can flatten while every mesh triangle stays positive.
        # Adjacent ports belong to different strands and must not be joined.
        for crossing in diagram['crossings']:
            if is_twist_box(crossing):
                continue
            for port in (0, 1):
                a, b = crossing['ports'][port], crossing['ports'][port+2]
                na, left, ha = fair_ends[a['edge']][a['end']]
                nb, right, hb = fair_ends[b['edge']][b['end']]
                h = (ha+hb)/2
                scale = np.sqrt(h**3/((left+right)/2))
                bends.append((na, self.crossing_nodes[crossing['id']], nb))
                coefficients.append((scale/left, -scale*(1/left+1/right), scale/right))
        self.n_moving = len(nodes)
        self.sampling_pairs = np.asarray(sampling_pairs, int)
        self.bends = np.asarray(bends, int)
        self.bend_coefficients = np.asarray(coefficients)
        self.bending_weight = 20.
        # Solve the sparse bending coupling instead of damping each landmark
        # independently. The old diagonal damping made even smooth, large-scale
        # motion take hundreds of tiny steps to avoid exciting stiff wiggles.
        # I + Hessian(E_bending) is positive definite, so its inverse gives a
        # descent direction for the same combined energy. Factor only the free
        # coordinates, once per mesh: exterior points remain exactly fixed.
        bending = coo_matrix((self.bend_coefficients.ravel(),
                              (np.repeat(np.arange(len(self.bends)), 3), self.bends.ravel())),
                             shape=(len(self.bends), self.n_moving)).tocsr()
        self.preconditioner = (eye(self.n_moving)+self.bending_weight*(bending.T@bending)).tocsc()
        self._configure_box_attachments()
        self.base_preconditioner = self.preconditioner
        self.adjacency = np.zeros((self.n_graph, self.n_graph), bool)
        for a, b in links:
            self.adjacency[a, b] = self.adjacency[b, a] = True
        # Include the lasso's vertices and regular boundary samples in the mesh.
        for a, b in zip(self.region, np.roll(self.region, -1, axis=0)):
            count = max(1, int(math.ceil(np.linalg.norm(b-a)/self.unit)))
            for t in np.arange(count)/count:
                node(a+t*(b-a))
        # Fixed rectangle boundaries are holes in the ambient flow. Include
        # corners, ports, and boundary samples without adding them to the
        # all-pairs strand energy or expanding the implicit twists.
        for box, boundary in zip(self.boxes, self.box_boundaries):
            for p in box_port_points(box):
                node(p)
            for a, b in zip(boundary, np.roll(boundary, -1, axis=0)):
                count = max(1, int(math.ceil(np.linalg.norm(b-a)/(self.unit*.5))))
                for t in np.arange(count)/count:
                    node(a+t*(b-a))
        bounds = np.vstack([nodes, self.region])
        lo, hi = bounds.min(axis=0)-2*self.unit, bounds.max(axis=0)+2*self.unit
        for p in ([lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]):
            node(p)
        self.region_unit = self.region/self.unit
        self.box_boundaries_unit = [p/self.unit for p in self.box_boundaries]
        self._configure_mesh(np.asarray(nodes)/self.unit)
        self.sampling_lengths = np.linalg.norm(
            self.positions[self.sampling_pairs[:, 1]]-self.positions[self.sampling_pairs[:, 0]], axis=1)
        self.bindings = [self._bind_path(p) for p in self.edge_geometry]
        self._configure_box_tips()
        self._configure_fair_guard()
        self._configure_center(_center_target)
        self.energy, _ = self._energy_gradient(self.positions)
        self.initial_energy = self.energy
        self.iterations = 0
        self.remeshes = 0
        self.resamples = 0
        self.sampling_iteration = 0
        self.full_sampling_iteration = 0
        self.last_resample_attempt = 0
        self.sampling_warning = ''
        self.reason = ''
        _check_stop(stop)

    def _configure_box_attachments(self):
        """Penalize tangential departure near a fixed box port.

        The first few fairing samples supply a short attachment neighborhood.
        Squared lateral displacement normalized by material distance is minimized
        by a perpendicular strand, with no preferred axial length. The
        diagonal Hessian bound improves the existing sparse preconditioner.
        Passage ports use the normal of the side they actually meet.
        """
        nodes, anchors, tangents, weights = [], [], [], []
        self.attachment_weight = 40.
        for box, boundary in zip(self.boxes, self.box_boundaries):
            for port, ref in zip(box_port_points(box), box['ports']):
                p = np.asarray(port, float)
                a, b = boundary, np.roll(boundary, -1, axis=0)
                delta = b-a
                t = np.clip(np.sum((p-a)*delta, axis=1)/np.sum(delta*delta, axis=1), 0, 1)
                side = int(np.argmin(np.linalg.norm(p-a-t[:, None]*delta, axis=1)))
                tangent = delta[side]/np.linalg.norm(delta[side])
                ids = self.fair_ids[ref['edge']][::1 if ref['end'] == 0 else -1]
                samples = self.fair_distances[ref['edge']]
                if ref['end']:
                    samples = samples[-1]-samples[::-1]
                for rank, ident in enumerate(ids[1:4], 1):
                    nodes.append(ident)
                    anchors.append(ids[0])
                    tangents.append(tangent)
                    distance = max(float(samples[rank]), .06)
                    weights.append(self.attachment_weight*(1., .4, .15)[rank-1]/distance**2)
        self.attachment_nodes = np.asarray(nodes, int)
        self.attachment_anchors = np.asarray(anchors, int)
        self.attachment_tangents = np.asarray(tangents, float).reshape(-1, 2)
        self.attachment_weights = np.asarray(weights)
        if nodes:
            diagonal = np.bincount(nodes, weights=weights, minlength=self.n_moving)
            self.preconditioner += coo_matrix((diagonal, (np.arange(self.n_moving), np.arange(self.n_moving))),
                                             shape=(self.n_moving, self.n_moving)).tocsc()

    def _attachment_energy_gradient(self, positions):
        gradient = np.zeros_like(positions)
        ids, anchors = self.attachment_nodes, self.attachment_anchors
        tangent = self.attachment_tangents
        sideways = np.sum((positions[ids]-positions[anchors])*tangent, axis=1)
        forces = (self.attachment_weights*sideways)[:, None]*tangent
        np.add.at(gradient, ids, forces)
        np.add.at(gradient, anchors, -forces)
        return .5*float(np.sum(self.attachment_weights*sideways**2)), gradient

    def _configure_box_tips(self):
        """Bind the actual entry tangent, which can fall between fair samples.

        Each polyline is already split at mesh faces, so its first segment is
        an exact affine image. Its angular energy is independent of segment
        length and hence unchanged by remeshing or redundant subdivisions.
        Normalize the sparse row by the initial segment length to keep even
        very short, previously subdivided attachment segments well scaled.
        """
        self.tip_matrix = None
        self.tip_weight = 40.
        if not self.boxes:
            return
        bindings = {edge['id']: binding for edge, binding in zip(self.original['edges'], self.bindings)}
        rows, columns, values, directions = [], [], [], []
        for box in self.boxes:
            for ref in box['ports']:
                _check_stop(self.stop)
                reverse = 1 if ref['end'] == 0 else -1
                points, ids, bary = [a[::reverse] for a in bindings[ref['edge']]]
                distances = np.linalg.norm(points[1:]-points[0], axis=1)
                next_point = int(np.flatnonzero(distances > 1e-7)[0])+1
                vector = (points[next_point]-points[0])/self.unit
                length = float(np.linalg.norm(vector))
                row = len(directions)
                directions.append(vector/length)
                for sign, index in ((-1., 0), (1., next_point)):
                    rows.extend([row]*3)
                    columns.extend(ids[index])
                    values.extend(sign*bary[index]/length)
        self.tip_matrix = coo_matrix((values, (rows, columns)),
                                      shape=(len(directions), len(self.positions))).tocsr()
        self.tip_directions = np.asarray(directions)
        self.tip_tangents = self.attachment_tangents[::3]
        active = self.tip_matrix[:, :self.n_moving]
        self.preconditioner = (self.base_preconditioner+self.tip_weight*(active.T@active)).tocsc()
        self.descent_solver = splu(self.preconditioner[self.free_nodes][:, self.free_nodes])
        self._cached_energy_evaluation = None

    def _tip_energy_gradient(self, positions):
        direction = self.tip_directions+self.tip_matrix@(positions-self.rest)
        length2 = np.maximum(np.sum(direction*direction, axis=1), 1e-20)
        lateral = np.sum(direction*self.tip_tangents, axis=1)
        energy = .5*self.tip_weight*float(np.sum(lateral**2/length2))
        force = self.tip_weight*((lateral/length2)[:, None]*self.tip_tangents-
                                (lateral**2/length2**2)[:, None]*direction)
        return energy, self.tip_matrix.T@force

    def _energy_gradient(self, positions):
        """Selected strand energy, with fixed-box attachment terms."""
        _check_stop(self.stop)
        cached = getattr(self, '_cached_energy_evaluation', None)
        if (cached is not None and cached[3] == self.bending_weight and
                np.array_equal(positions, cached[0])):
            # The accepted line-search trial is the next step's starting
            # point. Reuse its exact evaluation, while protecting the cache
            # against callers mutating either input positions or gradients.
            return cached[1], cached[2].copy()
        energy, coarse = energy_gradient(positions[:self.n_graph], self.adjacency, self.stop)
        gradient = np.zeros_like(positions)
        gradient[:self.n_graph] = coarse
        a, b, c = self.bends.T
        ca, cb, cc = self.bend_coefficients.T[:, :, None]
        turn = ca*positions[a]+cb*positions[b]+cc*positions[c]
        energy += .5*self.bending_weight*float(np.sum(turn*turn))
        force = self.bending_weight*turn
        np.add.at(gradient, a, ca*force)
        np.add.at(gradient, b, cb*force)
        np.add.at(gradient, c, cc*force)
        if len(self.attachment_nodes):
            attachment_energy, attachment_gradient = self._attachment_energy_gradient(positions)
            energy += attachment_energy
            gradient += attachment_gradient
            tip_energy, tip_gradient = self._tip_energy_gradient(positions)
            energy += tip_energy
            gradient += tip_gradient
        self._cached_energy_evaluation = positions.copy(), energy, gradient.copy(), self.bending_weight
        return energy, gradient

    def _configure_mesh(self, rest):
        """Build an ambient chart with fixed exterior and box boundaries."""
        _check_stop(self.stop)
        mesh = Delaunay(rest)
        self.rest = rest.copy()
        self.positions = rest.copy()
        self.mesh = mesh
        self.triangles = self.mesh.simplices
        if len(np.unique(self.triangles)) != len(self.rest):
            raise ValueError('Nearly coincident layout points could not be meshed reliably. Separate them before relaxing this region.')
        free = np.zeros(len(rest), bool)
        collar = .22
        free[:self.n_moving] = _inside(self.rest[:self.n_moving], self.region_unit) & (_boundary_distance(self.rest[:self.n_moving], self.region_unit) > collar)
        # Freeze any face that is outside or cut by the lasso. Therefore the
        # affine extension is exactly identity on the entire exterior/boundary.
        faces = self.rest[self.triangles]
        bad = ~_inside(faces.mean(axis=1), self.region_unit)
        mesh_edges = np.sort(np.concatenate([self.triangles[:, [0, 1]], self.triangles[:, [1, 2]], self.triangles[:, [2, 0]]]), axis=1)
        self.mesh_edges = np.unique(mesh_edges, axis=0)
        self._index_mesh_edges()
        hit, _, t, u = _segment_hits(self.rest[self.mesh_edges[:, 0]], self.rest[self.mesh_edges[:, 1]],
                                    self.region_unit, np.roll(self.region_unit, -1, axis=0))
        cut = set(map(tuple, self.mesh_edges[hit[(t > 1e-8) & (t < 1-1e-8) & (u > 1e-8) & (u < 1-1e-8)]]))
        for i, tri in enumerate(self.triangles):
            if any(tuple(sorted((tri[j], tri[(j+1) % 3]))) in cut for j in range(3)):
                bad[i] = True
        free[np.unique(self.triangles[bad])] = False
        # Freeze every triangle meeting a box interior. With positive triangle
        # area throughout a step, the affine extension then fixes each entire
        # box pointwise and cannot push an exterior strand through its border.
        # Unlike the lasso's protective collar, no extra distance is frozen:
        # nearby strand samples need freedom to align with the box normal.
        for boundary in self.box_boundaries_unit:
            _check_stop(self.stop)
            inside = _inside(rest, boundary)
            on_border = _boundary_distance(rest, boundary) < 1e-8
            free[inside | on_border] = False
            box_faces = _inside(faces.mean(axis=1), boundary)
            hit, _, t, u = _segment_hits(rest[self.mesh_edges[:, 0]], rest[self.mesh_edges[:, 1]],
                                        boundary, np.roll(boundary, -1, axis=0))
            cut = set(map(tuple, self.mesh_edges[hit[(t > 1e-8) & (t < 1-1e-8) & (u > 1e-8) & (u < 1-1e-8)]]))
            for i, tri in enumerate(self.triangles):
                if any(tuple(sorted((tri[j], tri[(j+1) % 3]))) in cut for j in range(3)):
                    box_faces[i] = True
            free[np.unique(self.triangles[box_faces])] = False
        self.free = free
        if not np.any(free):
            raise ValueError('No movable strand lies safely inside this region. Circle a larger area.')
        self.free_nodes = np.flatnonzero(free[:self.n_moving])
        self.fixed_binding_node = int(np.flatnonzero(~free)[0])
        self.preconditioner = self.base_preconditioner
        self.descent_solver = splu(self.preconditioner[self.free_nodes][:, self.free_nodes])
        _check_stop(self.stop)
        self.initial_area = _cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])

    def _configure_center(self, target=None):
        """Keep the actual arclength centroid fixed for whole-diagram runs.

        The target survives resampling; a new sampling's landmark mean must
        not become the new translation reference. Partial tangles retain their
        original boundary-only constraints.
        """
        self.center_target = None
        self.center_bindings = None
        self._center_arrays = None
        if target is False or (target is None and not np.all(self.free[:self.n_moving])):
            return
        self.center_bindings = []
        for points, ids, bary in self.bindings:
            # Translating free mesh nodes by t moves this polyline point by
            # a*t. A may be below one near fixed scaffold triangles, even
            # when every energy landmark is initially free.
            influence = np.sum(bary*self.free[ids], axis=1)
            self.center_bindings.append((points, ids, bary, influence))
        # Batch the integrals across edges. Keep explicit segment indices so
        # concatenation never connects unrelated strands at an edge break.
        starts, offset = [], 0
        for points, _, _, _ in self.center_bindings:
            starts.append(np.arange(offset, offset+len(points)-1))
            offset += len(points)
        self._center_arrays = (*[np.concatenate([binding[k] for binding in self.center_bindings])
                                 for k in range(4)], np.concatenate(starts))
        self.center_target = (self._curve_center(self.positions)[0] if target is None
                              else np.asarray(target, float).copy())
        _check_stop(self.stop)

    def _curve_center(self, positions):
        """Actual polyline arclength centroid and its 2-D translation Jacobian."""
        _check_stop(self.stop)
        displacement = positions-self.rest
        original, ids, bary, influence, a = self._center_arrays
        b = a+1
        points = original+self.unit*np.einsum('ni,nij->nj', bary, displacement[ids])
        segments = points[b]-points[a]
        lengths = np.linalg.norm(segments, axis=1)
        midpoints = (points[a]+points[b])/2
        total = lengths.sum()
        moment = np.sum(lengths[:, None]*midpoints, axis=0)
        # Derivatives with respect to a free-node translation in pixels.
        grad_length = ((influence[b]-influence[a])/np.maximum(lengths, 1e-20))[:, None]*segments
        length_gradient = grad_length.sum(axis=0)
        moment_gradient = midpoints.T@grad_length
        moment_gradient += np.eye(2)*np.sum(lengths*(influence[a]+influence[b])/2)
        center = moment/total
        jacobian = (moment_gradient-np.outer(center, length_gradient))/total
        _check_stop(self.stop)
        return center, jacobian

    def _center_trial(self, proposal):
        """Center before certification, never by moving an accepted drawing.

        Newton's two-variable correction accounts for fixed scaffold influence
        on the exact polylines. Every corrected move still needs the ordinary
        whole-step nonfolding and actual energy-decrease checks.
        """
        if self.center_target is None:
            return proposal
        proposal = proposal.copy()
        for _ in range(8):
            center, jacobian = self._curve_center(proposal)
            error = self.center_target-center
            if np.linalg.norm(error) < 1e-7:
                return proposal
            try:
                shift = np.linalg.solve(jacobian, error)/self.unit
            except np.linalg.LinAlgError:
                return None
            if not np.isfinite(shift).all():
                return None
            proposal[self.free_nodes] += shift
        return None

    def _center_gradient(self):
        """Derivative of the geometric centroid with respect to free nodes."""
        _check_stop(self.stop)
        original, ids, bary, _, a = self._center_arrays
        b = a+1
        points = original+self.unit*np.einsum('ni,nij->nj', bary, (self.positions-self.rest)[ids])
        segments = points[b]-points[a]
        lengths = np.linalg.norm(segments, axis=1)
        total = lengths.sum()
        middle = (points[a]+points[b])/2
        center = np.sum(lengths[:, None]*middle, axis=0)/total
        tangent = segments/np.maximum(lengths[:, None], 1e-20)
        change = (middle-center)[:, :, None]*tangent[:, None, :]
        diagonal = lengths[:, None, None]*np.eye(2)/2
        vertex = np.zeros((len(points), 2, 2))
        np.add.at(vertex, a, (diagonal-change)*(self.unit/total))
        np.add.at(vertex, b, (diagonal+change)*(self.unit/total))
        nodes = np.zeros((len(self.positions), 2, 2))
        for corner in range(3):
            np.add.at(nodes, ids[:, corner], bary[:, corner, None, None]*vertex)
        _check_stop(self.stop)
        return nodes[self.free_nodes]

    def _index_mesh_edges(self):
        """Build one exact broad-phase index for all strand bindings in a chart.

        Re-running an all-pairs sweep on the mesh for every graph edge spent
        most remeshing time rediscovering mesh/mesh contacts that binding never
        uses. Unit grid cells index only mesh segments. Long outside scaffolds
        use an overflow list rather than filling an enormous grid rectangle.
        """
        self._mesh_a = self.rest[self.mesh_edges[:, 0]]
        self._mesh_b = self.rest[self.mesh_edges[:, 1]]
        self._mesh_low = np.minimum(self._mesh_a, self._mesh_b)-1e-9
        self._mesh_high = np.maximum(self._mesh_a, self._mesh_b)+1e-9
        grid, overflow = defaultdict(list), []
        lower, upper = np.floor(self._mesh_low).astype(int), np.floor(self._mesh_high).astype(int)
        for index, ((x0, y0), (x1, y1)) in enumerate(zip(lower, upper)):
            if index % 256 == 0:
                _check_stop(self.stop)
            if (x1-x0+1)*(y1-y0+1) > 128:
                overflow.append(index)
                continue
            for x in range(x0, x1+1):
                for y in range(y0, y1+1):
                    grid[x, y].append(index)
        self._mesh_grid = {key: np.asarray(values, int) for key, values in grid.items()}
        self._mesh_overflow = np.asarray(overflow, int)

    def _mesh_segment_hits(self, a, b):
        """Intersect strand segments against the reusable mesh index exactly."""
        low, high = np.minimum(a, b)-1e-9, np.maximum(a, b)+1e-9
        lower, upper = np.floor(low).astype(int), np.floor(high).astype(int)
        groups = defaultdict(list)
        for index, (lo, hi) in enumerate(zip(lower, upper)):
            groups[tuple((*lo, *hi))].append(index)
        segment_ids, fractions = [], []
        for (x0, y0, x1, y1), group in groups.items():
            _check_stop(self.stop)
            if (x1-x0+1)*(y1-y0+1) > 128:
                possible = np.arange(len(self.mesh_edges))
            else:
                candidates = [self._mesh_overflow]
                candidates.extend(self._mesh_grid.get((x, y), ())
                                  for x in range(x0, x1+1) for y in range(y0, y1+1))
                possible = np.unique(np.concatenate([c for c in candidates if len(c)])) if any(map(len, candidates)) else np.empty(0, int)
            if not len(possible):
                continue
            # Bound temporary pair arrays as well as Stop latency when many
            # densely sampled segments occupy one cell.
            batch = max(1, min(128, 65536//len(possible)))
            for left in range(0, len(group), batch):
                _check_stop(self.stop)
                i = np.repeat(group[left:left+batch], len(possible))
                j = np.tile(possible, len(group[left:left+batch]))
                keep = np.all(low[i] <= self._mesh_high[j], axis=1) & np.all(self._mesh_low[j] <= high[i], axis=1)
                i, j = i[keep], j[keep]
                v, w, q = b[i]-a[i], self._mesh_b[j]-self._mesh_a[j], self._mesh_a[j]-a[i]
                det = _cross(v, w)
                safe = np.where(abs(det) > 1e-12, det, 1.)
                t, u = _cross(q, w)/safe, _cross(q, v)/safe
                keep = (abs(det) > 1e-12) & (t >= -1e-9) & (t <= 1+1e-9) & (u >= -1e-9) & (u <= 1+1e-9)
                segment_ids.extend(i[keep].tolist())
                fractions.extend(np.clip(t[keep], 0, 1).tolist())
        return np.asarray(segment_ids, int), np.asarray(fractions)

    def _remesh(self):
        """Refresh numerical triangles without changing the current diagram.

        Triangulation edges are coordinate scaffolding, not physical strands.
        Keeping them forever eventually traps the gradient against an arbitrary
        skinny triangle. Rebind the *current exact polylines* to a fresh mesh;
        no vertices or crossings move, and the same graph energy continues.
        Construct on a temporary object so cancellation cannot partly commit it.
        """
        _check_stop(self.stop)
        candidate = copy(self)
        candidate.original = self.diagram()
        candidate.edge_geometry = [np.asarray(e['points'], float) for e in candidate.original['edges']]
        candidate._configure_mesh(self.positions)
        candidate.bindings = [candidate._bind_path(p) for p in candidate.edge_geometry]
        candidate._configure_box_tips()
        candidate._configure_fair_guard(self._fair_reference)
        candidate._configure_center(self.center_target if self.center_target is not None else False)
        _check_stop(self.stop)
        candidate.remeshes += 1
        self.__dict__.update(candidate.__dict__)

    def _sampling_strain(self):
        """Arclength-weighted upper-quartile strain of fairing intervals.

        A few tiny feature intervals can become extremely short without
        making the overall strand sampling poor. Letting their maximum trigger
        a rebuild repeatedly is expensive and creates needless subdivisions.
        Periodic refreshing still discovers isolated new bends.
        """
        lengths = np.linalg.norm(
            self.positions[self.sampling_pairs[:, 1]]-self.positions[self.sampling_pairs[:, 0]], axis=1)
        ratio = np.maximum(lengths, 1e-12)/np.maximum(self.sampling_lengths, 1e-12)
        strain = np.maximum(ratio, 1/ratio)
        order = np.argsort(strain)
        weights = np.cumsum(self.sampling_lengths[order])
        return float(strain[order[np.searchsorted(weights, .75*weights[-1])]])

    def _underresolved_neighborhood(self, errors):
        """Refresh both sides of nearby crossings, not isolated graph edges."""
        selected = errors > np.maximum(.015, 1.1*self._fair_reference)
        threatened = set(map(tuple, self._fair_arrays[-1][selected]))
        affected = {eid for eid, fine in self.fair_ids.items()
                    if any(pair in threatened for pair in zip(fine[:-1], fine[1:]))}
        for _ in range(2):
            neighbors = set()
            for crossing in self.original['crossings']:
                incident = {ref['edge'] for ref in crossing['ports']}
                if incident & affected:
                    neighbors.update(incident)
            affected.update(neighbors)
        return affected

    def _retained_sampling_anchors(self, diagram, affected):
        """Keep all unaffected energy points at their current material positions."""
        anchors = {}
        for edge in diagram['edges']:
            _check_stop(self.stop)
            eid = edge['id']
            if eid in affected:
                continue
            original = np.asarray(edge['points'], float)
            low, high = self.region_bounds
            if np.any(original.max(axis=0) < low) or np.any(original.min(axis=0) > high):
                # These paths are exactly fixed but are intentionally not
                # split at their fairing points in the ambient mesh.
                continue
            fine = self.positions[self.fair_ids[eid]]*self.unit
            distance, location = cKDTree(original).query(fine)
            location[0], location[-1] = 0, len(original)-1
            if distance.max() > 1e-4*self.unit or np.any(np.diff(location) <= 0):
                raise ValueError('Fairing landmarks could not be retained in strand order.')
            lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(original, axis=0), axis=1))]
            anchors[eid] = (self.positions[self.coarse_ids[eid]]*self.unit,
                            lengths[location], fine)
        return anchors

    def _resample(self, affected=None):
        """Redistribute energy landmarks on the exact current curves, atomically.

        This is a new discretization, not a geometric move. Never replace the
        drawing with chords between the new landmarks. Keep the physical unit
        and selection fixed, but refresh bending weights and the whole solve.
        Energy values across this boundary are not directly comparable.
        """
        _check_stop(self.stop)
        current = self.diagram()
        retained = (None if affected is None else self._retained_sampling_anchors(current, affected))
        candidate = type(self)(current, self.region, self.stop,
                               dynamic_sampling=self.dynamic_sampling, _unit=self.unit,
                               _coarse_counts=self.coarse_counts,
                               _center_target=self.center_target if self.center_target is not None else False,
                               _retained_anchors=retained)
        candidate.iterations = self.iterations
        candidate.resample_min_interval = self.resample_min_interval
        candidate.resample_interval = self.resample_interval
        candidate.remeshes = self.remeshes
        candidate.resamples = self.resamples+1
        candidate.sampling_iteration = self.iterations
        candidate.full_sampling_iteration = (self.iterations if affected is None
                                            else self.full_sampling_iteration)
        if retained is not None:
            # Retaining an arc's landmarks must also retain its allowance;
            # repeated refreshes elsewhere cannot gradually loosen its guard.
            for eid in retained:
                candidate._fair_reference[candidate._fair_slices[eid]] = self._fair_reference[self._fair_slices[eid]]
        candidate.last_resample_attempt = self.iterations
        _check_stop(self.stop)
        self.__dict__.update(candidate.__dict__)

    def _refresh_sampling(self, *, stalled=False):
        if not self.dynamic_sampling:
            return False
        age = self.iterations-self.last_resample_attempt
        if age < 1:
            return False
        full_age = self.iterations-self.full_sampling_iteration
        periodic = full_age >= self.resample_interval
        errors = self._fair_guard_errors(self.positions)
        underresolved = age >= 1 and np.any(errors > np.maximum(.015, 1.1*self._fair_reference))
        if not underresolved and not periodic and age < (1 if stalled else self.resample_min_interval):
            return False
        strained = full_age >= self.resample_min_interval and self._sampling_strain() >= 1.5
        if not underresolved and not stalled and not periodic and not strained:
            return False
        try:
            if underresolved and not periodic and not strained:
                self._resample(self._underresolved_neighborhood(errors))
            else:
                self._resample()
        except (ValueError, QhullError) as exc:
            self.last_resample_attempt = self.iterations
            self.sampling_warning = str(exc)
            return False
        return True

    def _bind_path(self, points):
        """Split at mesh edges so subsequent affine images are exact polylines."""
        _check_stop(self.stop)
        low, high = self.region_bounds
        if np.any(points.max(axis=0) < low) or np.any(points.min(axis=0) > high):
            # An entire outside strand is acted on by the identity. There is
            # no reason to split it at every outside scaffold edge first.
            ids = np.full((len(points), 3), self.fixed_binding_node, int)
            bary = np.zeros((len(points), 3))
            bary[:, 0] = 1.
            return points.copy(), ids, bary
        normalized = points/self.unit
        simplex_at_vertices = self.mesh.find_simplex(normalized, tol=0.)
        # A triangle is convex: a segment with both ends strictly inside the
        # same face needs no mesh cuts. Most vertices from earlier charts are
        # such redundant subdivisions. Keep the full intersection test for
        # boundary points and every numerically ambiguous case.
        transforms = self.mesh.transform[np.maximum(simplex_at_vertices, 0)]
        uv = np.einsum('nij,nj->ni', transforms[:, :2], normalized-transforms[:, 2])
        bary_at_vertices = np.column_stack([uv, 1-uv.sum(axis=1)])
        strict_interior = ((simplex_at_vertices >= 0) &
                           np.isfinite(bary_at_vertices).all(axis=1) &
                           (bary_at_vertices.min(axis=1) > 1e-9))
        uncertain = np.flatnonzero(~strict_interior[:-1] | ~strict_interior[1:] |
                                   (simplex_at_vertices[:-1] != simplex_at_vertices[1:]))
        a, b = normalized[:-1], normalized[1:]
        i, t = self._mesh_segment_hits(a[uncertain], b[uncertain])
        i = uncertain[i]
        cuts = [[] for _ in a]
        for segment, fraction in zip(i, t):
            if 1e-8 < fraction < 1-1e-8:
                cuts[segment].append(float(fraction))
        expanded = []
        for j, (x, y) in enumerate(zip(points[:-1], points[1:])):
            expanded.append(x)
            for v in sorted(set(round(v, 12) for v in cuts[j])):
                point = x+v*(y-x)
                # Merge numerically duplicated *new* mesh cuts, without
                # removing a preexisting material vertex in a short segment.
                if np.linalg.norm(point-expanded[-1]) >= 1e-8 and np.linalg.norm(point-y) >= 1e-8:
                    expanded.append(point)
        expanded.append(points[-1])
        original = np.asarray(expanded)
        p = original/self.unit
        simplex = self.mesh.find_simplex(p, tol=1e-8)
        if np.any(simplex < 0):
            raise ValueError('A strand lies outside the relaxation mesh.')
        transform = self.mesh.transform[simplex]
        uv = np.einsum('nij,nj->ni', transform[:, :2], p-transform[:, 2])
        bary = np.column_stack([uv, 1-uv.sum(axis=1)])
        ids = self.triangles[simplex]
        return original, ids, bary

    def _configure_fair_guard(self, reference=None):
        """Bind a shape-resolution check to the actual curves, not the mesh.

        Positive triangle area allows arbitrary shear and a tiny unsampled
        bend can become a hairpin. Monitor each exact polyline between its
        ordered fairing nodes. The tolerance belongs to the sampling epoch;
        changing the triangulation must never reset it.
        """
        originals, mesh_ids, barys, groups, starts, pairs = [], [], [], [], [], []
        self._fair_slices = {}
        point_offset = interval_offset = 0
        for edge, (original, ids, bary) in zip(self.original['edges'], self.bindings):
            _check_stop(self.stop)
            low, high = self.region_bounds
            if np.any(original.max(axis=0) < low) or np.any(original.min(axis=0) > high):
                # _bind_path leaves these identity-bound exterior polylines
                # unsplit. They cannot deform and need no resolution guard.
                continue
            fine = self.fair_ids[edge['id']]
            distance, location = cKDTree(original).query(self.positions[fine]*self.unit)
            # A loop's two ends have the same position but distinct order.
            location[0], location[-1] = 0, len(original)-1
            if distance.max() > 1e-4*self.unit or np.any(np.diff(location) <= 0):
                raise ValueError('Fairing landmarks could not be located in strand order.')
            bracket = (np.searchsorted(location, np.arange(len(original)), side='right')-1).clip(0, len(fine)-2)
            originals.append(original/self.unit)
            mesh_ids.append(ids)
            barys.append(bary)
            groups.append(bracket+interval_offset)
            starts.append(np.arange(len(original)-1)+point_offset)
            pairs.append(np.column_stack((fine[:-1], fine[1:])))
            self._fair_slices[edge['id']] = slice(interval_offset, interval_offset+len(fine)-1)
            point_offset += len(original)
            interval_offset += len(fine)-1
        self._fair_arrays = tuple(np.concatenate(a) for a in
                                  (originals, mesh_ids, barys, groups, starts, pairs))
        self._fair_reference = (self._fair_guard_errors(self.positions) if reference is None
                                else reference.copy())
        _check_stop(self.stop)

    def _fair_guard_errors(self, positions):
        """Unrepresented bend/backtracking per fairing interval, in units."""
        _check_stop(self.stop)
        original, ids, bary, group, starts, pairs = self._fair_arrays
        points = original+np.einsum('ni,nij->nj', bary, (positions-self.rest)[ids])
        a = positions[pairs[:, 0]]
        vector = positions[pairs[:, 1]]-a
        length2 = np.maximum(np.sum(vector*vector, axis=1), 1e-25)
        offset = points-a[group]
        fraction = np.clip(np.sum(offset*vector[group], axis=1)/length2[group], 0., 1.)
        distance = np.linalg.norm(offset-fraction[:, None]*vector[group], axis=1)
        errors = np.zeros(len(pairs))
        np.maximum.at(errors, group, distance)
        # Distance alone misses a thin hairpin lying along/inside its chord.
        # Its reverse travel is also unresolved geometry and needs landmarks.
        reverse = np.maximum(-np.sum((points[starts+1]-points[starts])*vector[group[starts]], axis=1)
                             /np.sqrt(length2[group[starts]]), 0.)
        backtracking = np.bincount(group[starts], weights=reverse, minlength=len(pairs))
        _check_stop(self.stop)
        return np.maximum(errors, backtracking)

    def _fair_step_resolved(self, positions):
        return bool(np.all(self._fair_guard_errors(positions) <=
                           np.maximum(.02, 1.25*self._fair_reference)))

    def _valid_step(self, delta):
        p, d = self.positions[self.triangles], delta[self.triangles]
        a, b = p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]
        da, db = d[:, 1]-d[:, 0], d[:, 2]-d[:, 0]
        c0, c1, c2 = _cross(a, b), _cross(da, b)+_cross(a, db), _cross(da, db)
        # A positive quadratic can dip below zero between positive endpoints.
        t = np.clip(-c1/np.where(abs(c2) > 1e-20, 2*c2, 1.), 0, 1)
        minimum = np.minimum(c0, c0+c1+c2)
        minimum = np.where(c2 > 0, np.minimum(minimum, c0+t*c1+t*t*c2), minimum)
        return bool(np.all(minimum > self.initial_area*1e-5))

    def _descent_direction(self, gradient):
        direction = np.zeros_like(self.positions)
        direction[self.free_nodes] = self.descent_solver.solve(-gradient[self.free_nodes])
        if self.center_target is not None and not np.all(self.free[:self.n_moving]):
            # The whole diagram may reach the fixed collar later in the run.
            # A translation of only its remaining free nodes is no longer an
            # energy symmetry. Project the descent solve onto the centroid's
            # tangent constraint; the subsequent Newton correction is then
            # second order, so line search can still find a descending step.
            constraint = self._center_gradient()  # node, output, coordinate
            response = self.descent_solver.solve(constraint.reshape(len(self.free_nodes), 4)).reshape(-1, 2, 2)
            gram = np.einsum('noi,npi->op', constraint, response)
            drift = np.einsum('noi,ni->o', constraint, direction[self.free_nodes])
            multiplier = np.linalg.pinv(gram, rcond=1e-12)@drift
            direction[self.free_nodes] -= np.einsum('noi,o->ni', response, multiplier)
        _check_stop(self.stop)
        return direction

    def step(self, speed=1., *, _sampling_retry=False):
        """One topology-preserving descent step within the current sampling.

        ``speed`` (0.25 to 8) can change on every call, including during a live
        minimization. It scales the trial time step and displacement ceiling,
        not the number of iterations. Backtracking still limits a step when
        energy descent or the all-time, nonfolding certificate requires it.
        """
        if not np.isfinite(speed) or not .25 <= speed <= 8:
            raise ValueError('Energy speed must be between 0.25 and 8.')
        _check_stop(self.stop)
        self._refresh_sampling()
        energy, grad = self._energy_gradient(self.positions)
        direction = self._descent_direction(grad)
        maximum = np.linalg.norm(direction, axis=1).max()
        if not np.isfinite(energy) or not np.isfinite(maximum):
            raise ValueError('Energy is singular at touching landmarks; separate those strands first.')
        if maximum < 1e-7:
            if not _sampling_retry and self._refresh_sampling(stalled=True):
                return self.step(speed, _sampling_retry=True)
            self.reason = 'The selected region has reached a stationary layout.'
            return None
        # Rebuild a strained chart before it squeezes a strand into a nearly
        # degenerate triangle. The rebind is identity, not an animation step.
        faces = self.positions[self.triangles]
        area_ratio = _cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])/self.initial_area
        if np.min(area_ratio) < .08:
            self._remesh()
            # Rebuild the gradient on the refreshed chart before line search.
            energy, grad = self._energy_gradient(self.positions)
            direction = self._descent_direction(grad)
            maximum = np.linalg.norm(direction, axis=1).max()
        for attempt in range(2):
            alpha = speed*min(.08/max(maximum, 1e-20), 1.)
            for _ in range(22):
                _check_stop(self.stop)
                proposal = self._center_trial(self.positions+alpha*direction)
                if proposal is None:
                    alpha *= .5
                    continue
                delta = proposal-self.positions
                slope = float(np.sum(grad*delta))
                distance = np.linalg.norm(delta, axis=1).max()
                if (slope < 0 and distance <= speed*.08+1e-12 and
                        self._valid_step(delta) and self._fair_step_resolved(proposal)):
                    trial, _ = self._energy_gradient(proposal)
                    if trial <= energy+1e-4*slope:
                        _check_stop(self.stop)
                        if distance*self.unit < 1e-5:
                            break
                        self.positions = proposal
                        self.energy = trial
                        self.iterations += 1
                        return self.diagram()
                alpha *= .5
            if attempt == 0:
                self._remesh()
                energy, grad = self._energy_gradient(self.positions)
                direction = self._descent_direction(grad)
                maximum = np.linalg.norm(direction, axis=1).max()
        if not _sampling_retry and self._refresh_sampling(stalled=True):
            return self.step(speed, _sampling_retry=True)
        self.reason = 'Relaxation settled against the fixed boundary or reached numerical tolerance.'
        return None

    def diagram(self):
        displacement = self.positions-self.rest
        memo = {}
        for edge, (original, ids, bary) in zip(self.original['edges'], self.bindings):
            mapped = original+self.unit*np.einsum('ni,nij->nj', bary, displacement[ids])
            # Deepcopy metadata and incidence, but directly substitute the
            # newly mapped coordinate lists instead of recursively copying
            # thousands of old coordinates that would immediately be thrown
            # away. Every returned list remains independent of the engine.
            memo[id(edge['points'])] = mapped.tolist()
        result = deepcopy({key: value for key, value in self.original.items()
                           if key not in ('spatial_curves', 'spatial_fingerprint')}, memo)
        for crossing in result['crossings']:
            if not is_twist_box(crossing):
                crossing['point'] = (np.asarray(crossing['point'])+displacement[self.crossing_nodes[crossing['id']]]*self.unit).tolist()
        # Share the exact crossing endpoint coordinates on each incident edge.
        cross = {c['id']: c for c in result['crossings']}
        ports = {b['id']: box_port_points(b) for b in self.boxes}
        for edge in result['edges']:
            if edge.get('start') is not None:
                for key, index in (('start', 0), ('end', -1)):
                    cid, port = edge[key]
                    edge['points'][index] = (ports[cid][port] if cid in ports else cross[cid]['point']).copy()
            elif edge.get('closed'):
                edge['points'][-1] = edge['points'][0].copy()
        result.pop('spatial_curves', None)
        result.pop('spatial_fingerprint', None)
        result['relaxation'] = dict(iterations=self.iterations, remeshes=self.remeshes, energy=self.energy,
                                    energy_model='layout',
                                    initial_energy=self.initial_energy, boundary=self.region.tolist(),
                                    resamples=self.resamples, sampling_iteration=self.sampling_iteration,
                                    sampling_warning=self.sampling_warning,
                                    center_target=None if self.center_target is None else self.center_target.tolist())
        if self.boxes:
            result['relaxation']['fixed_boxes'] = [b['id'] for b in self.boxes]
        return result
