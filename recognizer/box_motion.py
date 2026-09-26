"""Smooth external strand motion while compact twist tangles stay fixed.

Only a parity-sized geometric proxy is used by the spatial motion engine.
Every accepted frame is recompressed before it is returned to the interface;
failure to certify a boundary attachment or passage also rolls back the
underlying gesture.  Stored twist magnitudes never determine motion cost.
"""
from __future__ import annotations

from copy import deepcopy
import math

import cv2
import numpy as np

from .diagram import component_walks, copy_diagram, validate
from .motion import SmoothDrag, MoveRejected
from .twist_boxes import (box_corners, box_port_points, bundle_ports,
                          expand_boxes, is_twist_box)
from .box_geometry import collapse_boxes


def _point_segment_distance(point, a, b):
    delta = b-a
    scale = float(delta @ delta)
    t = 0. if scale == 0 else float(np.clip((point-a) @ delta/scale, 0., 1.))
    return float(np.linalg.norm(point-a-t*delta))


def _nearest_edge(diagram, point, original_edge=None):
    point = np.asarray(point, float)
    best = None
    allowed = None
    lineage = diagram.get('twist_box_edge_lineage', {})
    if original_edge is not None and lineage:
        witnesses = {int(eid) for eid, records in lineage.items()
                     if any(old == original_edge for old, _ in records)}
        for walk in component_walks(diagram):
            if witnesses.intersection(eid for eid, _ in walk):
                allowed = {eid for eid, _ in walk}
                break
    for edge in diagram['edges']:
        if allowed is not None and edge['id'] not in allowed:
            continue
        points = np.asarray(edge['points'], float)[:, :2]
        delta = np.diff(points, axis=0)
        square = np.einsum('ij,ij->i', delta, delta)
        offset = point-points[:-1]
        along = np.divide(np.einsum('ij,ij->i', offset, delta), square,
                          out=np.zeros(len(square)), where=square > 0)
        offset -= np.clip(along, 0., 1.)[:, None]*delta
        distance = math.sqrt(float(np.min(np.einsum('ij,ij->i', offset, offset)))) if len(square) else math.inf
        candidate = (distance, edge['id'])
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise MoveRejected('Pick an external strand first.')
    return best[1]


def _densify_box_neighborhood(diagram, boxes):
    """Keep cubic fitting local at fixed attachment points and proxy corners."""
    rectangles = []
    for box in boxes:
        corners = np.asarray(box_corners(box))
        rectangles.append((corners.min(axis=0)-5., corners.max(axis=0)+5.))
    for edge in diagram['edges']:
        points = np.asarray(edge['points'], float)[:, :2]
        # Most samples of a large knot are nowhere near a box. Retain their
        # original coordinates and visit only segments that can hit a collar.
        lower = np.minimum(points[:-1], points[1:])
        upper = np.maximum(points[:-1], points[1:])
        nearby = np.zeros(len(points)-1, bool)
        for low, high in rectangles:
            nearby |= np.all(upper >= low, axis=1) & np.all(lower <= high, axis=1)
        if not nearby.any():
            continue
        refined = []
        last = 0
        for index in np.flatnonzero(nearby):
            refined.extend(points[last:index].tolist())
            a, b = points[index:index+2]
            delta = b-a
            intervals = []
            for low, high in rectangles:
                enter, leave = 0., 1.
                for axis in (0, 1):
                    if abs(delta[axis]) < 1e-12:
                        if a[axis] < low[axis] or a[axis] > high[axis]:
                            enter, leave = 1., 0.
                            break
                    else:
                        u, v = sorted(((low[axis]-a[axis])/delta[axis],
                                       (high[axis]-a[axis])/delta[axis]))
                        enter, leave = max(enter, u), min(leave, v)
                if leave > enter:
                    intervals.append((enter, leave))
            cuts = sorted({0., 1., *(t for interval in intervals for t in interval)})
            for start, finish in zip(cuts, cuts[1:]):
                near = any(lo <= (start+finish)/2 <= hi for lo, hi in intervals)
                # Refine only inside a collar, not an entire distant segment
                # that happens to pass through the box's bounding rectangle.
                count = max(1, int(math.ceil(np.linalg.norm(delta)*(finish-start)))) if near else 1
                refined.extend((a + delta*(start+(finish-start)*t/count)).tolist() for t in range(count))
            last = index+1
        refined.extend(points[last:].tolist())
        edge['points'] = refined
    return diagram


def _pinned_material_parameters(engine, boxes):
    """Identify bundle arcs in the selected curve, excluding existing passers."""
    curve = engine.curves[engine.ci]
    grid = curve.samples()
    points = curve.spline(grid)[:, :2]
    pinned = []
    # Work on the periodic grid without its repeated endpoint.
    grid, points = grid[:-1], points[:-1]
    # OpenCV evaluates Point2f coordinates, including rounding the query point.
    # Match that representation in the broad phase: a double just beyond a
    # boundary can round onto it, especially far from the canvas origin.
    point32 = points.astype(np.float32)
    for box in boxes:
        polygon = np.asarray(box_corners(box), np.float32)
        nearby = np.flatnonzero(np.all(point32 >= polygon.min(axis=0), axis=1)
                                & np.all(point32 <= polygon.max(axis=0), axis=1))
        inside = np.zeros(len(points), bool)
        inside[nearby] = [cv2.pointPolygonTest(polygon, tuple(map(float, p)), False) >= 0
                          for p in points[nearby]]
        if not inside.any():
            continue
        if inside.all():
            raise MoveRejected('The selected component lies entirely inside a twist box.')
        top, bottom = bundle_ports(box)
        anchors = np.asarray(box_port_points(box))[top+bottom]
        starts = np.flatnonzero(inside & ~np.roll(inside, 1))
        for start in starts:
            indexes, current = [], int(start)
            while inside[current]:
                indexes.append(current)
                current = (current+1) % len(grid)
            ends = points[[indexes[0], indexes[-1]]]
            # The endpoint sample is within one sampling chord of the boundary.
            # A complete passage elsewhere on the same link component remains
            # movable, even if that component also traverses the bundle.
            tolerance = max(3., min(box['size']) * .025)
            if any(float(np.linalg.norm(anchors-end, axis=1).min()) <= tolerance for end in ends):
                pinned.extend(float(grid[index]) for index in indexes)
    return np.asarray(pinned)


class BoxSmoothDrag:
    """Same public gesture API as SmoothDrag, with fixed compact box interiors."""
    def __init__(self, diagram, edge_id, point, radius=65., under=False):
        verdict = validate(diagram)
        if not verdict['valid']:
            raise MoveRejected('Repair the diagram before dragging: ' + '; '.join(verdict['errors']))
        if not any(e['id'] == edge_id for e in diagram['edges']):
            raise MoveRejected('Pick an external strand first.')
        self.original = copy_diagram(diagram)
        self.last_diagram = self.original
        self.boxes = [deepcopy(c) for c in diagram['crossings'] if is_twist_box(c)]
        self.anchor = np.asarray(point, float)
        self.current = np.zeros(2)
        self.under = bool(under)
        self.last_moves = []
        self.radius = float(radius)
        for box in self.boxes:
            if cv2.pointPolygonTest(np.asarray(box_corners(box), np.float32), tuple(map(float, point)), True) >= -1.:
                # A visible strand above the panel is editable there. The
                # hidden bundle and hidden underpassages remain fixed picks.
                on_passer = False
                for passage in box.get('passages', []):
                    if not passage['over'] or not any(box['ports'][p]['edge'] == edge_id for p in passage['ports']):
                        continue
                    path = np.asarray(passage['points'], float)
                    if min((_point_segment_distance(self.anchor, a, b) for a, b in zip(path, path[1:])),
                           default=math.inf) <= 4.:
                        on_passer = True
                        break
                if not on_passer:
                    raise MoveRejected('Pick a strand outside the twist box; its bundle and boundary stay fixed.')
        proxy = _densify_box_neighborhood(expand_boxes(diagram, proxy=True, keep_lineage=True), self.boxes)
        proxy_edge = _nearest_edge(proxy, point, original_edge=edge_id)
        self.engine = SmoothDrag(proxy, proxy_edge, point, radius=radius, under=under)
        pinned = _pinned_material_parameters(self.engine, self.boxes)
        if len(pinned):
            curve = self.engine.curves[self.engine.ci]
            distance = np.abs((pinned-self.engine.center+curve.length/2) % curve.length-curve.length/2)
            available = float(distance.min())-3.
            if available < 4.:
                raise MoveRejected('Pick farther from the box attachment to move a smooth neighborhood.')
            self.engine.radius = min(self.engine.radius, available)
            self.engine._prepare_regular()
        self.radius = self.engine.radius
        # Check smoothing once before permitting any motion. This cannot
        # silently accept a proxy whose boundary permutation changed at grab.
        try:
            collapse_boxes(self.engine.last_projection, self.boxes,
                           default_over=not self.under, attachment_tolerance=2., _array_geometry=True)
        except ValueError as exc:
            raise MoveRejected(str(exc)) from exc

    @property
    def last_diagram(self):
        """Expose an independent JSON frame while retaining arrays internally."""
        memo = {}
        for edge in self._last_diagram['edges']:
            points = edge['points']
            if isinstance(points, np.ndarray):
                memo[id(points)] = points.tolist()
        return copy_diagram(self._last_diagram, memo)

    @last_diagram.setter
    def last_diagram(self, diagram):
        self._last_diagram = diagram

    @property
    def checks(self):
        return self.engine.checks

    def move(self, target, under=None):
        requested = self.under if under is None else bool(under)
        # SmoothDrag uses replacement values for its physical state; saving
        # references is sufficient and avoids copying dense spline arrays.
        names = ('current', 'depth_bumps', 'last_diagram', 'last_projection',
                 'checks', 'last_moves', 'under', 'lift', 'last_step_stats')
        missing = object()
        saved = {name: getattr(self.engine, name, missing) for name in names}
        active = set(self.engine._active_components)
        try:
            # Recompression consumes the engine frame read-only. Exporting a
            # dense public XYZ snapshot here would immediately throw it away.
            moved = self.engine.move(target, under=requested, _export=False)
            compact = collapse_boxes(moved, self.boxes, default_over=not requested,
                                     attachment_tolerance=2., _array_geometry=True)
            compact.pop('twist_box_expansion', None)
            compact.pop('twist_box_edge_lineage', None)
        except Exception as exc:
            for name, value in saved.items():
                if value is missing:
                    self.engine.__dict__.pop(name, None)
                else:
                    setattr(self.engine, name, value)
            self.engine._active_components = active
            if isinstance(exc, ValueError) and not isinstance(exc, MoveRejected):
                raise MoveRejected(str(exc)) from exc
            raise
        self.current = self.engine.current.copy()
        self.last_moves = list(self.engine.last_moves)
        self.last_diagram = compact
        self.last_step_stats = getattr(self.engine, 'last_step_stats', {})
        compact.setdefault('motion', {}).update(twist_boxes_fixed=len(self.boxes),
            box_proxy=True, note='Smooth external motion with fixed twist bundles and boundary attachments.')
        return self.last_diagram
