"""One guarded curve fit shared by the live canvas and its TikZ export.

Preparation has no GUI dependencies and may run in a worker. The model remains
polygonal for editing and topology; all returned coordinates are in its image
frame, with read-only arrays. Zoom changes only the curve tessellation.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from . import render
from .bezier import flatten_cubics


def _readonly(points):
    result = np.array(points, dtype=float).reshape(-1, 2)
    result.setflags(write=False)
    return result


def _projection(points, query):
    """Nearest polyline point and its traversal tangent, including short edges."""
    a, vectors = points[:-1], np.diff(points, axis=0)
    lengths2 = np.sum(vectors*vectors, axis=1)
    fractions = np.divide(np.sum((query-a)*vectors, axis=1), lengths2,
                          out=np.zeros(len(vectors)), where=lengths2 > 0.)
    projected = a+np.clip(fractions, 0., 1.)[:, None]*vectors
    distances2 = np.sum((projected-query)**2, axis=1)
    index = int(np.argmin(distances2))
    tangent = vectors[index]/max(math.sqrt(lengths2[index]), 1e-20)
    return float(distances2[index]), projected[index], tangent


def _on_curve(points, curves, query):
    """Project an attachment onto its cubic and evaluate its analytic tangent."""
    if not curves:
        _, point, tangent = _projection(points, query)
        return point, tangent
    controls = np.asarray(curves)

    def evaluate(t):
        t = np.asarray(t)[:, None]
        v = 1-t
        return (v**3*controls[:, 0]+3*v*v*t*controls[:, 1]
                +3*v*t*t*controls[:, 2]+t**3*controls[:, 3])

    # Use one nearby seed per compact segment, then minimize distance along
    # every segment. Explicit endpoints protect against clamped Newton steps.
    seeds = np.linspace(0., 1., 17)
    samples = np.array([evaluate(np.full(len(controls), t)) for t in seeds])
    t = seeds[np.argmin(np.sum((samples-query)**2, axis=2), axis=0)]
    for _ in range(8):
        u, v = t[:, None], (1-t)[:, None]
        first = 3*(v*v*(controls[:, 1]-controls[:, 0])
                   +2*v*u*(controls[:, 2]-controls[:, 1])
                   +u*u*(controls[:, 3]-controls[:, 2]))
        second = 6*(v*(controls[:, 2]-2*controls[:, 1]+controls[:, 0])
                    +u*(controls[:, 3]-2*controls[:, 2]+controls[:, 1]))
        difference = evaluate(t)-query
        denominator = np.sum(first*first+difference*second, axis=1)
        step = np.divide(np.sum(difference*first, axis=1), denominator,
                         out=np.zeros(len(t)), where=abs(denominator) > 1e-20)
        t = np.clip(t-step, 0., 1.)
    candidates = np.array([controls[:, 0], evaluate(t), controls[:, 3]])
    choice, index = np.unravel_index(np.argmin(np.sum((candidates-query)**2, axis=2)),
                                    candidates.shape[:2])
    parameter = (0., t[index], 1.)[choice]
    c = controls[index]
    tangent = 3*((1-parameter)**2*(c[1]-c[0])
                 +2*(1-parameter)*parameter*(c[2]-c[1])+parameter**2*(c[3]-c[2]))
    if np.linalg.norm(tangent) < 1e-12:
        tangent = c[3]-c[0]
    tangent /= max(float(np.linalg.norm(tangent)), 1e-20)
    return candidates[choice, index], tangent


def _middle(points):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    return np.array([np.interp(arc[-1]/2, arc, points[:, axis]) for axis in (0, 1)])


class PreparedDisplayCurves:
    """Owned diagram snapshot plus fitted paths, attachments, and picking data.

    ``strokes`` and ``passages`` return the usual (points, color, width) records.
    Color changes do not refit or resample. At most one zoom tessellation is
    retained, bounded by both 0.2 screen pixels and the fitting safety margin.
    ``nearest`` selects the visible geometry but returns an original-model
    anchor, so dragging never bakes a cosmetic approximation into the diagram.
    """

    def __init__(self, diagram):
        self.diagram = deepcopy(diagram)
        strokes = render.paths(self.diagram, with_ids=True)
        passages = render.box_passages(self.diagram, with_ids=True)
        self._stroke_count = len(strokes)
        self._records = [(_readonly(points), color, width, eid)
                         for points, color, width, eid in strokes+passages]
        triples = [(points, color, width) for points, color, width, _ in self._records]
        fitted = render._tikz_curves(triples)
        self._curves = tuple(None if curves is None else tuple(_readonly(c) for c in curves)
                             for curves in fitted)
        self.stroke_curves = self._curves[:self._stroke_count]
        self.passage_curves = self._curves[self._stroke_count:]
        crossing_map = render._crossings(self.diagram)
        self._edges = {edge['id']: edge for edge in self.diagram.get('edges', [])}
        model_points = {eid: _readonly(render.edge_points(edge, crossing_map))
                        for eid, edge in self._edges.items()}
        # Above-box passage pieces belong to the entry edge but live inside the
        # box record, so their anchors must remain on that stored passage.
        self._anchors = [model_points.get(eid, points) if i < self._stroke_count else points
                         for i, (points, _, _, eid) in enumerate(self._records)]
        self._flat_scale = None
        self._flatten(1.)
        self._arrow_records = []
        indices = {eid: i for i, (_, _, _, eid) in enumerate(self._records[:self._stroke_count])}
        for point, tangent, color, eid in render.arrows(self.diagram, with_ids=True):
            index = indices.get(eid)
            if index is not None:
                points = self._records[index][0]
                mapped, direction = _on_curve(points, self._curves[index], np.asarray(point))
                if np.dot(direction, tangent) < 0:
                    direction = -direction
                point, tangent = tuple(mapped), tuple(direction)
            self._arrow_records.append((point, tangent, color))
        self._marker_records = []
        for i, (points, _, _, eid) in enumerate(self._records[:self._stroke_count]):
            point, _ = _on_curve(points, self._curves[i], _middle(self._flat[i]))
            component = self._edges[eid].get('component', 0)
            self._marker_records.append((component, tuple(point)))

    def _flatten(self, scale):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Display scale must be positive and finite.')
        if scale == self._flat_scale:
            return
        tolerances = tuple(min(.2/scale, .01, record[2]*.005) for record in self._records)
        if tolerances == getattr(self, '_flat_tolerances', None):
            self._flat_scale = scale
            return
        # The .01/width bounds match the precision used by the shared clearance
        # validator. A coarse zoom cannot erase narrow, verified crossing gaps.
        self._flat = tuple(points if curves is None else _readonly(flatten_cubics(curves, error))
            for (points, _, _, _), curves, error in zip(self._records, self._curves, tolerances))
        self._low = np.array([points.min(axis=0) for points in self._flat]).reshape(-1, 2)
        self._high = np.array([points.max(axis=0) for points in self._flat]).reshape(-1, 2)
        self._flat_scale = scale
        self._flat_tolerances = tolerances

    def _triples(self, start, stop, colored, points):
        return [(points[i], self._records[i][1] if colored else '#171717', self._records[i][2])
                for i in range(start, stop)]

    def strokes(self, colored=True, scale=1.0):
        self._flatten(scale)
        return self._triples(0, self._stroke_count, colored, self._flat)

    def passages(self, colored=True, scale=1.0):
        self._flatten(scale)
        return self._triples(self._stroke_count, len(self._records), colored, self._flat)

    def raw_strokes(self, colored=True):
        return self._triples(0, self._stroke_count, colored, [r[0] for r in self._records])

    def raw_passages(self, colored=True):
        return self._triples(self._stroke_count, len(self._records), colored,
                             [r[0] for r in self._records])

    def arrows(self, colored=True):
        return [(point, tangent, color if colored or color == render.DEFAULT_ORIENTATION_COLOR
                 else '#171717') for point, tangent, color in self._arrow_records]

    def markers(self, component):
        return [point for owner, point in self._marker_records if owner == component]

    def nearest(self, point, limit):
        if not math.isfinite(limit) or limit < 0:
            return None
        query = np.asarray(point, dtype=float)
        nearby = np.flatnonzero(np.all(self._low <= query+limit, axis=1)
                                & np.all(self._high >= query-limit, axis=1))
        best = (float('inf'), None, None)
        for index in nearby:
            distance2, displayed, _ = _projection(self._flat[index], query)
            # Later passage layers win exact ties over strands below a box.
            if distance2 <= best[0] and self._records[index][3] is not None:
                _, model_anchor, _ = _projection(self._anchors[index], displayed)
                best = distance2, self._records[index][3], tuple(model_anchor)
        if best[0] <= limit*limit:
            return 'edge', best[1], best[2]
        return None


def prepare_display_curves(diagram):
    """Prepare a reusable, independent snapshot without changing the model."""
    return PreparedDisplayCurves(diagram)
