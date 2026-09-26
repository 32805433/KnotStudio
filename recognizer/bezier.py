"""Compact, endpoint-preserving cubic approximations for vector export.

The fitting error is measured against the whole source polyline, not just its
vertices. A common parametrization bounds the distance in both directions.
These are cosmetic approximations; callers remain responsible for checking
clearance between different strands before accepting them.
"""
from __future__ import annotations

import math

import numpy as np


def _unit(vector, fallback):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if math.isfinite(length) and length > 1e-12:
        return vector / length
    fallback = np.asarray(fallback, dtype=float)
    length = float(np.linalg.norm(fallback))
    return fallback / length if length > 1e-12 else np.array([1., 0.])


def _evaluate(curve, parameters):
    u = np.asarray(parameters, dtype=float)[:, None]
    v = 1-u
    return v**3*curve[0] + 3*v*v*u*curve[1] + 3*v*u*u*curve[2] + u**3*curve[3]


def _derivative(curve, parameters):
    u = np.asarray(parameters, dtype=float)[:, None]
    v = 1-u
    return 3*(v*v*(curve[1]-curve[0]) + 2*v*u*(curve[2]-curve[1])
              + u*u*(curve[3]-curve[2]))


def _chord_distances(points, start, end):
    chord = end-start
    length2 = float(np.dot(chord, chord))
    if length2 < 1e-24:
        return np.linalg.norm(points-start, axis=1)
    fractions = np.clip((points-start) @ chord / length2, 0., 1.)
    return np.linalg.norm(points-start-fractions[:, None]*chord, axis=1)


def flatten_cubics(curves, tolerance):
    """Return a polyline within ``tolerance`` of contiguous cubic segments.

    Subdivision uses the convex hull of the control points, including distance
    to the *finite* chord. Unlike fixed-rate sampling this also handles loops,
    stationary points, and controls extending beyond the endpoints. The bound
    holds in both directions, by continuity of projection onto each chord.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('The flattening tolerance must be positive and finite.')
    result = []
    for value in curves:
        curve = np.asarray(value, dtype=float)
        if curve.shape != (4, 2) or not np.isfinite(curve).all():
            raise ValueError('Each cubic must contain four finite 2D control points.')
        if not result:
            result.append(curve[0].copy())
        stack = [curve]
        while stack:
            part = stack.pop()
            if np.max(_chord_distances(part[1:3], part[0], part[3])) <= tolerance:
                result.append(part[3].copy())
                continue
            a, b, c = (part[:-1]+part[1:])/2
            d, e = (a+b)/2, (b+c)/2
            middle = (d+e)/2
            stack.append(np.array([middle, e, c, part[3]]))
            stack.append(np.array([part[0], a, d, middle]))
    return np.asarray(result, dtype=float).reshape(-1, 2)


def _fit_controls(points, u, first_tangent, last_tangent, length):
    """Solve the two handle lengths while fixing endpoint tangent directions."""
    v = 1-u
    b1, b2 = 3*v*v*u, 3*v*u*u
    a = b1[:, None]*first_tangent
    b = -b2[:, None]*last_tangent
    baseline = (v**3+b1)[:, None]*points[0] + (b2+u**3)[:, None]*points[-1]
    residual = points-baseline
    matrix = np.array([[np.sum(a*a), np.sum(a*b)],
                       [np.sum(a*b), np.sum(b*b)]])
    rhs = np.array([np.sum(a*residual), np.sum(b*residual)])
    determinant = float(np.linalg.det(matrix))
    if determinant > 1e-12*max(1., float(np.trace(matrix))**2):
        handles = np.linalg.solve(matrix, rhs)
    else:
        handles = np.full(2, length/3)
    # Negative handles reverse prescribed tangents. Very long handles can
    # create hidden loops even when their parameterized vertices fit well.
    handles = np.where(handles > length*1e-6, handles, length/3)
    handles = np.minimum(handles, length*.75)
    return np.array([points[0], points[0]+first_tangent*handles[0],
                     points[-1]-last_tangent*handles[1], points[-1]])


def _error_bounds(curve, points, u):
    """Convex-hull bounds for cubic minus each parameterized source segment.

    A cubic restricted to [u[i], u[i+1]] minus the corresponding linear source
    segment is another cubic. All its control vectors inside the tolerance
    disk certify the entire interval, including curve excursions between data
    points. This is conservative but considerably cheaper than extremum roots.
    """
    values = _evaluate(curve, u)
    derivatives = _derivative(curve, u)
    step = np.diff(u)[:, None]/3
    errors = np.stack([values[:-1]-points[:-1],
                       values[:-1]+step*derivatives[:-1]-(2*points[:-1]+points[1:])/3,
                       values[1:]-step*derivatives[1:]-(points[:-1]+2*points[1:])/3,
                       values[1:]-points[1:]], axis=1)
    return np.max(np.linalg.norm(errors, axis=2), axis=1)


def _reparameterize(curve, points, u):
    difference = _evaluate(curve, u)-points
    first = _derivative(curve, u)
    second = 6*((1-u)[:, None]*(curve[2]-2*curve[1]+curve[0])
                + u[:, None]*(curve[3]-2*curve[2]+curve[1]))
    denominator = np.sum(first*first+difference*second, axis=1)
    safe = np.where(abs(denominator) > 1e-20, denominator, np.inf)
    candidate = u-np.sum(difference*first, axis=1)/safe
    candidate[0], candidate[-1] = 0., 1.
    if np.any(np.diff(candidate) <= 1e-10):
        return u
    return candidate


def _short_curve(points, first_tangent, last_tangent, tolerance):
    """A tangent-preserving two-point curve with a certified chord corridor."""
    first, last = points
    length = float(np.linalg.norm(last-first))
    handles = np.full(2, length/3)
    controls = np.array([first+first_tangent*handles[0], last-last_tangent*handles[1]])
    distances = _chord_distances(controls, first, last)
    # A convex capsule around the segment contains the complete cubic. Since
    # both endpoints are fixed, it also gives the reverse approximation bound.
    scale = np.minimum(1., tolerance*.95/np.maximum(distances, 1e-30))
    handles *= scale
    return np.array([first, first+first_tangent*handles[0],
                     last-last_tangent*handles[1], last])


def smooth_polyline(points, segment_tolerances, radius):
    """Soften freehand jitter within a local displacement budget.

    Average in arclength, not sample index: densely sampled mouse events must
    not attract the curve. Endpoints stay fixed. Return the smoothed points
    and the *remaining* segment budgets for a subsequent cubic fit. Subtracting
    the larger endpoint displacement bounds the whole intervening segment.
    """
    from scipy.ndimage import gaussian_filter1d

    points = np.asarray(points, dtype=float)
    limits = np.asarray(segment_tolerances, dtype=float)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    length = arc[-1]
    if length < 1e-12 or len(points) < 3:
        return points.copy(), limits.copy()
    closed = np.linalg.norm(points[0]-points[-1]) < 1e-12
    radius = min(radius, length/(12 if closed else 6))
    count = max(12, min(20000, int(np.ceil(length/(radius/12)))))
    locations = np.linspace(0, length, count, endpoint=not closed)
    uniform = np.column_stack([np.interp(locations, arc, points[:, axis]) for axis in (0, 1)])
    averaged = gaussian_filter1d(uniform, radius/(locations[1]-locations[0]),
                                 axis=0, mode='wrap' if closed else 'nearest')
    if closed:
        locations = np.r_[locations, length]
        averaged = np.vstack([averaged, averaged[0]])
    target = np.column_stack([np.interp(arc, locations, averaged[:, axis]) for axis in (0, 1)])
    shift = target-points
    # A gradual endpoint collar avoids introducing a corner where a freely
    # smoothed arc meets a fixed crossing or box attachment.
    collar = (1-np.exp(-(arc/radius)**2))*(1-np.exp(-((length-arc)/radius)**2))
    shift *= collar[:, None]
    vertex_limits = np.r_[limits[0], np.minimum(limits[:-1], limits[1:]), limits[-1]]
    distances = np.linalg.norm(shift, axis=1)
    shift *= np.minimum(1., .7*vertex_limits/np.maximum(distances, 1e-30))[:, None]
    shift[[0, -1]] = 0
    distances = np.linalg.norm(shift, axis=1)
    remaining = limits-np.maximum(distances[:-1], distances[1:])
    return points+shift, remaining


def fit_cubics(points, tolerance, *, start_tangent=None, end_tangent=None,
               segment_tolerances=None):
    """Approximate a polyline with a deterministic sequence of cubic Béziers.

    Endpoints remain exact. Both optional tangent vectors follow the direction
    of traversal, including ``end_tangent``. Adjacent cubics have the same
    tangent direction; a repeated endpoint gives a smooth periodic seam.
    ``tolerance`` bounds bidirectional distance to the source polyline. The
    returned controls are unrounded; exporters should validate rounded output.
    Optional ``segment_tolerances`` tighten that bound separately on each
    source segment, so a nearby strand need not constrain the entire path.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('The fitting tolerance must be positive and finite.')
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return []
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('A path must contain finite 2D points.')
    if len(points) < 2:
        return []
    limits = np.full(len(points)-1, tolerance)
    if segment_tolerances is not None:
        local = np.asarray(segment_tolerances, dtype=float)
        if (local.shape != limits.shape or not np.isfinite(local).all()
                or np.any(local <= 0)):
            raise ValueError('Provide one positive finite tolerance per source segment.')
        limits = np.minimum(limits, local)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]
    if not keep.all():
        retained = np.flatnonzero(keep)
        # A repeated sample may carry a tighter limit. Retain it on the
        # adjoining interval instead of discarding that safety constraint.
        original_limits = limits
        limits = np.array([limits[a:b].min() for a, b in zip(retained[:-1], retained[1:])])
        if len(limits) and retained[-1] < len(points)-1:
            limits[-1] = min(limits[-1], float(original_limits[retained[-1]:].min()))
    points = points[keep].copy()
    if len(points) < 2:
        return []
    closed = np.linalg.norm(points[0]-points[-1]) < 1e-12
    if closed:
        points[-1] = points[0]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    length = float(arc[-1])

    def at(distances):
        if closed:
            distances = np.mod(distances, length)
        return np.column_stack([np.interp(distances, arc, points[:, axis]) for axis in (0, 1)])

    tangent_cache = {}

    def tangent(index):
        if closed and index == len(points)-1:
            index = 0
        if index not in tangent_cache:
            # A broad freehand smoothing budget must not average the two
            # sides of a tight hairpin into a reversed tangent. Use this
            # vertex's remaining local fitting budget for its neighborhood.
            adjacent = [limits[max(0, index-1)], limits[min(index, len(limits)-1)]]
            if closed and index == 0:
                adjacent.append(limits[-1])
            radius = min(6*min(adjacent), length/(8 if closed else 3))
            location = arc[index]
            left, right = -radius, radius
            if not closed:
                left, right = max(left, -location), min(right, length-location)
                # Use a full one-sided neighborhood at the two open ends.
                if index == 0:
                    right = min(2*radius, length)
                elif index == len(points)-1:
                    left = -min(2*radius, length)
            offsets = np.linspace(left, right, 9)
            scale = max(radius, 1e-20)
            coordinates = offsets/scale
            design = np.column_stack([np.ones(9), coordinates, coordinates**2])
            values = at(location+offsets)
            derivative = np.linalg.lstsq(design, values-points[index], rcond=None)[0][1]
            fallback = values[-1]-values[0]
            if np.linalg.norm(fallback) < 1e-12:
                fallback = points[min(index+1, len(points)-1)]-points[max(0, index-1)]
            tangent_cache[index] = _unit(derivative, fallback)
        return tangent_cache[index]

    first = tangent(0) if start_tangent is None else _unit(start_tangent, tangent(0))
    last = tangent(len(points)-1) if end_tangent is None else _unit(end_tangent, tangent(len(points)-1))
    if closed and start_tangent is None and end_tangent is not None:
        first = last
    elif closed and end_tangent is None and start_tangent is not None:
        last = first

    result = []
    stack = [(0, len(points)-1, first, last)]
    while stack:
        begin, end, left_tangent, right_tangent = stack.pop()
        section = points[begin:end+1]
        section_limits = limits[begin:end]
        section_length = float(arc[end]-arc[begin])
        if len(section) == 2:
            result.append(_short_curve(section, left_tangent, right_tangent, section_limits[0]))
            continue
        u = (arc[begin:end+1]-arc[begin])/section_length
        best_error, best_curve, best_bounds = math.inf, None, None
        for _ in range(4):
            curve = _fit_controls(section, u, left_tangent, right_tangent, section_length)
            bounds = _error_bounds(curve, section, u)/section_limits
            error = float(np.max(bounds))
            if error < best_error:
                best_error, best_curve, best_bounds = error, curve, bounds
            if error <= 1.:
                break
            if error > 4.:
                break
            updated = _reparameterize(curve, section, u)
            if np.array_equal(updated, u):
                break
            u = updated
        # A complete loop cannot be represented by the collinear handles at a
        # single coincident endpoint, even when a large tolerance permits it.
        if best_error <= 1. and not (closed and begin == 0 and end == len(points)-1):
            result.append(best_curve)
            continue
        worst = int(np.argmax(best_bounds))
        split = max(1, min(len(section)-2, worst+1))
        # Keep pathological noisy input from producing quadratic recursion.
        if len(section) > 24:
            split = int(np.clip(split, len(section)//12, len(section)-1-len(section)//12))
        middle = begin+split
        shared = tangent(middle)
        stack.append((middle, end, shared, right_tangent))
        stack.append((begin, middle, left_tangent, shared))
    return result
