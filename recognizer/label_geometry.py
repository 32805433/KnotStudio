"""Small, pixel-only checks that keep broken strands out of label proposals."""
from __future__ import annotations

from math import ceil, hypot

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .knotfolio_backend import _adjacency, _prune_spurs
from .photo_arrows import paired_shaft_arrows


def arrow_component(component):
    """Protect a paired arrow on a continuing shaft or at its narrow tip.

    Used only to protect ink from label deletion. A plain bar has no head;
    a serifed I/T has no taper from an interior broad head to a narrow tip,
    and an italic letter's one-sided hook is not a two-sided arrowhead.
    Projections make this check independent of the arrow's direction.
    """
    # A strand arrow can lie in the middle of a short, wide component. Share
    # the tracer's two backward wings plus continuing shaft certificate before
    # applying the separate, narrow terminal-arrow test below.
    if paired_shaft_arrows(component):
        return True
    points = np.argwhere(component)[:, ::-1].astype(float)
    if len(points) < 12:
        return False
    points -= points.mean(axis=0)
    _, axes = np.linalg.eigh(points.T @ points)
    along, across = points @ axes[:, 1], points @ axes[:, 0]
    length, breadth = np.ptp(along), np.ptp(across)
    if length < 12 or length < 3.2 * max(1., breadth):
        return False
    bins = np.floor(along-along.min()).astype(int)
    low = np.full(bins.max()+1, np.inf)
    high = np.full(bins.max()+1, -np.inf)
    np.minimum.at(low, bins, across)
    np.maximum.at(high, bins, across)
    if not np.isfinite(low).all():
        return False
    widths = high-low+1.
    centers = (low+high)*.5
    for widths, centers in ((widths, centers), (widths[::-1], centers[::-1])):
        n = len(widths)
        tail = widths[2:max(3, round(.55*n))]
        shaft = float(np.median(tail))
        peak = int(np.argmax(widths))
        if not (.60*n <= peak <= .95*n and widths[peak] >= 2.5*max(1., shaft)):
            continue
        if np.quantile(tail, .9) > max(shaft+1.5, 1.5*shaft):
            continue
        if np.ptp(centers[2:round(.55*n)]) > max(2., shaft):
            continue
        # An italic l can end in a wide hook that tapers back to a narrow tip.
        # Fit the *shaft* centerline, then require head ink on both sides of
        # its extrapolation. The overall PCA axis alone is biased toward such
        # a hook and would make its one-sided bulge look more symmetrical.
        shaft_bins = np.arange(2, max(3, round(.55*n)))
        slope, intercept = np.polyfit(shaft_bins, centers[shaft_bins], 1)
        shaft_center = slope * peak + intercept
        half_head_span = .5 * (widths[peak] - 1.)
        low_wing = shaft_center - (centers[peak] - half_head_span)
        high_wing = centers[peak] + half_head_span - shaft_center
        if min(low_wing, high_wing) < .75 * shaft:
            continue
        head = widths[peak:]
        if (len(head) >= 3 and np.mean(head[-2:]) <= .55*widths[peak]
                and np.mean(np.diff(head) <= .75) >= .75):
            return True
    return False


def _open_ends(component, stroke_width, allow_branches=False):
    """Return endpoints and outward directions after removing pixel spurs."""
    graph = _adjacency(skeletonize(component))
    _prune_spurs(graph, max(2., stroke_width * .85))
    ends = [point for point, neighbors in graph.items() if len(neighbors) == 1]
    if len(ends) < 2 or (not allow_branches and (
            len(ends) != 2 or any(len(neighbors) > 2 for neighbors in graph.values()))):
        return []
    result = []
    for start in ends:
        point, previous, length = start, None, 0.
        while length < max(4., 2.5 * stroke_width):
            following = [p for p in graph[point] if p != previous]
            if len(following) != 1:
                break
            next_point = following[0]
            length += hypot(point[0] - next_point[0], point[1] - next_point[1])
            previous, point = point, next_point
        vector = np.subtract(start, point, dtype=float)
        norm = np.linalg.norm(vector)
        if norm < max(1., .5 * stroke_width):
            return []
        result.append((np.asarray(start, dtype=float), vector / norm))
    return result


def strand_continuation_group(labels, group_ids, pieces, *, allow_branches=False,
                              ignore_ids=()):
    """Whether every end of an unsupported label group meets nearby strands.

    ``labels`` is a connected-component label image. ``pieces`` maps its IDs to
    dictionaries with ``bbox`` (x1, y1, x2, y2), ``stroke_width``, ``ends``,
    ``branches``, and ``closed``. Only groups consisting entirely of ordinary
    two-ended arcs can be rejected by default. ``allow_branches=True`` also
    checks branched fragments: all their ends must match nearby strands. This
    allows callers to protect individual fragments within grouped typography.
    Closed glyphs and dots retain their independent evidence for being text.
    ``ignore_ids`` excludes known companion glyphs from the neighboring ink
    search without imposing any shape requirements on those companions. This
    lets callers check a single suspect piece inside an already formed group
    without mistaking its adjacent letters, subscripts or dots for strands.

    Both ends of every arc must point towards nearby non-group ink of similar
    thickness. The neighboring strand need not be parallel: an underpassing
    arc normally stops alongside a transverse overpassing strand. The search
    follows the outgoing end direction, so ink behind or beside a glyph does
    not alone make it a continuation. An isolated handwritten C, l or 5 remains
    eligible even when skeleton pruning removes its serif.

    Skeletons and distance transforms cover the proposed glyphs and small
    endpoint neighborhoods only; the diagram's full size does not affect the
    work after its connected components have been computed.
    """
    group = set(group_ids)
    excluded = list(group | set(ignore_ids))
    if not group or any(i not in pieces for i in group):
        return False
    if any(pieces[i].get('closed') or pieces[i].get('ends', 0) < 2
           or (not allow_branches and (pieces[i].get('branches')
                                     or pieces[i].get('ends') != 2)) for i in group):
        return False
    height, width = labels.shape
    for index in group:
        piece = pieces[index]
        stroke = float(piece['stroke_width'])
        if not np.isfinite(stroke) or stroke <= 0:
            return False
        x1, y1, x2, y2 = map(int, piece['bbox'])
        endpoints = _open_ends(labels[y1:y2, x1:x2] == index, stroke, allow_branches)
        if len(endpoints) < 2:
            return False
        for local_end, direction in endpoints:
            endpoint = local_end + [y1, x1]
            # Extra margin prevents the crop edge from changing the width of
            # the neighboring strand inside the actual five-width search.
            radius = max(6, ceil(7. * stroke))
            y, x = map(int, endpoint)
            ax, ay = max(0, x-radius), max(0, y-radius)
            bx, by = min(width, x+radius+1), min(height, y+radius+1)
            patch = labels[ay:by, ax:bx]
            outside = (patch > 0) & ~np.isin(patch, excluded)
            if not outside.any():
                return False
            # Zero padding makes distance estimates valid even at image edges.
            padded = np.pad(outside, 1)
            widths = 2. * cv2.distanceTransform(padded.astype(np.uint8),
                                                cv2.DIST_L2, 5)[1:-1, 1:-1]
            centerline = skeletonize(padded)[1:-1, 1:-1]
            points = np.argwhere(centerline & (widths >= .55 * stroke)
                                 & (widths <= 1.8 * stroke))
            if not len(points):
                return False
            displacement = points + [ay, ax] - endpoint
            distance = np.linalg.norm(displacement, axis=1)
            # Measure the visible gap, excluding half the nearby line width.
            clearance = distance - widths[points[:, 0], points[:, 1]] * .5
            ahead = displacement @ direction
            if not np.any((clearance <= 5. * stroke)
                          & (ahead >= .25 * distance)):
                return False
    return True
