"""Remove sub-stroke endpoint hooks that recross their own inferred gap.

This is a review-only geometric cleanup after gap matching.  It changes both
joined polylines, rather than suppressing an intersection during assembly.
Resolved curls and anything enclosing another strand are deliberately retained.
"""
from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np

from .geometry import intersection


def _points(extraction, match):
    if 'points' in match:
        return np.asarray(match['points'], dtype=float)
    return np.asarray([extraction['paths'][match[key][0]]['points'][
        0 if match[key][1] == 0 else -1] for key in ('a', 'b')], dtype=float)


def _segments_touch_region(points, boundary, allowed_join=None):
    """Whether a polyline enters a lobe; only its immediate join may touch."""
    contour = np.asarray(boundary, dtype=np.float32)
    polygon = np.vstack([boundary, boundary[0]])
    low, high = np.min(boundary, axis=0), np.max(boundary, axis=0)
    for index, (a, b) in enumerate(zip(points, points[1:])):
        if np.any(np.maximum(a, b) < low) or np.any(np.minimum(a, b) > high):
            continue
        for point in (a, b, (a+b)/2):
            touches = cv2.pointPolygonTest(contour, tuple(map(float, point)), False) >= 0
            if (touches and not (allowed_join is not None and index == 0
                                 and np.linalg.norm(point-allowed_join) < 1e-6)):
                return True
        for c, d in zip(polygon, polygon[1:]):
            hit = intersection(a, b, c, d)
            if hit is None:
                continue
            if (allowed_join is not None and index == 0
                    and np.linalg.norm(hit[2]-allowed_join) < 1e-6):
                continue
            return True
    return False


def _clear_lobe(extraction, match_index, path_index, remainder, boundary):
    if _segments_touch_region(remainder, boundary, allowed_join=boundary[-1]):
        return False
    for index, path in enumerate(extraction['paths']):
        if index == path_index:
            continue
        points = np.asarray(path['points'], dtype=float)
        if path.get('closed') and np.linalg.norm(points[0]-points[-1]) > 1e-8:
            points = np.vstack([points, points[0]])
        if _segments_touch_region(points, boundary):
            return False
    for index, match in enumerate(extraction['matches']):
        if index != match_index and _segments_touch_region(_points(extraction, match), boundary):
            return False
    return True


def normalize_endpoint_hooks(extraction, *, stroke_width=None):
    """Return ``(independent_extraction_copy, {'repairs': [...]})``.

    An open visible endpoint can acquire a tiny hook from raster skeletonizing.
    If its own gap crosses that hook, trim the visible prefix exactly to the
    crossing and shorten the gap to the same point.  Every candidate lobe must
    fit within one stroke width, have perimeter below three widths and visible
    arclength below two widths, and enclose at most a quarter squared width.
    No other visible strand or inferred gap may touch or enter the lobe.

    This cannot establish the intention of genuinely sub-resolution curls;
    callers must expose the repair diagnostics and require visual review.
    """
    result = deepcopy(extraction)
    diagnostic = {'repairs': []}
    width = float(stroke_width if stroke_width is not None else
                  result.get('diagnostics', {}).get('stroke_width', 0.))
    if not np.isfinite(width) or width <= 0:
        return result, diagnostic
    for match_index, match in enumerate(result['matches']):
        for side, key in enumerate(('a', 'b')):
            path_index, end = match[key]
            path = result['paths'][path_index]
            if path.get('closed'):
                continue
            visible = np.asarray(path['points'], dtype=float)
            visible = visible if end == 0 else visible[::-1]
            gap = _points(result, match)
            outward_gap = gap if side == 0 else gap[::-1]
            if len(visible) < 4 or len(gap) < 2:
                continue
            if np.linalg.norm(visible[0]-outward_gap[0]) > 1e-6:
                continue
            lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(visible, axis=0), axis=1))]
            # Only the gap's adjacent straight segment may take part in a nib.
            # A curved gap passing back near its end needs a separate model.
            start, next_point = outward_gap[:2].copy()
            for segment in range(1, len(visible)-1):
                if lengths[segment] > 2*width:
                    break
                hit = intersection(start, next_point, visible[segment], visible[segment+1])
                if hit is None:
                    continue
                gap_t, visible_t, point = hit
                gap_length = float(np.linalg.norm(point-start))
                if not (1e-6 < gap_t < 1-1e-6 and 1e-6 < gap_length < width):
                    continue
                visible_length = float(lengths[segment] + np.linalg.norm(point-visible[segment]))
                boundary = np.vstack([visible[:segment+1], point])
                diameter = float(np.linalg.norm(boundary[:, None]-boundary[None, :], axis=2).max())
                area = abs(float(np.sum(boundary[:, 0]*np.roll(boundary[:, 1], -1)
                                        - boundary[:, 1]*np.roll(boundary[:, 0], -1))/2))
                if (visible_length > 2*width or diameter >= width
                        or visible_length+gap_length >= 3*width
                        or not 1e-8 < area < .25*width*width):
                    continue
                remainder = np.vstack([point, visible[segment+1:]])
                if np.linalg.norm(remainder[1]-remainder[0]) < 1e-8:
                    remainder = remainder[1:]
                if len(remainder) < 2 or not _clear_lobe(
                        result, match_index, path_index, remainder, boundary):
                    continue
                # No intersection remains: both pieces now meet exactly at the
                # former crossing, and the entire hooked prefix is absent.
                path['points'] = (remainder if end == 0 else remainder[::-1]).tolist()
                gap[0 if side == 0 else -1] = point
                match['points'] = gap.tolist()
                match['distance'] = float(np.linalg.norm(np.diff(gap, axis=0), axis=1).sum())
                if len(gap) == 2:
                    direction = gap[1]-gap[0]
                    for crossing in match.get('crossings', []):
                        crossing['gap_t'] = float(np.dot(np.asarray(crossing['point'])-gap[0], direction)
                                                  / np.dot(direction, direction))
                repair = {'path': int(path_index), 'end': int(end), 'match': int(match_index),
                          'original_endpoint': start.tolist(), 'joined_endpoint': point.tolist(),
                          'removed_visible_length': visible_length, 'shortened_gap_length': gap_length,
                          'lobe_diameter': diameter, 'lobe_area': area, 'stroke_width': width,
                          'reason': 'sub-stroke endpoint hook recrossed its own inferred gap'}
                diagnostic['repairs'].append(repair)
                break
    if diagnostic['repairs']:
        result.setdefault('diagnostics', {})['endpoint_hook_repairs'] = deepcopy(diagnostic['repairs'])
    return result, diagnostic
