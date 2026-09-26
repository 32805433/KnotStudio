"""Small learned ink-component classifier for label proposals.

The model classifies geometry, never text contents or filenames.  It supplements
the label grouper and does not erase anything.  Its compact tree representation
uses NumPy only at inference, so the desktop app needs no ML runtime/download.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .diagram_labels import _glyph_geometry


MODEL_PATH = Path(__file__).with_name('models') / 'label_components_v2.json'
FEATURE_VERSION = 1
SCALAR_FEATURES = (
    'log_width_strokes', 'log_height_strokes', 'log_area_strokes',
    'log_aspect', 'fill', 'log_ends', 'log_branches', 'closed',
    'log_width_variation', 'width_image', 'height_image', 'area_ink',
    'area_largest', 'stroke_image', 'solidity', 'perimeter_strokes',
    'holes', 'eccentricity', 'angle_cos2', 'angle_sin2',
    'nearest_bbox_strokes', 'nearest_large_strokes', 'nearby_similar_height',
    'nearby_baseline', 'horizontal_overlap', 'vertical_overlap',
    'saturation', 'contrast', 'dark_background',
)
FEATURE_NAMES = SCALAR_FEATURES + tuple(f'occupancy_{i}' for i in range(64))


def candidate_component_ids(stats, shape, *, max_components=2048):
    """Broad, cheap proposal pool; tiny dust and whole diagrams are excluded."""
    height, width = shape[:2]
    ids = np.arange(1, len(stats))
    if not len(ids):
        return ids
    s = stats[1:]
    bound = .8 if len(s) <= 8 else .5
    keep = ((s[:, cv2.CC_STAT_AREA] >= 4)
            & (s[:, cv2.CC_STAT_WIDTH] <= bound * width)
            & (s[:, cv2.CC_STAT_HEIGHT] <= bound * height)
            & (s[:, cv2.CC_STAT_AREA] <= .60 * max(1, s[:, cv2.CC_STAT_AREA].sum())))
    ids = ids[keep]
    if len(ids) > max_components:
        # Prefer substantive ink over photo compression dust. Never use an
        # image identity or the presence of a catalogue annotation here.
        ids = ids[np.argsort(stats[ids, cv2.CC_STAT_AREA])[-max_components:]]
    return np.sort(ids)


def extract_component_features(rgb, labels, stats, pieces=None, *, component_ids=None):
    """Return ``(component_ids, float32 feature matrix)`` for detached ink.

    ``labels`` and ``stats`` are from connectedComponentsWithStats. ``pieces``
    may cache _glyph_geometry results from the caller. Features are normalized
    by stroke width, image extent, and neighboring ink rather than resolution.
    The feature ordering is published as FEATURE_NAMES.
    """
    height, width = labels.shape
    ids = (candidate_component_ids(stats, labels.shape) if component_ids is None
           else np.asarray(component_ids, dtype=int))
    if not len(ids):
        return ids, np.empty((0, len(FEATURE_NAMES)), np.float32)
    pieces = pieces or {}
    s = stats[1:].astype(float)
    areas = s[:, cv2.CC_STAT_AREA]
    total = max(1., areas.sum())
    largest = max(1., areas.max(initial=1.))
    x, y, w, h = (s[:, k] for k in range(4))
    right, bottom = x+w, y+h
    substantial = (areas >= max(20., .12*largest)) | (w >= .2*width) | (h >= .2*height)
    background = np.median(rgb.reshape(-1, 3)[::max(1, rgb.size//30000)], axis=0)
    dark = float(background.mean() < 115.)
    rows = []
    for index in ids:
        xx, yy, ww, hh, area = map(int, stats[index])
        component = labels[yy:yy+hh, xx:xx+ww] == index
        geometry = pieces.get(int(index)) or _glyph_geometry(component)
        stroke = max(1., float(geometry['stroke_width']))
        ys, xs = np.nonzero(component)
        centered = np.column_stack((xs-xs.mean(), ys-ys.mean())).astype(float)
        covariance = centered.T @ centered / max(1, len(xs))
        eig, vec = np.linalg.eigh(covariance)
        major = vec[:, -1]
        eccentricity = float(np.sqrt(max(0., 1.-eig[0]/max(eig[-1], 1.))))
        contours, hierarchy = cv2.findContours(component.astype(np.uint8), cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        hull_area = cv2.contourArea(cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32)))
        holes = 0 if hierarchy is None else sum(v[3] >= 0 for v in hierarchy[0])
        dx = np.maximum(np.maximum(xx-right, x-(xx+ww)), 0.)
        dy = np.maximum(np.maximum(yy-bottom, y-(yy+hh)), 0.)
        distance = np.hypot(dx, dy)
        distance[index-1] = np.inf
        close = (distance <= 2.*max(hh, 4))
        similar = (h >= .55*hh) & (h <= 1.8*hh)
        baseline = np.abs((y+bottom)-(2*yy+hh)) <= .7*max(hh, 1)
        h_overlap = np.minimum(right, xx+ww)-np.maximum(x, xx)
        v_overlap = np.minimum(bottom, yy+hh)-np.maximum(y, yy)
        ink_color = np.median(rgb[yy:yy+hh, xx:xx+ww][component], axis=0)
        nearest = min(100., float(distance.min(initial=np.inf))/stroke)
        near_large = min(100., float(distance[substantial].min(initial=np.inf))/stroke)
        scalars = [
            np.log1p(ww/stroke), np.log1p(hh/stroke), np.log1p(area/stroke**2),
            np.log(max(.01, ww/max(hh, 1))), area/(ww*hh),
            np.log1p(geometry['ends']), np.log1p(geometry['branches']), float(geometry['closed']),
            np.log1p(geometry['width_variation']), ww/width, hh/height,
            area/total, area/largest, stroke/max(height, width), min(2., area/max(1., hull_area)),
            np.log1p(perimeter/stroke), min(8, holes), eccentricity,
            major[0]**2-major[1]**2, 2*major[0]*major[1],
            np.log1p(nearest), np.log1p(near_large),
            min(12, np.count_nonzero(close & similar)),
            min(12, np.count_nonzero(close & similar & baseline)),
            min(12, np.count_nonzero(close & (h_overlap > .5*ww))),
            min(12, np.count_nonzero(close & (v_overlap > .5*hh))),
            (ink_color.max()-ink_color.min())/255., np.linalg.norm(ink_color-background)/442., dark,
        ]
        # Aspect-normalized pixels distinguish serif/glyph shapes from smooth
        # fragments; aspect ratio is retained above rather than thrown away.
        occupancy = cv2.resize(component.astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA)
        rows.append(np.r_[scalars, occupancy.ravel()])
    matrix = np.asarray(rows, np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite label-component features.')
    return ids, matrix


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        return None
    model = json.loads(MODEL_PATH.read_text())
    if model.get('feature_version') != FEATURE_VERSION or model.get('feature_names') != list(FEATURE_NAMES):
        raise ValueError('Label-component model feature schema does not match the program.')
    return model


def predict_features(features, model):
    """Average finite leaf probabilities from the compact bagged tree model."""
    # sklearn trees compare float32 feature values against float64 thresholds.
    # Explicit promotion prevents NumPy's weak scalar casting from rounding a
    # threshold onto the feature itself and choosing the wrong child.
    features = np.asarray(features, np.float32).astype(np.float64)
    values = np.zeros(len(features), dtype=float)
    for tree in model['trees']:
        nodes = np.zeros(len(features), dtype=int)
        active = np.ones(len(features), dtype=bool)
        while active.any():
            rows = np.flatnonzero(active)
            for node in np.unique(nodes[rows]):
                selected = rows[nodes[rows] == node]
                feature, threshold, left, right, probability = tree[int(node)]
                if feature < 0:
                    values[selected] += probability
                    active[selected] = False
                else:
                    nodes[selected] = np.where(features[selected, feature] <= threshold, left, right)
    return values / max(1, len(model['trees']))


def predict_components(rgb, labels, stats, pieces=None, *, component_ids=None):
    """Map component IDs to learned label probabilities; {} without a model."""
    model = load_model()
    if model is None:
        return {}
    ids, features = extract_component_features(rgb, labels, stats, pieces,
                                                component_ids=component_ids)
    probabilities = predict_features(features, model)
    return dict(zip(map(int, ids), map(float, probabilities)))
