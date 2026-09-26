"""Portable spatial-move regressions on known two/three-component unlinks."""
from copy import deepcopy
from itertools import permutations
import json
import math

import numpy as np
import pytest

from recognizer.diagram import component_walks, pd_code, reverse_component, validate
from recognizer.motion import SmoothDrag, from_curves, to_curves
from recognizer.projection_equivalence import same_pd_projection
from recognizer.reidemeister import crossing_sign


def circle(cx, cy, radius, height):
    t = np.linspace(.031, math.tau+.031, 182)
    p = np.column_stack((cx+radius*np.cos(t), cy+radius*np.sin(t),
                         np.full(len(t), height)))
    p[-1] = p[0]
    return dict(points=p.tolist())


def edge_near(diagram, point):
    query = np.asarray(point)
    hits = []
    for edge in diagram['edges']:
        p = np.asarray(edge['points'])
        a, v = p[:-1], np.diff(p, axis=0)
        t = np.clip(np.sum((query-a)*v, axis=1)/np.maximum(np.sum(v*v, axis=1), 1e-16), 0, 1)
        hits.append((np.linalg.norm(query-a-t[:, None]*v, axis=1).min(), edge['id']))
    return min(hits)[1]


def area_signs(diagram):
    result = []
    for curve in to_curves(diagram):
        p = np.asarray(curve['points'])
        result.append(int(np.sign(np.sum(p[:-1, 0]*p[1:, 1]-p[1:, 0]*p[:-1, 1]))))
    return result


@pytest.mark.parametrize('under', [False, True])
def test_r2_creation_and_new_gesture_removal_keep_orientations_after_json(under):
    source = from_curves([circle(300, 180, 80, 20), circle(300, 400, 100, -20)], 600, 600)
    source = reverse_component(source, 0)
    snapshot = deepcopy(source)
    anchor, target = [300., 260.], [300., 350.]
    drag = SmoothDrag(source, edge_near(source, anchor), anchor, radius=180., under=under)
    moved = drag.move(target)
    assert validate(moved)['valid']
    assert len(moved['crossings']) == 2
    assert sorted(crossing_sign(moved, c['id']) for c in moved['crossings']) == [-1, 1]
    assert area_signs(moved) == area_signs(source)
    for c in moved['crossings']:
        moving = max(c['motion_branches'], key=lambda b: b['weight'])
        assert moving['over'] is not under
    saved = json.loads(json.dumps(moved))
    result = SmoothDrag(saved, edge_near(saved, target), target, radius=180.).move(anchor)
    assert pd_code(result) == []
    assert len(component_walks(result)) == 2
    assert area_signs(result) == area_signs(source)
    assert source == snapshot


def r3_curves(offset, heights):
    y = 300-math.sqrt(130**2-80**2)
    return [circle(220, 300, 130, heights[0]), circle(380, 300, 130, heights[1]),
            circle(300, y-65+offset, 65, heights[2])]


@pytest.mark.parametrize('heights', list(permutations((-40., 0., 40.))))
def test_r3_all_height_orders_from_uncached_drawing_and_back(heights):
    source = from_curves(r3_curves(-4., heights), 600, 600)
    source.pop('spatial_curves')
    source.pop('spatial_fingerprint')
    snapshot = deepcopy(source)
    anchor = np.array([300., 296-math.sqrt(130**2-80**2)])
    drag = SmoothDrag(source, edge_near(source, anchor), anchor, radius=95.)
    moved = drag.move(anchor+[0, 8])
    expected = from_curves(r3_curves(4., heights), 600, 600)
    assert validate(moved)['valid']
    assert len(moved['crossings']) == 6
    assert len(component_walks(moved)) == 3
    assert same_pd_projection(pd_code(moved), pd_code(expected))
    assert area_signs(moved) == area_signs(source)
    restored = drag.move(anchor)
    assert same_pd_projection(pd_code(restored), pd_code(source))
    assert source == snapshot
