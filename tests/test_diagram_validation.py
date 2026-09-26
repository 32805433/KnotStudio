"""Reject malformed imported geometry before rendering or editing it."""
from copy import deepcopy

import numpy as np
import pytest

from recognizer.diagram import validate
from recognizer.render import render_image


def triangle():
    return dict(width=100, height=100, crossings=[], edges=[
        dict(id=0, start=None, end=None, closed=True,
             points=[[20, 20], [80, 20], [50, 80], [20, 20]])])


@pytest.mark.parametrize('points', [
    [[20], [80]], [[20, 20, 0], [80, 20, 0]],
    [[20, 20], [], [80, 20]], [[20, '20'], [80, 20]],
    [[20, float('nan')], [80, 20]], [[20, 20], [float('inf'), 20]],
    [20, 80], np.array([20., 80.]), np.zeros((4, 3)),
    np.zeros((4, 2), dtype=complex),
])
def test_invalid_edge_geometry_is_not_reported_valid(points):
    diagram = triangle()
    diagram['edges'][0]['points'] = points
    assert not validate(diagram)['valid']


@pytest.mark.parametrize('point', [[], [30], [30, 40, 50], [float('nan'), 40],
                                    [30, float('inf')], [30, '40']])
def test_invalid_crossing_geometry_is_not_reported_valid(point):
    # Incidence is a valid one-crossing immersed circle. Its vertex must also
    # be a usable planar point, regardless of the combinatorial PD validity.
    diagram = dict(width=100, height=100, crossings=[dict(
        id=0, point=point, over=1,
        ports=[dict(edge=0, end=0), dict(edge=0, end=1),
               dict(edge=1, end=0), dict(edge=1, end=1)])], edges=[
        dict(id=0, start=[0, 0], end=[0, 1], closed=False,
             points=[[50, 50], [80, 20], [20, 20], [50, 50]]),
        dict(id=1, start=[0, 2], end=[0, 3], closed=False,
             points=[[50, 50], [20, 80], [80, 80], [50, 50]])])
    assert not validate(diagram)['valid']


@pytest.mark.parametrize('array', [False, True])
def test_planar_geometry_remains_valid_without_mutation(array):
    diagram = triangle()
    if array:
        diagram['edges'][0]['points'] = np.asarray(diagram['edges'][0]['points'])
    before = deepcopy(diagram['edges'][0]['points'])
    assert validate(diagram)['valid']
    assert np.array_equal(diagram['edges'][0]['points'], before)
    assert render_image(diagram).size == (200, 200)


@pytest.mark.parametrize('name', ['width', 'height'])
@pytest.mark.parametrize('value', [-100, 0, float('nan'), float('inf'), '100', None])
def test_invalid_canvas_dimensions_are_rejected_before_rendering(name, value):
    diagram = triangle()
    diagram[name] = value
    assert not validate(diagram)['valid']


def test_missing_canvas_dimensions_keep_the_default_frame():
    diagram = triangle()
    diagram.pop('width')
    diagram.pop('height')
    assert validate(diagram)['valid']
    assert render_image(diagram, scale=1).size == (800, 600)
