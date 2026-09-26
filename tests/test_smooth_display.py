"""Visible freehand smoothing with local clearance and unchanged editing data."""
from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from recognizer import render
from recognizer.bezier import fit_cubics, flatten_cubics, smooth_polyline
from recognizer.diagram import pd_code
from recognizer.display_curves import prepare_display_curves
from recognizer.drawing import PencilGesture
from recognizer.pipeline import recognize_array


def circle(radius=90, center=(150, 150), wobble=0):
    t = np.linspace(0, 2*np.pi, 241)
    r = radius+wobble*(np.sin(19*t)+.6*np.sin(31*t+.4))
    p = np.array(center)+np.c_[r*np.cos(t), r*np.sin(t)]
    p[-1] = p[0]
    return p


def total_turn(points):
    """Absolute turning after uniform sampling, independent of fit tessellation."""
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    locations = np.linspace(0, arc[-1], max(4, int(arc[-1]/2)))
    p = np.column_stack([np.interp(locations, arc, points[:, axis]) for axis in (0, 1)])
    v = np.diff(p, axis=0)
    angles = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    return np.abs(np.diff(angles)).sum()


def distance_to_path(queries, path):
    a, v = path[:-1], np.diff(path, axis=0)
    lengths2 = np.maximum(np.sum(v*v, axis=1), 1e-30)
    result = []
    for q in queries:
        t = np.clip(np.sum((q-a)*v, axis=1)/lengths2, 0., 1.)
        result.append(np.linalg.norm(a+t[:, None]*v-q, axis=1).min())
    return np.asarray(result)


def test_wobbly_pencil_recognition_is_visibly_smoothed_without_changing_model():
    pen = PencilGesture(Image.new('RGB', (410, 410), 'white'), 4)
    p = circle(135, (200, 200), wobble=5)
    for a, b in zip(p, p[1:]):
        image = pen.add(a, b)
    result = recognize_array(np.asarray(image))
    assert result['status'] == 'ok'
    diagram = result['diagram']
    original = deepcopy(diagram)
    prepared = prepare_display_curves(diagram)
    source = prepared.raw_strokes()[0][0]
    actual = prepared.strokes()[0][0]
    assert prepared.stroke_curves[0] is not None
    # Curve fitting alone left most of this freehand waviness in place.
    assert total_turn(actual) < .4*total_turn(source)
    assert not render._tikz_has_intersections([(actual, 'black', 2.)])
    assert diagram == original and pd_code(diagram) == pd_code(original)
    assert not actual.flags.writeable


def test_local_tight_gap_does_not_disable_smoothing_of_a_distant_loop():
    loose = (circle(100, (0, 0), 3), 'black', 2.)
    tight = [(circle(r, (400, 0)), 'black', 2.) for r in (60, 62.4)]
    solo = render._tikz_curves([loose])[0]
    together = render._tikz_curves([loose]+tight)
    assert all(c is not None for c in together)
    np.testing.assert_array_equal(together[0], solo)


def test_close_parallel_loops_keep_their_gap():
    source = [(circle(r), 'black', 3.) for r in (90, 93.6)]
    curves = render._tikz_curves(source)
    assert all(c is not None for c in curves)
    first, second = [flatten_cubics(c, .001) for c in curves]
    assert distance_to_path(first, second).min() > 3.2
    assert distance_to_path(second, first).min() > 3.2
    assert not render._tikz_has_intersections([(first, 'black', 3.), (second, 'black', 3.)])


def test_smoothing_does_not_create_small_local_self_intersections():
    points = np.array([[.1868, -.1059], [.8909, -.8054], [1.6829, .5830],
                       [2.2957, .6067], [2.6954, .2001], [3.4919, -.8052],
                       [3.7922, -.4743], [4.5502, -.9465], [5.0749, .3811],
                       [5.8411, -.4956], [6.5575, .3784], [7.3272, .6870],
                       [7.3510, -.5228], [7.4252, .7830]])
    curves = render._tikz_curves([(points, 'black', 2.)])[0]
    actual = points if curves is None else flatten_cubics(curves, .0001)
    # Independent strict segment crossing check, including very short arcs.
    v = np.diff(actual, axis=0)
    for i in range(len(v)-2):
        w, delta = v[i+2:], actual[i+2:-1]-actual[i]
        det = v[i, 0]*w[:, 1]-v[i, 1]*w[:, 0]
        valid = abs(det) > 1e-12
        safe = np.where(valid, det, 1.)
        t = (delta[:, 0]*w[:, 1]-delta[:, 1]*w[:, 0])/safe
        u = (delta[:, 0]*v[i, 1]-delta[:, 1]*v[i, 0])/safe
        assert not np.any(valid & (t > 1e-8) & (t < 1-1e-8) & (u > 1e-8) & (u < 1-1e-8))


def test_small_closed_loop_is_not_erased():
    p = circle(2, (0, 0))
    fitted = render._tikz_curves([(p, 'black', 4.)])[0]
    q = p if fitted is None else flatten_cubics(fitted, .001)
    area = abs(np.sum(q[:-1, 0]*q[1:, 1]-q[1:, 0]*q[:-1, 1]))/2
    assert area > 8
    np.testing.assert_allclose(q[0], q[-1], atol=.001)


@pytest.mark.parametrize('radius', [1., 2., 4., 8.])
def test_tight_hairpin_does_not_gain_reverse_bends(radius):
    y = np.linspace(80, 0, 81)
    angle = np.linspace(0, np.pi, 60)
    p = np.concatenate([np.c_[np.zeros(len(y)), y],
                        np.c_[radius-radius*np.cos(angle), -radius*np.sin(angle)][1:],
                        np.c_[np.full(len(y)-1, 2*radius), y[-2::-1]]])
    curves = render._tikz_curves([(p, 'black', 3.)])[0]
    assert curves is not None
    q = flatten_cubics(curves, .001)
    v = np.diff(q, axis=0)
    angles = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    # One half-turn; a too-wide tangent neighborhood creates extra wiggles
    # when it straddles both sides of the narrow bend.
    assert np.abs(np.diff(angles)).sum() < 1.2*np.pi


def test_averaging_and_fitting_share_a_local_displacement_budget():
    p = circle(100, (0, 0), 2)
    limits = np.linspace(.05, 6, len(p)-1)
    softened, remaining = smooth_polyline(p, limits, 12.)
    shift = np.linalg.norm(softened-p, axis=1)
    np.testing.assert_array_equal(softened[[0, -1]], p[[0, -1]])
    assert np.all(remaining > 0)
    assert np.all(np.maximum(shift[:-1], shift[1:])+remaining <= limits+1e-12)
    # The tightly constrained part still receives a close cubic fit.
    curves = fit_cubics(softened, 6., segment_tolerances=remaining)
    actual = flatten_cubics(curves, .001)
    assert distance_to_path(p[:10], actual).max() < max(limits[:10])+.01


def test_averaging_depends_on_arclength_not_mouse_event_density():
    p = circle(100, (0, 0), 3)
    dense = np.concatenate([np.linspace(a, b, 31 if i < 70 else 2)[:-1]
                            for i, (a, b) in enumerate(zip(p, p[1:]))]+[p[-1:]])
    first, _ = smooth_polyline(p, np.full(len(p)-1, 8.), 12.)
    second, _ = smooth_polyline(dense, np.full(len(dense)-1, 8.), 12.)
    assert distance_to_path(first, second).max() < .04


@pytest.mark.parametrize('limits', [[.1], [0, 1], [np.nan, 1], [-1, 1]])
def test_invalid_segment_limits_are_rejected(limits):
    with pytest.raises(ValueError, match='tolerance per source segment'):
        fit_cubics([[0, 0], [10, 1], [20, 0]], 2., segment_tolerances=limits)


def test_repeated_samples_keep_their_local_fit_limits():
    p = [[0, 0], [0, 0], [10, 0], [20, 0], [20, 0]]
    curves = fit_cubics(p, 3., segment_tolerances=[.01, 3., 3., .01],
                        start_tangent=[1, 1], end_tangent=[1, -1])
    actual = flatten_cubics(curves, .0001)
    assert abs(actual[:, 1]).max() < .0101


def test_overpass_halves_have_shared_tangents_and_fixed_endpoints():
    left = np.array([[-80, 20], [-60, 11], [-40, 13], [-20, 4], [0, 0.]])
    right = np.array([[0, 0], [20, -3], [40, -12], [60, -9], [80, -20.]])
    curves = render._tikz_curves([(left, 'black', 3.), (right, 'black', 3.)])
    assert all(c is not None for c in curves)
    for source, fitted in zip((left, right), curves):
        np.testing.assert_allclose(fitted[0][0], source[0], atol=.001)
        np.testing.assert_allclose(fitted[-1][-1], source[-1], atol=.001)
    a = curves[0][-1][3]-curves[0][-1][2]
    b = curves[1][0][1]-curves[1][0][0]
    assert np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b) > .999999


def test_prepared_display_reuses_fit_for_zoom_color_picking_and_tikz(monkeypatch):
    from test_render import one_crossing

    diagram = one_crossing()
    original = deepcopy(diagram)
    prepared = prepare_display_curves(diagram)

    def fail(*args, **kwargs):
        raise AssertionError('A prepared display must reuse its fit.')

    monkeypatch.setattr(render, '_tikz_curves', fail)
    monochrome = prepared.strokes(colored=False, scale=8)
    assert all(c == '#171717' for _, c, _ in monochrome)
    assert '.. controls' in render.render_tikz(diagram, prepared=prepared)
    point = monochrome[0][0][len(monochrome[0][0])//2]
    hit = prepared.nearest(point, .001)
    assert hit[:2] == ('edge', 0)
    source = np.asarray(diagram['edges'][0]['points'])
    assert distance_to_path([hit[2]], source)[0] < 1e-9
    assert diagram == original and pd_code(diagram) == pd_code(original)
