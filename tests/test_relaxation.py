"""Energy decrease and ambient isotopy, including the fixed exterior."""
from copy import deepcopy
import threading

import numpy as np
import pytest

from recognizer.diagram import pd_code, validate
from recognizer.projection_equivalence import same_pd_projection
from recognizer.motion import from_curves, to_curves
from recognizer.relaxation import RelativeRelaxation, RelaxationCancelled, energy_gradient, validate_region, _inside


def circle(cx, cy, radius, z=0., n=181):
    """Synthetic oriented spatial circle with an exact closure point."""
    angles = np.linspace(.031, 2*np.pi+.031, n+1)
    points = np.column_stack((cx+radius*np.cos(angles),
                              cy+radius*np.sin(angles), np.full(n+1, z)))
    points[-1] = points[0]
    return {"points": points.tolist()}


def riii_curves(offset):
    crossing_y = 300-np.sqrt(130**2-80**2)
    return [circle(220, 300, 130, -40), circle(380, 300, 130, 40),
            circle(300, crossing_y-65+offset, 65, 0)]


def test_energy_gradient_is_the_derivative_of_the_reported_energy():
    points = np.array([[0., 0.], [.7, .2], [1.9, -.4], [.2, 2.3]])
    adjacency = np.zeros((4, 4), bool)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        adjacency[a, b] = adjacency[b, a] = True
    _, gradient = energy_gradient(points, adjacency)
    for i in range(4):
        for axis in range(2):
            step = np.zeros_like(points)
            step[i, axis] = 1e-6
            difference = (energy_gradient(points+step, adjacency)[0]-energy_gradient(points-step, adjacency)[0])/2e-6
            assert gradient[i, axis] == pytest.approx(difference, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize('region', [
    [[50, 30], [550, 30], [550, 360], [50, 360]],
    [[50, 30], [550, 30], [550, 520], [360, 520], [360, 310], [240, 310], [240, 520], [50, 520]],
])
def test_relative_flow_decreases_energy_and_fixes_entire_exterior(region):
    diagram = from_curves(riii_curves(-4), 600, 600)
    original = deepcopy(diagram)
    engine = RelativeRelaxation(diagram, region)
    before = engine.energy
    for _ in range(35):
        sampling = engine.resamples
        result = engine.step()
        if result is None:
            break
        assert engine.energy < (before if engine.resamples == sampling else engine.initial_energy)
        before = engine.energy
        assert pd_code(result) == pd_code(diagram)
        assert validate(result)['valid']
        # Boundary vertices and exterior vertices stay bit-for-bit fixed.
        assert np.array_equal(engine.positions[~engine.free], engine.rest[~engine.free])
        for edge, (points, _, _) in zip(result['edges'], engine.bindings):
            outside = ~_inside(points, engine.region)
            assert np.allclose(np.asarray(edge['points'])[outside], points[outside], atol=1e-10, rtol=0)
    assert engine.iterations > 0
    assert engine.energy < engine.initial_energy*.99
    # Test boundary points between mesh vertices, not just the pinned vertices.
    boundary = np.concatenate([a+np.linspace(0, 1, 41)[:, None]*(b-a)
                               for a, b in zip(engine.region, np.roll(engine.region, -1, axis=0))])
    simplex = engine.mesh.find_simplex(boundary/engine.unit)
    tr = engine.mesh.transform[simplex]
    uv = np.einsum('nij,nj->ni', tr[:, :2], boundary/engine.unit-tr[:, 2])
    bary = np.column_stack([uv, 1-uv.sum(axis=1)])
    moved = np.einsum('ni,nij->nj', bary, (engine.positions-engine.rest)[engine.triangles[simplex]])
    assert np.max(abs(moved)) < 1e-10
    # Rebuild from the resulting actual curve geometry, independently of the
    # unchanged combinatorial labels, to check that no extra crossings arose.
    rebuilt = from_curves(to_curves(engine.diagram()), 600, 600)
    assert same_pd_projection(pd_code(diagram), pd_code(rebuilt))
    assert diagram == original


def test_positive_endpoint_triangles_do_not_hide_a_midstep_collapse():
    diagram = from_curves([circle(300, 300, 100)], 600, 600)
    engine = RelativeRelaxation(diagram, [[100, 100], [500, 100], [500, 500], [100, 500]])
    # 180-degree rotation has positive final area, but its straight homotopy
    # collapses every triangle halfway through. The all-time check rejects it.
    assert not engine._valid_step(-2*(engine.positions-engine.positions.mean(axis=0)))
    assert engine._valid_step(np.zeros_like(engine.positions))


def test_cancellation_never_commits_a_new_step():
    event = threading.Event()
    diagram = from_curves([circle(300, 300, 100)], 600, 600)
    region = [[100, 100], [500, 100], [500, 500], [100, 500]]
    engine = RelativeRelaxation(diagram, region, event)
    old = engine.positions.copy()
    event.set()
    with pytest.raises(RelaxationCancelled):
        engine.step()
    assert np.array_equal(old, engine.positions)
    with pytest.raises(RelaxationCancelled):
        RelativeRelaxation(diagram, region, event)


@pytest.mark.parametrize('region', [
    [[0, 0], [100, 100], [0, 100], [100, 0]],
    [[0, 0], [100, 0], [100, 100], [50, 0], [0, 100]],
    [[0, 0], [1, 0], [1, 1]],
])
def test_invalid_lassos_are_rejected(region):
    with pytest.raises(ValueError):
        validate_region(region)


def test_empty_selection_and_boundary_through_crossing_are_rejected():
    diagram = from_curves(riii_curves(-4), 600, 600)
    with pytest.raises(ValueError, match='No movable'):
        RelativeRelaxation(diagram, [[5, 5], [30, 5], [30, 30], [5, 30]])
    point = diagram['crossings'][0]['point']
    x, y = point
    with pytest.raises(ValueError, match='crossing'):
        RelativeRelaxation(diagram, [[x, y-60], [x+80, y-60], [x+80, y+60], [x, y+60]])
