"""R1 topology, independent PD signs, physical orientation and empty-face checks."""
from copy import deepcopy
import math

import numpy as np
import pytest

from recognizer.diagram import assign_components, component_walks, pd_code, reverse_component, validate
from recognizer.geometry import check_embedding
from recognizer.reidemeister import add_curl, crossing_sign, removable_monogons, remove_curl


def circle(center=(200, 200), radius=120, count=121):
    t = np.linspace(0, math.tau, count)
    points = np.column_stack([center[0]+radius*np.cos(t), center[1]+radius*np.sin(t)]).tolist()
    return assign_components(dict(width=400, height=400, crossings=[], edges=[
        dict(id=0, points=points, start=None, end=None, closed=True)], component_orientations={}))


def test_dense_monogon_sampling_keeps_the_same_removable_regions_and_pd():
    original=add_curl(circle(),0,[200,80],sign=-1,side_point=[200,20])
    dense=deepcopy(original)
    for edge in dense['edges']:
        points=np.array(edge['points'])
        edge['points']=np.vstack([*[np.linspace(a,b,10,endpoint=False)
                                   for a,b in zip(points,points[1:])],points[-1:]]).tolist()
    expected=removable_monogons(original)
    assert {r['id'] for r in removable_monogons(dense)}=={r['id'] for r in expected}
    for region in expected:
        result=remove_curl(dense,region['crossing_id'],region['edge_id'])
        assert validate(result)['crossing_free_components']==1
        assert pd_code(result)==[]


def test_batched_polygon_check_matches_strict_interior_at_concave_boundaries():
    from recognizer.reidemeister import _any_inside,_inside
    polygon=np.array([[0,0],[10,0],[10,10],[4,4],[0,10]],float)
    points=np.vstack([polygon,[[0,5],[5,0],[4,9],[5,2],[12,12]],
                      np.random.default_rng(17).uniform(-5,15,(270,2))])
    for point in points:
        assert _any_inside(np.array([point]),polygon)==_inside(point,polygon)
    assert _any_inside(points,polygon)


def oriented_area(diagram):
    edges = {e['id']: e for e in diagram['edges']}
    orientations = diagram.get('component_orientations', {})
    areas = []
    for ci, walk in enumerate(component_walks(diagram)):
        area = 0.
        for eid, direction in walk:
            p = np.asarray(edges[eid]['points'])
            value = float(np.sum(p[:-1, 0]*p[1:, 1]-p[1:, 0]*p[:-1, 1]))
            area += value*(-1 if direction else 1)
        areas.append(area*orientations.get(str(ci), 1))
    return areas


@pytest.mark.parametrize('angle', [0., math.pi/2, math.pi*.73])
@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('side', [-1, 1])
def test_curl_sign_side_rotation_and_roundtrip(angle, sign, side):
    original = reverse_component(circle(), 0)
    snapshot = deepcopy(original)
    radial = np.array([math.cos(angle), math.sin(angle)])
    chosen = np.array([200, 200])+120*radial
    side_point = chosen+side*60*radial
    changed = add_curl(original, 0, chosen, sign, side_point)
    assert original == snapshot
    assert validate(changed)['valid']
    assert check_embedding(changed)['valid']
    assert crossing_sign(changed, 0) == sign
    assert oriented_area(original)[0]*oriented_area(changed)[0] > 0
    # Independently recorded from installed SnapPy/Spherogram, not from the
    # geometric sign helper: positive R1 has PD [a,a,b,b], negative [a,b,b,a].
    x, = pd_code(changed)
    assert (x[0] == x[1] and x[2] == x[3]) if sign == 1 else (x[0] == x[3] and x[1] == x[2])
    region = next(r for r in removable_monogons(changed) if r['edge_id'] == 1)
    polygon = np.asarray(region['polygon'])
    assert float((polygon.mean(axis=0)-chosen)@(side_point-chosen)) > 0
    removed = remove_curl(changed, region['crossing_id'], region['edge_id'])
    assert validate(removed)['crossing_free_components'] == 1
    assert pd_code(removed) == []
    assert oriented_area(original)[0]*oriented_area(removed)[0] > 0
    assert check_embedding(removed)['valid']


def imported_curl():
    # Hand-built two-lobe immersed curve. There is no creation-history metadata.
    c = [200, 180]
    upper = [c, [260, 120], [245, 75], [200, 60], [155, 75], [140, 120], c]
    lower = [c, [260, 240], [320, 360], [200, 420], [80, 360], [140, 240], c]
    return assign_components(dict(width=400, height=500, component_orientations={},
        crossings=[dict(id=7, point=c, over=1, ports=[dict(edge=10, end=1), dict(edge=10, end=0),
                                                       dict(edge=9, end=0), dict(edge=9, end=1)])],
        edges=[dict(id=9, points=upper, start=[7, 2], end=[7, 3], closed=False),
               dict(id=10, points=lower, start=[7, 1], end=[7, 0], closed=False)]))


def test_imported_monogons_and_selected_region_removal():
    diagram = imported_curl()
    assert validate(diagram)['valid']
    assert check_embedding(diagram)['valid']
    regions = removable_monogons(diagram)
    assert {r['edge_id'] for r in regions} == {9, 10}
    assert len({r['id'] for r in regions}) == 2
    for selected in (9, 10):
        changed = remove_curl(diagram, 7, edge_id=selected)
        assert validate(changed)['crossing_free_components'] == 1
        points = np.asarray(changed['edges'][0]['points'])
        assert (points[:, 1].mean() > 180) if selected == 9 else (points[:, 1].mean() < 180)


def test_enclosed_disconnected_component_excludes_monogon():
    diagram = imported_curl()
    extra = circle(center=(200, 100), radius=8, count=41)['edges'][0]
    extra['id'] = 88
    diagram['edges'].append(extra)
    assign_components(diagram)
    assert check_embedding(diagram)['valid']
    assert {r['edge_id'] for r in removable_monogons(diagram)} == {10}
    with pytest.raises(ValueError, match='shaded empty monogon'):
        remove_curl(diagram, 7, edge_id=9)


def test_second_curl_on_arc_and_orientation_after_splicing():
    d = reverse_component(circle(), 0)
    once = add_curl(d, 0, [200, 80], sign=1, side_point=[200, 20])
    twice = add_curl(once, 0, [200, 320], sign=-1, side_point=[200, 380])
    assert validate(twice)['crossings'] == 2
    assert sorted(crossing_sign(twice, c['id']) for c in twice['crossings']) == [-1, 1]
    assert oriented_area(d)[0]*oriented_area(twice)[0] > 0
    region = next(r for r in removable_monogons(twice) if r['crossing_id'] == 1)
    removed = remove_curl(twice, 1, edge_id=region['edge_id'])
    assert validate(removed)['crossings'] == 1
    assert crossing_sign(removed, 0) == 1
    assert oriented_area(d)[0]*oriented_area(removed)[0] > 0


def test_curl_rejects_crossing_adjacent_point_and_unclear_side():
    d = imported_curl()
    original = deepcopy(d)
    with pytest.raises(ValueError, match='farther from crossings'):
        add_curl(d, 9, [201, 179], side_point=[250, 180])
    with pytest.raises(ValueError, match='one side'):
        add_curl(circle(), 0, [320, 200], side_point=[320, 200])
    with pytest.raises(ValueError, match='sign'):
        add_curl(d, 9, [200, 60], sign=0, side_point=[200, 20])
    assert d == original


def test_curl_cannot_pass_through_neighbor_or_leave_canvas():
    d = circle(center=(200, 150), radius=120)
    with pytest.raises(ValueError, match='canvas'):
        add_curl(d, 0, [200, 30], side_point=[200, -10], size=30)
    # An entire concentric neighboring component blocks an exterior curl.
    d = circle()
    second = circle(radius=131)['edges'][0]
    second['id'] = 1
    d['edges'].append(second)
    assign_components(d)
    with pytest.raises(ValueError):
        add_curl(d, 0, [200, 80], side_point=[200, 20])


def test_curl_does_not_jump_across_component_in_swept_region():
    # Both endpoint drawings can be valid even if replacing a bent strand
    # sweeps over this small circle. R1 requires an empty local disk throughout.
    d = circle()
    extra = circle(center=(181, 86), radius=2.3, count=25)['edges'][0]
    extra['id'] = 8
    d['edges'].append(extra)
    assign_components(d)
    assert check_embedding(d)['valid']
    with pytest.raises(ValueError, match='local region'):
        add_curl(d, 0, [200, 80], 1, [200, 160], size=22.22)
