"""Geometry and cache regressions for empty-bigon cancellation."""
from copy import deepcopy
import math

import numpy as np
import pytest

from recognizer.diagram import component_walks, pd_code, validate
from recognizer.geometry import check_embedding
from recognizer.motion import from_curves
from recognizer.reidemeister_ii import removable_bigons, remove_bigon


def overlap(angle=0., reversed_first=False, count=121):
    theta = np.linspace(.031, 2*math.pi+.031, count)
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    curves = []
    for ci, (cx, height) in enumerate(((-40, 20), (40, -20))):
        p = np.column_stack([cx+100*np.cos(theta), 100*np.sin(theta)])
        p = p@rotation.T+300
        curve = dict(points=np.column_stack([p, np.full(len(p), height)]).tolist())
        if ci == 0 and reversed_first:
            curve['points'].reverse()
        curves.append(curve)
    return from_curves(curves, 600, 600)


@pytest.mark.parametrize('angle', [.37, math.pi/2, 2.7])
@pytest.mark.parametrize('reverse', [False, True])
def test_rotated_bounded_bigons_cancel_without_a_creation_history(angle, reverse):
    original = overlap(angle, reverse)
    original.pop('spatial_curves', None)
    original.pop('spatial_fingerprint', None)
    regions = removable_bigons(original)
    assert len(regions) == 3
    for region in regions:
        result = remove_bigon(original, region['id'])
        assert validate(result)['valid']
        assert check_embedding(result)['valid']
        assert len(component_walks(result)) == 2
        assert pd_code(result) == []


def test_dense_straight_subdivisions_do_not_change_the_selected_faces():
    original = overlap()
    dense = deepcopy(original)
    for edge in dense['edges']:
        p = np.array(edge['points'])
        edge['points'] = np.vstack([*[np.linspace(a, b, 9, endpoint=False)
                                      for a, b in zip(p, p[1:])], p[-1:]]).tolist()
    assert {r['id'] for r in removable_bigons(dense)} == {r['id'] for r in removable_bigons(original)}
    for region in removable_bigons(dense):
        result = remove_bigon(dense, region['id'])
        assert validate(result)['crossing_free_components'] == 2
        assert check_embedding(result)['valid']


def test_cancellation_clears_cached_spatial_lift_and_is_atomic():
    original = overlap()
    assert original['spatial_curves']
    original['source'] = {'user_notes': 'Keep this drawing provenance.'}
    snapshot = deepcopy(original)
    result = remove_bigon(original, removable_bigons(original)[0]['id'])
    assert 'spatial_curves' not in result
    assert 'spatial_fingerprint' not in result
    assert result['source'] == original['source']
    assert original == snapshot
    with pytest.raises(ValueError, match='shaded empty bigon'):
        remove_bigon(original, 'this-region-no-longer-exists')
    assert original == snapshot


def test_crossing_free_diagram_has_no_bigon_and_no_deletion_side_effect():
    original = overlap()
    out = remove_bigon(original, removable_bigons(original)[0]['id'])
    snapshot = deepcopy(out)
    assert removable_bigons(out) == []
    with pytest.raises(ValueError, match='shaded empty bigon'):
        remove_bigon(out, '0:1:0:2')
    assert out == snapshot




def test_becoming_crossing_free_preserves_component_colors_and_order():
    theta = np.linspace(.031, math.tau+.031, 101)
    original = from_curves([dict(points=np.column_stack([
        center+110*np.cos(theta), 300+110*np.sin(theta), np.full(len(theta), z)]).tolist())
        for center, z in ((200, 40), (350, 0), (500, -40))], 720, 600)
    record = min(removable_bigons(original), key=lambda r: np.mean(r['polygon'], axis=0)[0])
    result = remove_bigon(original, record['id'])
    old_segments = {}
    for ci, walk in enumerate(component_walks(original)):
        for eid, _ in walk:
            edge = next(e for e in original['edges'] if e['id'] == eid)
            for a, b in zip(edge['points'], edge['points'][1:]):
                old_segments[tuple(sorted((tuple(a), tuple(b))))] = ci
    common = 0
    for ci, walk in enumerate(component_walks(result)):
        for eid, _ in walk:
            edge = next(e for e in result['edges'] if e['id'] == eid)
            for a, b in zip(edge['points'], edge['points'][1:]):
                key = tuple(sorted((tuple(a), tuple(b))))
                if key in old_segments:
                    common += 1
                    assert ci == old_segments[key]
    assert common > 20
