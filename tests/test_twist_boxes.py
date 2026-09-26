"""Synthetic twist-box topology: no recognition corpus images are used here."""
from copy import deepcopy
from collections import Counter
import json
import math

import pytest

from recognizer.diagram import component_walks, pd_code, validate
from recognizer.reidemeister import crossing_sign
from recognizer.twist_boxes import (
    box_corners, box_port_points, bundle_ports, compact_component_walks,
    expand_boxes, make_twist_box, parse_twist_label, transition_port,
    twist_label, validate_box,
)


def closed_braid(label, strands=2):
    box = make_twist_box(10, [200, 180], [120, 160], label, strand_count=strands)
    points = box_port_points(box)
    top, bottom = bundle_ports(box)
    edges = []
    for lane, (a, b) in enumerate(zip(top, bottom)):
        eid = lane + 30
        box['ports'][a] = dict(edge=eid, end=0)
        box['ports'][b] = dict(edge=eid, end=1)
        # The topology is the braid closure. Geometry is only used to determine
        # crossing tangents, which expansion provides locally inside the box.
        edges.append(dict(id=eid, start=[10, a], end=[10, b], closed=False,
                          points=[points[a], [50 - lane*10, 60 - lane*10],
                                  [50-lane*10, 290+lane*10], points[b]]))
    return dict(width=400, height=360, crossings=[box], edges=edges,
                component_orientations={})


@pytest.mark.parametrize('value,half', [('0', 0), ('1', 2), ('-2', -4), ('+1/2', 1),
                                        ('−3/2', -3), ('½', 1), ('-½', -1),
                                        ('2/4', 1), (1.5, 3), (-.5, -1)])
def test_exact_half_integer_values(value, half):
    assert parse_twist_label(value) == half
    assert parse_twist_label(twist_label(half)) == half


@pytest.mark.parametrize('value', [True, 'n', '3/4', '.125', '1/0', 'nan', float('inf')])
def test_ambiguous_or_non_half_integral_values_rejected(value):
    with pytest.raises(ValueError):
        parse_twist_label(value)


@pytest.mark.parametrize('strands', [2, 3, 4, 6])
@pytest.mark.parametrize('label', ['0', '1/2', '-1/2', '1', '-1', '3/2', '-3/2'])
def test_braid_expansion_preserves_components_planarity_and_signed_crossings(strands, label):
    compact = closed_braid(label, strands)
    original = deepcopy(compact)
    expanded = expand_boxes(compact)
    half = parse_twist_label(label)
    expected_crossings = abs(half) * strands * (strands-1) // 2
    expected_components = (strands+1)//2 if half % 2 else strands
    assert len(expanded['crossings']) == expected_crossings
    assert len(compact_component_walks(compact)) == expected_components
    assert len(component_walks(expanded)) == expected_components
    assert validate(expanded)['valid'], validate(expanded)
    assert len(pd_code(expanded)) == expected_crossings
    if half:
        assert {crossing_sign(expanded, c['id']) for c in expanded['crossings']} == {1 if half > 0 else -1}
    else:
        assert all(e['closed'] for e in expanded['edges'])
    assert compact == original


@pytest.mark.parametrize('strands', [2, 3, 6])
@pytest.mark.parametrize('label', ['-3/2', '-1', '0', '1/2', '2'])
def test_proxy_preserves_boundary_connectivity_but_only_parity(strands, label):
    compact = closed_braid(label, strands)
    expanded = expand_boxes(compact, proxy=True)
    parity = abs(parse_twist_label(label)) % 2
    assert len(expanded['crossings']) == parity * strands * (strands-1) // 2
    assert len(component_walks(expanded)) == len(compact_component_walks(compact))
    assert validate(expanded)['valid']
    record = expanded['twist_box_expansion'][0]
    assert record['box'] == compact['crossings'][0]
    assert record['proxy'] is True


def test_full_twist_links_each_pair_once():
    expanded = expand_boxes(closed_braid('1', 5))
    edge_components = {e['id']: e['component'] for e in expanded['edges']}
    pairs = Counter(tuple(sorted((edge_components[c['ports'][0]['edge']],
                                  edge_components[c['ports'][1]['edge']])))
                    for c in expanded['crossings'])
    assert len(pairs) == 10
    assert set(pairs.values()) == {2}
    assert all(a != b for a, b in pairs)


def test_reversing_one_component_changes_only_mixed_crossing_signs():
    compact = closed_braid('1', 3)
    compact['component_orientations']['0'] = -1
    expanded = expand_boxes(compact)
    assert sorted(crossing_sign(expanded, c['id']) for c in expanded['crossings']) == [-1, -1, -1, -1, 1, 1]
    assert validate(expanded)['valid']


def test_zero_splicing_preserves_directions_on_closed_components():
    compact = closed_braid('0', 3)
    compact['component_orientations'] = {'0': -1, '2': -1}
    expanded = expand_boxes(compact)
    assert expanded['component_orientations'] == {'0': -1, '2': -1}
    assert validate(expanded)['crossing_free_components'] == 3


def test_large_twist_count_stays_compact_and_expansion_is_explicitly_bounded():
    compact = closed_braid('1000000000000000000000000000000', 6)
    validate_box(compact['crossings'][0])
    assert len(json.dumps(compact)) < 3000
    assert len(compact_component_walks(compact)) == 6
    with pytest.raises(ValueError, match='expansion needs.*limit is'):
        expand_boxes(compact, max_crossings=500)
    assert not expand_boxes(compact, proxy=True)['crossings']


def test_geometry_rotates_and_two_strand_ports_remain_compatible():
    box = make_twist_box(2, [100, 80], [40, 60], '-1/2', axis=(1, 0))
    assert box_corners(box) == [[70, 100], [130, 100], [130, 60], [70, 60]]
    assert box_port_points(box) == [[70, 92], [130, 92], [130, 68], [70, 68]]
    assert [transition_port(box, p) for p in range(4)] == [2, 3, 0, 1]
    box['half_twists'], box['label'] = 2, '1'
    assert [transition_port(box, p) for p in range(4)] == [1, 0, 3, 2]


def test_general_ports_and_separate_passage_pairing():
    box = make_twist_box(2, [100, 80], [40, 60], '1/2', strand_count=3)
    points = box_port_points(box)
    box['ports'] += [None, None]
    box['passages'] = [dict(ports=[6, 7], over=True, points=[[80, 80], [120, 80]])]
    box['port_points'] = points+[[80, 80], [120, 80]]
    validate_box(box, require_ports=False)
    top, bottom = bundle_ports(box)
    assert [transition_port(box, p) for p in top] == list(reversed(bottom))
    assert transition_port(box, 6) == 7
    assert transition_port(box, 7) == 6
    box['passages'][0]['ports'] = [5, 7]
    with pytest.raises(ValueError, match='exactly one'):
        validate_box(box, require_ports=False)


def test_invalid_geometry_and_label_disagreement_rejected():
    box = closed_braid('1')['crossings'][0]
    box['label'] = '2'
    with pytest.raises(ValueError, match='disagrees'):
        validate_box(box)
    with pytest.raises(ValueError, match='positive'):
        make_twist_box(1, [0, 0], [0, 1], '1')
    with pytest.raises(ValueError, match='axis'):
        make_twist_box(1, [0, 0], [1, 1], '1', axis=[0, 0])


def two_boxes(first_label, second_label, strands=3):
    boxes = [make_twist_box(i, [200, 120+220*i], [120, 120], value, strand_count=strands)
             for i, value in enumerate((first_label, second_label))]
    edge_id, edges = 20, []
    for a, b in ((0, 1), (1, 0)):
        _, bottom = bundle_ports(boxes[a])
        top, _ = bundle_ports(boxes[b])
        a_points, b_points = box_port_points(boxes[a]), box_port_points(boxes[b])
        for lane, (start, end) in enumerate(zip(bottom, top)):
            boxes[a]['ports'][start] = dict(edge=edge_id, end=0)
            boxes[b]['ports'][end] = dict(edge=edge_id, end=1)
            points = [a_points[start], b_points[end]] if a == 0 else [
                a_points[start], [40-10*lane, 480+10*lane], [40-10*lane, 20-10*lane], b_points[end]]
            edges.append(dict(id=edge_id, start=[a, start], end=[b, end], points=points, closed=False))
            edge_id += 1
    return dict(width=500, height=500, crossings=boxes, edges=edges, component_orientations={'0': -1})


@pytest.mark.parametrize('labels', [('0', '0'), ('0', '1/2'), ('1/2', '0'),
                                    ('1/2', '-1/2'), ('1', '3/2'), ('-1', '0')])
def test_adjacent_boxes_and_zero_splicing(labels):
    compact = two_boxes(*labels)
    expanded = expand_boxes(compact)
    assert validate(expanded)['valid'], validate(expanded)
    assert len(component_walks(expanded)) == len(compact_component_walks(compact))
    assert len(expanded['crossings']) == 3 * sum(abs(parse_twist_label(label)) for label in labels)
    # A physical orientation witness survives on every component, even when
    # expansion splices several differently-directed old edges into one edge.
    for ci, walk in enumerate(component_walks(expanded)):
        assert all(next(e for e in expanded['edges'] if e['id'] == eid)['component'] == ci
                   for eid, _ in walk)


def test_array_backed_external_geometry_is_accepted():
    import numpy as np
    compact = closed_braid('3/2', 3)
    for edge in compact['edges']:
        edge['points'] = np.asarray(edge['points'])
    expanded = expand_boxes(compact)
    assert validate(expanded)['valid']


def test_legacy_two_strand_box_without_explicit_strand_count_is_valid():
    # Two-strand boxes predate the optional strand_count field. Every box
    # reader uses two as its default, including the PD expansion.
    current = closed_braid('1')
    legacy = deepcopy(current)
    legacy['crossings'][0].pop('strand_count')
    snapshot = deepcopy(legacy)
    assert validate(legacy) == validate(current)
    assert validate(legacy)['implicit_crossings'] == 2
    assert pd_code(legacy) == pd_code(current)
    assert legacy == snapshot
