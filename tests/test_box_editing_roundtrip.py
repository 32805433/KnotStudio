"""Box editing keeps physical strand directions and the full signed twist."""
from copy import deepcopy
import json

import pytest

from recognizer.box_operations import combine_twist_boxes, delete_zero_box, expand_twist_box
from recognizer.diagram import component_walks, pd_code, validate
from recognizer.pd_preview import prepare_compact_pd_preview
from recognizer.twist_boxes import expand_boxes, is_twist_box, parse_twist_label
from test_twist_boxes import closed_braid, two_boxes


def rooted_visits(diagram, roots):
    """Record oriented over/under visits rooted on unchanged exterior arcs."""
    diagram = expand_boxes(diagram)
    edges = {e['id']: e for e in diagram['edges']}
    vertices = {v['id']: v for v in diagram['crossings']}
    orientations = diagram.get('component_orientations', {})
    directions = {eid: direction ^ (orientations.get(str(ci), 1) == -1)
                  for ci, walk in enumerate(component_walks(diagram)) for eid, direction in walk}
    seen, labels, result = set(), {}, []
    for root in sorted(roots):
        if root in seen:
            continue
        first = dart = root, int(directions[root])
        word = []
        while True:
            eid, direction = dart
            seen.add(eid)
            endpoint = edges[eid]['end' if direction == 0 else 'start']
            if endpoint is None:
                break
            cid, port = endpoint
            labels.setdefault(cid, len(labels))
            word.append((labels[cid], port % 2 == vertices[cid]['over']))
            next_port = vertices[cid]['ports'][(port+2) % 4]
            dart = next_port['edge'], next_port['end']
            if dart == first:
                break
        result.append(word)
    assert seen == set(edges)
    return result


@pytest.mark.parametrize('strands', [2, 3, 4])
@pytest.mark.parametrize('label,amount', [('2', '1/2'), ('-2', '-1/2')])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_partial_expansion_preserves_rooted_orientations_and_json(strands, label, amount, side):
    diagram = closed_braid(label, strands)
    diagram['component_orientations'] = {'0': -1}
    # Give four-strand explicit braids ample space without changing incidence.
    for vertex in diagram['crossings']:
        vertex['point'] = [v*4 for v in vertex['point']]
        vertex['size'] = [v*4 for v in vertex['size']]
    for edge in diagram['edges']:
        edge['points'] = [[v*4 for v in point] for point in edge['points']]
    original = deepcopy(diagram)
    changed = expand_twist_box(diagram, 10, amount, side)
    assert validate(changed)['valid']
    residual = next(v for v in changed['crossings'] if is_twist_box(v))
    assert residual['half_twists'] == parse_twist_label(label)-parse_twist_label(amount)
    roots = {e['id'] for e in original['edges']}
    assert rooted_visits(changed, roots) == rooted_visits(original, roots)
    restored = json.loads(json.dumps(changed))
    assert pd_code(restored) == pd_code(changed)
    assert diagram == original


@pytest.mark.parametrize('strands', [2, 3, 4])
def test_cancel_boxes_and_delete_zero_preserves_components_and_orientations(strands):
    diagram = two_boxes('1/2', '-1/2', strands)
    original = deepcopy(diagram)
    merged = combine_twist_boxes(diagram, 0, 1)
    assert merged['crossings'][0]['half_twists'] == 0
    result = delete_zero_box(merged, 0)
    assert validate(result)['valid']
    assert pd_code(result) == []
    assert len(component_walks(result)) == strands
    assert result['component_orientations'] == merged['component_orientations']
    assert diagram == original


def test_expand_one_box_preserves_huge_neighbor_and_bounded_preview():
    diagram = two_boxes('1/2', str(10**40))
    original = deepcopy(diagram)
    changed = expand_twist_box(diagram, 0)
    boxes = [v for v in changed['crossings'] if is_twist_box(v)]
    assert len(boxes) == 1
    assert boxes[0]['half_twists'] == 2*10**40
    assert validate(changed)['valid']
    pd, count, error = prepare_compact_pd_preview(changed)
    assert pd is None and count is None and '10,000' in error
    assert len(json.dumps(changed)) < 30_000
    assert diagram == original


def boxed_loops(label):
    """Two disjoint exterior circle arcs attached to one compact box."""
    import math
    import numpy as np
    from recognizer.twist_boxes import make_twist_box, bundle_ports
    theta = math.acos(.44)
    box = make_twist_box(10, [250, 200], [120, 2*math.sqrt(100**2-44**2)], label)
    top, bottom = bundle_ports(box)
    edges = []
    for lane, (cx, angles) in enumerate(((170, np.linspace(-theta, -math.tau+theta, 221)),
                                         (330, np.linspace(-math.pi+theta, math.pi-theta, 221)))):
        points = np.column_stack((cx+100*np.cos(angles), 200+100*np.sin(angles))).tolist()
        edges.append(dict(id=30+lane, start=[10, top[lane]], end=[10, bottom[lane]],
                          closed=False, points=points))
        box['ports'][top[lane]] = dict(edge=30+lane, end=0)
        box['ports'][bottom[lane]] = dict(edge=30+lane, end=1)
    return dict(width=520, height=650, crossings=[box], edges=edges,
                component_orientations={'0': -1})


@pytest.mark.parametrize('label', ['1/2', str(10**40)])
def test_box_drag_remains_compact_and_can_continue_after_serialization(label):
    from recognizer.box_motion import BoxSmoothDrag
    diagram = boxed_loops(label)
    original = deepcopy(diagram)
    moved = BoxSmoothDrag(diagram, 30, [70, 200], radius=70).move([60, 205])
    assert validate(moved)['valid']
    box = next(v for v in moved['crossings'] if is_twist_box(v))
    for field in ('point', 'size', 'axis', 'half_twists', 'strand_count'):
        assert box[field] == original['crossings'][0][field]
    assert len(moved['crossings']) == 1
    assert diagram == original
    saved = json.loads(json.dumps(moved))
    # The second gesture must not require a cached lift from the first one.
    from test_motion_roundtrip import edge_near
    edge = edge_near(saved, [60, 205])
    result = BoxSmoothDrag(saved, edge, [60, 205], radius=65).move([55, 208])
    assert validate(result)['valid']
    assert len(result['crossings']) == 1
    assert next(v for v in result['crossings'] if is_twist_box(v))['half_twists'] == parse_twist_label(label)


def test_energy_fixes_box_and_orientation_and_keeps_previous_frames_independent():
    from recognizer.relaxation import RelativeRelaxation
    diagram = boxed_loops('1')
    original = deepcopy(diagram)
    before_pd = pd_code(diagram)
    engine = RelativeRelaxation(diagram, [[30, 50], [490, 50], [490, 360], [30, 360]])
    first = engine.step()
    assert first is not None
    snapshot = deepcopy(first)
    for _ in range(8):
        result = engine.step()
        if result is None:
            break
        assert validate(result)['valid']
        assert pd_code(result) == before_pd
        assert result['crossings'][0] == original['crossings'][0]
    assert engine.iterations > 1
    assert engine.energy < engine.initial_energy
    assert first == snapshot
    assert diagram == original
