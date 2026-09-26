"""Exact reuse, deadline safety, and equivalence of the faster geometry work."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
import pytest

from recognizer import knotfolio_backend as backend
from recognizer.bundled_crossings import EndpointTangents, stable_tangent
from recognizer.pipeline import recognize_array
from recognizer.recognition_runtime import (RecognitionDeadline, checkpoint, current,
                                          recognition_runtime)


def pixels():
    image = Image.new('RGB', (320, 240), 'white')
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 35, 270, 205), outline='black', width=5)
    draw.rectangle((142, 27, 152, 48), fill='white')
    draw.line((150, 20, 150, 65), fill='black', width=4)
    return np.asarray(image)


def comparable(trace):
    return json.dumps(trace, sort_keys=True)


def test_threshold_equivalent_masks_reuse_and_results_are_isolated():
    rgb = pixels()
    with recognition_runtime(None) as state:
        first = backend.extract(rgb, {'threshold': 100})
        expected = deepcopy(first)
        first['paths'][0]['points'][0][0] = -999
        first['diagnostics']['unmatched_endpoints'].clear()
        second = backend.extract(rgb, {'threshold': 230})
        expected['diagnostics']['threshold'] = 230
        assert comparable(second) == comparable(expected)
        assert state.counts['trace_hits'] == state.counts['extract_hits'] == 1
        changed = rgb.copy(); changed[70:90, 35:45] = 0
        backend.extract(changed, {'threshold': 230})
        backend.extract(rgb, {'threshold': 230, 'stop_at_branches': True})
        assert state.counts['extract_misses'] == 3
    assert current() is None


def test_trace_reuse_keeps_matching_options_distinct():
    rgb = pixels()
    for options in ({'max_endpoints': 0}, {'tangent_weight': 2},
                    {'bundle_gaps': True}, {'arrow_wings': True}):
        expected = backend.extract(rgb, options)
        with recognition_runtime(None):
            backend.extract(rgb)
            actual = backend.extract(rgb, options)
        assert comparable(actual) == comparable(expected)


def test_radius_filter_preserves_brute_force_candidates(monkeypatch):
    rgb = pixels()
    class AllEnds:
        def __init__(self, points): self.count = len(points)
        def query_ball_point(self, points, radius):
            return [list(range(self.count)) for _ in points]
    for opts in ({}, {'bundle_gaps': True}, {'max_gap_widths': 2}):
        actual = backend.extract(rgb, opts)
        with monkeypatch.context() as patch:
            patch.setattr(backend, 'cKDTree', AllEnds)
            expected = backend.extract(rgb, opts)
        assert comparable(actual) == comparable(expected)


def test_lightweight_graph_preserves_cycle_order():
    rng = np.random.default_rng(316)
    for _ in range(40):
        skeleton = rng.random((20, 22)) < .5
        graph = backend._adjacency(skeleton)
        assert nx.cycle_basis(backend._PixelGraphView(graph)) == nx.cycle_basis(nx.Graph(graph))


def test_cached_tangents_are_exact():
    rng = np.random.default_rng(99)
    arrays = [np.cumsum(rng.normal(size=(80, 2)), axis=0) for _ in range(3)]
    endpoints = [{'path': i, 'end': end} for i in range(3) for end in (0, 1)]
    cache = EndpointTangents(arrays, endpoints)
    for index, endpoint in enumerate(endpoints):
        for distance in (0., .1, 4., 15., 40., 1000.):
            expected = stable_tangent(arrays[endpoint['path']], endpoint['end'], distance)
            np.testing.assert_array_equal(cache.get(index, distance), expected)
            np.testing.assert_array_equal(cache.get(index, distance), expected)


def test_color_packing_preserves_lexicographic_bins():
    rng = np.random.default_rng(721)
    for _ in range(12):
        colors = rng.integers(0, 5, (1000, 3), dtype=np.uint8)
        expected, inverse, counts = np.unique(colors, axis=0, return_inverse=True, return_counts=True)
        packed = colors[:, 0]*25 + colors[:, 1]*5 + colors[:, 2]
        unique, index, sizes = np.unique(packed, return_inverse=True, return_counts=True)
        np.testing.assert_array_equal(np.column_stack((unique//25, unique//5 % 5, unique % 5)), expected)
        np.testing.assert_array_equal(index, inverse)
        np.testing.assert_array_equal(sizes, counts)


def test_budget_returns_partial_evidence_without_a_pd(monkeypatch):
    from recognizer import pipeline
    def interrupted(rgb, *, options):
        state = current()
        state.boxes = {'boxes': [{'id': 'box', 'half_twists': None}]}
        state.record_failure({'diagram': None, 'pd_code': None, 'diagnostics': {
            'unmatched_endpoints': [{'point': [8, 9]}]}, 'preview_paths': []})
        state.deadline = state.started - 1
        checkpoint('local crossing search')
    monkeypatch.setattr(pipeline, '_recognize_with_runtime', interrupted)
    result = recognize_array(pixels(), options={'time_budget': .01})
    assert result['diagram'] is result['pd_code'] is None
    assert result['diagnostics']['budget_exhausted']
    assert result['diagnostics']['twist_boxes']['boxes'][0]['half_twists'] is None
    assert result['diagnostics']['unmatched_endpoints'][0]['point'] == [8, 9]
    assert current() is None


@pytest.mark.parametrize('seconds', [0, -1, float('inf'), float('nan')])
def test_invalid_budgets_are_rejected(seconds):
    with pytest.raises(ValueError, match='time_budget'):
        recognize_array(pixels(), options={'time_budget': seconds})


def test_runtime_does_not_leak_across_threads_or_calls():
    def worker(number):
        assert current() is None
        with recognition_runtime(None) as state:
            state.put(('test',), [number])
            assert state.get(('test',)) == [number]
        assert current() is None
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(worker, range(12)))


def test_cache_is_bounded_and_does_not_change_recognition():
    with recognition_runtime(None) as state:
        state.limit = 1024
        for i in range(20): state.put(('sample', i), np.arange(60))
        assert state.cache_bytes <= state.limit
        assert state.get(('sample', 19)) is not None
        assert state.get(('sample', 0)) is None


def test_shortened_solver_cannot_export_an_unfinished_incumbent(monkeypatch):
    from scipy.optimize import OptimizeResult
    graph = nx.Graph()
    candidates = {(0, 2): {'cost': 1.}, (1, 3): {'cost': 1.},
                  (0, 1): {'cost': 3.}, (2, 3): {'cost': 3.}}
    for edge, value in candidates.items(): graph.add_edge(*edge, weight=value['cost'])
    endpoints = [{'point': np.array(p, float)} for p in ((0, 0), (1, 0), (1, 1), (0, 1))]
    monkeypatch.setattr(backend, 'remaining', lambda cap: .05)
    monkeypatch.setattr(backend, 'milp', lambda *a, **k:
                        OptimizeResult(x=np.array([0., 0., 1., 1.]), status=1))
    with pytest.raises(RecognitionDeadline):
        backend._noncrossing_matching(graph, endpoints, candidates)
