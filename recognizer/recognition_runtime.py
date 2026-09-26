"""One recognition's shared deadline and bounded, pixel-keyed work cache.

Context-local state keeps concurrent UI workers independent. Nothing survives
the call or uses image names, dataset metadata, or previous review decisions.
"""
from collections import Counter, OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import math
import pickle
import time
import json
from functools import wraps

import numpy as np

DEFAULT_SECONDS = 10.0
_current = ContextVar('recognition_runtime', default=None)


class RecognitionDeadline(Exception):
    pass


class Runtime:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.seconds = seconds
        self.deadline = None if seconds is None else self.started + seconds
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.limit = 48 * 1024 * 1024
        self.counts = Counter()
        self.stage = 'starting'
        self.boxes = None
        self.failure = None

    def get(self, key):
        if key not in self.cache:
            self.counts[key[0] + '_misses'] += 1
            return None
        data = self.cache.pop(key)
        self.cache[key] = data
        self.counts[key[0] + '_hits'] += 1
        # Serialization isolates every consumer's mutable graph/diagnostics.
        return pickle.loads(data)

    def put(self, key, value):
        data = pickle.dumps(value, protocol=5)
        if len(data) > self.limit:
            return
        old = self.cache.pop(key, None)
        self.cache_bytes -= len(old) if old else 0
        while self.cache and self.cache_bytes + len(data) > self.limit:
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= len(old)
        self.cache[key] = data
        self.cache_bytes += len(data)

    def record_failure(self, result):
        if result.get('diagram') is None:
            # This is an inspected partial trace, never an accepted diagram.
            diagnostic = result.get('diagnostics', {})
            score = sum(len(diagnostic.get(k, [])) for k in
                        ('branch_points', 'unmatched_endpoints', 'gap_conflicts'))
            if self.failure is None or score < self.failure[0]:
                self.failure = (score, deepcopy(result))

    def record_trace_failure(self, trace, *, scale=1., preprocessing=None):
        """Keep partial evidence before the next extraction can time out.

        A reconstruction may try several masks without returning. Save only
        incomplete traces, in original-image coordinates, so the editor can
        still show actual unresolved endpoints after a deadline. No candidate
        topology or inferred PD is promoted to a recognized diagram here.
        """
        if trace.get('complete'):
            return
        source = trace.get('diagnostics', {})
        keys = ('branch_points', 'unmatched_endpoints', 'gap_conflicts')
        score = sum(len(source.get(k, [])) for k in keys)
        if (self.failure is not None and score >= self.failure[0]
                or not trace.get('paths') and not score):
            return
        diagnostic = {key: deepcopy(source.get(key, [])) for key in keys}
        diagnostic['branch_points'] = [[v/scale for v in point]
                                       for point in diagnostic['branch_points']]
        for endpoint in diagnostic['unmatched_endpoints']:
            endpoint['point'] = [v/scale for v in endpoint['point']]
        diagnostic['stroke_width'] = source.get('stroke_width', 1.5)/scale
        for key in ('threshold', 'endpoint_count', 'matching_skipped'):
            if key in source:
                diagnostic[key] = deepcopy(source[key])
        if preprocessing is not None:
            diagnostic['preprocessing'] = deepcopy(preprocessing)
        preview = deepcopy(trace.get('paths', []))
        for path in preview:
            path['points'] = [[v/scale for v in point] for point in path['points']]
        self.failure = (score, dict(diagram=None, pd_code=None,
                                   diagnostics=diagnostic, preview_paths=preview,
                                   unlinked_unknot_components=None))


def current():
    return _current.get()


def remaining(cap=None):
    state = current()
    left = (max(0., state.deadline - time.monotonic())
            if state and state.deadline is not None else math.inf)
    return min(left, cap) if cap is not None else left


def checkpoint(stage):
    state = current()
    if state is not None:
        state.stage = stage
        if remaining() <= 0:
            raise RecognitionDeadline(stage)


def array_key(array):
    array = np.ascontiguousarray(array)
    return array.shape, array.dtype.str, hashlib.blake2b(array.view(np.uint8), digest_size=20).digest()


def memoized_attempt(function):
    @wraps(function)
    def attempt(rgb, *, options=None):
        checkpoint('trying a reconstruction')
        state = current()
        if state is None:
            return function(rgb, options=options)
        key = ('reconstruction', array_key(rgb), json.dumps(options or {}, sort_keys=True))
        result = state.get(key)
        if result is None:
            result = function(rgb, options=options)
            state.put(key, result)
        state.record_failure(result)
        return result
    return attempt


@contextmanager
def recognition_runtime(seconds):
    if seconds is not None:
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('time_budget must be a positive finite number of seconds, or None.')
    state = Runtime(seconds)
    token = _current.set(state)
    try:
        yield state
    finally:
        _current.reset(token)
        state.cache.clear()
