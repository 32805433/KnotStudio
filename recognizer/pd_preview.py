"""Exact, bounded PD preparation for accepted compact editor diagrams.

The displayed encoding is always the full diagrammatic PD. In particular a
box's parity proxy must never be used here, even when a drag only changes the
outside geometry.
"""
from __future__ import annotations

import numpy as np

from .twist_boxes import is_twist_box


PD_PREVIEW_LIMIT = 10000
PD_PREVIEW_LIMIT_MESSAGE = (
    'This compact diagram exceeds the 10,000-crossing PD preview limit. '
    'Its twist boxes remain editable and can be saved as JSON.'
)


def pd_preview_crossings(diagram):
    """Lower bound on expanded size; passage crossings are checked on expansion."""
    return sum(abs(v['half_twists'])*v.get('strand_count', 2)*(v.get('strand_count', 2)-1)//2
               if is_twist_box(v) else 1 for v in diagram.get('crossings', []))


def _has_passages(diagram):
    return any(is_twist_box(v) and v.get('passages') for v in diagram['crossings'])


def _encoding_graph(diagram):
    """Discard exterior samples only when no passage can intersect them.

    Without passages, expansion uses exterior geometry solely to draw the
    resulting arcs. Its incidence, row order and oriented component witnesses
    are independent of the samples. Keep endpoint coordinates for zero-box
    splicing, while avoiding a recursive copy of thousands of drag samples.

    Passage intersections depend on actual geometry. Retain it unchanged;
    replacing an exterior path by its endpoint chord could introduce crossings
    through a box and corrupt the exported PD.
    """
    if _has_passages(diagram) or not any(is_twist_box(v) for v in diagram['crossings']):
        return diagram
    return {
        'crossings': diagram['crossings'],
        'edges': [dict(id=e['id'], start=e.get('start'), end=e.get('end'),
                       closed=e.get('closed', False),
                       points=[e['points'][0], e['points'][-1]])
                  for e in diagram['edges']],
        'component_orientations': diagram.get('component_orientations', {}),
    }


def prepare_compact_pd_preview(diagram):
    """Return ``(full_pd, crossing_free_components, error)`` without UI work."""
    if pd_preview_crossings(diagram) > PD_PREVIEW_LIMIT:
        return None, None, PD_PREVIEW_LIMIT_MESSAGE
    from .diagram import crossing_free_count, pd_code
    try:
        pd = pd_code(_encoding_graph(diagram))
        return pd, crossing_free_count(diagram, pd), None
    except Exception as exc:
        # The geometry has already been accepted. An encoding failure must not
        # turn that accepted engine state into a rejected drag frame.
        return None, None, str(exc) or type(exc).__name__


def _freeze(value):
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, dict):
        # Legacy in-memory orientation maps can contain both integer and
        # string component IDs; model traversal deliberately prefers the
        # string entry. Do not conflate those keys when caching its encoding.
        return tuple(sorted((type(k).__module__, type(k).__qualname__, repr(k), _freeze(v))
                            for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def pd_preview_key(diagram):
    """Exact cache key for encoding, including row order and component direction.

    A geometric move with no passages cannot affect PD. A Reidemeister move,
    port reassignment, crossing switch, coefficient edit or orientation edit
    does affect it and invalidates the key. Passage expansion also uses strand
    geometry and crossing positions, so all of those coordinates are retained
    in that case, including stored passage crossing order.
    """
    geometry = _has_passages(diagram)
    vertices = []
    for vertex in diagram['crossings']:
        if is_twist_box(vertex):
            # Box geometry also participates in validation. Retain the complete
            # compact record, including passage points and layer decisions.
            vertices.append(_freeze(vertex))
        else:
            vertices.append((vertex['id'], vertex.get('over', 1),
                             _freeze(vertex['ports']),
                             _freeze(vertex.get('point')) if geometry else None))
    edges = tuple((e['id'], _freeze(e.get('start')), _freeze(e.get('end')),
                   bool(e.get('closed', False)),
                   np.asarray(e['points'], dtype=float).tobytes() if geometry else None)
                  for e in diagram['edges'])
    return tuple(vertices), edges, _freeze(diagram.get('component_orientations', {}))


class PDPreviewCache:
    """One gesture's most recent exact encoding, independent of graph identity.

    A motion worker owns this cache; accepted graphs and returned PD lists must
    be treated as values. Only successful results are reused so a corrected
    geometry or encoding failure is always retried.
    """

    def __init__(self):
        self._key = None
        self._prepared = None

    def prepare(self, diagram):
        # Keep arbitrarily large compact labels cheap before walking geometry.
        if pd_preview_crossings(diagram) > PD_PREVIEW_LIMIT:
            return None, None, PD_PREVIEW_LIMIT_MESSAGE
        key = pd_preview_key(diagram)
        if self._prepared is not None and key == self._key:
            return self._prepared
        prepared = prepare_compact_pd_preview(diagram)
        if prepared[2] is None:
            self._key, self._prepared = key, prepared
        return prepared
