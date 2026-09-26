"""Exact exterior embedding checks with compact twist rectangles as holes.

No proxy braid is required: implicit crossings and passing strands remain
inside fixed boxes. Box attachments are distinct boundary points, never fake
four-valent crossings with an artificial collision exemption.
"""
from __future__ import annotations

import numpy as np

from .geometry import check_embedding, embedding_stroke_contacts
from .twist_boxes import box_port_points, is_twist_box


def _exterior(diagram):
    boxes = {b['id']: b for b in diagram['crossings'] if is_twist_box(b)}
    if not boxes:
        return diagram
    edges = []
    for original in diagram['edges']:
        edge = original
        for key in ('start', 'end'):
            location = edge.get(key)
            if location is not None and location[0] in boxes:
                if edge is original:
                    edge = dict(original)
                edge[key] = None
        edges.append(edge)
    return {**diagram, 'crossings': [c for c in diagram['crossings'] if not is_twist_box(c)],
            'edges': edges}


def _box_contact_error(diagram, active_edge_ids=None):
    """Slab-clipping detects any exterior segment in an open box rectangle.

    Checking vertices alone misses a segment which crosses a whole box. The
    rectangle is shrunk only by numerical tolerance; true boundary attachments
    remain allowed, but a curve cannot enter through one side and leave another.
    """
    boxes = [b for b in diagram['crossings'] if is_twist_box(b)]
    if not boxes:
        return None
    for edge in diagram['edges']:
        if active_edge_ids is not None and edge['id'] not in active_edge_ids:
            continue
        points = np.asarray(edge['points'], float)
        for box in boxes:
            axis = np.asarray(box['axis'], float)
            basis = np.array([[axis[1], -axis[0]], axis])
            local = (points-box['point'])@basis.T
            half = np.asarray(box['size'], float)/2-1e-7
            a, delta = local[:-1], np.diff(local, axis=0)
            candidates = (np.all(np.minimum(a, a+delta) < half, axis=1)
                          & np.all(np.maximum(a, a+delta) > -half, axis=1))
            if np.any(candidates):
                a, delta = a[candidates], delta[candidates]
                lower, upper = np.zeros(len(a)), np.ones(len(a))
                for dim in (0, 1):
                    moving = np.abs(delta[:, dim]) > 1e-14
                    denominator = np.where(moving, delta[:, dim], 1.)
                    left = (-half[dim]-a[:, dim])/denominator
                    right = (half[dim]-a[:, dim])/denominator
                    lower = np.maximum(lower, np.where(moving, np.minimum(left, right), 0.))
                    upper = np.minimum(upper, np.where(moving, np.maximum(left, right), 1.))
                    upper[~moving & (np.abs(a[:, dim]) >= half[dim])] = -1.
                if np.any(upper > lower+1e-12):
                    return 'An exterior strand enters a twist box; its boundary must stay fixed.'
            for end, key in enumerate(('start', 'end')):
                location = edge.get(key)
                if location is not None and location[0] == box['id']:
                    expected = box_port_points(box)[location[1]]
                    if np.linalg.norm(points[-1 if end else 0]-expected) > 1e-6:
                        return 'A strand attachment moved away from its twist-box port.'
    return None


def check_exterior_embedding(diagram, **options):
    error = _box_contact_error(diagram, options.get('active_edge_ids'))
    if error:
        return {'valid': False, 'errors': [error]}
    return check_embedding(_exterior(diagram), **options)


def exterior_stroke_contacts(diagram):
    error = _box_contact_error(diagram)
    if error:
        raise ValueError(error)
    return embedding_stroke_contacts(_exterior(diagram))
