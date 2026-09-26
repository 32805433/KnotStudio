"""Align the two ends of a compact box without moving its exterior drawing.

A shorter concentric box frees two collars inside its former rectangle. Each
collar fans the old ordered attachments to the same evenly spaced lane grid.
All cubic control polygons stay ordered and inside that collar, which proves
that the connectors introduce no crossing. Port IDs and topology do not change.
"""
from copy import deepcopy

import cv2
import numpy as np

from .box_operations import _box, _finish, _segment_inside
from .twist_boxes import box_corners, box_port_points, bundle_ports, is_twist_box


def _fan_parameters(controls, tolerance):
    """A shared adaptive cubic subdivision keeps ordered lanes compact."""
    values = [0.]
    def split(c, left, right, depth):
        error = max(float(np.max(np.abs(c[1]-(2*c[0]+c[3])/3))),
                    float(np.max(np.abs(c[2]-(c[0]+2*c[3])/3))))
        if error <= tolerance or depth == 8:
            values.append(right)
            return
        a, b, d = (c[:-1]+c[1:])/2
        e, f = (a+b)/2, (b+d)/2
        mid = (e+f)/2
        t = (left+right)/2
        split(np.array([c[0], a, e, mid]), left, t, depth+1)
        split(np.array([mid, f, d, c[3]]), t, right, depth+1)
    split(controls, 0., 1., 0)
    # Tiny end intervals retain the intended tangent in the sampled geometry.
    return np.unique(np.r_[values, .001, .999])


def _clear_interior(diagram, box):
    """Do not uncover a hidden passage, overlapping box or malformed geometry."""
    if box.get('passages'):
        raise ValueError('This box has a passing strand. Align its ends before drawing passages, or expand it first.')
    center, axis = np.asarray(box['point']), np.asarray(box['axis'])
    right = np.array([axis[1], -axis[0]])
    frame = np.column_stack([right, axis])
    corners = np.asarray(box_corners(box))
    low, high = corners.min(axis=0), corners.max(axis=0)
    bounds = [-np.asarray(box['size'])/2, np.asarray(box['size'])/2]
    for vertex in diagram['crossings']:
        if vertex['id'] == box['id']:
            continue
        if is_twist_box(vertex):
            area, _ = cv2.intersectConvexConvex(corners.astype(np.float32),
                                               np.asarray(box_corners(vertex), np.float32))
            if area > 1e-5:
                raise ValueError('The box overlaps another box; separate them before aligning strands.')
        elif np.all(np.abs((np.asarray(vertex['point'])-center)@frame) < np.asarray(box['size'])/2-1e-6):
            raise ValueError('An ordinary crossing lies inside this box; the attachments cannot be aligned safely.')
    for edge in diagram['edges']:
        points = np.asarray(edge['points'], float)
        if np.any(points.max(axis=0) < low-1e-6) or np.any(points.min(axis=0) > high+1e-6):
            continue
        local = (points-center)@frame
        candidates = np.flatnonzero(np.all(np.maximum(local[:-1], local[1:]) > bounds[0]+1e-6, axis=1)
                                    & np.all(np.minimum(local[:-1], local[1:]) < bounds[1]-1e-6, axis=1))
        if any(_segment_inside(local[i], local[i+1], bounds) for i in candidates):
            raise ValueError('A strand enters the box interior without a stored passage; align only boxes with a clear interior.')


def _lane_targets(box):
    """Return a common lane grid, or None when no alignment is needed."""
    top, bottom = bundle_ports(box)
    axis = np.asarray(box['axis'], float)
    right = np.array([axis[1], -axis[0]])
    across = (np.asarray(box_port_points(box), float)-box['point'])@right
    # Keep an already evenly spaced bundle, including its width and offset.
    targets = np.linspace((across[top[0]]+across[bottom[0]])/2,
                          (across[top[-1]]+across[bottom[-1]])/2, len(top))
    if all(np.allclose(across[ports], targets, atol=1e-6, rtol=0) for ports in (top, bottom)):
        return None
    return targets


def align_twist_box(diagram, box_id):
    """Return aligned endpoints and smooth internal collars; input is immutable.

    Existing edge samples stay byte-for-byte intact as subsequences. The box is
    shortened axially, never enlarged. This also works for rotated boxes and
    odd half-twist counts: alignment is by lane position, not strand identity.
    """
    box = _box(diagram, box_id)
    top, bottom = bundle_ports(box)
    width, height = box['size']
    center, axis = np.asarray(box['point'], float), np.asarray(box['axis'], float)
    right = np.array([axis[1], -axis[0]])
    points = np.asarray(box_port_points(box), float)
    across = (points-center)@right
    targets = _lane_targets(box)
    if targets is None:
        return deepcopy(diagram)
    _clear_interior(diagram, box)
    from .render import line_width
    minimum_height = max(12., 3*line_width(diagram))
    available = min(.32*height, (height-minimum_height)/2)
    if available < 1e-3:
        raise ValueError('This box is too short to make room for aligned strand attachments.')
    displacement = max(float(np.max(np.abs(across[ports]-targets))) for ports in (top, bottom))
    collar = min(available, max(6*line_width(diagram), 1.25*displacement))
    edges = {edge['id']: edge for edge in diagram['edges']}
    connectors = {}
    for sign, ports in ((-1, top), (1, bottom)):
        inward = -sign*axis
        starts = across[ports]
        slopes = []
        for port in ports:
            ref = box['ports'][port]
            edge = edges[ref['edge']]
            if edge['start' if ref['end'] == 0 else 'end'] != [box_id, port]:
                raise ValueError('The box attachment disagrees with its connected strand.')
            path = np.asarray(edge['points'][::1 if ref['end'] == 0 else -1], float)
            if np.linalg.norm(path[0]-points[port]) > 1e-5:
                raise ValueError('The box attachment is detached from its strand.')
            tangent = next((path[0]-p for p in path[1:] if np.linalg.norm(p-path[0]) > 1e-6), None)
            forward = float(tangent@inward) if tangent is not None else 0.
            slopes.append(float(tangent@right)/forward if forward > 1e-6 else 0.)
        slopes = np.asarray(slopes)
        # Limit the tangent length for the whole bundle together. Ordered
        # Bezier control points imply ordered strands for every parameter t.
        strength = 1.
        while True:
            control = starts+strength*slopes*collar/3
            if (np.all(np.abs(control) < width/2-1e-6)
                    and np.all(np.diff(control) > 1e-5)):
                break
            strength *= .5
            if strength < 1e-6:
                control = starts.copy()
                break
        # Shared samples preserve the strict ordering in the stored polylines.
        # Include a short normal segment at the box to prevent visible corners.
        controls = np.vstack([starts, control, targets, targets])
        t = _fan_parameters(controls, max(.1, line_width(diagram)*.05))
        weight = np.column_stack([(1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3])
        coordinates = weight@controls
        along = sign*(height/2-collar*t)
        for lane, port in enumerate(ports):
            path = center+coordinates[:, lane, None]*right+along[:, None]*axis
            path[0] = points[port]
            connectors[port] = path.tolist()
    result = deepcopy(diagram)
    new_box = _box(result, box_id)
    new_box['size'][1] = height-2*collar
    new_box['port_points'] = points.tolist()
    for port, connector in connectors.items():
        new_box['port_points'][port] = connector[-1]
    for edge in result['edges']:
        start, end = edge.get('start'), edge.get('end')
        if start and start[0] == box_id:
            edge['points'] = connectors[start[1]][:0:-1]+edge['points']
        if end and end[0] == box_id:
            edge['points'] = edge['points']+connectors[end[1]][1:]
    return _finish(result)


def align_created_box(diagram, box_id):
    """Best-effort layout after an otherwise valid box edit.

    Lack of collar room must not turn a successful algebraic operation into a
    failure. The original layout is retained when alignment is unsafe.
    """
    try:
        return align_twist_box(diagram, box_id)
    except ValueError:
        return diagram


def align_twist_boxes(diagram):
    """Automatically align every safe box at an editor transaction boundary.

    Inputs are never mutated. Aligned boxes need neither a copy nor a geometry
    scan; an obstructed box does not prevent the other boxes from aligning.
    Call when opening/recognizing a diagram or committing a box edit, never
    on redraw, individual motion frames, or undo/redo restoration.
    """
    if diagram is None:
        return None
    result = diagram
    for box in diagram.get('crossings', []):
        if is_twist_box(box) and _lane_targets(box) is not None:
            result = align_created_box(result, box['id'])
    return result
