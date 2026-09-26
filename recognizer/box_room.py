"""Reclaim a clear parallel collar to give an explicit twist room to display."""
from copy import deepcopy
import math

import cv2
import numpy as np

from .box_geometry import collapse_box
from .box_insert import _rectangle_ports
from .diagram import assign_components, validate
from .render import line_width
from .twist_boxes import box_corners, expand_boxes, is_twist_box, twist_label


def _certify_room_proxy(box, trails, binding, vertices):
    from .box_compression import certify_selected_bundle
    proxy_box = deepcopy(box)
    count = box['half_twists']
    proxy_box['half_twists'] = (1 if count >= 0 else -1)*(abs(count) % 2)
    proxy_box['label'] = twist_label(proxy_box)
    return certify_selected_bundle(proxy_box, trails, binding, vertices)


def grow_twist_box_room(diagram, box_id, minimum_length, side='bottom'):
    """Extend one box into certified empty collars without moving exterior ink.

    Only the selected box is expanded to a small parity proxy while finding its
    longer rectangle. All original twists remain discrete after recompression.
    With ``side='both'``, half of the extra length is reclaimed at each end.
    A crossing, foreign strand/box, or turning bundle blocks growth. The input
    is untouched on both success and rejection.
    """
    box = next((v for v in diagram['crossings'] if v['id'] == box_id and is_twist_box(v)), None)
    if box is None:
        raise ValueError('Select a twist box first.')
    if side not in ('top', 'bottom', 'both'):
        raise ValueError('Choose the top, bottom, or both ends for expansion.')
    if not isinstance(minimum_length, (int, float)) or not math.isfinite(minimum_length) or minimum_length <= 0:
        raise ValueError('The required box length must be finite and positive.')
    if minimum_length <= box['size'][1]:
        return deepcopy(diagram)
    if box.get('passages'):
        raise ValueError('A box with passing strands needs enough room before expansion; move those strands clear to enlarge it.')
    extra = minimum_length-box['size'][1]
    axis = np.asarray(box['axis'], float)
    grown = deepcopy(box)
    direction = {'top': -1, 'bottom': 1, 'both': 0}[side]
    grown['point'] = (np.asarray(box['point'])+direction*axis*extra/2).tolist()
    grown['size'] = [box['size'][0], float(minimum_length)]
    # Measured ports will be re-established at the actual new boundary cuts.
    grown.pop('port_points', None)
    grown.pop('bundle_ports', None)
    grown['ports'] = [None]*(2*grown['strand_count'])
    polygon = np.asarray(box_corners(grown), np.float32)
    width = line_width(diagram)
    for vertex in diagram['crossings']:
        if vertex['id'] == box_id:
            continue
        if is_twist_box(vertex):
            area, _ = cv2.intersectConvexConvex(polygon, np.asarray(box_corners(vertex), np.float32))
            occupied = area > 1e-5
        else:
            occupied = cv2.pointPolygonTest(polygon, tuple(map(float, vertex['point'])), True) >= -2*width
        if occupied:
            raise ValueError('A crossing or another box blocks the required space. Try the other end, expand less, or move it clear.')
    proxy = expand_boxes(diagram, proxy=True, box_ids=[box_id])
    try:
        _rectangle_ports(proxy, grown, width)
        result = collapse_box(proxy, grown, attachment_tolerance=max(1e-4, width*.1),
                              bundle_certificate=_certify_room_proxy)
        created = next(v for v in result['crossings'] if is_twist_box(v)
                       and v['id'] not in {c['id'] for c in diagram['crossings'] if c['id'] != box_id})
        if created.get('passages'):
            raise ValueError('An unselected strand occupies the extra space.')
    except ValueError as exc:
        raise ValueError('There is not enough clear parallel room at this end. Try the other end, expand less, or straighten the nearby strands. '+str(exc)) from exc
    old_id = created['id']
    created['id'] = box_id
    for edge in result['edges']:
        for key in ('start', 'end'):
            if edge.get(key) is not None and edge[key][0] == old_id:
                edge[key][0] = box_id
    assign_components(result)
    verdict = validate(result)
    if not verdict['valid']:
        raise ValueError('Cannot enlarge this box safely: '+'; '.join(verdict['errors']))
    result['twist_box_room'] = {'box_id': box_id, 'side': side,
                              'original_length': box['size'][1], 'length': minimum_length}
    return result
