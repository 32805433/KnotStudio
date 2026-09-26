"""Local box-side resizing by an injective piecewise-affine plane deformation.

The displacement is along one box axis only. It is affine inside the selected
box and fades to zero in a finite collar. Every triangle has positive slope
along that axis. Splitting edges at all cell/triangle boundaries maps the
polylines exactly, preserving crossings without expanding implicit twists.
"""
import math

import cv2
import numpy as np

from .box_operations import _box, _finish
from .diagram import copy_diagram
from .twist_boxes import box_corners, box_port_points, is_twist_box

# Corner order is top-left, bottom-left, bottom-right, top-right.
SIDES = ((0, -1), (1, 1), (0, 1), (1, -1))


def box_side_hit(diagram, point, tolerance):
    """Nearest side in diagram units; works independently of viewport zoom."""
    point = np.asarray(point, float)
    best = None
    for box in diagram.get('crossings', []):
        if not is_twist_box(box):
            continue
        corners = np.asarray(box_corners(box), float)
        for side, (a, b) in enumerate(zip(corners, np.roll(corners, -1, axis=0))):
            delta = b-a
            t = float(np.clip((point-a)@delta/(delta@delta), 0., 1.))
            anchor = a+t*delta
            distance = float(np.linalg.norm(point-anchor))
            if distance <= tolerance and (best is None or distance < best[0]):
                best = (distance, box['id'], side, anchor.tolist(), a.tolist(), b.tolist())
    return best


class _ResizeMap:
    def __init__(self, box, side, displacement, radius):
        dimension, sign = SIDES[side]
        self.dimension, self.sign = dimension, sign
        self.center = np.asarray(box['point'], float)
        axis = np.asarray(box['axis'], float)
        frame = [np.array([axis[1], -axis[0]]), axis]
        self.direction, self.transverse = frame[dimension], frame[1-dimension]
        self.frame = np.column_stack([self.direction, self.transverse])
        length, across = box['size'][dimension], box['size'][1-dimension]
        collar = max(24., radius, 2*max(0., sign*displacement))
        margin = max(24., min(radius, across*.5))
        # The opposite side is fixed. Uniform affine scaling inside the box
        # retains all stored port and passage geometry, including crossings.
        if sign > 0:
            self.s = np.r_[-length/2, np.linspace(length/2, length/2+collar, 9)]
            t = np.linspace(0., 1., 9)
            profile = np.r_[0., 1-(3*t*t-2*t*t*t)]
        else:
            self.s = np.r_[np.linspace(-length/2-collar, -length/2, 9), length/2]
            t = np.linspace(0., 1., 9)
            profile = np.r_[3*t*t-2*t*t*t, 0.]
        t = np.linspace(0., 1., 5)
        self.t = np.r_[np.linspace(-across/2-margin, -across/2, 5),
                       np.linspace(across/2, across/2+margin, 5)]
        fade = np.r_[3*t*t-2*t*t*t, 1-(3*t*t-2*t*t*t)]
        self.values = displacement*profile[:, None]*fade[None, :]
        slopes = 1+np.diff(self.values, axis=0)/np.diff(self.s)[:, None]
        if float(slopes.min()) <= 1e-6:
            raise ValueError('The requested resize would fold the surrounding drawing.')
        self.low = np.array([self.s[0], self.t[0]])
        self.high = np.array([self.s[-1], self.t[-1]])

    def local(self, points):
        return (np.asarray(points, float)-self.center)@self.frame

    def shift(self, points):
        xy = self.local(points)
        inside = np.all(xy >= self.low, axis=1) & np.all(xy <= self.high, axis=1)
        output = np.zeros(len(xy))
        x, y = xy[inside].T
        i = np.clip(np.searchsorted(self.s, x, side='right')-1, 0, len(self.s)-2)
        j = np.clip(np.searchsorted(self.t, y, side='right')-1, 0, len(self.t)-2)
        u = (x-self.s[i])/(self.s[i+1]-self.s[i])
        v = (y-self.t[j])/(self.t[j+1]-self.t[j])
        a, b, c, d = (self.values[i, j], self.values[i+1, j],
                       self.values[i, j+1], self.values[i+1, j+1])
        output[inside] = np.where(u >= v, (1-u)*a+(u-v)*b+v*d,
                                 (1-v)*a+(v-u)*c+u*d)
        return output

    def points(self, points):
        points = np.asarray(points, float)
        return (points+self.shift(points)[:, None]*self.direction).tolist()

    def path(self, points):
        """Exact image of a polyline; only segments meeting the collar split."""
        local = self.local(points)
        indexes = np.flatnonzero(np.all(np.maximum(local[:-1], local[1:]) >= self.low, axis=1)
                                & np.all(np.minimum(local[:-1], local[1:]) <= self.high, axis=1))
        if not len(indexes):
            return points
        pieces, last = [], 0
        for index in indexes:
            pieces.extend(points[last:index])
            a, b = local[index:index+2]
            delta = b-a
            cuts = [0., 1.]
            for k, grid in enumerate((self.s, self.t)):
                if abs(delta[k]) > 1e-12:
                    cuts.extend(float(t) for t in (grid-a[k])/delta[k] if 1e-9 < t < 1-1e-9)
            cuts = sorted(set(cuts))
            diagonal = []
            for left, right in zip(cuts, cuts[1:]):
                mid = a+(left+right)*.5*delta
                i, j = np.searchsorted(self.s, mid[0])-1, np.searchsorted(self.t, mid[1])-1
                if not (0 <= i < len(self.s)-1 and 0 <= j < len(self.t)-1):
                    continue
                size = np.array([self.s[i+1]-self.s[i], self.t[j+1]-self.t[j]])
                uv = (a-[self.s[i], self.t[j]])/size
                speed = delta/size
                denominator = speed[0]-speed[1]
                if abs(denominator) > 1e-12:
                    t = (uv[1]-uv[0])/denominator
                    if left+1e-9 < t < right-1e-9:
                        diagonal.append(float(t))
            cuts = sorted(set(cuts+diagonal))[:-1]
            pa, pb = np.asarray(points[index]), np.asarray(points[index+1])
            pieces.extend((pa+fraction*(pb-pa)).tolist() for fraction in cuts)
            last = index+1
        pieces.extend(points[last:])
        return self.points(pieces)


class BoxResize:
    """Gesture engine compatible with the editor's cancellable drag worker."""
    def __init__(self, diagram, box_id, side, anchor, radius=65.):
        if side not in range(4):
            raise ValueError('Choose a side of the twist box.')
        self.original = copy_diagram(diagram)
        self.box = _box(self.original, box_id)
        self.box_id, self.side = box_id, side
        self.anchor = np.asarray(anchor, float)
        self.radius = max(12., float(radius))
        self.last_moves = []
        self.last_diagram = self.original

    def move(self, target, under=None):
        from .render import line_width
        dimension, sign = SIDES[self.side]
        axis = np.asarray(self.box['axis'], float)
        direction = np.array([axis[1], -axis[0]]) if dimension == 0 else axis
        displacement = float((np.asarray(target, float)-self.anchor)@direction)
        if not math.isfinite(displacement):
            raise ValueError('Choose a finite box size.')
        new_length = self.box['size'][dimension]+sign*displacement
        if new_length < max(12., 4*line_width(self.original)):
            raise ValueError('The box is too thin to display clearly; keep its opposite sides apart.')
        if abs(displacement) < 1e-9:
            self.last_diagram = copy_diagram(self.original)
            return self.last_diagram
        mapping = _ResizeMap(self.box, self.side, displacement, self.radius)
        # Other compact boxes must remain rectangles with their own fixed
        # tangles. Do not deform one inadvertently through a partial collar.
        for other in self.original['crossings']:
            if other['id'] == self.box_id or not is_twist_box(other):
                continue
            polygon = mapping.local(box_corners(other)).astype(np.float32)
            low, high = mapping.low, mapping.high
            region = np.array([[low[0],low[1]], [high[0],low[1]],
                               [high[0],high[1]], [low[0],high[1]]], np.float32)
            area, _ = cv2.intersectConvexConvex(polygon, region)
            if area > 1e-5:
                raise ValueError('Another twist box is too close to this side. Try a smaller resize or the opposite side.')
        result = copy_diagram(self.original)
        for vertex in result['crossings']:
            if vertex['id'] == self.box_id:
                ports = box_port_points(vertex)
                vertex['point'] = (np.asarray(vertex['point'])+displacement*.5*direction).tolist()
                vertex['size'] = list(vertex['size'])
                vertex['size'][dimension] = new_length
                vertex['port_points'] = mapping.points(ports)
                for passage in vertex.get('passages', []):
                    passage['points'] = mapping.points(passage['points'])
                for crossing in vertex.get('passage_crossings', []):
                    crossing['point'] = mapping.points([crossing['point']])[0]
            elif not is_twist_box(vertex):
                vertex['point'] = mapping.points([vertex['point']])[0]
        for edge in result['edges']:
            edge['points'] = mapping.path(edge['points'])
        self.last_diagram = _finish(result)
        return self.last_diagram
