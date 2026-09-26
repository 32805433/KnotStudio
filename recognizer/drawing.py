"""Raster pencil gestures with explicit over/under clearance.

Compose from a fixed pre-gesture image so subsequent mouse events cannot fill
an underpass back in. Shift state is recorded for each incoming segment.
"""
import math

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw


def dark_background(image):
    """Cheap polarity hint for editing even before the first recognition."""
    sample = image.resize((min(128, image.width), min(128, image.height))).convert('RGB')
    return float(np.median(np.asarray(sample).mean(axis=2))) < 115.


def _ink_mask(rgb, dark):
    if not dark:
        return rgb.min(axis=2) < 235
    sample = rgb[::max(1, rgb.shape[0]//128), ::max(1, rgb.shape[1]//128)]
    background = float(np.median(sample.mean(axis=2)))
    return rgb.max(axis=2) > min(210., background+45.)


def erase_connected_ink(image, point, *, dark=None):
    """Erase one 8-connected raster ink region; crossing gaps remain gaps.

    Return (new_image, pixel_count), or (None, 0) for background/outside clicks.
    A one-pixel antialias fringe is erased without absorbing other ink regions.
    The source and its in-memory rendering metadata are preserved.
    """
    x, y = (math.floor(v) for v in point)
    if not (0 <= x < image.width and 0 <= y < image.height):
        return None, 0
    rgb = np.asarray(image.convert('RGB'))
    sample = rgb[::max(1, image.height//128), ::max(1, image.width//128)]
    level = float(np.median(sample.mean(axis=2)))
    dark = dark_background(image) if dark is None else dark
    light = rgb.max(axis=2) if dark else rgb.min(axis=2)
    ink = _ink_mask(rgb, dark)
    if not ink[y, x]:
        return None, 0
    filled = ink.astype(np.uint8)
    cv2.floodFill(filled, None, (x, y), 2, flags=8)
    selected = filled == 2
    around = cv2.dilate(selected.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    weak = light > level+12. if dark else light < 254
    erase = selected | (around & weak & ~ink)
    # Estimate the nearby paper/board color, leaving every unrelated pixel intact.
    ring = cv2.dilate(around.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~around & ~weak
    background = rgb[ring]
    if not len(background):
        background = sample.reshape(-1, 3)
    color = (0, 0, 0) if dark else tuple(np.median(background, axis=0).astype(np.uint8).tolist())
    result = image.convert('RGB').copy()
    result.info = image.info.copy()
    result.paste(color, (0, 0), Image.fromarray(erase.astype(np.uint8)*255))
    return result, int(erase.sum())


def _line(mask, points, width, fill=255):
    draw = ImageDraw.Draw(mask)
    draw.line(points, fill=fill, width=width)
    radius = (width - 1) / 2
    for x, y in (points[0], points[-1]):
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill)


class PencilGesture:
    def __init__(self, image, width, *, dark=None):
        self.base = image.convert('RGB').copy()
        self.dark = dark_background(image) if dark is None else dark
        self.foreground = 'white' if self.dark else '#111111'
        self.background = 'black' if self.dark else 'white'
        self.width = max(1, round(width))
        self.clearance = max(2, round(self.width * .65))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (2*self.clearance+1, 2*self.clearance+1))
        self.ink = _ink_mask(np.asarray(self.base), self.dark)
        ink = self.ink.astype(np.uint8)*255
        self.blocked = Image.fromarray(cv2.dilate(ink, self.kernel))
        self.over = Image.new('L', image.size)
        self.under = Image.new('L', image.size)
        self.points = []
        self.self_crossings = []

    def add(self, a, b, under=False):
        a, b = tuple(a), tuple(b)
        _line(self.under if under else self.over, [a,b], self.width)
        if not self.points:
            self.points.append(a)
        if math.dist(a,b) > 1e-6:
            # Detect transverse crossings of earlier parts of this gesture.
            # Adjacent segments share an endpoint and are ordinary continuation.
            v = np.subtract(b,a)
            for i, (c,d) in enumerate(zip(self.points[:-2],self.points[1:-1])):
                w = np.subtract(d,c)
                determinant = v[0]*w[1]-v[1]*w[0]
                if abs(determinant) < 1e-8:
                    continue
                delta = np.subtract(c,a)
                t = (delta[0]*w[1]-delta[1]*w[0])/determinant
                u = (delta[0]*v[1]-delta[1]*v[0])/determinant
                if 1e-5 < t <= 1 and 0 <= u <= 1:
                    point = np.asarray(a)+t*v
                    if i == 0 and math.dist(point,self.points[0]) < self.width:
                        continue  # Closing the stroke joins its initial endpoint.
                    if any(math.dist(point,r['point']) < self.width for r in self.self_crossings):
                        continue
                    sine = abs(determinant) / (np.linalg.norm(v)*np.linalg.norm(w))
                    self.self_crossings.append({'point':point.tolist(),
                        'over_segment':i if under else len(self.points)-1,
                        'under_segment':len(self.points)-1 if under else i,
                        'sine_angle':float(sine)})
            self.points.append(b)
        return self.render()

    def render(self):
        halo = Image.fromarray(cv2.dilate(np.asarray(self.over), self.kernel))
        visible_under = ImageChops.subtract(self.under, self.blocked)
        # Starting/ending on an existing endpoint should join it. Limit this
        # exception to endpoint neighborhoods; the rest of the stroke has gaps.
        if self.points:
            for point in (self.points[0], self.points[-1]):
                x,y=map(round,point)
                if not (0 <= x < self.base.width and 0 <= y < self.base.height):
                    continue
                if not self.ink[y, x]:
                    continue
                r=self.width+self.clearance
                join=Image.new('L',self.base.size)
                ImageDraw.Draw(join).ellipse((x-r,y-r,x+r,y+r),fill=255)
                halo=ImageChops.subtract(halo,join)
                visible_under=ImageChops.lighter(visible_under,ImageChops.darker(self.under,join))
        image=self.base.copy()
        image.paste(self.background,(0,0),halo)
        visible=ImageChops.lighter(self.over,visible_under)
        image.paste(self.foreground,(0,0),visible)
        for crossing in self.self_crossings:
            x,y=crossing['point']
            # A fixed circular hole leaves acute-angle strands touching: their
            # separation at radius r is only r*sin(angle). Allow a full stroke
            # width plus white clearance before restoring the overpass.
            r=min(math.hypot(*self.base.size),
                  (self.width+self.clearance)/crossing['sine_angle'])
            disk=Image.new('L',self.base.size)
            ImageDraw.Draw(disk).ellipse((x-r,y-r,x+r,y+r),fill=255)
            over=self._branch_mask(crossing['over_segment'],crossing['under_segment'],r)
            under=self._branch_mask(crossing['under_segment'],crossing['over_segment'],r)
            # Erase only the under-strand pixels close to the overpass. Clearing
            # the entire angle-dependent disk would damage nearby unrelated arcs.
            halo=Image.fromarray(cv2.dilate(np.asarray(over),self.kernel))
            erase=ImageChops.darker(disk,ImageChops.darker(under,halo))
            image.paste(self.background,(0,0),erase)
            image.paste(self.foreground,(0,0),ImageChops.darker(over,erase))
        return image

    def _branch_mask(self,index,other_index,radius):
        lo,hi=index,index+1
        length=0
        while lo>0 and lo-1!=other_index and length<2*radius:
            length+=math.dist(self.points[lo],self.points[lo-1]);lo-=1
        length=0
        while hi<len(self.points)-1 and hi!=other_index and length<2*radius:
            length+=math.dist(self.points[hi],self.points[hi+1]);hi+=1
        mask=Image.new('L',self.base.size)
        # Match the rounded caps added by each original mouse segment, including
        # interior vertices on a curved hand-drawn branch.
        for a,b in zip(self.points[lo:hi],self.points[lo+1:hi+1]):
            _line(mask,[a,b],self.width)
        return mask
