"""Pixels-only proposals for discrete twist boxes.

Unlike ordinary labels, a twist coefficient is mathematical input. This module
therefore decodes only signed integers and half integers; it never interprets a
filename, census identity, source metadata, or an arbitrary algebraic expression.
Uncertain coefficients and attachments remain review proposals. No source pixel
is changed, and callers must not build a tangle from a proposal marked for review.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import subprocess
import time
from fractions import Fraction

import cv2
import numpy as np
from PIL import Image, ImageOps

from .text_labels import _tesseract_executable


def parse_twist_value(text):
    """Return an exact count of half twists, or ``None`` for unsupported text.

    Labels count full twists: ``1`` becomes 2 and ``-3/2`` becomes -3.
    Arbitrary expressions, decimals, and fractions outside half integers are not
    guessed. Whitespace and the ordinary Unicode minus are presentation only.
    """
    if not isinstance(text, str):
        return None
    value = re.sub(r'\s+', '', text).replace('−', '-').replace('–', '-')
    if not re.fullmatch(r'[+-]?\d{1,5}(?:/[12])?', value):
        return None
    result = Fraction(value) * 2
    return int(result) if result.denominator == 1 else None


def format_twist_value(half_twists):
    if isinstance(half_twists, bool) or not isinstance(half_twists, (int, np.integer)):
        raise ValueError('A twist coefficient must be an integer number of half twists.')
    value = int(half_twists)
    return str(value // 2) if value % 2 == 0 else f'{value}/2'


def _foreground(rgb):
    from .chalk_foreground import photographic_chalk
    chalk = photographic_chalk(rgb)
    if chalk is not None:
        return chalk[0], True
    level = float(np.median(rgb.mean(axis=2)))
    if level < 115:
        # Board chalk is light or saturated; broad variations of illumination
        # should not become a giant dark rectangle.
        gray = rgb.max(axis=2)
        return gray > min(210., level + 43.), True
    return rgb.min(axis=2) < 200, False


def _ordered_quad(points):
    points = np.asarray(points, dtype=float).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0])
    points = points[np.argsort(angles)]
    # Image-clockwise, starting at the upper-left-like corner. The actual port
    # sides, not the shape's long dimension, determine the twist axis.
    first = int(np.argmin(points.sum(axis=1)))
    return np.roll(points, -first, axis=0)


def _closed_quads(ink):
    contours, hierarchy = cv2.findContours(ink.astype(np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    image_area = ink.size
    proposals = []
    for index, contour in enumerate(contours):
        if hierarchy is None or hierarchy[0, index, 3] < 0:
            continue  # A box interior is a bounded background region.
        area = abs(cv2.contourArea(contour))
        if not 60 <= area <= .65 * image_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, .028 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        quad = _ordered_quad(polygon[:, 0])
        edges = np.roll(quad, -1, axis=0)-quad
        lengths = np.linalg.norm(edges, axis=1)
        if min(lengths) < 9 or max(lengths)/min(lengths) > 18:
            continue
        angles = np.abs(np.sum(edges * np.roll(edges, 1, axis=0), axis=1) /
                        (lengths * np.roll(lengths, 1)))
        if max(angles) > .43:
            continue
        rectangularity = area / max(1., cv2.minAreaRect(contour)[1][0] *
                                    cv2.minAreaRect(contour)[1][1])
        if rectangularity < .83:
            continue
        proposals.append((quad, float(rectangularity)))
    return proposals


def _offset_quad(quad, amount):
    output = []
    for i in range(4):
        a, b, c = quad[(i-1)%4], quad[i], quad[(i+1)%4]
        e1, e2 = b-a, c-b
        n1 = np.array([e1[1], -e1[0]])/np.linalg.norm(e1)
        n2 = np.array([e2[1], -e2[0]])/np.linalg.norm(e2)
        bisector = n1+n2
        output.append(b + amount*bisector/max(.2, float(np.dot(bisector,n1))))
    return np.array(output)


def _quad_candidates(ink):
    proposals = _closed_quads(ink)
    # Broken hand-drawn corners are common. Small isotropic closing of gaps is
    # used only to propose geometry; all text and port evidence uses source ink.
    # Dilation is needed at open right-angle corners where closing alone leaves
    # an opening. Restore its inward contour displacement before sampling ink.
    maximum = max(2, min(12, round(min(ink.shape)*.013)))
    for radius in sorted(set([max(1, maximum//3), maximum])):
        dilated = cv2.dilate(ink.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
        for quad, score in _closed_quads(dilated):
            quad = _offset_quad(quad, radius)
            center = quad.mean(axis=0)
            if any(np.linalg.norm(center-q.mean(axis=0)) < .12*np.linalg.norm(q[0]-q[2])
                   and abs(cv2.contourArea(quad.astype(np.float32))/max(1.,cv2.contourArea(q.astype(np.float32)))-1)<.35
                   for q,_ in proposals):
                continue
            proposals.append((quad, score*.96))
    return proposals


def _rectified(ink, quad):
    edges = np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)
    width, height = max(12, round((edges[0]+edges[2])/2)), max(12, round((edges[1]+edges[3])/2))
    destination = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    crop = cv2.warpPerspective(ink.astype(np.uint8), transform, (width, height),
                               flags=cv2.INTER_NEAREST)
    # Contour points lie against the inner edge of the border. Exclude that
    # border and tiny antialiasing islands; never remove text from source ink.
    margin = max(2, round(min(width, height) * .045))
    crop[:margin] = 0; crop[-margin:] = 0
    crop[:, :margin] = 0; crop[:, -margin:] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    keep = np.zeros(count, bool)
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        # A numeral can legitimately sit close to a frame. Border residue is
        # both near the edge and thin along it; proximity and height alone
        # would erase the whole printed 1 in a tightly fitted coefficient.
        if (area >= max(2, width*height*.00035) and w < .93*width and h < .93*height
                and not ((x <= margin or x+w >= width-margin) and h > 3*w)
                and not ((y <= margin or y+h >= height-margin) and w > 3*h)):
            keep[i] = True
    return keep[labels], transform


def _ocr_one(mask, executable, psm, timeout):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    mask = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
    factor = min(6., max(1., 96 / max(1, mask.shape[0])))
    picture = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).resize(
        (max(1, round(mask.shape[1]*factor)), max(1, round(mask.shape[0]*factor))),
        Image.Resampling.LANCZOS)
    picture = ImageOps.expand(picture, 24, fill=255)
    stream = io.BytesIO(); picture.save(stream, format='PNG')
    try:
        result = subprocess.run([executable, 'stdin', 'stdout', '--psm', str(psm), 'tsv'],
                                input=stream.getvalue(), capture_output=True, check=True,
                                timeout=max(.05, timeout))
        words = [row for row in csv.DictReader(io.StringIO(result.stdout.decode()), delimiter='\t')
                 if row.get('level') == '5' and row.get('text', '').strip()]
        if not words:
            return None
        text = ''.join(row['text'].strip() for row in words)
        confidence = min(float(row['conf']) for row in words)
        half_twists = parse_twist_value(text)
        return {'text': text, 'half_twists': half_twists, 'confidence': confidence/100}
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError, KeyError):
        return None


def _leading_sign(mask):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    pieces = [i for i in range(1,count) if stats[i,4] >= max(3,mask.sum()*.015)]
    if len(pieces)<2:
        return None,mask
    first = min(pieces,key=lambda i:centers[i,0])
    other = [i for i in pieces if i!=first]
    x,y,w,h,area = stats[first]
    minx = min(stats[i,0] for i in other)
    top = min(stats[i,1] for i in other); bottom=max(stats[i,1]+stats[i,3] for i in other)
    if not top+.1*(bottom-top) <= centers[first,1] <= top+.9*(bottom-top):
        return None,mask
    if x+w > minx:
        # A handwritten digit may lean left above/below the minus, overlapping
        # its bounding box while the actual strokes remain cleanly separated.
        # Require the sign to lie to the left on every occupied common row;
        # relaxing bounding-box overlap alone would also admit exponent marks.
        digit_mask = np.isin(labels, other)
        common_rows = [row for row in range(y, y+h)
                       if np.any(labels[row] == first) and np.any(digit_mask[row])]
        if (len(common_rows) < .6*h or
                centers[first,0] > min(centers[i,0] for i in other)-.3*(bottom-top) or
                any(np.flatnonzero(labels[row] == first)[-1]+1 >=
                    np.flatnonzero(digit_mask[row])[0] for row in common_rows)):
            return None,mask
    if not .12*(bottom-top) <= w <= 1.5*(bottom-top):
        return None,mask
    sign=labels[y:y+h,x:x+w]==first
    value=None
    if w>=2.2*h and area/(w*h)>.4:
        value=-1
    elif (.55<=w/max(h,1)<=1.8 and max(sign.sum(0))/h>=.75
          and max(sign.sum(1))/w>=.75 and area/(w*h)<.7):
        value=1
    return (value, mask & (labels!=first)) if value else (None,mask)


def _slash_fraction(mask, executable, deadline):
    """Read separated integer digits around an independently supported slash.

    Whole-word OCR commonly drops a slash and changes ``7/2`` into ``72``.
    Component geometry must first establish a genuine diagonal separator; the
    numerator and denominator then need confident, nonconflicting digit OCR.
    A vertical middle ``1``, a bent digit, and touching glyphs cannot qualify.
    """
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    pieces = [i for i in range(1, count) if stats[i, 4] >= max(3, mask.sum()*.015)]
    if len(pieces) < 3:
        return None
    separators = []
    for index in pieces:
        x, y, w, h, area = stats[index]
        if h < 8 or not .15 <= w/h <= .80:
            continue
        yy, xx = np.where(labels == index)
        covariance = np.cov(np.stack([xx, yy]))
        eigenvalues = np.linalg.eigvalsh(covariance)
        slope = float(covariance[0, 1] / max(1., covariance[1, 1]))
        if eigenvalues[0]/max(1., eigenvalues[1]) > .035 or not -.70 <= slope <= -.15:
            continue
        left = [i for i in pieces if stats[i, 0]+stats[i, 2] <= x]
        right = [i for i in pieces if stats[i, 0] >= x+w]
        if not left or not right or len(left)+len(right)+1 != len(pieces):
            continue
        # A baseline slash spans approximately the same height as the digits.
        # Fractions stacked vertically need a different structural decoder.
        if any(not .65*h <= stats[i, 3] <= 1.5*h or
               abs(centers[i, 1]-centers[index, 1]) > .35*h
               for i in left+right):
            continue
        separators.append((left, right))
    if len(separators) != 1:
        return None
    left, right = separators[0]
    values = []
    evidence = []
    confidence = .95
    for component_ids in (left, right):
        current = []
        plausible_values = set()
        for mode in (7, 13):
            remaining = deadline-time.monotonic()
            if remaining <= .05:
                return None
            reading = _ocr_one(np.isin(labels, component_ids), executable, mode,
                               min(1.5, remaining))
            if reading:
                evidence.append({**reading, 'psm': mode})
                if re.fullmatch(r'\d{1,5}', reading['text']):
                    if reading['confidence'] >= .45:
                        plausible_values.add(int(reading['text']))
                    if reading['confidence'] >= .75:
                        current.append(reading)
        if (not current or len(plausible_values) != 1 or
                (len(current) < 2 and current[0]['confidence'] < .90)):
            return None
        values.append(int(current[0]['text']))
        confidence = min(confidence, max(r['confidence'] for r in current))
    numerator, denominator = values
    if denominator not in (1, 2):
        return None
    text = f'{numerator}/{denominator}'
    return {'text': text, 'half_twists': parse_twist_value(text),
            'confidence': confidence, 'fraction_geometry': True,
            'digit_readings': evidence}


def _signed_one(mask):
    """Confirm the common handwritten +1/-1 using independent stroke geometry."""
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    pieces = [i for i in range(1,count) if stats[i,4] >= max(3, mask.sum()*.04)]
    if len(pieces) != 2:
        return None
    left,right = sorted(pieces, key=lambda i:centers[i,0])
    x,y,w,h,area = stats[right]
    component = labels[y:y+h,x:x+w]==right
    # Allow a serif foot, but require a long nearly vertical stem. This cannot
    # turn arbitrary OCR letters or a numerator into an unverified numeral.
    if h < 4 or w/h > .50 or max(component.sum(axis=0))/h < .80:
        return None
    sx,sy,sw,sh,sa = stats[left]
    sign = labels[sy:sy+sh,sx:sx+sw]==left
    if not .1*h <= sw <= 1.4*h or not .10 <= (centers[left,1]-y)/h <= .85:
        return None
    if sw >= 2.2*sh and sa/(sw*sh) >= .4:
        return -2
    # A plus has both a through-going horizontal and vertical bar and four
    # largely empty corners. Multiplication signs and detached curve corners
    # fail the row/column occupancy checks.
    if (.55 <= sw/max(1,sh) <= 1.8 and max(sign.sum(axis=0))/sh >= .75
            and max(sign.sum(axis=1))/sw >= .75 and sa/(sw*sh)<.7):
        return 2
    return None


def _decode(mask, executable, deadline):
    signed = _signed_one(mask)
    if signed is not None:
        return {'text': format_twist_value(signed), 'half_twists': signed, 'confidence': .92}, {'reason': 'Signed one confirmed by independent sign/stem geometry.', 'readings': []}
    if executable is None:
        return None, {'reason': 'Numeric OCR is unavailable.', 'readings': []}
    readings = []
    # Upright mathematical text is the first hypothesis. Rotate only when the
    # ink is tall; reading both 6 and a rotated 9 is inherently ambiguous.
    ys, xs = np.where(mask)
    rotations = [0, 1, 3]
    if len(xs) and ys.max()-ys.min() > 1.05 * max(1, xs.max()-xs.min()):
        
        number, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        substantial = [i for i in range(1, number) if stats[i, 4] >= max(3, mask.sum()*.03)]
        if len(substantial) >= 2 and np.ptp(centers[substantial,1]) > .7*np.ptp(centers[substantial,0]):
            rotations = [1, 3, 0]
    for rotation in rotations:
        rotated = np.rot90(mask, rotation)
        sign, digits = _leading_sign(rotated)
        fraction = _slash_fraction(digits, executable, deadline)
        if fraction is not None:
            if sign is not None:
                fraction['half_twists'] *= sign
                fraction['text'] = ('-' if sign < 0 else '+')+fraction['text']
                fraction['sign_geometry'] = sign
            readings.append({**fraction, 'rotation': rotation*90})
            return fraction, {'reason': 'Diagonal slash and separately confirmed integer digits.',
                              'readings': readings}
        for mode in (7, 13):
            remaining = deadline-time.monotonic()
            if remaining <= .05:
                break
            reading = _ocr_one(digits, executable, mode, min(1.5, remaining))
            if reading:
                if sign is not None and reading['half_twists'] is not None:
                    if reading['half_twists'] < 0:
                        reading['half_twists'] = None
                    else:
                        reading['half_twists'] *= sign
                        reading['text'] = ('-' if sign < 0 else '+') + reading['text']
                        reading['sign_geometry'] = sign
                readings.append({**reading, 'rotation': rotation*90, 'psm': mode})
        valid = [r for r in readings if r['rotation'] == rotation*90 and r['half_twists'] is not None
                 and r['confidence'] >= .45]
        values = {r['half_twists'] for r in valid}
        if len(values) == 1 and (len(valid) >= 2 or max(r['confidence'] for r in valid) >= .85):
            best = max(valid, key=lambda r:r['confidence'])
            return best, {'reason': 'Constrained integer/half-integer reading.', 'readings': readings}
    return None, {'reason': 'The coefficient needs confirmation.', 'readings': readings}


def _runs(values):
    switches = np.flatnonzero(np.diff(np.pad(values.astype(np.int8), (1, 1))))
    return list(zip(switches[::2], switches[1::2]))


def _ports(ink, quad, *, adaptive=False):
    # Side scans deliberately take place outside the box, avoiding the border
    # connected to all entering strands. Two distances reject a border serif or
    # a one-pixel blemish; they also expose additional strands instead of
    # coercing every box to a two-strand tangle.
    widths = []
    limit = max(8, min(60, float(np.linalg.norm(quad[0]-quad[2]))*.25))
    offsets = np.arange(0., limit, .5)
    for side in range(4):
        a,b = quad[side],quad[(side+1)%4]
        direction = (b-a)/np.linalg.norm(b-a)
        normal = np.array([direction[1],-direction[0]])
        for position in (.2,.4,.6,.8):
            locations = a+(b-a)*position+offsets[:,None]*normal
            pixels = cv2.remap(ink.astype(np.uint8), locations[:,0].astype(np.float32)[None],
                               locations[:,1].astype(np.float32)[None], cv2.INTER_NEAREST).ravel()
            occupied = np.flatnonzero(pixels)
            if len(occupied) and occupied[0] < 5:
                first = occupied[0]
                empty = np.flatnonzero((pixels[first:-2]==0)&(pixels[first+1:-1]==0)&(pixels[first+2:]==0))
                if len(empty):
                    widths.append(float(offsets[first+empty[0]]-offsets[first]))
    stroke = max(1., float(np.median(widths))) if widths else 1.
    ports = []
    side_counts = []
    for side in range(4):
        a, b = quad[side], quad[(side+1)%4]
        edge = b-a; length = float(np.linalg.norm(edge)); direction = edge/length
        normal = np.array([direction[1], -direction[0]])
        extent = max(2*stroke, .08*length)
        samples = np.arange(-extent, length+extent+.5, .5)
        scans = []
        offsets = [max(2., factor*stroke) for factor in ((1.35, 2., 2.8, 4., 6., 8.) if adaptive else (2.8,5.))]
        for offset in offsets:
            locations = a + samples[:, None]*direction + offset*normal
            pixels = cv2.remap(ink.astype(np.uint8), locations[:, 0].astype(np.float32)[None],
                               locations[:, 1].astype(np.float32)[None], cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT).ravel().astype(bool)
            groups = [(float(samples[x:y].mean()), float((y-x)*.5)) for x, y in _runs(pixels)
                      if (y-x)*.5 >= .75 and (y-x)*.5 < max(6*stroke, .23*length)]
            scans.append(groups)
        chosen = None
        for index in range(len(scans)-1):
            near,far=scans[index],scans[index+1]
            if near and len(near)==len(far) and all(
                    abs(p[0]-q[0]) <= max(5*stroke,.08*length)
                    for p,q in zip(near,far)):
                if chosen is None or len(near)>len(scans[chosen]):
                    chosen=index
        if chosen is None:
            scans=[[],[]];near_offset,far_offset=1.,2.
        else:
            near_offset,far_offset=offsets[chosen],offsets[chosen+1]
            scans=[scans[chosen],scans[chosen+1]]
        current = []
        for position, width in scans[0]:
            if not -.035*length <= position <= 1.035*length:
                continue
            other = [v for v in scans[1] if abs(v[0]-position) <= max(5*stroke, .08*length)]
            if not other:
                continue
            far = min(other, key=lambda v:abs(v[0]-position))
            # Extrapolate to the box edge, rather than snapping to the line at
            # an arbitrary offset outside it.
            edge_position = position - (far[0]-position)*near_offset/max(.5,far_offset-near_offset)
            # A nearby curl may intersect both outward scan lines without
            # entering this edge. Extrapolation must actually reach the edge;
            # clipping such a curve to a corner manufactures an extra port.
            if not -.035*length <= edge_position <= 1.035*length:
                continue
            edge_position = float(np.clip(edge_position, 0, length))
            current.append({'point': [float(v) for v in a+edge_position*direction],
                            'side': side, 'position': edge_position/length,
                            'stroke_width': float((width+far[1])/2), 'confidence': .9})
        current.sort(key=lambda p:p['position'])
        ports.extend(current); side_counts.append(len(current))
    pairs = [(a, (a+2)%4) for a in (0, 1) if side_counts[a] >= 2
             and side_counts[a] == side_counts[(a+2)%4]
             and side_counts[(a+1)%4] == side_counts[(a+3)%4] == 0]
    sides = list(pairs[0]) if len(pairs) == 1 else None
    if sides is None and not adaptive:
        alternative = _ports(ink, quad, adaptive=True)
        if alternative[1] is not None or len(alternative[0]) > len(ports):
            return alternative
    return ports, sides, side_counts, stroke


def _canonical_descriptor(record):
    """Add the discrete model's geometric frame and CCW port indexing."""
    if record['port_sides'] is None:
        return record
    quad=np.array(record['outline'],float)
    side, opposite=record['port_sides']
    entry=(quad[side]+quad[(side+1)%4])/2
    exit=(quad[opposite]+quad[(opposite+1)%4])/2
    axis=exit-entry; length=float(np.linalg.norm(axis)); axis/=length
    right=np.array([axis[1],-axis[0]])
    width=float((np.linalg.norm(quad[(side+1)%4]-quad[side])+
                 np.linalg.norm(quad[(opposite+1)%4]-quad[opposite]))/2)
    top=sorted([p['point'] for p in record['ports'] if p['side']==side],
               key=lambda p:float(np.dot(p,right)))
    bottom=sorted([p['point'] for p in record['ports'] if p['side']==opposite],
                  key=lambda p:float(np.dot(p,right)))
    count=len(top)
    record.update(point=quad.mean(axis=0).tolist(), axis=axis.tolist(), size=[width,length],
                  port_points=[top[0],*bottom,*reversed(top[1:])],
                  bundle_ports={'top':[0,*range(2*count-1,count,-1)],
                                'bottom':list(range(1,count+1))})
    return record


def inspect_twist_box(image, outline, label=None, *, use_ocr=True, max_ocr_seconds=4.):
    """Inspect a user-selected quadrilateral, including missed automatic boxes.

    Returns one candidate, with the same fields as ``detect_twist_boxes`` records.
    ``label`` is optional explicit user input, in full twists; unsupported text
    raises ValueError. The returned status remains review if ports are ambiguous.
    Geometry uses image coordinates, ``size=[transverse_width, axial_length]``,
    and ``axis`` points from the entry to the exit side. ``port_points`` and
    ``bundle_ports`` directly describe the model's CCW boundary port indexing.
    The caller still confirms the region before discarding its raster interior.
    """
    rgb=np.asarray(image)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or not rgb.size:
        raise ValueError('Twist-box review requires a nonempty RGB uint8 image.')
    supplied=np.asarray(outline,float)
    if supplied.shape!=(4,2) or not np.isfinite(supplied).all():
        raise ValueError('Choose four finite corners of the twist box.')
    quad=_ordered_quad(supplied)
    if not cv2.isContourConvex(quad.astype(np.float32)) or cv2.contourArea(quad.astype(np.float32))<9:
        raise ValueError('The twist box must be a nondegenerate convex quadrilateral.')
    if max_ocr_seconds<0:
        raise ValueError('OCR seconds cannot be negative.')
    ink,dark=_foreground(rgb)
    mask,_=_rectified(ink,quad)
    ports,sides,counts,stroke=_ports(ink,quad)
    if label is not None:
        value=parse_twist_value(label)
        if value is None:
            raise ValueError('Enter a signed integer or half integer, for example -2 or 3/2.')
        numeric={'reason':'Coefficient supplied explicitly by the user.','readings':[]}
        confidence=1.
    else:
        reading,numeric=_decode(mask,_tesseract_executable() if use_ocr else None,
                                time.monotonic()+max_ocr_seconds)
        value=reading['half_twists'] if reading else None
        confidence=reading['confidence'] if reading else .45
    record={'id':'twist-box-1','outline':quad.tolist(),
            'bbox':[float(quad[:,0].min()),float(quad[:,1].min()),float(quad[:,0].max()),float(quad[:,1].max())],
            'numeric_label':format_twist_value(value) if value is not None else None,
            'half_twists':value,'ports':ports,'port_sides':sides,
            'strand_count':counts[sides[0]] if sides else None,'confidence':confidence,
            'status':'ready' if sides and value is not None else 'review',
            'numeric_source':'user' if label is not None else 'pixels',
            'source_sha256':hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
            'image_size':[rgb.shape[1],rgb.shape[0]],
            'diagnostics':{'side_counts':counts,'stroke_width':stroke,'numeric':numeric,
                           'attachment_reason':'Paired opposite-side strand ports.' if sides
                           else 'Strand attachments need confirmation.','dark_background':dark}}
    return _canonical_descriptor(record)


def _suppress_clipped_candidates(records):
    """Discard a clipped unreadable sub-frame of an independently ready box."""
    # Dilation can join a numeral to a border and make a partial rectangle.
    # Readable nested candidates and conflicting port axes remain reviewable.
    def clipped_duplicate(candidate):
        if candidate['half_twists'] is not None:
            return False
        quad = np.asarray(candidate['outline'], np.float32)
        area = abs(cv2.contourArea(quad))
        for other in records:
            if other is candidate or other['status'] != 'ready':
                continue
            if candidate['port_sides'] != other['port_sides']:
                continue
            outer = np.asarray(other['outline'], np.float32)
            outer_area = abs(cv2.contourArea(outer))
            overlap, _ = cv2.intersectConvexConvex(quad, outer)
            if .30*outer_area <= area < .92*outer_area and overlap >= .97*area:
                return True
        return False
    return [record for record in records if not clipped_duplicate(record)]


def detect_twist_boxes(image, *, use_ocr=True, max_boxes=64, max_ocr_seconds=12.):
    """Return non-destructive JSON review records from an RGB uint8 image.

    ``outline`` lists four corners clockwise in image coordinates. Side ``i``
    goes from corner i to i+1. Accepted attachments occupy two opposite sides
    with equal numbers of ports; ``strand_count`` may exceed two. Port positions
    are measured along their side. A record is ``ready`` only when its coefficient
    and these attachment checks succeed; all other records require review.
    """
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError('Twist-box review requires a nonempty RGB uint8 image.')
    if max_boxes < 1 or max_ocr_seconds < 0:
        raise ValueError('Twist-box limits must be positive (OCR seconds may be zero).')
    ink, dark = _foreground(rgb)
    executable = _tesseract_executable() if use_ocr else None
    deadline = time.monotonic()+max_ocr_seconds
    records = []
    from .box_frames import open_frame_candidates
    closed = _quad_candidates(ink)
    candidates = [(quad, score, 'closed_interior') for quad, score in closed]
    for quad, score in open_frame_candidates(ink):
        area = abs(cv2.contourArea(quad.astype(np.float32)))
        # Preserve existing closed-frame geometry and its measured ports.
        # An adjoining strand can supply a longer fourth side of an otherwise
        # identical rectangle. Compare overlap, not only center proximity.
        duplicate = False
        for q, _ in closed:
            intersection, _ = cv2.intersectConvexConvex(quad.astype(np.float32), q.astype(np.float32))
            union = area+abs(cv2.contourArea(q.astype(np.float32)))-intersection
            if intersection/max(1., union) > .6:
                duplicate = True
                break
        if duplicate:
            continue
        candidates.append((quad, score, 'four_borders'))
    for quad, rectangularity, frame_source in sorted(candidates, key=lambda q: (q[0][:,1].min(), q[0][:,0].min())):
        mask, transform = _rectified(ink, quad)
        if mask.sum() < 4:
            continue
        text_y,text_x=np.where(mask)
        # A coefficient sits within the box. Tiny residual ink hugging one
        # boundary usually belongs to an adjoining strand or damaged frame,
        # not to a label inside a rectangular complementary region.
        if not (.12 <= text_x.mean()/mask.shape[1] <= .88 and
                .12 <= text_y.mean()/mask.shape[0] <= .88):
            continue
        if frame_source == 'four_borders':
            # With no closed interior, a nearly filled panel, frame residue,
            # or one straight crossing remnant is not coefficient evidence.
            # In particular, do not interpret a lone underpass stub as "1".
            spread = np.linalg.eigvalsh(np.cov(np.stack([text_x, text_y])))
            if (mask.mean() > .55 or
                    (np.ptp(text_x)+1)/mask.shape[1] > .88 or
                    (np.ptp(text_y)+1)/mask.shape[0] > .88 or
                    spread[0]/max(1., spread[1]) < .025):
                continue
        ports, sides, counts, stroke = _ports(ink, quad)
        # A text rectangle without visible attachments is not interpreted as
        # a twist box. Partial pairs remain reviewable, but isolated text frames
        # and unrelated rectangular diagram regions do not flood the overlay.
        if sum(counts) < 3 or max(counts) < 2:
            continue
        # An open outline is weaker evidence than a bounded interior: require
        # both complete bundles, even when OCR finds a plausible numeral.
        if frame_source == 'four_borders' and sides is None:
            continue
        reading, numeric = _decode(mask, executable, deadline)
        value = reading['half_twists'] if reading else None
        if value is None and sides is None:
            continue
        center = quad.mean(axis=0)
        if any(np.linalg.norm(center-np.array(r['outline']).mean(axis=0)) < 2*stroke for r in records):
            continue
        records.append({'id': f'twist-box-{len(records)+1}',
                        'outline': quad.tolist(),
                        'bbox': [float(quad[:,0].min()), float(quad[:,1].min()),
                                 float(quad[:,0].max()), float(quad[:,1].max())],
                        'numeric_label': format_twist_value(value) if value is not None else None,
                        'half_twists': value, 'ports': ports, 'port_sides': sides,
                        'strand_count': counts[sides[0]] if sides else None,
                        'confidence': float(min(rectangularity, reading['confidence'] if reading else .45)),
                        'status': 'ready' if sides and value is not None else 'review',
                        'diagnostics': {'side_counts': counts, 'rectangularity': rectangularity,
                                        'frame_source': frame_source,
                                        'stroke_width': stroke, 'numeric': numeric,
                                        'attachment_reason': 'Paired opposite-side strand ports.' if sides
                                        else 'Strand attachments need confirmation.'}})
        if len(records) >= max_boxes:
            break
    records = [_canonical_descriptor(record) for record in _suppress_clipped_candidates(records)]
    for index, record in enumerate(records, 1):
        record['id'] = f'twist-box-{index}'
    return {'boxes': records, 'source_sha256': hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
            'image_size': [rgb.shape[1], rgb.shape[0]],
            'diagnostics': {'dark_background': dark, 'ocr_available': bool(executable),
                            'ocr_budget_exhausted': time.monotonic() >= deadline,
                            'candidate_limit_reached': len(records) >= max_boxes}}
