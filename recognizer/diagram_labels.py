"""Conservative local OCR for detached labels on printed link diagrams.

This fallback proposes whole connected glyphs, never OCR rectangles.  Simple
closed components and ordinary short arcs are protected before OCR is called.
Its output is only a cleaned copy; labels never supply knot identities.
"""
from __future__ import annotations

import csv
import io
import re
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.morphology import skeletonize

from .knotfolio_backend import _adjacency, _prune_spurs
from .text_labels import _tesseract_executable
from .desktop_platform import hidden_subprocess_options


def _ocr_label(crop):
    executable = _tesseract_executable()
    if executable is None:
        return None
    original = Image.fromarray(crop)
    scales = dict.fromkeys(max(2, min(6, round(size / original.height))) for size in (120, 260))
    for scale in scales:
        picture = ImageOps.expand(original.resize(
            (original.width * scale, original.height * scale), Image.Resampling.LANCZOS),
            24, fill='white')
        stream = io.BytesIO()
        picture.save(stream, format='PNG')
        for mode in ('7', '13'):
            try:
                result = subprocess.run([executable, 'stdin', 'stdout', '--psm', mode, 'tsv'],
                                        input=stream.getvalue(), capture_output=True,
                                        check=True, timeout=2, **hidden_subprocess_options())
                words = [r for r in csv.DictReader(io.StringIO(result.stdout.decode()), delimiter='\t')
                         if r.get('level') == '5' and r.get('text', '').strip()]
                if len(words) != 1:
                    continue
                text = words[0]['text'].strip()
                confidence = float(words[0]['conf'])
            except (OSError, subprocess.SubprocessError, UnicodeError, ValueError, KeyError):
                return None
            # No whitelist is sent to OCR: it must identify text independently.
            if confidence >= 80 and re.fullmatch(r'[A-Za-zα-ωΑ-Ω][A-Za-zα-ωΑ-Ω0-9_()]*|\([A-Za-z]\)', text):
                return {'text': text, 'confidence': confidence}
    return None


def _glyph_geometry(component):
    skeleton = skeletonize(component)
    graph = _adjacency(skeleton)
    # Tight connected-component crops touch the array boundary. Without an
    # explicit background frame, a vertical glyph can appear many times wider
    # than its actual stroke and its serif/subscript gets pruned away.
    distance = cv2.distanceTransform(np.pad(component.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    widths = 2 * distance[skeleton]
    stroke_width = float(np.median(widths)) if len(widths) else 1.
    _prune_spurs(graph, max(2., stroke_width * .85))
    ends = sum(len(v) == 1 for v in graph.values())
    branches = sum(len(v) > 2 for v in graph.values())
    variation = float(np.quantile(widths, .9)/max(1., np.quantile(widths, .1))) if len(widths) else 1.
    return {'ends': ends, 'branches': branches, 'stroke_width': stroke_width, 'width_variation': variation,
            'closed': bool(graph) and all(len(v) == 2 for v in graph.values())}


def remove_detached_labels(rgb, *, max_ocr_calls=24):
    """Return ``(cleaned_rgb, diagnostics)``; unchanged input if evidence is weak.

    Small branched glyphs seed horizontally aligned groups.  Unbranched ink is
    included only as a paired parenthesis or a small subscript of a branched
    glyph.  Every group must be detached from non-label ink by at least a stroke
    width.  OCR confirmation is required, and only connected glyph pixels and
    their immediately adjacent antialiasing are removed.

    The bound applies before launching OCR processes.  This deliberately leaves
    ambiguous labels, touching text, dashed strands, and circular components.
    """
    output = rgb.copy()
    diagnostics = {'removed_labels': [], 'ocr_available': _tesseract_executable() is not None,
                   'ocr_calls': 0}
    if not diagnostics['ocr_available']:
        return output, diagnostics
    ink = rgb.min(axis=2) < 205
    height, width = ink.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    candidates = {}
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if (area < 20 or h < 9 or h > .19 * height or w > .18 * width
                or not .12 <= w / h <= 2.8):
            continue
        shape = _glyph_geometry(labels[y:y+h, x:x+w] == index)
        if shape['closed']:
            continue
        candidates[index] = {'bbox': (x, y, x+w, y+h), **shape}
    if not candidates:
        return output, diagnostics
    used = set()
    considered = set()
    for seed, shape in candidates.items():
        if seed in used or not shape['branches']:
            continue
        x1, y1, x2, y2 = shape['bbox']
        h = y2-y1
        group = {seed}
        # Include only an immediately adjacent text-sized group.  The geometry
        # check below decides whether unbranched companions are punctuation.
        for index, other in candidates.items():
            if index == seed or index in used:
                continue
            a, b, c, d = other['bbox']
            oh = d-b
            if (.3 <= oh/h <= 2.4 and max(x1-c, a-x2, 0) <= .45*max(h, oh)
                    and max(y1-b, b-y1) <= .8*max(h, oh)):
                group.add(index)
        # Expand once for the last parenthesis in a compact expression B(K).
        for index, other in candidates.items():
            if index in group or index in used:
                continue
            a, b, c, d = other['bbox']
            if not .55 <= (d-b)/h <= 1.5:
                continue
            if any(max(candidates[j]['bbox'][0]-c, a-candidates[j]['bbox'][2], 0) <= .20*h
                   and abs(b-candidates[j]['bbox'][1]) <= .4*h for j in group):
                group.add(index)
        key = tuple(sorted(group))
        if key in considered:
            continue
        considered.add(key)
        boxes = [candidates[i]['bbox'] for i in group]
        gx1, gy1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        gx2, gy2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        if gx2-gx1 > 4.5*(gy2-gy1) or gy2-gy1 > .21*height:
            continue
        plain = [i for i in group if not candidates[i]['branches']]
        parentheses = []
        for index in plain:
            a, b, c, d = candidates[index]['bbox']
            # Only tall narrow companions can be parentheses; short unbranched
            # curves next to text remain protected actual strands.
            if (c-a)/(d-b) < .4 and d-b >= .9*h:
                parentheses.append(index)
            elif not (d-b <= .70*h and b >= y1+.25*h and a >= x2):
                break
        else:
            if len(parentheses) > 2:
                continue
            glyph_mask = np.isin(labels, list(group))
            # Separation is measured from all substantive non-group components.
            substantive = stats[:, cv2.CC_STAT_AREA] >= 12
            substantive[0] = False
            surrounding = substantive[labels] & ~glyph_mask
            distance = cv2.distanceTransform((~surrounding).astype(np.uint8), cv2.DIST_L2, 5)
            if float(distance[glyph_mask].min()) < max(2., shape['stroke_width']):
                continue
            if diagnostics['ocr_calls'] >= max_ocr_calls:
                break
            crop = np.full((gy2-gy1, gx2-gx1, 3), 255, dtype=np.uint8)
            local = glyph_mask[gy1:gy2, gx1:gx2]
            crop[local] = rgb[gy1:gy2, gx1:gx2][local]
            diagnostics['ocr_calls'] += 1
            reading = _ocr_label(crop)
            if reading is None:
                continue
            if parentheses and not ('(' in reading['text'] and ')' in reading['text']):
                continue
            # Recover antialiasing without a rectangular erase, and never take
            # pixels belonging to a substantive outside component.
            fringe = cv2.dilate(glyph_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            erase = glyph_mask | (fringe & (rgb.min(axis=2) < 250) & ~surrounding)
            output[erase] = 255
            diagnostics['removed_labels'].append({**reading, 'bbox': [gx1, gy1, gx2, gy2],
                'removed_pixels': int(erase.sum()), 'component_count': len(group),
                'reason': 'detached glyph geometry; local OCR confirmed'})
            used.update(group)
    return output, diagnostics
