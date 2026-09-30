"""Conservative OCR removal of detached numeric subscript labels.

Only a failed reconstruction invokes this fallback. Geometry proposes a small
exterior pair of glyphs; local Tesseract OCR must then confirm numeric text.
Removal uses entire connected components, never a rectangle that could cut a
strand. Label text is diagnostic only and never supplies a knot identity or PD.
"""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.morphology import skeletonize

from .desktop_platform import hidden_subprocess_options


def _tesseract_executable():
    from .resources import bundled_tesseract
    bundled = bundled_tesseract()
    if bundled:
        return bundled
    # A frozen release must not silently depend on Homebrew or a local install.
    import sys
    if getattr(sys, 'frozen', False):
        return None
    # Finder-launched applications may not inherit a Homebrew shell PATH.
    return shutil.which('tesseract') or shutil.which(
        'tesseract', path=os.pathsep.join(['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin']))


def _ocr_numeric(crop):
    executable = _tesseract_executable()
    if executable is None:
        return None
    image = Image.fromarray(crop)
    scale = max(2, min(6, round(100 / image.height)))
    image = ImageOps.expand(image.resize((image.width * scale, image.height * scale),
                                       Image.Resampling.LANCZOS), 20, fill="white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    try:
        result = subprocess.run([executable, "stdin", "stdout", "--psm", "7", "tsv"],
                                input=stream.getvalue(), capture_output=True, timeout=8, check=True,
                                **hidden_subprocess_options())
    except (OSError, subprocess.SubprocessError):
        return None
    words = [r for r in csv.DictReader(io.StringIO(result.stdout.decode("utf-8")), delimiter="\t")
             if r.get("level") == "5" and r.get("text", "").strip()]
    if len(words) != 1:
        return None
    text = words[0]["text"].strip()
    confidence = float(words[0]["conf"])
    if confidence < 85 or not re.fullmatch(r"[0-9]{2}", text):
        return None
    return {"text": text, "confidence": confidence}


def remove_detached_numeric_labels(rgb):
    """Return a cleaned copy and an auditable mask description, or no changes.

    This intentionally does not cover text attached to strands, arbitrary math,
    single-character component labels, or labels inside a diagram's interior.
    In particular a single small circular component must never be read as '0'.
    """
    mask = (rgb.min(axis=2) < 245).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    small = []
    large = []
    for i in range(1, count):
        x, y, cw, ch, area = map(int, stats[i])
        if area < 8:
            continue
        if ch <= .19 * h and cw <= .14 * w and ch >= 6 and .2 <= cw / ch <= 1.05:
            small.append(i)
        elif area > 20:
            large.append(i)
    if not large or len(small) < 2:
        return rgb, {"removed_labels": []}
    left = min(stats[i, 0] for i in large)
    top = min(stats[i, 1] for i in large)
    right = max(stats[i, 0] + stats[i, 2] for i in large)
    bottom = max(stats[i, 1] + stats[i, 3] for i in large)
    removed = []
    used = set()
    output = rgb.copy()
    for a in small:
        x, y, cw, ch, area = map(int, stats[a])
        if a in used or ch < 10:
            continue
        for b in small:
            if b == a or b in used:
                continue
            bx, by, bw, bh, ba = map(int, stats[b])
            # A smaller, lower glyph immediately to the right is a subscript.
            if not (.5 <= bh / ch <= .85 and 0 <= bx - (x + cw) <= .4 * ch
                    and .25 * ch <= by - y <= .8 * ch
                    and y + ch < by + bh <= y + 1.5 * ch):
                continue
            x1, y1, x2, y2 = x, y, bx + bw, max(y + ch, by + bh)
            if not ((x1 + x2) / 2 < left or (x1 + x2) / 2 > right
                    or (y1 + y2) / 2 < top or (y1 + y2) / 2 > bottom):
                continue
            if area + ba > .12 * mask.sum():
                continue
            glyph_mask = np.isin(labels[y1:y2, x1:x2], [a, b])
            # Two isolated ovals could be actual unknotted link components.
            # Require branching within at least one glyph before consulting OCR.
            skel = skeletonize(glyph_mask)
            neighbors = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
            if not np.any(skel & (neighbors >= 4)):
                continue
            crop = np.full((y2-y1, x2-x1, 3), 255, dtype=np.uint8)
            crop[glyph_mask] = rgb[y1:y2, x1:x2][glyph_mask]
            reading = _ocr_numeric(crop)
            if reading is None:
                continue
            # Do not erase nearby ink, including antialiasing from other strands.
            erase = np.isin(labels, [a, b])
            output[erase] = 255
            removed.append({**reading, "bbox": [x1, y1, x2, y2],
                            "removed_pixels": int(erase.sum()),
                            "reason": "detached exterior numeric subscript; local OCR confirmed"})
            used.update([a, b])
            break
    return output, {"removed_labels": removed,
                    "ocr_available": _tesseract_executable() is not None}
