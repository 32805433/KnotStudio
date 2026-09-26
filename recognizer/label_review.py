"""Non-destructive proposals for reviewing labels before image recognition.

Learned ink geometry supplies proposals without decoding text. A proposal is not a
decision: a small circle might be a zero or an actual link component, and the
user must decide whether to erase it.  Erasure is restricted to the proposed
connected ink and its antialiasing fringe, never to the bounding rectangle.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from .diagram_labels import _glyph_geometry
from .label_geometry import strand_continuation_group, arrow_component
from .label_model import candidate_component_ids, load_model, predict_components
from .label_grouping import (typography_neighbors as _neighbors,
                             _script as _script_neighbor, signed_prefix_owners,
                             blocked_typography_pairs, can_merge_typography_groups,
                             paired_horizontal_bars)


@dataclass(frozen=True)
class LabelReviewConfig:
    """Pixel-only proposal thresholds, also used by the reserved-data evaluator."""

    seed_probability: float = .20
    rescue_probability: float = .70
    printed_probability: float = .55
    companion_probability: float = .12
    anchor_probability: float = .65
    group_probability: float = .20
    bar_seed_probability: float = .70
    long_seed_probability: float = .60
    reconnect_companions: bool = True
    contextual_companions: bool = True

    def __post_init__(self):
        for name, value in vars(self).items():
            if name not in ('reconnect_companions', 'contextual_companions') and not 0. <= value <= 1.:
                raise ValueError(f'{name} must be between zero and one.')
        if type(self.reconnect_companions) is not bool:
            raise ValueError('reconnect_companions must be a boolean.')
        if type(self.contextual_companions) is not bool:
            raise ValueError('contextual_companions must be a boolean.')


def _contextual_separators(labels, pieces, seed_ids):
    """Original glyph seeds on both sides can support a slash or fraction bar.

    Recruited marks cannot provide this evidence recursively. In particular, a
    single nearby digit cannot promote a chain of broken diagram strands.
    """
    anchors = {i: p for i, p in pieces.items() if i in seed_ids and not p['bar']
               and p['label_probability'] >= .50 and p['compact']
               and p['width_variation'] >= 1.6}
    companions = {}
    for i, p in pieces.items():
        x,y,r,d = p['bbox']; w,h = r-x,d-y
        if p['bar']:
            if p.get('paired_bar'):
                continue
            above, below = set(), set()
            for j,a in anchors.items():
                ax,ay,ar,ad = a['bbox']; ah = ad-ay
                if (min(r,ar)-max(x,ax) < .50*(ar-ax)
                        or not .6*(ar-ax) <= w <= 3.*(ar-ax) or h > .25*ah):
                    continue
                if 0 <= y-ad <= .85*ah:
                    above.add(j)
                if 0 <= ay-d <= .85*ah:
                    below.add(j)
            if above and below:
                companions[i] = above | below
        elif p['ends'] == 2 and not p['branches'] and not p['closed'] and .15*h <= w <= h:
            # Slashes are nearly straight diagonals; curved strand fragments
            # and tall parentheses must not gain the same exemption.
            yy,xx = np.where(labels[y:d,x:r] == i)
            if len(xx) < 5 or np.std(xx) < 1 or np.std(yy) < 1:
                continue
            if abs(float(np.corrcoef(xx,yy)[0,1])) < .92:
                continue
            left, right = set(), set()
            for j,a in anchors.items():
                ax,ay,ar,ad = a['bbox']; ah = ad-ay
                if (a['label_probability'] < .65 or not .7*ah <= h <= 1.9*ah
                        or abs((ay+ad)-(y+d)) > .9*ah):
                    continue
                if 0 <= x-ar <= .5*ah:
                    left.add(j)
                if 0 <= ax-r <= .5*ah:
                    right.add(j)
            if left and right:
                companions[i] = left | right
    return companions


def _rgb(image):
    image = np.asarray(image)
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or not image.shape[0] or not image.shape[1]):
        raise ValueError("Label review requires an RGB uint8 image.")
    return image


def _digest(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def _foreground(rgb, diagnostics=None):
    from .chalk_foreground import photographic_chalk
    chalk = photographic_chalk(rgb, diagnostics)
    if chalk is not None:
        return chalk
    # A dark-board photograph needs the opposite polarity.  This affects only
    # proposals; no background correction is performed on the source image.
    level = float(np.median(rgb.mean(axis=2)))
    dark = level < 115.
    if dark:
        light = rgb.max(axis=2)
        return light > min(210., level + 45.), light > level + 12., True
    light = rgb.min(axis=2)
    return light < 215, light < 254, False


def _runs(mask, x0, y0):
    output = []
    for y in range(mask.shape[0]):
        switches = np.flatnonzero(np.diff(np.pad(mask[y].astype(np.int8), (1, 1))))
        output.extend([int(y+y0), int(a+x0), int(b+x0)]
                      for a, b in zip(switches[::2], switches[1::2]))
    return output


def _box_interiors(ink, pieces, boxes=None):
    """Protect coefficients even when a box's strand ports cannot be paired.

    A closed rectangular boundary is sufficient for protection; interpreting
    the twist itself still requires the separate box-review checks. Corners
    and a clear margin distinguish a surrounding box from holes in glyphs.
    """
    from .box_detection import _closed_quads
    quads = [q for q, score in _closed_quads(ink)]
    quads.extend(np.asarray(b['outline'], dtype=float)
                 for b in (boxes or {}).get('boxes', []) if b.get('outline'))
    protected = set()
    for i, piece in pieces.items():
        x, y, r, d = piece['bbox']
        for quad in quads:
            polygon = quad.astype(np.float32)
            if cv2.contourArea(polygon) < 1.8 * (r-x) * (d-y):
                continue
            # A thick printed coefficient can sit close to its frame. Its
            # stroke width is not a required amount of padding: demanding
            # that padding exposed a clearly enclosed digit to label deletion.
            # A full pixel inside the closed frame plus the area check above
            # still distinguishes an enclosing box from a glyph's own hole.
            margin = 1.
            if all(cv2.pointPolygonTest(polygon, (float(a), float(b)), True) >= margin
                   for a, b in ((x,y),(r-1,y),(r-1,d-1),(x,d-1))):
                protected.add(i)
                break
    return protected


def detect_labels(image, *, use_ocr=False, max_ocr_calls=16, max_ocr_seconds=3.,
                  max_candidates=256, twist_boxes=None, config=None):
    """Return JSON-serializable proposals without altering ``image``.

    Text is never decoded. Legacy ``use_ocr`` and OCR-budget arguments are
    accepted for compatibility but have no effect; ``text`` is always empty.
    ``bbox`` uses half-open image coordinates.  ``erase_runs`` contains half-open
    ``[y, x1, x2]`` spans.  Confidence describes the proposal, not a guarantee that
    the ink is text. Small closed components that pass the typography checks
    remain ambiguous. The user decides which proposed ink belongs to the link.
    """
    rgb = _rgb(image)
    config = config or LabelReviewConfig()
    height, width = rgb.shape[:2]
    foreground = {}
    ink, faint, dark = _foreground(rgb, foreground)
    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    diagnostics = {'text_decoding': False, 'ocr_available': False, 'ocr_calls': 0,
                   'component_count': number-1, 'dark_background': dark,
                   'candidate_limit_reached': False, 'ocr_budget_exhausted': False,
                   **foreground}
    ink_pixels = int(ink.sum())
    pieces = {}
    model = load_model()
    learned_pool = set(map(int, candidate_component_ids(stats, ink.shape))) if model else set()
    # Large connected arcs define a coarse occupied area.  This is used only
    # for weak, two-ended shapes: a short curve inside the diagram should not
    # become a label just because OCR guesses "C", "L" or "N" for it.
    tightly_cropped = number <= 8
    substantial = stats[:, cv2.CC_STAT_AREA] >= max(20, .50 * stats[1:, cv2.CC_STAT_AREA].max(initial=0))
    if not tightly_cropped:
        substantial |= ((stats[:, cv2.CC_STAT_HEIGHT] > .20 * height)
                        | (stats[:, cv2.CC_STAT_WIDTH] > .15 * width))
    substantial[0] = False
    contours, _ = cv2.findContours(substantial[labels].astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(contours)) if contours else None
    # Relative bounds allow the same drawing at different resolutions.  Short
    # strand fragments can pass this first gate; topology and grouping below
    # distinguish them from stronger label proposals.
    for i in range(1, number):
        x, y, w, h, area = map(int, stats[i])
        dash = w <= (.50 if model else .16) * width and h <= .06 * height and w >= 2.5*h
        legacy_eligible = not (area < 5 or h < 1 or h > (.45 if tightly_cropped else .24) * height
                or w > (.35 if tightly_cropped else .25) * width
                or (w > max(3.4*h, 10) and not dash)
                or area > (.25 if tightly_cropped else .10) * ink_pixels)
        if not legacy_eligible and i not in learned_pool:
            continue
        fill = area / (w*h)
        # Letter strokes occupy a compact box; long thin tangle arcs usually
        # leave nearly all of it empty. Keep bars for later sign/fraction grouping.
        if fill < .12 and i not in learned_pool:
            continue
        shape = _glyph_geometry(labels[y:y+h, x:x+w] == i)
        # A shallow U-shaped strand is not a fraction bar. Real bars occupy
        # most of their narrow box, or stay within two stroke widths vertically.
        dash = dash and (fill >= .45 or h <= 2.*shape['stroke_width'])
        if not dash and max(w,h)/max(shape['stroke_width'], 1.) > 32. and i not in learned_pool:
            continue
        exterior = (h <= (.45 if tightly_cropped else .18)*height
                    and w <= (.35 if tightly_cropped else .15)*width
                    and hull is not None and cv2.pointPolygonTest(
            hull, (float(x+w/2), float(y+h/2)), True) < -max(1., shape['stroke_width']*.4))
        pieces[i] = {'bbox': (x, y, x+w, y+h), 'area': area,
                     'legacy_eligible': legacy_eligible and fill >= .12 and (dash or max(w,h)/max(shape['stroke_width'],1.) <= 32.),
                     'exterior': exterior, 'bar': dash, 'fill': fill, 'compact': fill >= .20 and shape['width_variation'] >= 1.6 and 4.5 <= h/max(shape['stroke_width'],1.) <= 32., **shape}
    probabilities = predict_components(rgb, labels, stats, pieces, component_ids=list(pieces)) if model else {}
    for i, p in pieces.items():
        p['label_probability'] = probabilities.get(i, 0.)
    diagnostics['component_model'] = model.get('model_id', 'label_components_v2') if model else None
    protected_boxes = _box_interiors(ink, pieces, twist_boxes)
    diagnostics['box_components_protected'] = len(protected_boxes)
    pieces = {i: p for i, p in pieces.items() if i not in protected_boxes}
    protected_arrows = set()
    for i, p in pieces.items():
        x,y,r,d = p['bbox']
        if arrow_component(labels[y:d,x:r] == i):
            protected_arrows.add(i)
    diagnostics['arrow_components_protected'] = len(protected_arrows)
    pieces = {i: p for i, p in pieces.items() if i not in protected_arrows}
    font_heights = [p['bbox'][3]-p['bbox'][1] for p in pieces.values()
                    if not p['bar'] and p['fill'] >= .20 and p['width_variation'] >= 1.45
                    and (p['branches'] or p['closed'] or p['ends'] > 2)]
    for p in pieces.values():
        w, h = p['bbox'][2]-p['bbox'][0], p['bbox'][3]-p['bbox'][1]
        if (not p['bar'] and p['fill'] >= .32 and .30 <= w/h <= 1.8
                and 4.5 <= h/max(p['stroke_width'], 1.) <= 16.
                and any(.8 <= h/fh <= 1.25 for fh in font_heights)):
            p['compact'] = True
    diagnostics['strand_fragments_protected'] = 0
    # Topology alone is insufficient: a small curl/contact can also have a
    # loop or branches. Require compact ink or variable-width glyph strokes.
    for p in pieces.values():
        w, h = p['bbox'][2]-p['bbox'][0], p['bbox'][3]-p['bbox'][1]
        span = max(w,h)/max(p['stroke_width'],1.)
        structure = p['closed'] or p['branches'] or p['ends'] > 2
        p['glyph'] = bool(structure and (
            (p['fill'] >= .16 and p['width_variation'] >= 1.45)
            or (p['fill'] >= .25 and span <= 20.)
            or (p['ends'] == 4 and p['branches'] <= 2 and .6 <= w/h <= 1.6 and span <= 16.)))
    seeds = [i for i,p in pieces.items()
             if (not p['bar'] or (model and p['label_probability'] >= config.bar_seed_probability))
             and max(p['bbox'][2]-p['bbox'][0],p['bbox'][3]-p['bbox'][1]) >= max(4.,2.7*p['stroke_width'])
             and ((p['legacy_eligible'] and (p['glyph'] or p['exterior'] or p['compact'])
                   and (not model or p['label_probability'] >= (
                       config.long_seed_probability if not p['bar'] and max(p['bbox'][2]-p['bbox'][0], p['bbox'][3]-p['bbox'][1])/max(1.,p['stroke_width']) > 20.
                       else config.seed_probability)))
                  or (model and p['label_probability'] >= config.rescue_probability))]
    paired_bars = paired_horizontal_bars(pieces)
    for i, p in pieces.items():
        p['paired_bar'] = i in paired_bars
    prefix_owners = signed_prefix_owners(pieces)
    blocked_pairs = blocked_typography_pairs(pieces, prefix_owners)
    seed_ids = set(seeds)
    contextual = (_contextual_separators(labels, pieces, seed_ids)
                  if model and config.contextual_companions else {})
    # Start with the main glyph, so an accent/prime cannot impose its tiny
    # height as the size limit for the entire expression.
    seeds.sort(key=lambda i: (-(pieces[i]['bbox'][3]-pieces[i]['bbox'][1]),
                               pieces[i]['bbox'][1], pieces[i]['bbox'][0]))
    groups = []
    used = set()
    for seed in seeds:
        if seed in used:
            continue
        group = {seed}
        seed_height = pieces[seed]['bbox'][3]-pieces[seed]['bbox'][1]
        for _ in range(8):
            seed_height = max(pieces[j]['bbox'][3]-pieces[j]['bbox'][1]
                              for j in group)
            added = set()
            for i, piece in pieces.items():
                if i in used or i in group:
                    continue
                supported_separator = bool(contextual.get(i, set()) & group)
                strong_text = [j for j in group if not pieces[j]['bar']
                               and pieces[j]['label_probability'] >= .75
                               and pieces[j]['compact']]
                supported_glyph = (config.contextual_companions and len(strong_text) >= 2
                                   and piece['label_probability'] >= .05
                                   and piece['compact'] and piece['glyph']
                                   and piece['width_variation'] >= 1.8)
                # A weak seed cannot recruit an arbitrary chain of nearby
                # arcs. Uncertain punctuation needs an established text
                # anchor before typography rules may extend the group.
                if (model and piece['label_probability'] < config.companion_probability
                        and not supported_separator
                        and not any(pieces[j]['label_probability'] >= config.anchor_probability for j in group)):
                    continue
                x, y, r, d = piece['bbox']
                if i not in seed_ids and not piece['bar']:
                    # Non-label arcs must not be pulled into an expression by
                    # proximity. Admit only its punctuation/accent/subscript.
                    parenthesis = (r-x)/(d-y) < .4 and .75*seed_height <= d-y <= 1.45*seed_height
                    subscript = d-y <= .75*seed_height and piece['width_variation'] >= 1.3
                    dot = max(r-x,d-y) <= .30*seed_height and any(
                        (min(r,pieces[j]['bbox'][2]) > max(x,pieces[j]['bbox'][0]))
                        or (max(x-pieces[j]['bbox'][2],pieces[j]['bbox'][0]-r,0) < .35*seed_height
                            and y >= pieces[j]['bbox'][3]-.25*seed_height)
                        for j in group)
                    short_sign = (piece['ends'] == 2 and not piece['branches'] and r-x >= d-y
                                  and d-y <= .5*seed_height and r-x <= .8*seed_height)
                    script = any(_script_neighbor(pieces[j], piece) for j in group)
                    if not (parenthesis or subscript or dot or short_sign or script
                            or supported_separator or supported_glyph):
                        continue
                    if (model and piece['label_probability'] < config.companion_probability
                            and not (parenthesis or dot or short_sign or script
                                     or supported_separator or supported_glyph
                                     or (subscript and d-y <= .5*seed_height))):
                        continue
                span_limit = (24. if model and piece['label_probability'] >= .20 else 6.) if piece['bar'] else 2.3
                if d-y > 2.5 * seed_height or r-x > span_limit * seed_height:
                    continue
                if any(frozenset((j,i)) not in blocked_pairs and _neighbors(pieces[j], piece) for j in group):
                    boxes = [pieces[j]['bbox'] for j in group] + [piece['bbox']]
                    if (max(b[2] for b in boxes)-min(b[0] for b in boxes) <= 24. * seed_height
                            and max(b[3] for b in boxes)-min(b[1] for b in boxes) <= 3.2 * seed_height):
                        added.add(i)
            if not added:
                break
            group.update(added)
        used.update(group)
        strong = any(pieces[i]['closed'] or pieces[i]['branches'] or pieces[i]['ends'] > 2
                     for i in group)
        groups.append((group, strong))
    # Expressions are sometimes seeded by several glyphs. Merge neighboring
    # text runs using their shared baseline, and let a leading sign align with
    # the *whole fraction* rather than with its numerator alone.
    def bounds(group):
        boxes = [pieces[i]['bbox'] for i in group]
        return [min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)]
    def text_height(group):
        return max((pieces[i]['bbox'][3]-pieces[i]['bbox'][1] for i in group if not pieces[i]['bar']), default=1)
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            a, strong_a = groups[i]; x,y,r,d = bounds(a); h = text_height(a)
            for j in range(i+1, len(groups)):
                b, strong_b = groups[j]; xx,yy,rr,dd = bounds(b); hh = text_height(b)
                large = max(h, hh)
                horizontal_join = (min(h,hh) >= .5*large and max(xx-r,x-rr,0) <= .85*large
                        and min(r,rr)-max(x,xx) <= .3*min(r-x,rr-xx)
                        and abs((y+d)-(yy+dd)) <= .7*large
                        and max(d,dd)-min(y,yy) <= 1.6*large)
                fraction_join = False
                if min(r,rr)-max(x,xx) > 0 and max(d,dd)-min(y,yy) <= 4.*large:
                    for k in a|b:
                        p = pieces[k]
                        bx,by,br,bd = p['bbox']
                        if not p['bar'] or p.get('paired_bar') or br-bx < .6*max(r-x, rr-xx):
                            continue
                        tx,ty,tr,td = (xx,yy,rr,dd) if k in a else (x,y,r,d)
                        if (max(ty-bd,by-td,0) <= .85*large
                                and abs((bx+br)-(tx+tr)) <= 1.2*(br-bx)):
                            fraction_join = True; break
                if horizontal_join and not can_merge_typography_groups(a, b, pieces, prefix_owners):
                    horizontal_join = False
                if horizontal_join or fraction_join:
                    groups[i] = (a|b, strong_a or strong_b)
                    groups.pop(j); changed = True; break
    if config.reconnect_companions:
        # A sign may have lacked a text anchor before neighboring glyph runs
        # merged. Revisit its existing ownership now, using the whole expression.
        # Restrict this pass to short signs with a known owner; arbitrary nearby
        # arcs cannot be recruited simply because a label's rectangle grew.
        for group, _ in groups:
            if model and not any(pieces[j]['label_probability'] >= config.anchor_probability
                                 for j in group):
                continue
            for bar, owner in prefix_owners.items():
                if bar in used or owner not in group:
                    continue
                if _neighbors(pieces[bar], pieces[owner]):
                    group.add(bar)
                    used.add(bar)
    # Unseeded horizontal bars can be standalone signs immediately beside a
    # completed expression. A distant dash/guide never becomes its own label.
    for k, (group, strong) in enumerate(groups):
        x,y,r,d = bounds(group)
        for i,p in pieces.items():
            if i in used or not p['bar']:
                continue
            if i in prefix_owners and prefix_owners[i] not in group:
                continue
            xx,yy,rr,dd = p['bbox']
            if (max(xx-r,x-rr,0) <= .5*(d-y)
                    and min(r,rr)-max(x,xx) < 0
                    and abs((yy+dd)-(y+d)) <= .4*(d-y)
                    and rr-xx <= 1.2*(d-y)):
                group.add(i); used.add(i)
        groups[k] = (group, strong)
    candidates = []
    claimed = np.zeros((height, width), bool)
    def printed_glyph(i):
        p = pieces[i]
        x,y,r,d = p['bbox']
        return (model and p['label_probability'] >= config.printed_probability
                and p['width_variation'] >= 1.6 and p['fill'] >= .18
                and max(r-x,d-y)/max(p['stroke_width'],1.) <= 20.)
    for group, geometric in groups:
        protected = {i for i in group if
                     not printed_glyph(i) and
                     strand_continuation_group(labels, {i}, pieces, allow_branches=True, ignore_ids=group)}
        diagnostics['strand_fragments_protected'] += len(protected)
        group = group - protected
        if not group:
            continue
        if not group.intersection(seed_ids):
            # A sign recruited by a body cannot remain as an independent
            # proposal after that body's ink was identified as a strand.
            continue
        if model and np.mean([pieces[i]['label_probability'] for i in group]) < config.group_probability:
            diagnostics['weak_groups_rejected'] = diagnostics.get('weak_groups_rejected', 0) + 1
            continue
        geometric = any(pieces[i]['closed'] or pieces[i]['branches'] or pieces[i]['ends'] > 2 for i in group)
        if not geometric and not any(printed_glyph(i) for i in group) and strand_continuation_group(labels, group, pieces):
            diagnostics['strand_fragments_protected'] += len(group)
            continue
        boxes = [pieces[i]['bbox'] for i in group]
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        local_labels = labels[y1:y2, x1:x2]
        glyph = np.isin(local_labels, list(group))
        confidence = 0.
        # Exterior short strokes also remain reviewable without OCR: these
        # include handwritten n, 1 and minus signs, which often have no branch.
        closed = any(pieces[i]['closed'] for i in group)
        ambiguous = closed or not geometric
        if closed:
            reason = 'Possible label or small closed link component; review before deleting.'
        elif geometric:
            reason = 'Detached glyph-shaped ink; review before deleting.'
        else:
            reason = 'Small detached glyph-like ink; possible text or strand.'
        confidence = (float(np.mean([pieces[i]['label_probability'] for i in group]))
                      if model else (.45 if closed else .6 if geometric else .35))
        pad = max(1, min(3, round(max(pieces[i]['stroke_width'] for i in group)*.20)))
        ax, ay, bx, by = max(0, x1-pad), max(0, y1-pad), min(width, x2+pad), min(height, y2+pad)
        patch_labels = labels[ay:by, ax:bx]
        glyph = np.isin(patch_labels, list(group))
        fringe = cv2.dilate(glyph.astype(np.uint8), np.ones((2*pad+1, 2*pad+1), np.uint8)).astype(bool)
        # Every other connected component is protected, regardless of its size.
        distance_to_glyph = cv2.distanceTransform((~glyph).astype(np.uint8), cv2.DIST_L2, 5)
        # A neighboring strand can have its solid core just outside the erase
        # patch while its faint edge lies inside. Include enough surrounding
        # context to see every competing core closer than the glyph's fringe.
        # Fringe pixels are at most sqrt(2)*pad from glyph ink, so 2*pad+1
        # extra pixels cover all such competitors, including diagonal ones.
        context_pad = 2*pad+1
        cx, cy = max(0, ax-context_pad), max(0, ay-context_pad)
        cr, cd = min(width, bx+context_pad), min(height, by+context_pad)
        context_labels = labels[cy:cd, cx:cr]
        context_other = (context_labels > 0) & ~np.isin(context_labels, list(group))
        context_distance = cv2.distanceTransform((~context_other).astype(np.uint8),
                                                 cv2.DIST_L2, 5)
        distance_to_other = context_distance[ay-cy:by-cy, ax-cx:bx-cx]
        erase = glyph | (fringe & faint[ay:by, ax:bx] & (patch_labels == 0)
                         & (distance_to_glyph < distance_to_other))
        erase &= ~claimed[ay:by, ax:bx]
        if not erase.any():
            continue
        claimed[ay:by, ax:bx] |= erase
        spans = _runs(erase, ax, ay)
        # The surrounding background supplies a fill even for gray paper or
        # blackboards.  Only marked pixels change; no rectangular inpainting.
        ring = cv2.dilate(fringe.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        clean = ring & ~fringe & (patch_labels == 0)
        if not clean.any():
            clean = (patch_labels == 0) & ~erase
        background = (np.median(rgb[ay:by, ax:bx][clean], axis=0).astype(np.uint8).tolist()
                      if clean.any() else ([0, 0, 0] if dark else [255, 255, 255]))
        identifier = hashlib.sha256(np.asarray(spans, dtype='<i4').tobytes()).hexdigest()[:16]
        candidates.append({'id': identifier, 'bbox': [x1, y1, x2, y2],
                           'text': '', 'confidence': round(confidence, 4),
                           'reason': reason, 'ambiguous': ambiguous,
                           'erase_runs': spans, 'background_rgb': background,
                           'component_count': len(group), 'ink_pixels': int(erase.sum())})
        if len(candidates) >= max_candidates:
            diagnostics['candidate_limit_reached'] = True
            break
    candidates.sort(key=lambda p: (p['bbox'][1], p['bbox'][0]))
    return {'version': 1, 'candidates': candidates, 'diagnostics': diagnostics,
            'image_size': [width, height], 'image_sha256': _digest(rgb)}


def _validate(rgb, detection):
    if (detection.get('image_size') != [rgb.shape[1], rgb.shape[0]]
            or detection.get('image_sha256') != _digest(rgb)):
        raise ValueError('Label markings are stale; scan this image again before deleting.')


def delete_labels(image, detection, candidate_ids):
    """Return a copy with only explicitly selected proposals erased.

    A matching pixel hash is mandatory.  Thus a stale marking cannot erase a
    drawing that has since been edited, resized, replaced or restored by Undo.
    """
    rgb = _rgb(image)
    _validate(rgb, detection)
    wanted = set(candidate_ids)
    known = {candidate['id'] for candidate in detection['candidates']}
    if wanted - known:
        raise ValueError('Unknown label marking.')
    output = rgb.copy()
    height, width = rgb.shape[:2]
    for candidate in detection['candidates']:
        if candidate['id'] not in wanted:
            continue
        fill = candidate.get('background_rgb', [255, 255, 255])
        for y, x1, x2 in candidate['erase_runs']:
            if not (0 <= y < height and 0 <= x1 < x2 <= width):
                raise ValueError('Invalid label erase mask.')
            output[y, x1:x2] = fill
    return output


def retain_labels_after_deletion(image, detection, deleted_ids):
    """Retain disjoint remaining markings and bind them to the cleaned pixels.

    Call this with the output of ``delete_labels``.  Detection never overlaps
    candidate erase masks, so removing one label cannot invalidate another.
    """
    rgb = _rgb(image)
    if detection.get('image_size') != [rgb.shape[1], rgb.shape[0]]:
        raise ValueError('Label markings belong to a different image size.')
    removed = set(deleted_ids)
    known = {candidate['id'] for candidate in detection['candidates']}
    if removed - known:
        raise ValueError('Unknown label marking.')
    result = copy.deepcopy(detection)
    result['candidates'] = [candidate for candidate in result['candidates']
                            if candidate['id'] not in removed]
    result['image_sha256'] = _digest(rgb)
    return result
