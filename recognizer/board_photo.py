"""Classical board-photo recognition from pixels, with explicit uncertainty.

No source identities or reference encodings enter this module. Alternate
contrast extractions are hypotheses, not instructions to erase arbitrary ink.
Every successful photo reconstruction requires review of the inferred gaps.
"""
from copy import deepcopy
import hashlib
import time

import cv2
import numpy as np
from scipy import ndimage

from .diagram import pd_code, validate
from .geometry import assemble, beautify
from .knotfolio_backend import extract, _outward_tangent
from .recognition_runtime import current


MODES = ('auto', 'off', 'board', 'light', 'dark')


def looks_photographic(rgb):
    """Conservative routing hint; forced board modes bypass this heuristic."""
    small = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    middle = float(np.median(gray))
    # Flat white/black digital canvases stay on the original route. Board
    # photographs have a predominantly midtone, varying background.
    midtone = float(np.mean((gray > 20) & (gray < 240)))
    low, high = np.percentile(gray, [20, 80])
    detected = 25 < middle < 235 and midtone > .75 and high-low > 3
    return bool(detected), dict(background_median=middle, midtone_fraction=midtone)


class BoardContrast:
    def __init__(self, rgb, mode='board'):
        self.original_shape = rgb.shape
        self.scale = min(1., 1600/max(rgb.shape[:2]))
        if self.scale < 1:
            rgb = cv2.resize(rgb, (round(rgb.shape[1]*self.scale), round(rgb.shape[0]*self.scale)),
                             interpolation=cv2.INTER_AREA)
        self.rgb = rgb
        self.smooth = cv2.GaussianBlur(rgb, (0, 0), .7).astype(np.float32)
        self.radius = max(7, round(min(rgb.shape[:2])*.025))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.radius+1,)*2)
        lower = cv2.morphologyEx(self.smooth, cv2.MORPH_OPEN, kernel)
        upper = cv2.morphologyEx(self.smooth, cv2.MORPH_CLOSE, kernel)
        bright, dark = (self.smooth-lower).max(2), (upper-self.smooth).max(2)
        inferred = 'light' if np.percentile(dark, 99.5) > np.percentile(bright, 99.5) else 'dark'
        # Sparse chalk can occupy less than .5% of a full camera image. Broad
        # erased/shadowed edges can then win that percentile despite a dark
        # board with clear bright strokes. Use the background as a prior only
        # when the brighter, sparser tail provides comparable ink evidence.
        # Clearly dark ink on a dim whiteboard still wins; forced modes below
        # remain authoritative.
        bright_tail, dark_tail = np.percentile(bright, 99.8), np.percentile(dark, 99.8)
        if (np.median(self.smooth[::8, ::8]) < 115 and bright_tail > 25
                and bright_tail >= dark_tail/1.5):
            inferred = 'dark'
        self.polarity = mode if mode in ('light', 'dark') else inferred
        self.morph = dark if self.polarity == 'light' else bright
        background = cv2.GaussianBlur(self.smooth, (0, 0), self.radius)
        self.residual = background-self.smooth if self.polarity == 'light' else self.smooth-background
        self.mean = self.residual.max(2)
        # A separately estimated color field distinguishes colored marker ink
        # from neutral eraser residue without assuming blue/red/green ink.
        color_background = cv2.GaussianBlur(self.smooth, (0, 0), max(10, min(rgb.shape[:2])*.03))
        self.color_residual = (color_background-self.smooth if self.polarity == 'light'
                               else self.smooth-color_background)
        self.chalk = None
        if self.polarity == 'dark':
            from .chalk_foreground import ChalkEvidence
            self.chalk = ChalkEvidence(self.smooth, self.mean, self.radius)

    def _seeded(self, mask, seed):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        cores = np.bincount(labels[seed & mask], minlength=n)
        area_floor = max(25, min(mask.shape)**2*.00015)
        allowed = ((cores >= max(4, mask.size*.000012)) & (stats[:, 4] > area_floor))
        allowed[0] = False
        discarded = [stats[i, :4].tolist() for i in range(1, n)
                     if not allowed[i] and stats[i, 4] >= area_floor]
        return allowed[labels], discarded

    def candidates(self):
        if self.polarity == 'dark':
            peak = float(np.percentile(self.mean, 99.8))
            specs = [('local_mean', self.mean, max(25., peak*f), max(12., peak*f*.5), False)
                     for f in (.55, .70, .85)]
            peak = float(np.percentile(self.morph, 99.8))
            specs += [('local_envelope', self.morph, max(25., peak*f), max(12., peak*f*.35), False)
                      for f in (.45, .55)]
        else:
            peak = float(np.percentile(self.morph, 99.8))
            specs = [('local_envelope', self.morph, max(20., peak*f), max(8., peak*f*.4), False)
                     for f in (.25, .35, .45)]
            specs += [('color_and_neutral', self.morph, max(20., peak*f), max(8., peak*f*.4), True)
                      for f in (.25, .30)]
        for method, contrast, strong, weak, split_color in specs:
            dust_removed = 0
            if split_color:
                chroma = np.ptp(self.color_residual, axis=2)
                color = (chroma > 15) & (chroma > contrast*.25) & (contrast > weak)
                color, removed = self._seeded(color, color & (chroma > 30))
                distance = ndimage.distance_transform_edt(~color)
                gray, removed_gray = self._seeded(
                    (contrast > weak) & (distance > 2),
                    (contrast > strong) & (distance > max(5, min(contrast.shape)*.008)))
                removed += removed_gray
                keep = color | gray
            else:
                mask = contrast > weak
                if self.chalk is not None:
                    mask = self.chalk.filter(mask, contrast > strong)
                    dust_removed = int(np.count_nonzero((contrast > weak) & ~mask))
                keep, removed = self._seeded(mask, contrast > strong)
            out = np.full_like(self.rgb, 255)
            # Color is a matching cue, not a separate skeleton at every noisy
            # hue boundary. The tracer follows the union of the retained ink.
            base = self.rgb if self.polarity == 'light' else 255-self.rgb
            ink = base.astype(float)-base.min(2, keepdims=True)
            span = ink.max(2, keepdims=True)
            ink = np.where(span > 35, ink/np.maximum(span, 1)*160, 0).astype(np.uint8)
            out[keep] = ink[keep]
            yield out, dict(method=method, strong=strong, weak=weak,
                            dust_filter=self.chalk is not None,
                            dust_removed_pixels=dust_removed,
                            foreground_fraction=float(keep.mean()), discarded_regions=removed)


def prepared_board_contrast(rgb, mode='board'):
    """Reuse exact source masks within one recognition's bounded pixel cache.

    Tracing, fallback repairs, and source orientation otherwise repeat the
    same morphological background estimation. Store masks, not the much
    larger contrast work arrays. Cache deserialization gives each caller its
    own pixels; nothing persists between recognition calls.
    """
    from .recognition_runtime import array_key, checkpoint, current
    state = current()
    if state is None:
        return BoardContrast(rgb, mode)
    mode = 'board' if mode in ('auto', 'board') else mode
    identity = array_key(rgb)
    key = ('board_masks', identity, mode)
    data = state.get(key)
    if data is None and mode in ('dark', 'light'):
        # A forced polarity has identical masks if it agrees with a cached
        # inferred polarity. Do not duplicate all the mask bytes in the cache.
        inferred = state.get(('board_masks', identity, 'board'))
        if inferred is not None and inferred['polarity'] == mode:
            data = inferred
    if data is None:
        checkpoint('preparing board contrast')
        contrast = BoardContrast(rgb, mode)
        data = dict(scale=contrast.scale, polarity=contrast.polarity,
                    candidates=list(contrast.candidates()))
        state.put(key, data)

    class PreparedContrast:
        scale = data['scale']
        polarity = data['polarity']

        def candidates(self):
            yield from data['candidates']

    return PreparedContrast()


def _trim_to_plane(points, axis, value, limit):
    """Trim only a short, monotonically overdrawn end, in outward-first order."""
    projection = points @ axis-value
    travelled = 0.
    for i, (a, b) in enumerate(zip(points, points[1:])):
        segment_length = float(np.linalg.norm(b-a))
        if projection[i]*projection[i+1] <= 0 and abs(projection[i]-projection[i+1]) > 1e-8:
            fraction = projection[i]/(projection[i]-projection[i+1])
            if travelled+fraction*segment_length > limit:
                return None
            return np.vstack([a+fraction*(b-a), points[i+1:]])
        travelled += segment_length
        if travelled > limit:
            return None
    return None


def repair_overdrawn_seam(trace, *, allow_same_path=False, source_mask=None,
                          seam_widths=(2., 1.)):
    """Join one isolated pair of overlapping, parallel pen tips for review.

    This is not a crossing repair: reject any other strand or inferred gap in
    the local box. Both original tips are retained in the review diagnostics.
    """
    diagnostic = trace['diagnostics']
    ends = diagnostic['unmatched_endpoints']
    if diagnostic['branch_points'] or diagnostic['gap_conflicts'] or len(ends) != 2:
        return trace
    a, b = ends
    same_path = a['path'] == b['path']
    if same_path and (not allow_same_path or a['end'] == b['end']):
        return trace
    paths = trace['paths']
    if paths[a['path']].get('color') != paths[b['path']].get('color'):
        return trace
    pa = np.asarray(paths[a['path']]['points'])[::1 if a['end'] == 0 else -1]
    pb = np.asarray(paths[b['path']]['points'])[::1 if b['end'] == 0 else -1]
    width = diagnostic['stroke_width']
    ta = _outward_tangent(pa, 0, 3*width)
    tb = _outward_tangent(pb, 0, 3*width)
    delta = pb[0]-pa[0]
    overlap = -float(delta @ ta)
    minimum, maximum = (.5, 10.) if same_path else (1., 6.)
    supported_seam = source_mask is not None
    if (ta @ tb > -.90 or not minimum*width < overlap < maximum*width or
            (not supported_seam and np.linalg.norm(delta+(overlap*ta)) > 1.5*width)):
        return trace
    value = float((pa[0]+pb[0]) @ ta/2)
    qa, qb = (_trim_to_plane(p, ta, value, maximum*width) for p in (pa, pb))
    if qa is None or qb is None:
        return trace
    evidence = None
    if supported_seam:
        from .chalk_seams import chalk_overlap_evidence
        evidence = chalk_overlap_evidence(source_mask, qa[0], qb[0], width,
                                         center_widths=seam_widths[0], blank_widths=seam_widths[1])
        if evidence is None:
            return trace
    elif np.linalg.norm(qa[0]-qb[0]) > 1.5*width:
        return trace
    edit = None
    if same_path:
        left, right = len(pa)-len(qa)+1, len(qb)-1
        if left >= right:
            return trace
        edit = np.vstack([pa[:left], qa[0], qb[0], pb[:len(pb)-len(qb)+1][::-1]])
    low = np.minimum(pa[0], pb[0])-2*width
    high = np.maximum(pa[0], pb[0])+2*width
    if edit is not None:
        low, high = edit.min(axis=0)-2*width, edit.max(axis=0)+2*width
    # Bounding-box exclusion is deliberately conservative, including segments
    # that cross the box without having a vertex inside it.
    def enters(points):
        points = np.asarray(points)
        lo = np.minimum(points[:-1], points[1:])
        hi = np.maximum(points[:-1], points[1:])
        nearby = np.all(lo <= high, axis=1) & np.all(hi >= low, axis=1)
        if edit is None or not nearby.any():
            return bool(nearby.any())
        # A long overlap has a large bounding rectangle: other portions of an
        # oval may enter a corner without approaching the edited band. Check
        # actual swept tails and the new connector before rejecting it.
        from .motion import _segment_distances
        starts, ends = points[:-1][nearby], points[1:][nearby]
        n = len(edit)-1
        distances = _segment_distances(np.repeat(starts, n, axis=0),
                                      np.repeat(ends, n, axis=0),
                                      np.tile(edit[:-1], (len(starts), 1)),
                                      np.tile(edit[1:], (len(starts), 1)))
        return bool(np.any(distances < .75*width))
    for i, path in enumerate(paths):
        if same_path and i == a['path']:
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pa, axis=0), axis=1))]
            reach = max(8*width, overlap+3*width)
            if length[-1] <= 4*reach:
                return trace  # Do not collapse a tiny curl into its seam.
            first = np.searchsorted(length, reach)
            last = np.searchsorted(length, length[-1]-reach)
            if enters(pa[max(0, first-1):min(len(pa), last+1)]):
                return trace
        elif i in (a['path'], b['path']):
            # Only the first local tail may enter the edit box. Check the rest
            # too, so a nearby self-strand cannot be silently crossed.
            p = pa if i == a['path'] else pb
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            far = np.searchsorted(length, 8*width)
            if enters(p[max(0, far-1):]):
                return trace
        elif enters(path['points']):
            return trace
    for match in trace['matches']:
        x, y = match['a'], match['b']
        segment = [paths[x[0]]['points'][0 if x[1] == 0 else -1],
                   paths[y[0]]['points'][0 if y[1] == 0 else -1]]
        if enters(segment):
            return trace
    result = deepcopy(trace)
    if same_path:
        # Both shortened tips belong to one array. Independent assignments
        # would overwrite the first trim and leave the original overlap.
        points = np.vstack([qa[0], pa[left:right], qb[0]])
        result['paths'][a['path']]['points'] = points[::1 if a['end'] == 0 else -1].tolist()
        result['diagnostics']['same_path_seam_repair'] = True
    else:
        for endpoint, points in ((a, qa), (b, qb)):
            result['paths'][endpoint['path']]['points'] = points[::1 if endpoint['end'] == 0 else -1].tolist()
    result['matches'].append(dict(a=[a['path'], a['end']], b=[b['path'], b['end']],
                                  crossings=[], distance=float(np.linalg.norm(qa[0]-qb[0])),
                                  cost=0., alignment=[1., 1.], overdrawn_seam=True))
    result['diagnostics']['seam_repairs'] = [pa[0].tolist(), pb[0].tolist()]
    if evidence is not None:
        result['diagnostics']['chalk_seam_evidence'] = evidence
    result['diagnostics']['unmatched_endpoints'] = []
    result['complete'] = True
    return result


def recognize_board(rgb, mode='board', options=None, *, _contrast=None,
                    _minimum_agreement=1, _require_embedding=False):
    started = time.monotonic()
    # Experimental mask providers reuse the same classical reconstruction.
    # Their stricter acceptance requirements do not alter the established
    # contrast route used by ordinary callers.
    contrast = _contrast if _contrast is not None else prepared_board_contrast(rgb, mode)
    attempts, complete, best, failed_masks = [], [], None, []
    options = dict(options or {})
    photo_repairs = options.pop('photo_repairs', True)
    same_path_seams = options.pop('same_path_seams', False)
    def agreement(candidates):
        return len({p['_mask_id'] for _, _, p in candidates})

    def required_agreement(candidates):
        repaired = any(t['diagnostics'].get('pre_matching_seams') or
                       t['diagnostics'].get('seam_repairs') or
                       t['diagnostics'].get('photo_arrow_evidence') or
                       t['diagnostics'].get('photo_contact_repairs') for _, t, _ in candidates)
        return max(_minimum_agreement, 2 if repaired else 1)

    def remember_partial_trace(trace, preparation):
        state = current()
        if state is not None:
            state.record_trace_failure(trace, scale=contrast.scale,
                preprocessing=dict(transform='board_photo', polarity=contrast.polarity,
                                   method=preparation.get('method'),
                                   strong=preparation.get('strong'), weak=preparation.get('weak'),
                                   scale=contrast.scale))

    for pixels, preparation in contrast.candidates():
        # Threshold settings that produce identical ink are one observation,
        # not independent confirmation of a repaired seam.
        preparation = dict(preparation, _mask_id=hashlib.sha256(
            np.packbits(pixels.min(2) < 205).tobytes()).hexdigest())
        record = {k: v for k, v in preparation.items() if k != 'discarded_regions' and not k.startswith('_')}
        attempts.append(record)
        if not .0001 < preparation['foreground_fraction'] < .25:
            record['error'] = 'No sparse, sufficiently contrasted strand mask.'
            continue
        config = dict(max_gap_widths=85., tangent_widths=3., spur_widths=3.,
                      color_weight=10., separate_colors=False, min_object_pixels=10,
                      stop_at_branches=True, max_endpoints=120, photo_gap_policy=True)
        config.update(options or {})
        trace = repair_overdrawn_seam(extract(pixels, config), allow_same_path=same_path_seams,
                                     source_mask=pixels.min(2) < 205)
        diagnostic = trace['diagnostics']
        score = len(diagnostic['branch_points'])+len(diagnostic['unmatched_endpoints'])+len(diagnostic['gap_conflicts'])
        record.update(complete=trace['complete'], branches=len(diagnostic['branch_points']),
                      unmatched=len(diagnostic['unmatched_endpoints']), conflicts=len(diagnostic['gap_conflicts']))
        if best is None or score < best[0]:
            best = (score, trace, preparation)
        if not trace['complete']:
            remember_partial_trace(trace, preparation)
            failed_masks.append((pixels, preparation, trace))
            continue
        try:
            diagram = beautify(assemble(trace, pixels.shape[1], pixels.shape[0]))
            verdict = validate(diagram)
            if _require_embedding:
                from .geometry import check_embedding
                embedding = check_embedding(diagram, check_strokes=False)
                if not verdict['valid'] or not embedding['valid']:
                    raise ValueError('Candidate mask produced an invalid planar embedding.')
            record.update(crossings=verdict['crossings'], components=verdict['components'])
            complete.append((diagram, trace, preparation))
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            record['error'] = str(exc)
    if agreement(complete) < required_agreement(complete) and photo_repairs:
        from .photo_arrows import separate_photo_arrows
        from .bundle_contacts import repair_bundle_contacts
        repaired_complete = list(complete)
        for pixels, preparation, _ in failed_masks:
            # These hypotheses retain the original contrast mask and preserve
            # its unrelated ink. They never delete detached textual labels.
            cleaned, arrows = separate_photo_arrows(pixels)
            candidate = cleaned if cleaned is not None else pixels
            config = dict(max_gap_widths=85., tangent_widths=3., spur_widths=3.,
                          color_weight=10., separate_colors=False, min_object_pixels=10,
                          stop_at_branches=True, max_endpoints=120, micro_seams=True,
                          photo_gap_policy=True)
            config.update(options)
            trace = extract(candidate, config)
            contacts, contact_diagnostic = repair_bundle_contacts(candidate, trace, allow_single=True)
            if contacts is not None:
                trace = extract(contacts, config)
            trace = repair_overdrawn_seam(trace, allow_same_path=same_path_seams,
                                         source_mask=pixels.min(2) < 205)
            diagnostic = trace['diagnostics']
            diagnostic['photo_arrow_evidence'] = arrows
            diagnostic['photo_contact_repairs'] = contact_diagnostic['bundle_contact_repairs']
            score = len(diagnostic['branch_points'])+len(diagnostic['unmatched_endpoints'])+len(diagnostic['gap_conflicts'])
            record = {k:v for k,v in preparation.items() if k!='discarded_regions' and not k.startswith('_')}
            record.update(photo_repair=True, arrow_heads=len(arrows),
                          contacts=len(diagnostic['photo_contact_repairs']), complete=trace['complete'],
                          branches=len(diagnostic['branch_points']),
                          unmatched=len(diagnostic['unmatched_endpoints']), conflicts=len(diagnostic['gap_conflicts']))
            attempts.append(record)
            if best is None or score<best[0]:
                best = (score, trace, preparation)
            if not trace['complete']:
                remember_partial_trace(trace, preparation)
                continue
            try:
                diagram = beautify(assemble(trace, pixels.shape[1], pixels.shape[0]))
                verdict = validate(diagram)
                from .geometry import check_embedding
                embedding = check_embedding(diagram, check_strokes=False)
                if not embedding['valid']:
                    raise ValueError('Photo repair produced an invalid planar embedding: '+str(embedding.get('errors', [])))
                record.update(crossings=verdict['crossings'], components=verdict['components'])
                repaired_complete.append((diagram,trace,preparation))
            except (ValueError,RuntimeError,KeyError,IndexError) as exc:
                record['error'] = str(exc)
        # Extra interpretation of photographed ink requires independent
        # contrast extractions to agree; a single convenient mask is not enough.
        if agreement(repaired_complete)>=2:
            complete = repaired_complete
    conflicting = False
    if complete:
        from .motion import _same_projection
        conflicting = any(not _same_projection(complete[0][0], d) for d, _, _ in complete[1:])
    selected = complete[0] if agreement(complete) >= required_agreement(complete) and not conflicting else None
    if selected:
        diagram, trace, preparation = selected
    else:
        diagram = None
        _, trace, preparation = best if best else (0, {'paths': [], 'matches': [], 'diagnostics': {}}, {})
    diagnostic = deepcopy(trace['diagnostics'])
    scale = contrast.scale
    pre_seams = diagnostic.get('pre_matching_seams', [])
    if len(pre_seams) == 1 and 'chalk_seam_evidence' not in diagnostic:
        # Preserve the existing single-seam diagnostics contract while the
        # general list records every independently reserved repair.
        diagnostic['chalk_seam_evidence'] = {
            key: deepcopy(pre_seams[0][key]) for key in
            ('bridge_ends', 'length', 'unsupported_length', 'supported_fraction', 'stroke_width')}
    if any(e['same_path'] for e in pre_seams):
        diagnostic['same_path_seam_repair'] = True
    for evidence in diagnostic.get('pre_matching_seams', []):
        for key in ('bridge_ends', 'original_tips'):
            evidence[key] = [[v/scale for v in p] for p in evidence[key]]
        for key in ('length', 'unsupported_length', 'stroke_width'):
            evidence[key] /= scale
    if diagnostic.get('pre_matching_seams'):
        diagnostic.setdefault('seam_repairs', [])
        diagnostic['seam_repairs'].extend(
            [v*scale for v in p] for e in diagnostic['pre_matching_seams'] for p in e['original_tips'])
    if 'chalk_seam_evidence' in diagnostic:
        evidence = diagnostic['chalk_seam_evidence']
        evidence['bridge_ends'] = [[v/scale for v in p] for p in evidence['bridge_ends']]
        for key in ('length', 'unsupported_length', 'stroke_width'):
            evidence[key] /= scale
    # All exposed coordinates, including diagnostic marks, use the source frame.
    for key in ('branch_points', 'seam_repairs'):
        if key in diagnostic:
            diagnostic[key] = [[v/scale for v in p] for p in diagnostic[key]]
    for endpoint in diagnostic.get('unmatched_endpoints', []):
        endpoint['point'] = [v/scale for v in endpoint['point']]
    if diagnostic.get('micro_seam_repairs'):
        diagnostic['micro_seam_repairs'] = [
            [[v/scale for v in p] for p in repair]
            for repair in diagnostic['micro_seam_repairs']]
    for arrow in diagnostic.get('photo_arrow_evidence', []):
        arrow['point'] = [v/scale for v in arrow['point']]
        arrow['stroke_width'] /= scale
        arrow['wing_lengths'] = [v/scale for v in arrow['wing_lengths']]
    for repair in diagnostic.get('photo_contact_repairs', []):
        for key in ('junction','trimmed_under_tip','opposite_tip'):
            repair[key] = [v/scale for v in repair[key]]
        repair['crossings'] = [[v/scale for v in p] for p in repair['crossings']]
        repair['bbox'] = [v/scale for v in repair['bbox']]
    diagnostic['stroke_width'] = diagnostic.get('stroke_width', 1.5)/scale
    gaps = []
    for match in trace['matches']:
        a, b = match['a'], match['b']
        ends = [trace['paths'][i]['points'][0 if end == 0 else -1] for i, end in (a, b)]
        gaps.append(dict(ends=[[v/scale for v in p] for p in ends],
                         crossings=[[v/scale for v in hit['point']] for hit in match['crossings']],
                         seam=bool(match.get('overdrawn_seam'))))
    diagnostic.update(attempts=attempts, elapsed_seconds=time.monotonic()-started,
                      preprocessing=dict(transform='board_photo', polarity=contrast.polarity,
                                         method=preparation.get('method'), strong=preparation.get('strong'),
                                         weak=preparation.get('weak'), scale=scale),
                      board_photo=dict(successful_candidates=len(complete), distinct_successful_masks=agreement(complete),
                                       required_agreement=required_agreement(complete), conflicting_candidates=conflicting,
                                       gaps=gaps, review_points=[p for g in gaps for p in g['crossings']],
                                       discarded_regions=[[v/scale for v in box] for box in preparation.get('discarded_regions', [])]))
    warning = ('Board-photo cleanup and long-gap reconstruction were used. Inspect all crossings and small components against the photograph.')
    warnings = [warning]
    if conflicting:
        warnings.append('Different contrast extractions imply different diagrams; no PD was selected. Clarify the ambiguous strokes and recognize again.')
    elif diagram is None:
        warnings.append('The board photograph still has unresolved contacts, marks, or open ends. Crop away unrelated marks or clarify the marked strokes before trying again.')
    if diagnostic.get('seam_repairs'):
        warnings.append('Overlapping pen tips were joined at the marked seam; inspect this repair.')
    if diagnostic.get('photo_arrow_evidence'):
        warnings.append('Paired arrow wings were separated from continuing strands for tracing; their directions were retained as source evidence.')
    if diagnostic.get('photo_contact_repairs'):
        warnings.append('Touching underpass tips were separated at marked three-arm junctions; inspect these inferred crossings.')
    if diagram is not None:
        for crossing in diagram['crossings']:
            crossing['point'] = [v/scale for v in crossing['point']]
        for edge in diagram['edges']:
            edge['points'] = [[v/scale for v in p] for p in edge['points']]
        diagram.update(width=rgb.shape[1], height=rgb.shape[0],
                       recognition=dict(stroke_width=diagnostic['stroke_width'], warnings=warnings))
        verdict = validate(diagram)
        diagnostic['validation'] = verdict
        return dict(status='needs_review', diagram=diagram, pd_code=pd_code(diagram), warnings=warnings,
                    unlinked_unknot_components=verdict['crossing_free_components'], diagnostics=diagnostic)
    preview = deepcopy(trace['paths'])
    for path in preview:
        path['points'] = [[v/scale for v in p] for p in path['points']]
    return dict(status='failed', diagram=None, pd_code=None, warnings=warnings, diagnostics=diagnostic,
                preview_paths=preview, unlinked_unknot_components=0)
