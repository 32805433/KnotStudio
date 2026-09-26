"""Conservative continuation of unusually long, single underpass gaps.

This fallback only completes unmatched tips. It never changes visible ink or
an existing pairing, and never guesses a solid junction's crossing order.
The caller must still assemble and validate the complete planar embedding.
"""
from copy import deepcopy
from itertools import combinations
import time

import numpy as np

from .bundled_crossings import overlaps_visible_ink, stable_tangent
from .knotfolio_backend import _intersections, _segments_cross


def repair_long_underpasses(trace):
    """Return ``(trace, evidence)``; leave the input unchanged on refusal.

    Each endpoint must have exactly one supported partner among the remaining
    tips. A continuation has to be straight at two tangent scales and cross
    exactly one uninterrupted visible strand, well inside the gap. Ordinary
    gaps, ambiguous tip choices, open tangles and detached labels are untouched.
    """
    diagnostics = trace.get('diagnostics', {})
    tips = diagnostics.get('unmatched_endpoints', [])
    if (trace.get('complete') or diagnostics.get('branch_points')
            or diagnostics.get('gap_conflicts') or not 2 <= len(tips) <= 16
            or len(tips) % 2):
        return trace, []
    width = float(diagnostics.get('stroke_width', 0.))
    if not np.isfinite(width) or width <= 0:
        return trace, []
    paths = trace['paths']
    arrays = [np.asarray(path['points'], float) for path in paths]
    starts, ends, owners = [], [], []
    for index, points in enumerate(arrays):
        starts.extend(points[:-1]); ends.extend(points[1:])
        owners.extend([index] * (len(points)-1))
    starts, ends, owners = np.asarray(starts), np.asarray(ends), np.asarray(owners)
    if not len(starts):
        return trace, []
    extent = np.ptp(np.vstack(arrays), axis=0)
    limit = min(60*width, .3*float(np.linalg.norm(extent)))
    existing = []
    for match in trace['matches']:
        points = match.get('points')
        if points is None:
            points = [arrays[match[key][0]][0 if match[key][1] == 0 else -1]
                      for key in ('a', 'b')]
        existing.extend(zip(points[:-1], points[1:]))
    candidates = {}
    choices = [0] * len(tips)
    for i, j in combinations(range(len(tips)), 2):
        first, second = tips[i], tips[j]
        if paths[first['path']].get('color') != paths[second['path']].get('color'):
            continue
        a, b = np.asarray(first['point'], float), np.asarray(second['point'], float)
        delta = b-a; length = float(np.linalg.norm(delta))
        if not 18*width < length <= limit:
            continue
        direction = delta/length
        alignment = []
        for reach in (2.5*width, min(.2*length, 8*width)):
            ta = stable_tangent(arrays[first['path']], first['end'], reach)
            tb = stable_tangent(arrays[second['path']], second['end'], reach)
            alignment.append([float(direction @ ta), float(-direction @ tb)])
        if min(min(pair) for pair in alignment) < .86:
            continue
        hits = _intersections(a, b, starts, ends, owners, width)
        if (len(hits) != 1 or hits[0]['sine_angle'] < .45
                or not .15 <= hits[0]['gap_t'] <= .85):
            continue
        if overlaps_visible_ink(a, b, starts, ends, width):
            continue
        # A hidden underpass cannot cross a second hidden segment: there is no
        # visible evidence establishing which of those two would be on top.
        if any(_segments_cross(a, b, np.asarray(c), np.asarray(d)) for c, d in existing):
            continue
        record = {'a': [first['path'], first['end']],
                  'b': [second['path'], second['end']],
                  'distance': length, 'cost': length/width,
                  'alignment': alignment[-1],
                  'crossings': [{**hits[0], 'point': hits[0]['point'].tolist()}],
                  'long_underpass': True}
        candidates[i, j] = record
        choices[i] += 1; choices[j] += 1
    if any(count != 1 for count in choices):
        return trace, []
    proposals = list(candidates.values())
    segments = [(np.asarray(tips[i]['point']), np.asarray(tips[j]['point']))
                for i, j in candidates]
    if any(_segments_cross(a, b, c, d)
           for (a, b), (c, d) in combinations(segments, 2)):
        return trace, []
    result = deepcopy(trace)
    evidence = [{'ends': [a.tolist(), b.tolist()],
                 'crossing': match['crossings'][0]['point'],
                 'distance_widths': match['distance']/width,
                 'alignment': match['alignment']}
                for (a, b), match in zip(segments, proposals)]
    result['matches'].extend(proposals)
    result['complete'] = True
    result['diagnostics']['unmatched_endpoints'] = []
    result['diagnostics']['long_underpasses'] = evidence
    return result, evidence


def recognize_long_underpasses(rgb):
    """Review-only last resort for a failed, flat drawing; return None otherwise.

    Existing successful recognitions must take precedence over this routine.
    Two threshold masks must reconstruct the same diagram, and every complete
    candidate must agree. Photographs belong to the board-mask pipeline.
    """
    from .board_photo import looks_photographic
    from .diagram import pd_code, validate
    from .geometry import assemble, beautify, check_embedding
    from .knotfolio_backend import extract
    from .motion import _same_projection
    from .preprocess import prepare

    started = time.monotonic()
    if looks_photographic(rgb)[0]:
        return None
    pixels, preprocessing = prepare(rgb)
    complete, attempts = [], []
    for threshold in (150., 180., 205., 230.):
        trace = extract(pixels, {'threshold': threshold, 'arrow_wings': True,
                               'micro_seams': True, 'stop_at_branches': True,
                               'max_endpoints': 180})
        repaired, evidence = repair_long_underpasses(trace)
        record = {'threshold': threshold, 'repairs': len(evidence)}
        attempts.append(record)
        if not evidence:
            continue
        try:
            diagram = beautify(assemble(repaired, rgb.shape[1], rgb.shape[0]))
            verdict = validate(diagram)
            embedding = check_embedding(diagram, check_strokes=False)
            if not verdict['valid'] or not embedding['valid']:
                record['invalid_embedding'] = True
                continue
            code = pd_code(diagram)
            complete.append((diagram, repaired, verdict, code))
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            record['error'] = str(exc)
    if len(complete) < 2:
        return None
    diagram, trace, verdict, code = complete[0]
    if any(not _same_projection(diagram, candidate[0]) for candidate in complete[1:]):
        return None
    warning = ('Unusually long underpass gaps were joined using aligned strand '
               'tips and a visible over-strand. Inspect every marked crossing.')
    warnings = [warning]
    if trace['diagnostics'].get('arrow_pruning'):
        warnings.append('Paired arrow wings were separated from the strand trace; '
                        'their source directions are retained for orientation review.')
    if trace['diagnostics'].get('micro_seam_repairs'):
        warnings.append('Sub-width pen-tip seams were joined; inspect these repairs.')
    diagnostic = trace['diagnostics']
    diagnostic.update(validation=verdict, preprocessing=preprocessing,
                      long_underpass_attempts=attempts,
                      long_underpass_agreement=len(complete),
                      elapsed_seconds=time.monotonic()-started)
    diagram['recognition'] = {'stroke_width': diagnostic['stroke_width'],
                              'warnings': warnings}
    return {'status': 'needs_review', 'diagram': diagram, 'pd_code': code,
            'unlinked_unknot_components': verdict['crossing_free_components'],
            'diagnostics': diagnostic, 'warnings': warnings}
