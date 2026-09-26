"""Reserve isolated local pen seams before endpoints enter gap matching.

Overlapping tips point past one another, so a straight-gap direction score
rejects them. Repairing them only after matching is too late: a distant chord
may already have consumed an endpoint. Here every proposed local seam needs
source ink, antiparallel overlap, isolation, and a unique partner at each end.
"""
from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree

from .recognition_runtime import checkpoint


def reserve_local_seams(data, source_mask, tangent_widths=3., max_gap_widths=3.):
    """Return owned trace data, reserved matches, and source-mask evidence.

    Only branch-free traces qualify. Visible path indices and endpoint refs
    remain stable; reserved endpoints leave the ordinary matching pool. The
    ordinary short noncrossing-gap scale bounds the transverse connector,
    with at least one stroke width accounted for by its two ink shoulders.
    This does not authorize a long underpass or an over/under choice.
    """
    from .board_photo import repair_overdrawn_seam
    from .knotfolio_backend import _outward_tangent

    if data['branches'] or len(data['endpoints']) < 2:
        return data, [], []
    width = float(data['stroke_width'])
    endpoints = data['endpoints']
    points = np.array([e['point'] for e in endpoints])
    neighbors = cKDTree(points).query_pairs(11*width)
    lengths = [float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()) for a in data['arrays']]
    current_paths = data['paths']
    matches, evidence = [], []

    def proposal(i, j):
        a, b = endpoints[i], endpoints[j]
        if min(lengths[a['path']], lengths[b['path']]) < 12*width:
            return None
        ends = [dict(path=e['path'], end=e['end'], point=np.asarray(e['point']).tolist())
                for e in (a, b)]
        trace = dict(paths=current_paths, matches=matches, complete=False,
                     diagnostics=dict(stroke_width=width, branch_points=[], gap_conflicts=[],
                                      unmatched_endpoints=ends))
        repaired = repair_overdrawn_seam(trace, allow_same_path=True, source_mask=source_mask,
                                        seam_widths=(max_gap_widths, max(0., max_gap_widths-1.)))
        return repaired if repaired is not trace else None

    # A geometric ambiguity is not resolved by a cheaper pair or enumeration
    # order. Reserve only mutually unique local partners.
    proposals, degree = [], np.zeros(len(endpoints), dtype=int)
    for i, j in sorted(neighbors):
        checkpoint('checking local pen seams')
        if proposal(i, j) is not None:
            proposals.append((i, j))
            degree[i] += 1
            degree[j] += 1
    reserved = set()
    for i, j in proposals:
        if degree[i] != 1 or degree[j] != 1:
            continue
        # Recheck isolation after previously accepted disjoint edits. This
        # also handles two safe seams on opposite ends of the same path.
        repaired = proposal(i, j)
        if repaired is None:
            continue
        current_paths, matches = repaired['paths'], repaired['matches']
        record = deepcopy(repaired['diagnostics']['chalk_seam_evidence'])
        record['original_tips'] = repaired['diagnostics']['seam_repairs']
        record['endpoints'] = [[endpoints[k]['path'], endpoints[k]['end']] for k in (i, j)]
        record['same_path'] = endpoints[i]['path'] == endpoints[j]['path']
        record['center_limit_widths'] = max_gap_widths
        record['blank_limit_widths'] = max(0., max_gap_widths-1.)
        evidence.append(record)
        reserved.update((i, j))
    if not matches:
        return data, [], []
    result = deepcopy(data)
    result['paths'] = current_paths
    result['arrays'] = [np.asarray(p['points'], dtype=float) for p in current_paths]
    result['endpoints'] = []
    for i, endpoint in enumerate(endpoints):
        if i in reserved:
            continue
        item = deepcopy(endpoint)
        curve = result['arrays'][item['path']]
        item['point'] = curve[0 if item['end'] == 0 else -1].copy()
        item['tangent'] = _outward_tangent(curve, item['end'], tangent_widths*width)
        result['endpoints'].append(item)
    return result, matches, evidence
