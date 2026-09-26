"""Conservative gap evidence for imperfect, photographed strand drawings.

Every accepted gap must be supported before the global matching step. A
parallel bundle can explain a long hidden strand; merely intersecting several
unrelated strokes cannot. Digital drawings retain their separate gap policy.
"""
from __future__ import annotations


# The cost of leaving one end open. This lets matching prefer one well-supported
# join to two poor joins instead of treating closure as its first objective.
UNMATCHED_ENDPOINT_COST = 12.0


def photo_gap_evidence(length, width, alignment, hits, bundle):
    """Return (support, rejection reason) for a crossing-bearing photo gap.

    Margins are measured along the candidate strand in stroke widths. A single
    over-strand supplies only local evidence, whereas a verified parallel
    bundle explains the entire interval between its outermost strands.
    """
    if not hits:
        return None, None  # Noncrossing and micro-seam rules are separate.
    facing = min(alignment)
    scale = length / max(width, 1e-12)
    if len(hits) == 1:
        hit = hits[0]
        fraction = float(hit['gap_t'])
        margins = [fraction * length, (1 - fraction) * length]
        # A short gap can be asymmetric, especially after a touching arrow is
        # separated from an over-strand. Use actual ink scale at the nearer
        # shoulder; a fixed fraction incorrectly rejects those local gaps.
        if min(margins) < min(.8 * width, .2 * length) or hit['sine_angle'] < .3:
            return None, 'nonlocal_single_crossing'
        curved = (scale <= 50 and facing >= .45 and max(alignment) >= .9
                  and sum(alignment) >= 1.4 and .2 <= fraction <= .8
                  and hit['sine_angle'] >= .75)
        if scale > 25:
            # A long single underpass has no interior bundle to explain it.
            # It needs two nearly collinear, facing endpoint directions, a
            # central crossing and a nearly transverse over-strand instead.
            # Its full unsupported length still counts in the matching cost.
            aligned = facing >= .9 and .25 <= fraction <= .75
            # A hand-drawn approach can curve into the gap. A moderately
            # facing end alone is insufficient: the other must point almost
            # straight at it, their combined alignment must remain strong,
            # and one transverse crossing must explain the central interval.
            if hit['sine_angle'] < .75 or not .2 <= fraction <= .8:
                return None, 'nonlocal_single_crossing'
            if not aligned and not curved:
                return None, 'unsupported_single_crossing_length'
            return dict(kind='aligned_single_crossing' if aligned else 'curved_approach_single_crossing',
                        margins=margins), None
        if facing < .65:
            if curved:
                return dict(kind='curved_approach_single_crossing', margins=margins), None
            return None, 'endpoint_direction'
        return dict(kind='local_single_crossing', margins=margins), None
    if facing < .65:
        return None, 'endpoint_direction'
    if bundle is None:
        return None, 'unverified_bundle'
    if bundle['parallelism'] < .85 or min(h['sine_angle'] for h in hits) < .4:
        return None, 'nonparallel_bundle'
    # Wide bundle spacing cannot justify arbitrary blank tails on either side.
    # Interior spacing is unrestricted: real long bundles must remain possible.
    if max(bundle['margins']) > min(12 * width, max(3 * width, 1.3 * bundle['spacing'])):
        return None, 'unsupported_bundle_margin'
    return dict(kind='parallel_bundle', margins=bundle['margins']), None
