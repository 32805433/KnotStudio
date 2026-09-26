"""Exact comparison of classical crossing rotation systems.

A PD has two distinguished dart permutations: the cyclic rotation at a
crossing and the involution pairing the ends of each arc. Once one dart is
matched, those permutations force the entire connected component's mapping.
This avoids a general-purpose graph-isomorphism search during a drag.
"""
from collections import Counter


def _dart_arcs(pd):
    arcs = [None] * (4 * len(pd))
    first = {}
    for crossing, row in enumerate(pd):
        for port, label in enumerate(row):
            dart = 4 * crossing + port
            if label in first:
                other = first.pop(label)
                arcs[dart] = other
                arcs[other] = dart
            else:
                first[label] = dart
    if first or any(x is None for x in arcs):
        raise ValueError('Each PD arc label must occur twice.')
    return arcs


def same_pd_projection(left_pd, right_pd):
    """Match rotation, under/over, and arc incidence, ignoring arc names.

    The caller separately accounts for crossing-free components. This is the
    same colored directed multigraph equivalence as rotation edges, symmetric
    arc edges, and the alternating under/over color at each crossing. It does
    not use drawing coordinates, component IDs, or heuristic matching.
    """
    if len(left_pd) != len(right_pd):
        return False
    if left_pd == right_pd:
        return True
    left, right = _dart_arcs(left_pd), _dart_arcs(right_pd)
    count = len(left)
    rotate = [(i & ~3) + ((i + 1) & 3) for i in range(count)]
    reverse = [(i & ~3) + ((i - 1) & 3) for i in range(count)]
    colors_a = [i & 1 for i in range(count)]
    colors_b = colors_a.copy()
    # Refinement only rejects provably different graphs or orders candidate
    # roots. Four rounds have bounded linear cost even on large symmetric
    # alternating projections; exact verification below remains mandatory.
    for _ in range(4):
        codes = {}
        def refine(arcs, colors):
            return [codes.setdefault((colors[i], colors[rotate[i]],
                                      colors[reverse[i]], colors[arcs[i]]), len(codes))
                    for i in range(count)]
        newer_a, newer_b = refine(left, colors_a), refine(right, colors_b)
        frequencies = Counter(newer_a)
        if frequencies != Counter(newer_b):
            return False
        stable = len(set(colors_a)) == len(frequencies)
        colors_a, colors_b = newer_a, newer_b
        if stable:
            break
    unseen_a, unseen_b = set(range(count)), set(range(count))
    while unseen_a:
        root = min(unseen_a, key=lambda dart: frequencies[colors_a[dart]])
        candidates = [dart for dart in unseen_b if colors_b[dart] == colors_a[root]]
        for target in candidates:
            forward, backward = {}, {}
            pending = [(root, target)]
            valid = True
            while pending:
                a, b = pending.pop()
                if a in forward:
                    if forward[a] != b:
                        valid = False
                        break
                    continue
                if (a not in unseen_a or b not in unseen_b or b in backward
                        or colors_a[a] != colors_b[b]):
                    valid = False
                    break
                forward[a] = b
                backward[b] = a
                pending.extend(((rotate[a], rotate[b]), (left[a], right[b])))
            if valid:
                unseen_a.difference_update(forward)
                unseen_b.difference_update(backward)
                break
        else:
            return False
    return not unseen_b
