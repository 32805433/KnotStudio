"""Pixels-only raster tracing and crossing-gap reconstruction.

Copyright (C) 2026 contributors to this project.
SPDX-License-Identifier: GPL-2.0-or-later

The pipeline adapts ideas and control flow from Kyle Miller's KnotFolio,
Copyright (C) 2019 Kyle Miller, commit
a2cd207e7be5b3da384765bdfe7a818d7e3eaa4a (GPL-2.0-or-later).
This is a new Python implementation, with global matching and tangent scores.
See ../licenses/KnotFolio-GPL-2.0.md and ../docs/THIRD_PARTY.md.

No filenames, reference diagrams, or image metadata are accepted by this API.
Visible paths are over-strands wherever reconstructed gap segments cross them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import hypot
from typing import Any

import cv2
import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from .recognition_runtime import current, checkpoint, remaining, array_key, RecognitionDeadline
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from skimage.morphology import skeletonize


@dataclass
class Options:
    threshold: float = 205.0
    adaptive: bool = False
    min_object_pixels: int = 6
    spur_widths: float = 2.5
    max_gap_widths: float = 18.0
    tangent_widths: float = 2.5
    tangent_weight: float = 7.0
    distance_weight: float = 0.40
    color_weight: float = 2.0
    min_tangent_dot: float = -0.2
    noncrossing_max_widths: float = 3.0
    noncrossing_min_alignment: float = 0.65
    simplify_epsilon: float = 0.45
    include_skeleton: bool = False
    separate_colors: bool = True
    opening_widths: float = 0.0
    stop_at_branches: bool = False
    max_endpoints: int | None = None
    bundle_gaps: bool = False
    micro_seams: bool = False
    arrow_wings: bool = False
    adaptive_arrow_wings: bool = False
    photo_gap_policy: bool = False


def _adjacency(skeleton: np.ndarray, colors=None) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """8-connectivity without redundant diagonal edges around pixel corners."""
    skeleton = skeleton.copy()
    # Some digital skeletons retain a solid 2x2 cell at an arrow junction.
    # That cell makes a spurious four-edge cycle in the pixel graph. Remove
    # only a corner with no neighbors outside that cell, preserving all arms.
    cells = np.argwhere(skeleton[:-1, :-1] & skeleton[1:, :-1]
                        & skeleton[:-1, 1:] & skeleton[1:, 1:])
    for y, x in cells:
        square = {(y, x), (y + 1, x), (y, x + 1), (y + 1, x + 1)}
        if not all(skeleton[p] for p in square):
            continue
        if colors is not None and len({colors[p] for p in square}) != 1:
            continue
        for point in sorted(square):
            py, px = point
            external = [(py + dy, px + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if (py + dy, px + dx) not in square
                        and 0 <= py + dy < skeleton.shape[0]
                        and 0 <= px + dx < skeleton.shape[1]
                        and skeleton[py + dy, px + dx]
                        and not (dy and dx and (skeleton[py + dy, px] or skeleton[py, px + dx]))]
            if not external:
                skeleton[point] = False
                break
    y, x = np.nonzero(skeleton)
    # Retain the old set iteration order (and sorted neighbor order), since
    # traversal order can break otherwise equal matching scores differently.
    # Native ints have the same hashes/order as NumPy integer coordinates, but
    # make graph walking and per-attempt cache serialization much cheaper.
    pixels = set(zip(y.tolist(), x.tolist()))
    graph = {point: [] for point in pixels}
    padded = np.pad(skeleton, 1)
    color_pad = np.pad(colors, 1) if colors is not None else None
    center_color = colors[y, x] if colors is not None else None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            valid = padded[y+dy+1, x+dx+1].copy()
            if colors is not None:
                valid &= center_color == color_pad[y+dy+1, x+dx+1]
            if dy and dx:
                vertical = padded[y+dy+1, x+1]
                horizontal = padded[y+1, x+dx+1]
                if colors is not None:
                    vertical = vertical & (center_color == color_pad[y+dy+1, x+1])
                    horizontal = horizontal & (center_color == color_pad[y+1, x+dx+1])
                valid &= ~(vertical | horizontal)
            for py, px in zip(y[valid].tolist(), x[valid].tolist()):
                graph[(py, px)].append((py+dy, px+dx))
    return graph


def _color_layers(rgb, mask):
    """Cluster normalized chroma, invariant to white-background antialiasing."""
    labels = np.zeros(mask.shape, dtype=np.uint8)
    pixels = rgb[mask]
    if not len(pixels):
        return labels, 0
    chroma = pixels - pixels.min(axis=1, keepdims=True)
    ranges = chroma.max(axis=1, keepdims=True)
    chroma /= np.maximum(20, ranges)
    chroma[ranges[:, 0] < 20] = 0
    # The 3D bins have only 5^3 possibilities. Packed integers have precisely
    # the same lexicographic order as unique(..., axis=0), without row sorting.
    quantized = np.round(chroma * 4).astype(np.uint8)
    packed = quantized[:, 0]*25 + quantized[:, 1]*5 + quantized[:, 2]
    values, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    bins = np.column_stack((values//25, values//5 % 5, values % 5))
    centers = []
    for index in np.argsort(-counts):
        signature = bins[index] / 4.0
        if counts[index] < max(8, len(pixels) * 0.002) and centers:
            continue
        if all(np.linalg.norm(signature - center) > 0.36 for center in centers):
            centers.append(signature)
        if len(centers) == 12:
            break
    distances = np.linalg.norm(bins[:, None, :] / 4.0 - np.array(centers)[None, :, :], axis=2)
    labels[mask] = np.argmin(distances, axis=1)[inverse] + 1
    return labels, len(centers)


def _prune_spurs(graph, max_length):
    removed = 0
    while True:
        deleting = set()
        for start, neighbors in list(graph.items()):
            if len(neighbors) != 1:
                continue
            chain = [start]
            previous, point = start, neighbors[0]
            length = hypot(point[0] - start[0], point[1] - start[1])
            while len(graph[point]) == 2 and length <= max_length:
                chain.append(point)
                neighbors = graph[point]
                following = neighbors[1] if neighbors[0] == previous else neighbors[0]
                length += hypot(following[0] - point[0], following[1] - point[1])
                previous, point = point, following
            if len(graph[point]) > 2 and length <= max_length:
                deleting.update(chain)
        if not deleting:
            return removed
        for point in deleting:
            for neighbor in graph.pop(point, []):
                if neighbor in graph and point in graph[neighbor]:
                    graph[neighbor].remove(point)
        removed += len(deleting)


def _remove_pixel_cycles(graph, stroke_width):
    """Remove sub-stroke digital loops inside branched medial-axis junctions.

    A genuine isolated closed curve (all degrees two) is always preserved.
    These tiny cycles occur in filled arrowheads and solid 2x2 pixel cells.
    """
    if not any(len(neighbors) > 2 for neighbors in graph.values()):
        return 0
    removed = 0
    for _ in range(4):
        changed = False
        checkpoint('cleaning pixel junctions')
        for cycle in nx.cycle_basis(_PixelGraphView(graph)):
            if not any(len(graph[p]) > 2 for p in cycle):
                continue
            points = np.array(cycle)
            if max(np.ptp(points, axis=0)) > min(8.0, 1.2 * stroke_width):
                continue
            edges = list(zip(cycle, cycle[1:] + cycle[:1]))
            if not all(b in graph[a] for a, b in edges):
                continue
            a, b = max(edges, key=lambda e: len(graph[e[0]]) + len(graph[e[1]]))
            graph[a].remove(b); graph[b].remove(a)
            changed = True; removed += 1
        if not changed:
            break
    return removed


class _PixelGraphView:
    """Just the ordered adjacency used by cycle_basis; no edge attributes.

    Match nx.Graph(dict_of_lists)'s insertion order exactly. Changing its DFS
    order could select a different tiny cycle to remove at an ink junction.
    """
    def __init__(self, graph):
        self.adj = {point: {} for point in graph}
        for point, neighbors in graph.items():
            for neighbor in neighbors:
                self.adj[point][neighbor] = None
                self.adj[neighbor][point] = None

    def __iter__(self):
        return iter(self.adj)

    def __getitem__(self, node):
        return self.adj[node]

    def is_directed(self):
        return False

    def is_multigraph(self):
        return False


def _trace(graph):
    """Trace maximal paths; branch nodes are retained for diagnostics."""
    visited = set()
    paths = []

    def edge(a, b):
        return (a, b) if a < b else (b, a)

    def walk(start, second):
        chain = [start]
        previous, point = start, second
        visited.add(edge(start, second))
        while True:
            chain.append(point)
            if point == start or len(graph[point]) != 2:
                break
            neighbors = graph[point]
            following = neighbors[1] if neighbors[0] == previous else neighbors[0]
            if edge(point, following) in visited:
                break
            visited.add(edge(point, following))
            previous, point = point, following
        return chain

    for start in sorted(graph):
        if len(graph[start]) == 2:
            continue
        for second in graph[start]:
            if edge(start, second) not in visited:
                paths.append(walk(start, second))
    for start in sorted(graph):
        for second in graph[start]:
            if edge(start, second) not in visited:
                paths.append(walk(start, second))
    return paths


def _outward_tangent(points, end, distance):
    points = points if end == 0 else points[::-1]
    length = 0.0
    i = 1
    while i < len(points):
        length += np.linalg.norm(points[i] - points[i - 1])
        if length >= distance:
            break
        i += 1
    i = min(i, len(points) - 1)
    tangent = points[0] - points[i]
    norm = np.linalg.norm(tangent)
    return tangent / norm if norm > 0 else np.array([1.0, 0.0])


def _intersections(a, b, segments_a, segments_b, owners, width):
    """Intersections strictly inside the candidate gap and visible segments."""
    v = b - a
    w = segments_b - segments_a
    delta = segments_a - a
    denominator = v[0] * w[:, 1] - v[1] * w[:, 0]
    stable = abs(denominator) > 1e-8
    safe = np.where(stable, denominator, 1.0)
    t = (delta[:, 0] * w[:, 1] - delta[:, 1] * w[:, 0]) / safe
    u = (delta[:, 0] * v[1] - delta[:, 1] * v[0]) / safe
    length = np.linalg.norm(v)
    margin = min(0.24, max(0.015, 0.5 * width / max(length, 1.0)))
    hits = np.flatnonzero(stable & (t > margin) & (t < 1 - margin)
                          & (u >= -1e-7) & (u <= 1 + 1e-7))
    result = []
    for index in sorted(hits, key=lambda k: t[k]):
        point = a + t[index] * v
        if result and np.linalg.norm(point - result[-1]["point"]) < 1.5:
            continue
        angle = abs(denominator[index]) / max(length * np.linalg.norm(w[index]), 1e-8)
        result.append({"point": point, "path": int(owners[index]),
                       "gap_t": float(t[index]), "sine_angle": float(angle)})
    return result


def _segments_cross(a, b, c, d):
    v, w = b - a, d - c
    cross = v[0] * w[1] - v[1] * w[0]
    if abs(cross) < 1e-8:
        return False
    delta = c - a
    t = (delta[0] * w[1] - delta[1] * w[0]) / cross
    u = (delta[0] * v[1] - delta[1] * v[0]) / cross
    return 1e-5 < t < 1 - 1e-5 and 1e-5 < u < 1 - 1e-5


def _noncrossing_matching(matching_graph, endpoints, candidates, *, unmatched_cost=None):
    """Blossom first; add planar conflict cuts only when they are necessary."""
    checkpoint('matching strand ends')
    if unmatched_cost is None:
        pairs = sorted(tuple(sorted(pair)) for pair in nx.min_weight_matching(matching_graph, weight="weight"))
    else:
        # An unmatched photographic endpoint is preferable to an unsupported
        # closure. Benefits include the two avoided unmatched-end penalties;
        # negative-benefit edges can simply remain unused.
        optional = matching_graph.copy()
        for _, _, edge in optional.edges(data=True):
            edge['benefit'] = 2*unmatched_cost-edge['weight']
        pairs = sorted(tuple(sorted(pair)) for pair in nx.max_weight_matching(
            optional, maxcardinality=False, weight='benefit'))
    checkpoint('checking strand connections')
    if unmatched_cost is None and len(pairs) * 2 != len(endpoints):
        return pairs, 0
    keys = list(candidates)
    key_index = {key: index for index, key in enumerate(keys)}
    forbidden = set()
    revisions = 0
    for _ in range(10):
        checkpoint('resolving crossing conflicts')
        conflicts = []
        for k, (i, j) in enumerate(pairs):
            for a, b in pairs[:k]:
                if _segments_cross(endpoints[i]["point"], endpoints[j]["point"],
                                   endpoints[a]["point"], endpoints[b]["point"]):
                    conflicts.append(tuple(sorted((key_index[i, j], key_index[a, b]))))
        if not conflicts:
            return pairs, revisions
        forbidden.update(conflicts)
        row_indices, col_indices = [], []
        for column, (i, j) in enumerate(keys):
            row_indices.extend((i, j)); col_indices.extend((column, column))
        for row, (a, b) in enumerate(sorted(forbidden), len(endpoints)):
            row_indices.extend((row, row)); col_indices.extend((a, b))
        matrix = coo_matrix((np.ones(len(row_indices)), (row_indices, col_indices)),
                            shape=(len(endpoints) + len(forbidden), len(keys))).tocsc()
        lower = np.r_[np.ones(len(endpoints)) if unmatched_cost is None else np.zeros(len(endpoints)),
                      np.zeros(len(forbidden))]
        upper = np.ones(len(endpoints) + len(forbidden))
        solver_seconds = max(.001, remaining(1.0))
        result = milp(np.array([candidates[key]["cost"]-(2*unmatched_cost if unmatched_cost is not None else 0)
                               for key in keys]),
                      integrality=np.ones(len(keys)), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={"time_limit": solver_seconds})
        checkpoint('resolving crossing conflicts')
        if solver_seconds < 1.0 and result.status == 1:
            # A shortened solver run may expose a different, unfinished
            # incumbent. Report the budget stop instead of adopting its PD.
            raise RecognitionDeadline('resolving crossing conflicts')
        if result.x is None:
            break
        pairs = sorted(keys[i] for i in np.flatnonzero(result.x > 0.5))
        revisions += 1
    return pairs, revisions


def _trace_pixels(rgb, opts):
    height, width = rgb.shape[:2]
    darkness = rgb.min(axis=2)
    if opts.adaptive:
        background = ndimage.gaussian_filter(darkness, 16.0)
        mask = (darkness < opts.threshold) & (darkness < background - 8)
    else:
        mask = darkness < opts.threshold
    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    mask &= sizes[labels] >= opts.min_object_pixels
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    state = current()
    trace_options = tuple((name, getattr(opts, name)) for name in (
        'separate_colors', 'opening_widths', 'spur_widths', 'arrow_wings',
        'adaptive_arrow_wings', 'simplify_epsilon', 'tangent_widths', 'include_skeleton'))
    rgb_key, mask_key = (array_key(rgb), array_key(mask)) if state else (None, None)
    key = ('trace', rgb_key, mask_key, trace_options) if state else None
    if state:
        cached = state.get(key)
        if cached is not None:
            return cached, key
    # Threshold/spur/tangent retries frequently share exactly the same ink.
    # Cache the immutable pre-pruning graph separately from full trace results;
    # Runtime serializes on put/get so no arrow/spur cleanup can contaminate
    # another attempt. Mask and RGB hashes, not threshold names, define reuse.
    ink_key = ('ink', rgb_key, mask_key, opts.separate_colors, opts.opening_widths)
    ink = state.get(ink_key) if state else None
    if ink is None:
        checkpoint('tracing ink')
        color_labels, color_count = _color_layers(rgb, mask) if opts.separate_colors else (mask.astype(np.uint8), 1)
        skeleton = np.zeros_like(mask)
        for color in range(1, color_count + 1):
            skeleton |= skeletonize(color_labels == color)
        radii = ndimage.distance_transform_edt(mask)[skeleton]
        stroke_width = max(1.5, float(np.median(radii) * 2)) if len(radii) else 1.5
        opening_radius = 0
        if opts.opening_widths > 0:
            # Thin arrow wings can occlude another component in a colored image.
            # Opening each color independently removes those narrow appendages;
            # opening the union would preserve them because the other color fills
            # their neighboring pixels. Visible stroke gaps are reconstructed below.
            opening_radius = max(1, int(round(stroke_width * opts.opening_widths)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (2 * opening_radius + 1, 2 * opening_radius + 1))
            kept = np.zeros_like(mask)
            for color in range(1, color_count + 1):
                kept |= cv2.morphologyEx((color_labels == color).astype(np.uint8),
                                         cv2.MORPH_OPEN, kernel).astype(bool)
            mask &= kept
            color_labels[~mask] = 0
            skeleton[:] = False
            for color in range(1, color_count + 1):
                skeleton |= skeletonize(color_labels == color)
        graph = _adjacency(skeleton, color_labels)
        ink = dict(mask=mask, color_labels=color_labels, color_count=color_count,
                   skeleton=skeleton, stroke_width=stroke_width,
                   opening_radius=opening_radius, graph=graph)
        if state:
            state.put(ink_key, ink)
    else:
        mask, color_labels = ink['mask'], ink['color_labels']
        color_count, skeleton = ink['color_count'], ink['skeleton']
        stroke_width, opening_radius = ink['stroke_width'], ink['opening_radius']
        graph = ink['graph']
    arrow_evidence = []
    if opts.arrow_wings:
        from .handdrawn_arrows import prune_arrow_wings
        arrow_evidence = prune_arrow_wings(graph, stroke_width)
    if opts.adaptive_arrow_wings:
        from .arrow_recovery import prune_curved_arrow_wings
        arrow_evidence += prune_curved_arrow_wings(graph, stroke_width)
    removed = _prune_spurs(graph, opts.spur_widths * stroke_width)
    # Rebuilding unchanged pixel adjacency is redundant. Retain the second
    # pass where pruning or a 2x2 digital cell could actually change it.
    has_cells = np.any(skeleton[:-1, :-1] & skeleton[1:, :-1]
                       & skeleton[:-1, 1:] & skeleton[1:, 1:])
    if removed or arrow_evidence or has_cells:
        pruned_skeleton = np.zeros_like(skeleton)
        for pixel in graph:
            pruned_skeleton[pixel] = True
        graph = _adjacency(pruned_skeleton, color_labels)
        removed += _prune_spurs(graph, opts.spur_widths * stroke_width)
    cycles_removed = _remove_pixel_cycles(graph, stroke_width)
    removed += _prune_spurs(graph, opts.spur_widths * stroke_width)
    branches = [p for p in graph if len(graph[p]) > 2]
    chains = _trace(graph)
    paths, arrays, path_colors = [], [], []
    color_centers = []
    for chain in chains:
        if len(chain) < 2:
            continue
        closed = chain[0] == chain[-1]
        points = np.array([(x, y) for y, x in chain], dtype=np.float32)
        sample_rgb = np.median(np.array([rgb[y, x, :3] for y, x in chain]), axis=0)
        # A normalized chromatic signature; grayscale remains neutral.
        signature = sample_rgb - sample_rgb.min()
        signature /= max(40.0, float(signature.max()))
        color = next((i for i, center in enumerate(color_centers)
                      if np.linalg.norm(signature - center) < 0.4), len(color_centers))
        if color == len(color_centers):
            color_centers.append(signature)
        simplified = cv2.approxPolyDP(points, opts.simplify_epsilon, closed).reshape(-1, 2)
        if closed and not np.array_equal(simplified[0], simplified[-1]):
            simplified = np.vstack([simplified, simplified[0]])
        if len(simplified) < 2:
            continue
        arrays.append(simplified.astype(float))
        path_colors.append(signature)
        paths.append({"points": simplified.tolist(), "closed": closed, "color": color})

    endpoints = []
    for index, (path, points) in enumerate(zip(paths, arrays)):
        if path["closed"]:
            continue
        for end in (0, 1):
            point = points[0 if end == 0 else -1]
            pixel = tuple(np.round(point[::-1]).astype(int))
            if len(graph.get(pixel, [])) != 1:
                continue
            endpoints.append({"path": index, "end": end, "point": point,
                              "tangent": _outward_tangent(points, end, opts.tangent_widths * stroke_width)})

    data = dict(width=width, height=height, paths=paths, arrays=arrays,
                path_colors=path_colors, endpoints=endpoints, branches=branches,
                arrow_evidence=arrow_evidence, stroke_width=stroke_width,
                color_count=color_count, opening_radius=opening_radius,
                removed=removed, cycles_removed=cycles_removed,
                foreground_pixels=int(mask.sum()), skeleton_pixels=len(graph),
                skeleton_points=[[int(x), int(y)] for y, x in sorted(graph)]
                    if opts.include_skeleton else None)
    if state:
        state.put(key, data)
    return data, key


def extract(rgb: np.ndarray, options: dict[str, Any] | Options | None = None) -> dict:
    """Return visible paths and matched endpoint gaps from an RGB image.

    Paths: {points: [[x,y], ...], closed: bool, color: int}.
    Matches: {a: [path_index, end], b: [path_index, end], ...}; end is 0
    for the first and 1 for the last point. A match is an under-strand.
    Check ``complete`` before accepting the output as a closed link diagram.
    All coordinates are in the original pixel coordinate system.
    """
    opts = options if isinstance(options, Options) else Options(**(options or {}))
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB, RGBA, or grayscale array")
    if not rgb.size or not np.isfinite(rgb).all():
        raise ValueError("Expected a nonempty image with finite pixel values")
    normalized_float = np.issubdtype(rgb.dtype, np.floating) and 0 <= float(rgb.min()) <= float(rgb.max()) <= 1
    rgb = rgb.astype(np.float32)
    if normalized_float:
        rgb *= 255
    if rgb.shape[2] == 4:
        alpha = rgb[:, :, 3:4] / 255.0
        rgb = rgb[:, :, :3] * alpha + 255 * (1 - alpha)
    checkpoint('preparing trace')
    data, trace_key = _trace_pixels(rgb, opts)
    state = current()
    photo_source_mask = rgb.min(2) < opts.threshold if opts.photo_gap_policy else None
    key = ('extract', trace_key, tuple((k, v) for k, v in asdict(opts).items()
                                      if k not in ('threshold', 'adaptive', 'min_object_pixels')),
           array_key(photo_source_mask) if opts.photo_gap_policy else None) if state else None
    if state:
        cached = state.get(key)
        if cached is not None:
            cached['diagnostics']['threshold'] = opts.threshold
            return cached
    width, height = data['width'], data['height']
    paths, arrays, path_colors = data['paths'], data['arrays'], data['path_colors']
    endpoints, branches = data['endpoints'], data['branches']
    arrow_evidence, stroke_width = data['arrow_evidence'], data['stroke_width']
    color_count, opening_radius = data['color_count'], data['opening_radius']
    removed, cycles_removed = data['removed'], data['cycles_removed']

    def finish(result):
        if state:
            state.put(key, result)
        return result

    starts, ends, owners = [], [], []
    # Photographic noise can produce thousands of tips. A board candidate with
    # unresolved junctions cannot be assembled; do not run a quadratic matching
    # search on it. Ordinary tracing keeps its existing behavior by default.
    if ((opts.stop_at_branches and branches) or
            (opts.max_endpoints is not None and len(endpoints) > opts.max_endpoints)):
        diagnostics = {"stroke_width": stroke_width, "threshold": opts.threshold,
                       "color_layers": color_count,
                       "morphological_opening_radius": opening_radius,
                       "foreground_pixels": data['foreground_pixels'], "skeleton_pixels": data['skeleton_pixels'],
                       "spur_pixels_removed": removed, "pixel_cycles_removed": cycles_removed,
                       "branch_points": [[int(x), int(y)] for y, x in branches],
                       "endpoint_count": len(endpoints), "candidate_count": 0,
                       "matching_revisions": 0, "gap_conflicts": [],
                       "unmatched_endpoints": [{"path": e["path"], "end": e["end"],
                                                 "point": e["point"].tolist()} for e in endpoints],
                       "matching_skipped": "unresolved branches" if branches else "endpoint limit"}
        if opts.include_skeleton:
            diagnostics["skeleton_points"] = data['skeleton_points']
        if arrow_evidence:
            diagnostics['arrow_pruning'] = arrow_evidence
        return finish({"width": width, "height": height, "paths": paths, "matches": [],
                "complete": False, "diagnostics": diagnostics})
    fixed_matches, seam_evidence = [], []
    endpoint_count = len(endpoints)
    if opts.photo_gap_policy:
        from .board_seams import reserve_local_seams
        data, fixed_matches, seam_evidence = reserve_local_seams(
            data, photo_source_mask, tangent_widths=opts.tangent_widths,
            max_gap_widths=opts.noncrossing_max_widths)
        paths, arrays, endpoints = data['paths'], data['arrays'], data['endpoints']
    fixed_connectors = [(arrays[m['a'][0]][0 if m['a'][1] == 0 else -1],
                         arrays[m['b'][0]][0 if m['b'][1] == 0 else -1])
                        for m in fixed_matches]
    for i, points in enumerate(arrays):
        starts.extend(points[:-1]); ends.extend(points[1:]); owners.extend([i] * (len(points) - 1))
    segment_starts = np.array(starts, dtype=float).reshape(-1, 2)
    segment_ends = np.array(ends, dtype=float).reshape(-1, 2)
    owners = np.array(owners, dtype=int)
    matching_graph = nx.Graph()
    matching_graph.add_nodes_from(range(len(endpoints)))
    candidates = {}
    deferred_singles=[]
    photo_rejections = {}
    bundle_enabled = opts.bundle_gaps or opts.photo_gap_policy
    # Exact radius filtering preserves the previous candidate order/costs.
    # There is no reason to inspect remote ends that the length check rejects.
    radius = max(opts.max_gap_widths * stroke_width,
                 .35*hypot(width, height) if bundle_enabled else 0.)
    nearby = (cKDTree([e['point'] for e in endpoints]).query_ball_point(
              [e['point'] for e in endpoints], radius*(1+1e-12)) if endpoints else [])
    from .bundled_crossings import EndpointTangents
    tangents = EndpointTangents(arrays, endpoints)
    for i, first in enumerate(endpoints):
        checkpoint('examining nearby strand ends')
        for j in sorted(index for index in nearby[i] if index > i):
            second = endpoints[j]
            delta = second["point"] - first["point"]
            length = np.linalg.norm(delta)
            ordinary_limit=opts.max_gap_widths * stroke_width
            extended=bool(length > ordinary_limit)
            if length < 1 or (extended and (not bundle_enabled or length > .35*hypot(width,height))):
                continue
            if any(_segments_cross(first['point'], second['point'], a, b) for a, b in fixed_connectors):
                continue
            direction = delta / length
            if opts.photo_gap_policy:
                # A distant endpoint does not justify looking farther around
                # a curved visible hook. Estimate the actual local direction,
                # averaging pixel stairs over the same fixed pen-width window.
                tangent1 = tangents.get(i, opts.tangent_widths*stroke_width)
                tangent2 = tangents.get(j, opts.tangent_widths*stroke_width)
            elif bundle_enabled and length > 6*stroke_width:
                reach=max(3*stroke_width,min(.22*length,12*stroke_width))
                tangent1=tangents.get(i,reach)
                tangent2=tangents.get(j,reach)
            else:
                tangent1,tangent2=first['tangent'],second['tangent']
            alignment1 = float(direction @ tangent1)
            alignment2 = float(-direction @ tangent2)
            if min(alignment1, alignment2) < opts.min_tangent_dot:
                continue
            hits = _intersections(first["point"], second["point"], segment_starts,
                                  segment_ends, owners, stroke_width)
            evidence=None
            if bundle_enabled:
                from .bundled_crossings import bundle_evidence, overlaps_visible_ink
                if overlaps_visible_ink(first['point'],second['point'],segment_starts,segment_ends,stroke_width):
                    continue
                evidence=bundle_evidence(first['point'],second['point'],hits,
                                         segment_starts,segment_ends,owners,stroke_width)
                if not opts.photo_gap_policy and extended and (evidence is None or min(alignment1,alignment2)<.7):
                    if len(hits)==1 and length<.15*hypot(width,height):
                        deferred_singles.append((i,j,float(length),alignment1,alignment2,hits))
                    continue
            photo_support = None
            if opts.photo_gap_policy:
                from .photo_gaps import photo_gap_evidence
                photo_support, reason = photo_gap_evidence(
                    float(length), stroke_width, [alignment1, alignment2], hits, evidence)
                if (reason in ('endpoint_direction', 'unsupported_single_crossing_length')
                        and first['path'] == second['path'] and len(hits) == 1):
                    from .photo_oval_gaps import oval_gap_evidence
                    ordered_tangents = [tangent1, tangent2] if first['end'] == 0 else [tangent2, tangent1]
                    photo_support = oval_gap_evidence(arrays[first['path']], stroke_width, ordered_tangents,
                                                     hits, segment_starts, segment_ends, owners, first['path'],
                                                     rejection_reason=reason, fixed_connectors=fixed_connectors)
                    if photo_support is not None:
                        reason = None
                if reason is not None:
                    photo_rejections[reason] = photo_rejections.get(reason, 0) + 1
                    continue
            # A pen lifted and put down a fraction of a stroke-width sideways
            # leaves opposing tips whose chord is oblique to both tangents.
            # Only the explicit fallback permits this sub-width seam; longer
            # breaks retain the ordinary alignment requirement.
            micro_seam = bool(opts.micro_seams and not hits
                              and length <= 1.1*stroke_width
                              and min(alignment1, alignment2) >= .2
                              and float(tangent1 @ tangent2) <= -.85
                              and all(np.linalg.norm(np.diff(arrays[e['path']], axis=0), axis=1).sum()
                                      >= 10*stroke_width for e in (first, second)))
            if not hits and not micro_seam and (length > opts.noncrossing_max_widths * stroke_width
                             or min(alignment1, alignment2) < opts.noncrossing_min_alignment):
                continue
            if hits and min(hit["sine_angle"] for hit in hits) < 0.12:
                continue
            color_distance = np.linalg.norm(path_colors[first["path"]] - path_colors[second["path"]])
            if (extended or (bundle_enabled and len(hits)>=2)) and color_distance > .5:
                continue
            # Missing ink between over-strands is explained by the bundle;
            # penalize the unsupported ends instead of every hidden pixel.
            distance_scale=(sum(evidence['margins'])/stroke_width + len(hits)*.5
                            if evidence else length/stroke_width/(1 if opts.photo_gap_policy else max(1,min(2,len(hits)))))
            cost = (opts.distance_weight * distance_scale
                    + opts.tangent_weight * (2 - alignment1 - alignment2)
                    # Short photographed fragments can have unstable median
                    # chroma. Bound that soft penalty so it cannot make two
                    # supported local joins lose to a bundle plus open ends.
                    # The hard color check above still protects long bundles.
                    + opts.color_weight * (min(color_distance, .5) if opts.photo_gap_policy else color_distance)
                    + (0 if hits else 2.0))
            matching_graph.add_edge(i, j, weight=float(cost))
            candidates[i, j] = {"cost": float(cost), "distance": float(length),
                                "alignment": [alignment1, alignment2], "hits": hits,
                                "bundle": evidence, "extended": extended, "micro_seam": micro_seam,
                                "photo_support": photo_support}

    if opts.bundle_gaps and deferred_singles:
        from .bundled_crossings import calibrated_single_evidence
        margins=[margin for c in candidates.values() if c['bundle'] and min(c['alignment'])>=.9
                 for margin in c['bundle']['margins']]
        calibration=float(np.median(margins)) if margins else None
        for i,j,length,a1,a2,hits in deferred_singles:
            first,second=endpoints[i],endpoints[j]
            evidence=calibrated_single_evidence(first['point'],second['point'],hits,[a1,a2],
                                                 stroke_width,calibration)
            if evidence is None:
                continue
            color_distance=np.linalg.norm(path_colors[first['path']]-path_colors[second['path']])
            if color_distance > .5:
                continue
            cost=float(opts.distance_weight*length/stroke_width + opts.tangent_weight*(2-a1-a2)
                       + opts.color_weight*color_distance)
            matching_graph.add_edge(i,j,weight=cost)
            candidates[i,j]={'cost':cost,'distance':length,'alignment':[a1,a2],'hits':hits,
                             'bundle':evidence,'extended':True}

    unmatched_cost = None
    if opts.photo_gap_policy:
        from .photo_gaps import UNMATCHED_ENDPOINT_COST
        unmatched_cost = UNMATCHED_ENDPOINT_COST
    pairs, matching_revisions = _noncrossing_matching(matching_graph, endpoints, candidates,
                                                      unmatched_cost=unmatched_cost)
    matches = []
    gap_conflicts = []
    for i, j in pairs:
        first, second = endpoints[i], endpoints[j]
        candidate = candidates[i, j]
        matches.append({"a": [first["path"], first["end"]],
                        "b": [second["path"], second["end"]],
                        "cost": candidate["cost"], "distance": candidate["distance"],
                        "alignment": candidate["alignment"],
                        "crossings": [{**hit, "point": hit["point"].tolist()}
                                      for hit in candidate["hits"]]})
        if bundle_enabled and candidate['bundle']:
            matches[-1]['bundle']={**candidate['bundle'],'extended':candidate['extended']}
        if candidate.get('micro_seam'):
            matches[-1]['micro_seam'] = True
        if candidate.get('photo_support'):
            matches[-1]['photo_gap_support'] = candidate['photo_support']
    matches.extend(fixed_matches)
    for k, (i, j) in enumerate(pairs):
        for m, (a, b) in enumerate(pairs[:k]):
            if _segments_cross(endpoints[i]["point"], endpoints[j]["point"],
                               endpoints[a]["point"], endpoints[b]["point"]):
                gap_conflicts.append([m, k])
    used = {i for pair in pairs for i in pair}
    unmatched = [{"path": endpoint["path"], "end": endpoint["end"],
                  "point": endpoint["point"].tolist()}
                 for i, endpoint in enumerate(endpoints) if i not in used]
    diagnostics = {"stroke_width": stroke_width, "threshold": opts.threshold,
                   "color_layers": color_count,
                   "morphological_opening_radius": opening_radius,
                   "foreground_pixels": data['foreground_pixels'], "skeleton_pixels": data['skeleton_pixels'],
                   "spur_pixels_removed": removed,
                   "pixel_cycles_removed": cycles_removed,
                   "branch_points": [[int(x), int(y)] for y, x in branches],
                   "endpoint_count": endpoint_count, "candidate_count": len(candidates),
                   "matching_revisions": matching_revisions,
                   "unmatched_endpoints": unmatched, "gap_conflicts": gap_conflicts}
    if opts.photo_gap_policy:
        diagnostics['photo_gap_policy'] = {'rejected_candidates': photo_rejections,
                                           'unmatched_endpoint_cost': unmatched_cost}
    if seam_evidence:
        diagnostics['pre_matching_seams'] = seam_evidence
    if opts.include_skeleton:
        diagnostics["skeleton_points"] = data['skeleton_points']
    if arrow_evidence:
        diagnostics['arrow_pruning'] = arrow_evidence
    if bundle_enabled:
        diagnostics['bundled_crossings']={'regions':[
            {'ends':[paths[m['a'][0]]['points'][0 if m['a'][1]==0 else -1],
                     paths[m['b'][0]]['points'][0 if m['b'][1]==0 else -1]],
             'crossings':[h['point'] for h in m['crossings']], **m['bundle']}
            for m in matches if m.get('bundle')],
            'review_points':[h['point'] for m in matches if len(m['crossings'])>1 for h in m['crossings']],
            'extended_candidates':sum(c['extended'] for c in candidates.values())}
    if any(m.get('micro_seam') for m in matches):
        diagnostics['micro_seam_repairs'] = [
            [paths[m[key][0]]['points'][0 if m[key][1] == 0 else -1] for key in ('a', 'b')]
            for m in matches if m.get('micro_seam')]
    return finish({"width": width, "height": height, "paths": paths, "matches": matches,
            "complete": bool(paths) and not branches and not unmatched and not gap_conflicts,
            "diagnostics": diagnostics})
