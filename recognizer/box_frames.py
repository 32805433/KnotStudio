"""Conservative rectangle proposals from four independently supported borders.

Unlike flood-filling a closed interior, this does not require corners to touch
and does not thicken ink (which can join a coefficient to a damaged frame).
These are only geometric proposals: box_detection must still require interior
text and two complete, opposite bundles before offering an open frame.
"""
import cv2
import numpy as np


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _support(quad, distance):
    """Each of the four actual sides needs ink, not just its two corners."""
    scores = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        length = np.linalg.norm(b-a)
        t = np.linspace(0, 1, max(24, int(length)))
        points = a+t[:, None]*(b-a)
        values = cv2.remap(distance, points[:, 0].astype(np.float32)[None],
                           points[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=1000).ravel()
        near = values <= 2.5
        # Broken corners are allowed; a large missing middle of a side is not.
        core = near[(t >= .2) & (t <= .8)]
        if near.mean() < .68 or core.mean() < .90:
            return None
        scores.append(float(near.mean()))
    return min(scores)


def _frame_cycles(neighbors, ends, directions, max_work=12000):
    """Bound traversal even for grids or hundreds of nested rectangles."""
    work = 0
    for a, adjacent in enumerate(neighbors):
        for b in sorted(adjacent):
            if b <= a:
                continue
            for c in sorted(neighbors[b]):
                work += 1
                if work > max_work:
                    return
                if c <= a or c == b or ends[b, a] == ends[b, c]:
                    continue
                if abs(directions[a]@directions[c]) < .94:
                    continue
                for d in sorted(adjacent & neighbors[c]):
                    work += 1
                    if work > max_work:
                        return
                    if d <= b or ends[a, b] == ends[a, d] or ends[c, b] == ends[c, d] or ends[d, a] == ends[d, c]:
                        continue
                    if abs(directions[b]@directions[d]) >= .94:
                        yield a, b, c, d


def open_frame_candidates(ink, *, max_segments=600, max_candidates=64):
    """Return bounded, scale-normalized quadrilaterals supported by four lines.

    Endpoint-local intersections make a sparse graph of possible corners.
    Only four-cycles with alternating near-perpendicular directions qualify.
    No virtual border or inferred corner is written into the source pixels.
    """
    scale = min(1., 1200/max(ink.shape))
    mask = (ink.astype(np.uint8)*255)
    if scale < 1:
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # Raster rotation/resizing can make a straight border a pixel staircase.
    # Smooth only the line-finding view; support, text and ports still use ink.
    line_view = cv2.GaussianBlur(mask, (3, 3), 1.)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(line_view)[0]
    if lines is None:
        return []
    segments = lines.reshape(-1, 2, 2).astype(float)
    lengths = np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1)
    order = np.flatnonzero(lengths >= 14.)
    order = order[np.argsort(-lengths[order], kind='stable')[:max_segments]]
    segments, lengths = segments[order], lengths[order]
    if len(segments) < 4:
        return []
    starts = segments[:, 0]
    directions = (segments[:, 1]-starts)/lengths[:, None]
    determinant = _cross(directions[:, None], directions[None])
    delta = starts[None]-starts[:, None]
    parameter = np.divide(_cross(delta, directions[None]), determinant,
                          out=np.full(determinant.shape, np.inf),
                          where=np.abs(determinant) > .90)
    # A corner must be close to an end of BOTH observed segments. A crossing
    # through the middle of a strand is not evidence of a box corner.
    tolerance = 3.+.24*lengths[:, None]
    near_start = np.abs(parameter) <= tolerance
    near_end = np.abs(parameter-lengths[:, None]) <= tolerance
    local = near_start | near_end
    adjacency = local & local.T
    np.fill_diagonal(adjacency, False)
    neighbors = [set(np.flatnonzero(row)) for row in adjacency]
    ends = np.abs(parameter-lengths[:, None]) < np.abs(parameter)
    distance = cv2.distanceTransform((mask < 128).astype(np.uint8), cv2.DIST_L2, 3)
    proposals = []
    seen = set()
    for cycle in _frame_cycles(neighbors, ends, directions):
        quad = np.array([starts[i]+parameter[i, j]*directions[i]
                         for i, j in zip(cycle, cycle[1:]+cycle[:1])])
        if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)):
            continue
        side_lengths = np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)
        if (min(side_lengths) < 18 or max(side_lengths)/min(side_lengths) > 12
                or max(side_lengths[0]/side_lengths[2], side_lengths[2]/side_lengths[0],
                       side_lengths[1]/side_lengths[3], side_lengths[3]/side_lengths[1]) > 1.5):
            continue
        area = abs(cv2.contourArea(quad.astype(np.float32)))
        if area > .65*mask.size:
            continue
        # Quantized footprint removes doubled stroke-edge cycles before the
        # more expensive pixel support checks.
        key = tuple(np.round(np.r_[quad.mean(0), np.sort(side_lengths)[::2]]/3).astype(int))
        if key in seen:
            continue
        support = _support(quad, distance)
        if support is None:
            continue
        seen.add(key)
        # Canonical image-clockwise order, matching box_detection.
        center = quad.mean(0)
        quad = quad[np.argsort(np.arctan2(quad[:, 1]-center[1], quad[:, 0]-center[0]))]
        quad = np.roll(quad, -np.argmin(quad.sum(1)), axis=0)/scale
        proposals.append((quad, support))
    # Double edges may suggest almost the same rectangle. Keep the strongest
    # support without spending the OCR budget repeatedly on the same label.
    chosen = []
    for quad, score in sorted(proposals, key=lambda item: -item[1]):
        area = abs(cv2.contourArea(quad.astype(np.float32)))
        if any(np.linalg.norm(quad.mean(0)-q.mean(0)) < .1*np.sqrt(area)
               and .7 < area/abs(cv2.contourArea(q.astype(np.float32))) < 1.4 for q, _ in chosen):
            continue
        chosen.append((quad, score))
        if len(chosen) >= max_candidates:
            break
    return chosen
