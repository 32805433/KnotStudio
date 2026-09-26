"""Read arrowheads from source pixels, then orient recovered closed components.

Arrow wings are intentionally absent from the reconstructed strand graph. This
pass uses the unpruned source skeleton: two short, backward-facing wings must
meet a continuing shaft, agree with its tangent, and lie away from crossings.
No filename, label text, reference PD, or source metadata is used.
"""
from itertools import combinations
import heapq
import time

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .diagram import component_walks, pd_code
from .knotfolio_backend import _adjacency, _color_layers, _remove_pixel_cycles, _trace, _prune_spurs
from .preprocess import prepare


def _strand_samples(diagram, scale, width):
    edges = {e['id']: e for e in diagram['edges']}
    points, tangents, components = [], [], []
    for ci, walk in enumerate(component_walks(diagram)):
        for eid, end in walk:
            p = np.asarray(edges[eid]['points'], float)*scale
            if end:
                p = p[::-1]
            lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            if lengths[-1] <= 0:
                continue
            s = np.linspace(0, lengths[-1], max(2, int(lengths[-1]/max(.5, width*.4))+1))
            def at(t):
                if edges[eid].get('closed'):
                    t = t % lengths[-1]
                return np.column_stack([np.interp(t, lengths, p[:, j]) for j in (0, 1)])
            q = at(s)
            tangent = at(s+width*1.5)-at(s-width*1.5)
            tangent /= np.maximum(1e-9, np.linalg.norm(tangent, axis=1))[:, None]
            points.extend(q); tangents.extend(tangent); components.extend([ci]*len(q))
    return np.asarray(points), np.asarray(tangents), np.asarray(components)


def _offset_candidates(graph, tree, width, *, flexible=False):
    # Resampling can put sub-stroke side branches on an otherwise continuous
    # arrow wing. Remove only branches shorter than our minimum accepted wing,
    # from an independent graph; retain the original graph for printed arrows.
    graph = {p: list(neighbors) for p, neighbors in graph.items()}
    _prune_spurs(graph, (.65 if flexible else .45)*width)
    chains = _trace(graph)
    lengths = [sum(np.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(c, c[1:])) for c in chains]
    parent = {p: p for p in graph if len(graph[p]) > 2}
    wings = {}
    for chain, length in zip(chains, lengths):
        if len(graph[chain[0]]) == 1:
            chain = chain[::-1]
        a, b = chain[0], chain[-1]
        if a not in parent or len(graph[b]) != 1 or not .5*width <= length <= 7*width:
            continue
        anchor, tail = np.array(a[::-1], float), np.array(b[::-1], float)
        # Hand-drawn wings can bend; the subsequent two-wing/shaft certificate
        # is stronger than a straight-wing shape threshold on its own.
        if np.linalg.norm(tail-anchor)/length < .65 or tree.query(tail)[0] < .45*width:
            continue
        wings.setdefault(a, []).append((anchor, tail, length))
    if not flexible:
        return wings, chains, lengths, parent
    # A hand-drawn wing may bend back onto its shaft, making a tiny loop.
    # Use only a local two-path lens: one path follows the shaft chord and the
    # other bows out. A lens alone never votes; it still needs an independent
    # terminal wing on the opposite side, checked by _offset_wing_arrows.
    lenses = {}
    for chain, length in zip(chains, lengths):
        a, b = chain[0], chain[-1]
        if a != b and a in parent and b in parent and length <= 5*width:
            lenses.setdefault(tuple(sorted((a, b))), []).append((chain, length))
    for ends, paths in lenses.items():
        if len(paths) != 2:
            continue
        (straight, short), (bowed, long) = sorted(paths, key=lambda item: item[1])
        a, b = np.asarray(ends, float)[:, ::-1]
        chord = np.linalg.norm(b-a)
        if not .75*width <= chord <= 3*width or short > 1.25*chord or long < 1.4*chord:
            continue
        if short+long > 7*width or max(tree.query([a, b])[0]) > 1.25*width:
            continue
        normal = np.array([-(b-a)[1], (b-a)[0]])/chord
        p = np.asarray(bowed, float)[:, ::-1]
        offsets = (p-a) @ normal
        tip = p[np.argmax(abs(offsets))]
        if not .5*width <= max(abs(offsets)) <= 2*width:
            continue
        for anchor in (a, b):
            length = np.linalg.norm(tip-anchor)
            if length >= .5*width:
                wings.setdefault(tuple(anchor[::-1].astype(int)), []).append(
                    (anchor, tip, length, 'closed_wing'))
    return wings, chains, lengths, parent


def _offset_wing_arrows(wings, chains, lengths, parent, tree, tangents,
                        components, crossings, width, scale, existing, *, flexible=False):
    """Pair offset arrow wings only across a short, observed common shaft.

    Hand-drawn wings need not meet at one pixel. Their individual directions
    are measured from their own attachment points, not an artificial common
    apex. A bounded source-graph path prevents pairing nearby disconnected ink.
    """
    candidates = [wing for group in wings.values() for wing in group]
    if len(candidates) < 2 or len(candidates) > 512:
        return []
    anchors = np.array([wing[0] for wing in candidates])
    distances, indices = tree.query(anchors)
    links = {}
    for chain, length in zip(chains, lengths):
        a, b = chain[0], chain[-1]
        # Up to 4 widths along the shaft plus two attachment tails of at
        # most 1.25 widths; connected() applies the tighter per-pair bound.
        if a not in parent or b not in parent or length > 6.5*width:
            continue
        # The connecting pixels must follow one reconstructed shaft, not a
        # crossing gap, arrow wing, or a neighboring strand.
        d, ix = tree.query(np.asarray(chain)[:, ::-1])
        if np.max(d) > 1.25*width or len(set(components[ix])) != 1:
            continue
        links.setdefault(a, []).append((b, length))
        links.setdefault(b, []).append((a, length))

    def connected(a, b, attachment_distance):
        # Thick arrow junctions can sit just off the shaft centerline. Their
        # short attachment tails do not consume the along-shaft span budget.
        limit = 4*width+attachment_distance
        queue = [(0., a)]; best = {a: 0.}
        while queue:
            length, node = heapq.heappop(queue)
            if node == b:
                return True
            if length > best[node]:
                continue
            for neighbor, step in links.get(node, ()):
                total = length+step
                if total <= limit and total < best.get(neighbor, float('inf')):
                    best[neighbor] = total
                    heapq.heappush(queue, (total, neighbor))
        return False

    proposals = []
    for i, j in sorted(cKDTree(anchors).query_pairs(4*width)):
        a, b = candidates[i], candidates[j]
        if len(a) > 3 and len(b) > 3:
            continue  # A closed curl or lens is not an arrow on its own.
        if max(distances[i], distances[j]) > 1.25*width or components[indices[i]] != components[indices[j]]:
            continue
        anchor = (a[0]+b[0])/2
        if any(np.linalg.norm(np.asarray(old['point'])*scale-anchor) < 2*width for old in existing):
            continue
        if len(crossings) and min(np.linalg.norm(crossings-q, axis=1).min()
                                  for q in (a[0], b[0], anchor)) < 2.5*width:
            continue
        if not connected(tuple(a[0][::-1].astype(int)), tuple(b[0][::-1].astype(int)),
                         distances[i]+distances[j]):
            continue
        if max(a[2], b[2])/min(a[2], b[2]) > (3.5 if flexible else 2.3):
            continue
        vectors = np.array([a[1]-a[0], b[1]-b[0]])
        units = vectors/np.linalg.norm(vectors, axis=1)[:, None]
        direction = -units.sum(axis=0)
        norm = np.linalg.norm(direction)
        if norm < .5:
            continue
        direction /= norm
        neighbors = tree.query_ball_point(anchor, 1.5*width)
        if not neighbors or len(set(components[neighbors])) != 1:
            continue
        shaft = np.mean(tangents[neighbors], axis=0)
        magnitude = np.linalg.norm(shaft)
        if magnitude < .85:
            continue
        shaft /= magnitude
        alignment = float(direction @ shaft)
        if abs(alignment) < (.8 if len(a) > 3 or len(b) > 3 else .86):
            continue
        # Both wings must point backward along the actual shaft and occupy
        # opposite sides. This rejects opposing tick marks and many stray forks.
        forward = shaft*np.sign(alignment)
        # Balance the wings around their own bisector, as for coincident
        # wings. A bent shaft need not coincide with that bisector exactly.
        sideways = vectors @ np.array([-direction[1], direction[0]])
        backward = -units @ forward
        if (sideways[0]*sideways[1] >= 0 or min(abs(sideways)) < .45*width
                or max(abs(sideways))/min(abs(sideways)) > (3.5 if flexible else 2.5)
                or min(backward) < .22
                or max(-units @ direction if flexible else backward) > .96):
            continue
        arrow = dict(point=(anchor/scale).tolist(), direction=direction.tolist(),
                     component=int(components[indices[i]]), sign=1 if alignment > 0 else -1,
                     alignment=abs(alignment), evidence=('closed_and_terminal_source_wings'
                         if len(a) > 3 or len(b) > 3 else 'offset_source_wings'))
        proposals.append((i, j, arrow))
    # A wing with multiple plausible partners does not establish an arrow.
    uses = {}
    for i, j, _ in proposals:
        uses[i] = uses.get(i, 0)+1; uses[j] = uses.get(j, 0)+1
    return [arrow for i, j, arrow in proposals if uses[i] == uses[j] == 1]


def infer_orientations(rgb, diagram, *, source_scale=1., arrow_evidence=()):
    """Return source-arrow evidence and signs relative to component_walks.

    Missing or conflicting evidence leaves a component's default orientation
    unspecified. A single clear arrow is sufficient; multiple arrows must all
    agree. Candidate coordinates are returned in the original image frame.
    """
    started = time.monotonic()
    pixels, _ = prepare(rgb)
    resize_scale = min(1., 1800/max(pixels.shape[:2]))
    if resize_scale < 1:
        pixels = cv2.resize(pixels, (round(pixels.shape[1]*resize_scale), round(pixels.shape[0]*resize_scale)),
                            interpolation=cv2.INTER_AREA)
    scale = source_scale*resize_scale
    mask = pixels.min(axis=2) < 150
    colors, count = _color_layers(pixels.astype(float), mask)
    skeleton = np.zeros_like(mask)
    for color in range(1, count+1):
        skeleton |= skeletonize(colors == color)
    radii = ndimage.distance_transform_edt(mask)[skeleton]
    width = max(1.5, float(np.median(radii)*2)) if len(radii) else 1.5
    points, tangents, components = _strand_samples(diagram, scale, width)
    arrows = []
    if len(points):
        tree = cKDTree(points)
        crossings = np.array([c['point'] for c in diagram['crossings']], float).reshape(-1, 2)*scale
        graph = _adjacency(skeleton, colors)
        _remove_pixel_cycles(graph, width)
        chains = _trace(graph)
        # Collapse only short *connected* branch junctions. Spatial proximity
        # alone must not join wings belonging to neighboring strands.
        parent = {p: p for p in graph if len(graph[p]) > 2}
        def root(p):
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p
        lengths = [sum(np.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(c, c[1:])) for c in chains]
        for chain, length in zip(chains, lengths):
            a, b = chain[0], chain[-1]
            if a in parent and b in parent and length <= 1.8*width:
                parent[root(a)] = root(b)
        wings = {}
        for chain, length in zip(chains, lengths):
            if len(graph[chain[0]]) == 1:
                chain = chain[::-1]
            a, b = chain[0], chain[-1]
            if a not in parent or len(graph[b]) != 1 or not .5*width <= length <= 7*width:
                continue
            anchor, tail = np.array(a[::-1], float), np.array(b[::-1], float)
            if np.linalg.norm(tail-anchor)/length < .78:
                continue
            # A short genuine strand ending at a crossing gap is not a wing.
            if tree.query(tail)[0] < .45*width:
                continue
            wings.setdefault(root(a), []).append((anchor, tail, length))
        for group in wings.values():
            for a, b in combinations(group, 2):
                anchor = (a[0]+b[0])/2
                if len(crossings) and np.linalg.norm(crossings-anchor, axis=1).min() < 2.5*width:
                    continue
                distance, index = tree.query(anchor)
                if distance > 1.25*width or max(a[2], b[2])/min(a[2], b[2]) > 2.3:
                    continue
                tangent = tangents[index]
                va, vb = a[1]-anchor, b[1]-anchor
                ua, ub = va/np.linalg.norm(va), vb/np.linalg.norm(vb)
                direction = -(ua+ub)
                direction /= max(1e-9, np.linalg.norm(direction))
                alignment = float(np.dot(direction, tangent))
                if abs(alignment) < .86:
                    continue
                normal = np.array([-direction[1], direction[0]])
                sideways = [float(v@normal) for v in (va, vb)]
                backward = [-float(u@direction) for u in (ua, ub)]
                if (sideways[0]*sideways[1] >= 0 or min(abs(v) for v in sideways) < .45*width
                        or min(backward) < .22 or max(backward) > .96
                        or max(abs(v) for v in sideways)/min(abs(v) for v in sideways) > 2.5):
                    continue
                neighbors = tree.query_ball_point(anchor, 1.5*width)
                if len(set(components[neighbors])) != 1:
                    continue
                ci = int(components[index]); sign = 1 if alignment > 0 else -1
                if any(np.linalg.norm(np.asarray(old['point'])*scale-anchor) < 2*width for old in arrows):
                    continue
                arrows.append(dict(point=(anchor/scale).tolist(), direction=direction.tolist(),
                                   component=ci, sign=sign, alignment=abs(alignment)))
        offset = _offset_candidates(graph, tree, width)
        arrows.extend(_offset_wing_arrows(*offset, tree, tangents,
                                          components, crossings, width, scale, arrows))
        # Preserve the established strict matches before considering uneven
        # or looped wings. New plausible partners must not suppress a clear
        # existing pair merely by making its junction more ambiguous.
        flexible = _offset_candidates(graph, tree, width, flexible=True)
        arrows.extend(_offset_wing_arrows(*flexible, tree, tangents,
                                          components, crossings, width, scale, arrows,
                                          flexible=True))
        # A geometric paired-wing cleanup can identify longer hand-drawn
        # arrowheads than the short printed-arrow detector above. Its evidence
        # was measured before removing any source ink. Attach it only to a
        # unique reconstructed shaft whose tangent agrees with that direction.
        for source in arrow_evidence:
            anchor = np.asarray(source['point'], float)*scale
            direction = np.asarray(source['direction'], float)
            direction /= max(1e-9, np.linalg.norm(direction))
            if len(crossings) and np.linalg.norm(crossings-anchor, axis=1).min() < 2.5*width:
                continue
            distance, index = tree.query(anchor)
            if distance > 1.5*width:
                continue
            # Average a short shaft neighborhood so a single raster kink at
            # the arrow junction cannot reverse or hide a clear direction.
            shaft_indices = tree.query_ball_point(anchor, 3*width)
            shaft_indices = [j for j in shaft_indices if components[j] == components[index]]
            shaft = np.mean(tangents[shaft_indices], axis=0)
            magnitude = np.linalg.norm(shaft)
            if magnitude < .8:
                continue
            alignment = float(direction @ (shaft/magnitude))
            if abs(alignment) < .8:
                continue
            neighbors = tree.query_ball_point(anchor, 1.8*width)
            if len(set(components[neighbors])) != 1:
                continue
            if any(np.linalg.norm(np.asarray(old['point'])*scale-anchor) < 2.5*width for old in arrows):
                continue
            arrows.append(dict(point=(anchor/scale).tolist(), direction=direction.tolist(),
                               component=int(components[index]), sign=1 if alignment > 0 else -1,
                               alignment=abs(alignment), evidence='paired_source_wings'))
    records, orientations = [], {}
    for ci, _ in enumerate(component_walks(diagram)):
        evidence = [a for a in arrows if a['component'] == ci]
        signs = {a['sign'] for a in evidence}
        status = 'consistent' if len(signs) == 1 else 'conflicting' if signs else 'not_detected'
        if status == 'consistent':
            orientations[str(ci)] = next(iter(signs))
        records.append(dict(component=ci, status=status, arrow_count=len(evidence)))
    return dict(arrows=arrows, components=records, component_orientations=orientations,
                elapsed_seconds=time.monotonic()-started)


def apply_source_orientations(result, rgb):
    """Apply once after all topology retries; keep rendered arrows and PD in sync."""
    diagram = result.get('diagram')
    if diagram is None:
        return result
    pixels, source_scale = rgb, 1.
    preparation = result.get('diagnostics', {}).get('preprocessing', {})
    if preparation.get('transform') == 'board_photo':
        # The accepted contrast mask comes from the source photograph before
        # skeleton/arrow pruning or any seam repair. Absolute dark-pixel
        # thresholding on the original photo would trace the board background
        # itself, missing white chalk and sometimes producing huge graphs.
        if preparation.get('method') == 'style_contrast_envelope':
            from .style_recovery import style_photo_mask
            prepared = style_photo_mask(rgb, preparation)
            if prepared is None:
                result['diagnostics']['orientation'] = {
                    'arrows': [], 'components': [], 'component_orientations': {},
                    'reason': 'Accepted source contrast mask could not be reproduced.'}
                return result
            pixels, source_scale = prepared
        else:
            from .board_photo import prepared_board_contrast
            contrast = prepared_board_contrast(rgb, preparation.get('polarity', 'board'))
            for candidate, record in contrast.candidates():
                if all(record.get(key) == preparation.get(key) for key in ('method', 'strong', 'weak')):
                    pixels, source_scale = candidate, contrast.scale
                    break
    retained = result.get('diagnostics', {}).get('photo_arrow_evidence', [])
    retained = retained + result.get('diagnostics', {}).get('arrow_pruning', [])
    evidence = infer_orientations(pixels, diagram, source_scale=source_scale,
                                 arrow_evidence=retained)
    if source_scale != 1. or pixels is not rgb:
        evidence['source_preparation'] = 'source_photo_contrast'
    diagram.setdefault('component_orientations', {}).update(evidence['component_orientations'])
    result['diagnostics']['orientation'] = evidence
    # Runtime measurements belong to diagnostics, not the deterministic saved
    # diagram. Identical pixels must still produce identical diagram values.
    diagram.setdefault('recognition', {})['orientation'] = {
        key: value for key, value in evidence.items() if key != 'elapsed_seconds'}
    conflicts = [str(c['component']+1) for c in evidence['components'] if c['status'] == 'conflicting']
    if conflicts:
        result['warnings'].append('Input arrows disagree on component(s) '+', '.join(conflicts)+
                                  '; orientation was not inferred there. Check and reverse the component if needed.')
        result['status'] = 'needs_review'
        diagram['recognition']['warnings'] = result['warnings']
    try:
        result['pd_code'] = pd_code(diagram)
    except ValueError as exc:
        if not (any(c.get('kind') == 'twist_box' for c in diagram['crossings'])
                and ('expansion needs' in str(exc) or 'crossing limit' in str(exc))):
            raise
        result['pd_code'] = None
        if str(exc) not in result['warnings']:
            result['warnings'].append(str(exc))
        result['status'] = 'needs_review'
    return result
