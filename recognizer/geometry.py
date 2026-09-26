"""Assemble a planar diagram from visible strands and inferred undercrossing gaps."""
import math
from collections import defaultdict
import numpy as np
from .diagram import assign_components, validate


def distinct_vertices(points, tolerance=1e-8):
    """Remove consecutive numerical duplicates before indexing segments.

    Skipping zero-length segments later leaves holes in their adjacency indices,
    causing the two surviving neighbors to look like an unencoded crossing.
    Compare against the last retained point so tiny steps cannot accumulate into
    a long, accidentally discarded arc. Work on a copy and retain the final end.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points.copy()
    # Almost every editable curve already has distinct vertices. Keep the
    # last-retained-point loop for true duplicates; adjacent filtering alone
    # would incorrectly erase an accumulated sequence of tiny steps.
    delta = np.diff(points, axis=0)
    if np.all(np.hypot(delta[:, 0], delta[:, 1]) >= tolerance):
        return points.copy()
    kept = [0]
    for i in range(1, len(points)):
        delta = points[i]-points[kept[-1]]
        if math.hypot(delta[0], delta[1]) >= tolerance:
            kept.append(i)
    result = points[kept].copy()
    if len(kept) > 1:
        result[-1] = points[-1]
    return result


def intersection(a,b,c,d):
    r,s = b-a,d-c
    den = r[0]*s[1]-r[1]*s[0]
    if abs(den)<1e-9:
        return None
    q=c-a
    t=(q[0]*s[1]-q[1]*s[0])/den
    u=(q[0]*r[1]-q[1]*r[0])/den
    if -1e-7<=t<=1+1e-7 and -1e-7<=u<=1+1e-7:
        return float(t),float(u),a+t*r
    return None


def _section(points, left, right):
    """Polyline subarc between fractional vertex indices."""
    def point(t):
        i=min(int(t),len(points)-2)
        return points[i]+(t-i)*(points[i+1]-points[i])
    out=[point(left)]
    return [out[0].tolist(), *points[int(left)+1:math.ceil(right)].tolist(), point(right).tolist()]


def assemble(extraction, width, height):
    paths=extraction['paths']; matches=extraction['matches']
    pieces=[]
    for i,path in enumerate(paths):
        pts=np.asarray(path['points'],dtype=float)
        if path.get('closed') and np.linalg.norm(pts[0]-pts[-1])>.01:
            pts=np.vstack([pts,pts[0]])
        pieces.append(dict(points=pts,closed=path.get('closed',False),
                           nodes=[('end',i,0),('end',i,1)],events=[],over=True))
    for mi,match in enumerate(matches):
        a,b=match['a'],match['b']
        pa=pieces[a[0]]['points'][0 if a[1]==0 else -1]
        pb=pieces[b[0]]['points'][0 if b[1]==0 else -1]
        pts=np.asarray(match.get('points',[pa,pb]),dtype=float)
        pieces.append(dict(points=pts,closed=False,nodes=[('end',*a),('end',*b)],events=[],over=False))
    # Index visible segments; a gap is under every visible segment it intersects.
    grid=defaultdict(list);cell=32
    for pi,piece in enumerate(pieces[:len(paths)]):
        for si,(a,b) in enumerate(zip(piece['points'],piece['points'][1:])):
            for x in range(math.floor(min(a[0],b[0])/cell),math.floor(max(a[0],b[0])/cell)+1):
                for y in range(math.floor(min(a[1],b[1])/cell),math.floor(max(a[1],b[1])/cell)+1):
                    grid[x,y].append((pi,si))
    crossings=[]
    for gi in range(len(paths),len(pieces)):
        gap=pieces[gi]
        for gs,(a,b) in enumerate(zip(gap['points'],gap['points'][1:])):
            candidates=set()
            for x in range(math.floor(min(a[0],b[0])/cell),math.floor(max(a[0],b[0])/cell)+1):
                for y in range(math.floor(min(a[1],b[1])/cell),math.floor(max(a[1],b[1])/cell)+1):
                    candidates.update(grid[x,y])
            for pi,si in sorted(candidates):
                piece=pieces[pi]; pts=piece['points']
                hit=intersection(a,b,pts[si],pts[si+1])
                if hit is None:continue
                t,u,p=hit
                # Endpoints coincide with their own connected visible paths.
                if min(np.linalg.norm(p-gap['points'][0]),np.linalg.norm(p-gap['points'][-1]))<1.5:continue
                if any(np.linalg.norm(p-np.asarray(c['point']))<1.5 for c in crossings):continue
                cid=len(crossings)
                crossings.append(dict(id=cid,point=p.tolist(),ports=[],over=1))
                gap['events'].append((gs+t,('cross',cid)))
                piece['events'].append((si+u,('cross',cid)))
    # Piece endpoints have degree two and will be contracted; crossing nodes
    # have degree four with over/under inherited from the visible/gap pieces.
    raw=[];loops=[]
    for piece in pieces:
        pts=piece['points'];events=sorted(piece['events'])
        if piece['closed']:
            if not events:
                loops.append(dict(points=pts.tolist(),closed=True,start=None,end=None))
                continue
            first=events[0][0]
            # Create intervals including the wrap from final event to first.
            for i,(t,node) in enumerate(events):
                nxt,other=events[(i+1)%len(events)]
                if i+1<len(events):points=_section(pts,t,nxt)
                else:points=_section(pts,t,len(pts)-1)+_section(pts,0,first)[1:]
                raw.append(dict(points=points,nodes=[node,other],over=[piece['over']]*2))
        else:
            events=[(0.,piece['nodes'][0])]+events+[(float(len(pts)-1),piece['nodes'][1])]
            for (t,node),(u,other) in zip(events,events[1:]):
                if u-t<1e-6:raise ValueError('Coincident gap and crossing endpoints.')
                raw.append(dict(points=_section(pts,t,u),nodes=[node,other],over=[piece['over']]*2))
    incidence=defaultdict(list)
    for i,e in enumerate(raw):
        for end,node in enumerate(e['nodes']):incidence[node].append((i,end))
    for node,refs in incidence.items():
        wanted=4 if node[0]=='cross' else 2
        if len(refs)!=wanted:raise ValueError(f'Expected {wanted} strands at {node}; found {len(refs)}.')
    used=set();edges=[];over_flags={}
    # Start at a crossing whenever possible; remaining pieces form free circles.
    seeds=[(i,end) for i,e in enumerate(raw) for end,n in enumerate(e['nodes']) if n[0]=='cross']
    seeds.extend((i,0) for i in range(len(raw)))
    for seed in seeds:
        if seed[0] in used:continue
        i,end=seed;node=raw[i]['nodes'][end];is_loop=node[0]!='cross'
        start=None if is_loop else node[1]
        points=[];start_over=raw[i]['over'][end];end_over=None;finish=None
        for _ in range(len(raw)+1):
            if i in used:break
            used.add(i);e=raw[i]
            pts=e['points'] if end==0 else list(reversed(e['points']))
            points.extend(pts if not points else pts[1:])
            nextnode=e['nodes'][1-end]
            if nextnode[0]=='cross':
                finish=nextnode[1];end_over=e['over'][1-end];break
            refs=incidence[nextnode]
            i,end=next(x for x in refs if x!=(i,1-end))
        eid=len(edges)
        edges.append(dict(id=eid,points=points,start=None if is_loop else [start,-1],
                          end=None if is_loop else [finish,-1],closed=is_loop,component=0))
        if not is_loop:
            over_flags[eid,0]=start_over;over_flags[eid,1]=end_over
    for loop in loops:edges.append(dict(id=len(edges),component=0,**loop))
    for c in crossings:
        refs=[];center=np.asarray(c['point'])
        for e in edges:
            for end,key in enumerate(('start','end')):
                if e[key] is None or e[key][0]!=c['id']:continue
                pts=e['points'] if end==0 else list(reversed(e['points']))
                # A local tangent avoids ordering errors from long curved arcs.
                q=next((np.asarray(p) for p in pts[1:] if np.linalg.norm(np.asarray(p)-center)>=2),np.asarray(pts[-1]))
                delta=q-center
                refs.append((math.atan2(-delta[1],delta[0]),e['id'],end))
        refs.sort()
        if len(refs)!=4:raise ValueError('Crossing does not have four incident ports.')
        flags=[]
        for p,(_,ei,end) in enumerate(refs):
            c['ports'].append(dict(edge=ei,end=end));edges[ei]['start' if end==0 else 'end']=[c['id'],p]
            flags.append(over_flags[ei,end])
        if flags[0]!=flags[2] or flags[1]!=flags[3] or flags[0]==flags[1]:
            raise ValueError('Inferred crossing is not transverse.')
        c['over']=0 if flags[0] else 1
    diagram=dict(crossings=crossings,edges=edges,width=width,height=height,component_orientations={})
    assign_components(diagram)
    verdict=validate(diagram)
    if not verdict['valid']:raise ValueError('; '.join(verdict['errors']))
    return diagram


def _crossing_order_error(diagram):
    edges = {edge['id']: edge for edge in diagram['edges']}
    for crossing in diagram['crossings']:
        if len(crossing['ports']) != 4:
            return 'Moving this point changes the cyclic strand order at a crossing.'
        angles = []
        center = crossing['point']
        for index, port in enumerate(crossing['ports']):
            points = edges[port['edge']]['points']
            sequence = iter(points) if port['end'] == 0 else reversed(points)
            next(sequence, None)
            point = next((p for p in sequence if math.dist(p, center) > 1.), None)
            if point is None:
                return 'An arc collapsed into a crossing.'
            angles.append((math.atan2(center[1]-point[1], point[0]-center[0]), index))
        order = [index for _, index in sorted(angles)]
        if any((order[(i+1) % 4]-order[i]) % 4 != 1 for i in range(4)):
            return 'Moving this point changes the cyclic strand order at a crossing.'
    return None


def _stroke_pair_key(a, b, i, j):
    """Identify exact geometric segments independently of edge IDs/direction."""
    first = tuple(sorted((tuple(a[i]), tuple(b[i]))))
    second = tuple(sorted((tuple(a[j]), tuple(b[j]))))
    return tuple(sorted((first, second)))


def embedding_stroke_contacts(diagram):
    """Record existing display-width contacts, without accepting intersections.

    Local surgery may join or reverse edges while leaving their distant pieces
    untouched. These exact segment-pair keys allow that inherited tight spacing
    to remain. Contacts exempted by an encoded crossing are NOT recorded: after
    removing that crossing, its former halo must satisfy ordinary clearance.
    """
    contacts = set()
    verdict = check_embedding(diagram, _collect_stroke_contacts=contacts)
    if not verdict['valid']:
        raise ValueError('; '.join(verdict['errors']))
    return frozenset(contacts)


def _embedding_candidate_pairs(a, b, owner, local, counts, arc_start, arc_end,
                               totals, closed, clearance, active=None):
    """Exact centerline candidates plus only nonlocal stroke candidates.

    A densely sampled strand can put hundreds of its own neighboring segments
    inside one stroke radius. Their stroke distance is exempt by material
    arclength, but their centerlines still require intersection checks. Query
    unpadded segment boxes for the latter. For stroke clearance, first query
    small contiguous blocks, discard wholly exempt block pairs, then expand
    the remaining blocks and apply the exact segment/material predicates.
    No vertices are removed, chords replaced, or clearance tolerances changed.
    When supplied, ``active`` is uniform for all segments of each edge owner.
    """
    from .motion import _box_pairs

    low, high = np.minimum(a, b), np.maximum(a, b)
    exact = _box_pairs(low, high, active=active)
    if len(exact):
        i, j = exact.T
        adjacent = (owner[i] == owner[j]) & ((local[j]-local[i] <= 1)
                    | (closed[owner[i]] & (local[j]-local[i] == counts[owner[i]]-1)))
        exact = exact[~adjacent]
    if not clearance or not len(a):
        return exact

    size = 32
    edge_starts = np.r_[0, np.flatnonzero(np.diff(owner))+1]
    edge_ends = np.r_[edge_starts[1:], len(a)]
    starts = np.concatenate([np.arange(left, right, size)
                             for left, right in zip(edge_starts, edge_ends)])
    ends = np.minimum(starts+size, np.repeat(edge_ends, (edge_ends-edge_starts+size-1)//size))
    low, high = low-clearance/2, high+clearance/2
    block_low = np.minimum.reduceat(low, starts, axis=0)
    block_high = np.maximum.reduceat(high, starts, axis=0)
    block_active = None if active is None else active[starts]
    blocks = _box_pairs(block_low, block_high, active=block_active)
    # A long/sparsely sampled block may itself contain nonlocal material.
    self_ids = np.arange(len(starts)) if block_active is None else np.flatnonzero(block_active)
    blocks = np.vstack((blocks, np.column_stack((self_ids, self_ids))))
    i, j = blocks.T
    same = owner[starts[i]] == owner[starts[j]]
    minimum = np.maximum(0., arc_start[starts[j]]-arc_end[ends[i]-1])
    maximum = np.maximum(0., arc_start[ends[j]-1]-arc_end[starts[i]])
    exempt = same & ((maximum < clearance*4)
                    | (closed[owner[starts[i]]] & (totals[owner[starts[i]]]-minimum < clearance*4)))
    blocks = blocks[~exempt]
    parts = [exact]
    offsets = np.arange(size)
    for begin in range(0, len(blocks), 128):
        first, second = blocks[begin:begin+128].T
        shape = (len(first), size, size)
        ii = np.broadcast_to(starts[first, None, None]+offsets[None, :, None], shape)
        jj = np.broadcast_to(starts[second, None, None]+offsets[None, None, :], shape)
        valid = ((ii < ends[first, None, None]) & (jj < ends[second, None, None]) & (ii < jj))
        i, j = ii[valid], jj[valid]
        same = owner[i] == owner[j]
        separation = np.maximum(0., arc_start[j]-arc_end[i])
        nearby = same & ((separation < clearance*4)
                        | (closed[owner[i]] & (totals[owner[i]]-separation < clearance*4)))
        i, j = i[~nearby], j[~nearby]
        overlaps = np.all(high[i] >= low[j], axis=1) & np.all(high[j] >= low[i], axis=1)
        if np.any(overlaps):
            parts.append(np.column_stack((i[overlaps], j[overlaps])))
    pairs = np.concatenate(parts)
    if not len(pairs):
        return pairs
    pairs = pairs[np.lexsort((pairs[:, 0], pairs[:, 1]))]
    return pairs[np.r_[True, np.any(pairs[1:] != pairs[:-1], axis=1)]]


def check_embedding(diagram, check_strokes=True, *, inherited_stroke_contacts=(),
                    _collect_stroke_contacts=None, active_edge_ids=None):
    """Check the exact stored polyline using a spatial broad phase and batches.

    Segment distance, encoded-crossing exceptions, local arclength exemptions,
    and cyclic port order retain the editor's tolerances. No vertices are
    decimated. This validates the final embedding, not the swept 3-D motion.
    Optional inherited contacts must come from embedding_stroke_contacts on
    the source diagram. Only identical segment pairs receive that exemption;
    new contacts, exact intersections, and crossing order are still checked.
    An active-edge subset is only for surgery on an already checked source:
    it tests changed edges against ALL possible neighbors, while omitting
    pairs of edges whose geometry and crossing incidences both stayed fixed.
    """
    error = _crossing_order_error(diagram)
    if error:
        return {'valid': False, 'errors': [error]}
    stroke = max(2., min(7., math.sqrt(diagram.get('width', 800)*diagram.get('height', 600))/220))
    clearance = stroke if check_strokes else 0.
    arrays, counts, arcs, totals, endpoints, closed, active = [], [], [], [], [], [], []
    crossings = {c['id']: i for i, c in enumerate(diagram['crossings'])}
    # The sentinel row can never exempt a segment pair from the collision test.
    crossing_points = np.asarray([c['point'] for c in diagram['crossings']]+[[np.inf, np.inf]], float)
    missing = len(crossings)
    for edge in diagram['edges']:
        points = distinct_vertices(edge['points'])
        if len(points) < 2:
            return {'valid': False, 'errors': ['A strand collapsed into a point.']}
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        arrays.append(points); counts.append(len(points)-1); arcs.append(s)
        totals.append(s[-1]); closed.append(bool(edge.get('closed')))
        active.append(active_edge_ids is None or edge['id'] in active_edge_ids)
        endpoints.append([crossings[loc[0]] if loc is not None else missing
                          for loc in (edge.get('start'), edge.get('end'))])
    if not arrays:
        return {'valid': True, 'errors': []}
    a = np.concatenate([p[:-1] for p in arrays]); b = np.concatenate([p[1:] for p in arrays])
    owner = np.repeat(np.arange(len(arrays)), counts)
    local = np.concatenate([np.arange(n) for n in counts])
    arc_start = np.concatenate([s[:-1] for s in arcs]); arc_end = np.concatenate([s[1:] for s in arcs])
    counts = np.asarray(counts); totals = np.asarray(totals); closed = np.asarray(closed)
    endpoints = np.asarray(endpoints)
    # Retaining the exact final endpoint can leave a sub-tolerance final
    # segment after duplicate cleaning. Match the scalar guard's skip while
    # keeping the original adjacency indices and material arclengths.
    keep = np.linalg.norm(b-a, axis=1) >= 1e-8
    a, b, owner, local = a[keep], b[keep], owner[keep], local[keep]
    arc_start, arc_end = arc_start[keep], arc_end[keep]
    active = None if active_edge_ids is None else np.asarray(active)[owner]
    pairs = _embedding_candidate_pairs(a, b, owner, local, counts, arc_start,
                                       arc_end, totals, closed, clearance, active=active)
    if len(pairs):
        i, j = pairs.T
        same = owner[i] == owner[j]
        adjacent = same & ((local[j]-local[i] <= 1)
                   | (closed[owner[i]] & (local[j]-local[i] == counts[owner[i]]-1)))
        pairs = pairs[~adjacent]
    # Bound temporary projection arrays even for a densely packed diagram.
    for begin in range(0, len(pairs), 32768):
        i, j = pairs[begin:begin+32768].T
        def encoded(ii, jj, points, radius):
            allowed = np.zeros(len(ii), bool)
            for left in (0, 1):
                cid = endpoints[owner[ii], left]
                shared = ((cid == endpoints[owner[jj], 0]) | (cid == endpoints[owner[jj], 1])) & (cid != missing)
                delta = points-crossing_points[cid]
                allowed |= shared & (np.hypot(delta[:, 0], delta[:, 1]) < radius)
            return allowed

        aa, bb, cc, dd = a[j], b[j], a[i], b[i]
        r, v, q = bb-aa, dd-cc, cc-aa
        den = r[:, 0]*v[:, 1]-r[:, 1]*v[:, 0]
        safe_den = np.where(abs(den) >= 1e-9, den, 1.)
        t = (q[:, 0]*v[:, 1]-q[:, 1]*v[:, 0])/safe_den
        u = (q[:, 0]*r[:, 1]-q[:, 1]*r[:, 0])/safe_den
        # Validate finite segments, not their supporting-line extensions.
        # Mesh transport can leave a genuine tiny intervening edge; extending
        # its neighbors across it invents an unencoded self-intersection.
        # Endpoint proximity is covered independently by stroke clearance.
        hit = ((abs(den) >= 1e-9) & (t >= 0.) & (t <= 1.)
               & (u >= 0.) & (u <= 1.))
        intersection_points = aa+t[:, None]*r
        bad_hit = np.zeros(len(i), bool)
        if np.any(hit):
            bad_hit[hit] = ~encoded(i[hit], j[hit], intersection_points[hit], .8)
        bad_gap = np.zeros(len(i), bool)
        centerline_contact = np.zeros(len(i), bool)
        gap_points = np.empty((len(i), 2))
        if check_strokes:
            same = owner[i] == owner[j]
            separation = np.maximum(0., arc_start[j]-arc_end[i])
            nearby = same & ((separation < stroke*4)
                            | (closed[owner[i]] & (totals[owner[i]]-separation < stroke*4)))
            # A dense polyline puts many nearby pieces of the same arc in the
            # broad phase. Their centerline intersections still need the test
            # above, but their stroke clearance is explicitly exempt. Filter
            # them before allocating four projections for every pair.
            distant = np.flatnonzero(~nearby)
            aa, bb, cc, dd = aa[distant], bb[distant], cc[distant], dd[distant]
            r, v = r[distant], v[distant]
            # Four endpoint projections, exactly as in the scalar validator.
            rr = np.maximum(1e-12, np.sum(r*r, axis=1))
            vv = np.maximum(1e-12, np.sum(v*v, axis=1))
            p0 = cc+np.clip(np.sum((aa-cc)*v, axis=1)/vv, 0., 1.)[:, None]*v
            p1 = cc+np.clip(np.sum((bb-cc)*v, axis=1)/vv, 0., 1.)[:, None]*v
            p2 = aa+np.clip(np.sum((cc-aa)*r, axis=1)/rr, 0., 1.)[:, None]*r
            p3 = aa+np.clip(np.sum((dd-aa)*r, axis=1)/rr, 0., 1.)[:, None]*r
            left = np.stack([aa, bb, p2, p3], axis=1)
            right = np.stack([p0, p1, cc, dd], axis=1)
            distance2 = np.sum((left-right)**2, axis=2)
            nearest = np.argmin(distance2, axis=1)
            row = np.arange(len(distant))
            too_close = distance2[row, nearest] < (stroke*.95)**2
            close = distant[too_close]
            if len(close):
                points = (left[row[too_close], nearest[too_close]]
                          + right[row[too_close], nearest[too_close]])/2
                gap_points[close] = points
                bad_gap[close] = ~encoded(i[close], j[close], points, stroke*3.5)
                # Parallel/retracing segments have zero determinant and are
                # not caught by the transverse intersection predicate. Never
                # inherit actual centerline contact as a cosmetic tight gap.
                centerline_contact[close] = distance2[row[too_close], nearest[too_close]] <= 1e-16
        if _collect_stroke_contacts is not None or inherited_stroke_contacts:
            for index in np.flatnonzero(bad_gap & ~centerline_contact):
                key = _stroke_pair_key(a, b, i[index], j[index])
                if _collect_stroke_contacts is not None:
                    _collect_stroke_contacts.add(key)
                    bad_gap[index] = False
                elif key in inherited_stroke_contacts:
                    bad_gap[index] = False
        bad = np.flatnonzero(bad_hit | bad_gap)
        if len(bad):
            index = bad[0]
            message, point = ('The move creates an intersection that has no crossing in the diagram.', intersection_points[index]) if bad_hit[index] else (
                'The move makes nonadjacent strands touch or overlap at the display stroke width.', gap_points[index])
            return {'valid': False, 'errors': [message], 'point': point.tolist()}
    return {'valid': True, 'errors': []}


def beautify(diagram):
    """Round pixel stair steps without changing crossing incidence or rotation.

    Two corner-cutting passes preserve crossing endpoints. The result is kept
    only when its centerlines still realize the same planar crossing graph.
    """
    from copy import deepcopy
    out=deepcopy(diagram)
    for edge in out['edges']:
        points=np.asarray(edge['points'],dtype=float)
        for _ in range(2):
            if len(points)<3:break
            a=.75*points[:-1]+.25*points[1:]
            b=.25*points[:-1]+.75*points[1:]
            middle=np.stack([a,b],axis=1).reshape(-1,2)
            points=np.vstack([points[0],middle,points[-1]])
        edge['points']=points.tolist()
    verdict=check_embedding(out, check_strokes=False)
    if verdict['valid']:
        out['geometry_processing']='Two endpoint-preserving corner-cutting passes; planar embedding checked.'
        return out
    diagram['geometry_processing']='Original traced centerlines retained: smoothing did not pass embedding checks.'
    return diagram
