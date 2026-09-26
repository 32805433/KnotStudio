"""Failure-only recognition of paired arrows on curved, hand-drawn shafts.

The shaft determines direction locally; the two wings supply independent
backward/side evidence. No single spur or solid crossing is interpreted as an
arrow, and a closed curl can never be a terminal wing.
"""
from itertools import combinations
import math
import time
import numpy as np


def prune_curved_arrow_wings(graph, stroke_width):
    from .knotfolio_backend import _trace
    width = max(1.5, float(stroke_width))
    chains = _trace(graph)
    lengths = [sum(math.dist(a,b) for a,b in zip(c,c[1:])) for c in chains]
    parent = {p:p for p in graph if len(graph[p])>2}
    def root(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]; p = parent[p]
        return p
    for chain, length in zip(chains,lengths):
        a,b = chain[0],chain[-1]
        if a in parent and b in parent and length <= 3.2*width:
            parent[root(a)] = root(b)
    groups = {}
    for p in parent:
        groups.setdefault(root(p), {'junction':set(), 'arms':[]})['junction'].add(p)
    for chain,length in zip(chains,lengths):
        a,b = chain[0],chain[-1]
        if a in parent and b in parent and root(a)==root(b):
            groups[root(a)]['junction'].update(chain)
            continue
        for points in (chain,chain[::-1]):
            if points[0] in parent:
                groups[root(points[0])]['arms'].append(dict(points=points,length=length,
                    terminal=len(graph[points[-1]])==1))
    deleting=set(); evidence=[]
    for group in groups.values():
        arms=group['arms']
        if len(arms)!=4: continue
        junction=np.asarray(list(group['junction']),float)[:,::-1]
        if np.ptp(junction,axis=0).max()>4*width: continue
        candidates=[]
        for ia,ib in combinations(range(4),2):
            wings=[arms[ia],arms[ib]]
            shaft=[a for i,a in enumerate(arms) if i not in (ia,ib)]
            if not all(a['terminal'] and .65*width<=a['length']<=10*width for a in wings): continue
            if max(a['length'] for a in wings)>4*min(a['length'] for a in wings): continue
            anchor=np.mean([a['points'][0][::-1] for a in wings],axis=0)
            shaft_vectors=[]
            for arm in shaft:
                p=np.asarray(arm['points'],float)[:,::-1]
                s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
                if s[-1]<5*width: break
                # Short reaches follow a curved shaft without treating the
                # large-scale arc chord as its tangent at the arrow.
                vectors=[]
                for reach in (1.75*width,2.5*width):
                    v=np.array([np.interp(reach,s,p[:,j]) for j in (0,1)])-anchor
                    norm=np.linalg.norm(v)
                    if norm<.6*reach: break
                    vectors.append(v/norm)
                if len(vectors)!=2: break
                shaft_vectors.append(vectors)
            if len(shaft_vectors)!=2: continue
            shaft_vectors=np.asarray(shaft_vectors)
            if np.any(np.sum(shaft_vectors[0]*shaft_vectors[1],axis=1)>-.82): continue
            axis=(shaft_vectors[0]-shaft_vectors[1]).mean(0)
            axis/=np.linalg.norm(axis)
            endpoints=[np.asarray(a['points'][-1][::-1],float)-anchor for a in wings]
            # The orientation is determined by both wings, not the identity or
            # arbitrary traversal direction of either shaft arm.
            if sum(float(v@axis) for v in endpoints)>0: axis=-axis
            normal=np.array([-axis[1],axis[0]])
            sides=[float(v@normal) for v in endpoints]
            if sides[0]*sides[1]>=0 or min(abs(v) for v in sides)<.65*width: continue
            if max(abs(v) for v in sides)>3*min(abs(v) for v in sides): continue
            units=[v/np.linalg.norm(v) for v in endpoints]
            bisector=-(units[0]+units[1])
            bisector/=max(1e-9,np.linalg.norm(bisector))
            if float(bisector@axis)<.85: continue
            good=True
            for arm,end in zip(wings,endpoints):
                norm=np.linalg.norm(end)
                if norm<1.2*width or norm/arm['length']<.5: good=False; break
                backward=-float(end@axis)/norm
                if not .1<backward<.99: good=False; break
                p=np.asarray(arm['points'],float)[:,::-1]-anchor
                if np.max(p@axis)>width: good=False; break
                # A hooked wing may curve back, but cannot make a near-closed
                # excursion substantially bigger than its endpoint evidence.
                if np.linalg.norm(p,axis=1).max()>1.45*norm: good=False; break
            if not good: continue
            removing=set().union(*(set(a['points'][1:]) for a in wings))
            if removing & group['junction'] or any(len(graph[p])>2 for p in removing): continue
            candidates.append((removing,dict(point=anchor.tolist(),direction=axis.tolist(),
                wing_endpoints=[[int(v) for v in a['points'][-1][::-1]] for a in wings],
                wing_lengths=[a['length'] for a in wings],removed_pixels=len(removing),
                method='paired_wings_local_curved_shaft')))
        # Multiple plausible choices at the same junction remain ambiguous.
        if len(candidates)==1:
            removing,record=candidates[0];deleting.update(removing);evidence.append(record)
    for p in deleting:
        for q in graph.pop(p,[]):
            if q in graph and p in graph[q]: graph[q].remove(p)
    return evidence


def recognize_drawn_arrows(rgb):
    """Require multiple masks to agree on a complete source-supported graph."""
    from .board_photo import looks_photographic
    from .diagram import pd_code,validate
    from .geometry import assemble,beautify,check_embedding
    from .knotfolio_backend import extract
    from .motion import _same_projection
    from .preprocess import prepare
    if looks_photographic(rgb)[0]: return None
    pixels,preprocessing=prepare(rgb)
    complete=[];attempts=[];started=time.monotonic()
    for threshold in (150.,180.,205.):
        trace=extract(pixels,dict(threshold=threshold,arrow_wings=True,
            adaptive_arrow_wings=True,micro_seams=True,bundle_gaps=True,
            stop_at_branches=True,max_endpoints=180))
        record=dict(threshold=threshold,complete=trace['complete'],
            branches=len(trace['diagnostics']['branch_points']))
        attempts.append(record)
        if not trace['complete'] or not any(a.get('method')=='paired_wings_local_curved_shaft'
                for a in trace['diagnostics'].get('arrow_pruning',[])): continue
        try:
            diagram=beautify(assemble(trace,rgb.shape[1],rgb.shape[0]))
            verdict=validate(diagram)
            if not verdict['valid'] or not check_embedding(diagram,check_strokes=False)['valid']: continue
            complete.append((diagram,trace,verdict,pd_code(diagram)))
        except (ValueError,RuntimeError,KeyError,IndexError): continue
    if len(complete)<2: return None
    diagram,trace,verdict,code=complete[0]
    if any(not _same_projection(diagram,c[0]) for c in complete[1:]): return None
    warning='Paired hand-drawn arrow wings were separated using the local curved shaft; inspect the marked arrows and crossings.'
    diagnostic=trace['diagnostics'];diagnostic.update(preprocessing=preprocessing,
        validation=verdict,drawn_arrow_attempts=attempts,drawn_arrow_agreement=len(complete),
        elapsed_seconds=time.monotonic()-started)
    diagram['recognition']=dict(stroke_width=diagnostic['stroke_width'],warnings=[warning])
    return dict(status='needs_review',diagram=diagram,pd_code=code,warnings=[warning],
        unlinked_unknot_components=verdict['crossing_free_components'],diagnostics=diagnostic)
