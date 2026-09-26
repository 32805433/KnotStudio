"""Conservative arrow-wing recognition on an unpruned strand skeleton.

Only a *pair* of terminal backward-facing wings may be removed, and only when
exactly two other arms continue a locally straight shaft through their connected
junction. Real crossing arms, single T-spurs and closed curls are not wings.
Coordinates returned for review are in the input skeleton's pixel frame (x,y).
"""
from itertools import combinations
import math
import numpy as np


def prune_arrow_wings(graph, stroke_width):
    """Remove geometrically supported paired arrow wings in place.

    Call before generic short-spur removal, so both arrow wings remain available
    as evidence. This does not infer component orientation: orientation must be
    read from the untouched original image after reconstructing the diagram.
    """
    from .knotfolio_backend import _trace
    width=max(1.5,float(stroke_width))
    chains=_trace(graph)
    if not chains:return []
    lengths=[sum(math.dist(a,b) for a,b in zip(c,c[1:])) for c in chains]
    parent={p:p for p in graph if len(graph[p])>2}
    if not parent:return []
    def root(p):
        while parent[p]!=p:
            parent[p]=parent[parent[p]];p=parent[p]
        return p
    for chain,length in zip(chains,lengths):
        a,b=chain[0],chain[-1]
        if a in parent and b in parent and length<=2.5*width:
            parent[root(a)]=root(b)
    groups={}
    for p in parent:groups.setdefault(root(p),{'junction':set(),'arms':[]})['junction'].add(p)
    for chain,length in zip(chains,lengths):
        a,b=chain[0],chain[-1]
        if a in parent and b in parent and root(a)==root(b):
            groups[root(a)]['junction'].update(chain)
            continue
        for points in (chain,chain[::-1]):
            if points[0] in parent:
                groups[root(points[0])]['arms'].append({'points':points,'length':length,
                    'terminal':len(graph[points[-1]])==1})
    evidence=[];deleting=set()
    for group in groups.values():
        arms=group['arms']
        # Ambiguous neighboring crossings / noisy stars are never simplified.
        if len(arms)!=4:continue
        junction=np.asarray(list(group['junction']),float)[:,::-1]
        if max(np.ptp(junction,axis=0))>3*width:continue
        for ia,ib in combinations(range(4),2):
            wings=[arms[ia],arms[ib]];shaft=[a for i,a in enumerate(arms) if i not in (ia,ib)]
            if not all(a['terminal'] and .65*width<=a['length']<=18*width for a in wings):continue
            if max(a['length'] for a in wings)>3.5*min(a['length'] for a in wings):continue
            anchor=np.mean([np.asarray(a['points'][0][::-1],float) for a in wings],axis=0)
            vectors=[np.asarray(a['points'][-1][::-1],float)-anchor for a in wings]
            norms=[np.linalg.norm(v) for v in vectors]
            if any(n/a['length']<.66 for n,a in zip(norms,wings)):continue
            units=[v/max(1e-12,n) for v,n in zip(vectors,norms)]
            direction=-(units[0]+units[1]);direction/=max(1e-12,np.linalg.norm(direction))
            backward=[-float(u@direction) for u in units]
            normal=np.array([-direction[1],direction[0]])
            sideways=[float(v@normal) for v in vectors]
            if (min(backward)<.28 or max(backward)>.94 or sideways[0]*sideways[1]>=0
                    or min(abs(v) for v in sideways)<.65*width
                    or max(abs(v) for v in sideways)>3*min(abs(v) for v in sideways)):
                continue
            shaft_vectors=[];shaft_ok=True
            for arm in shaft:
                points=np.asarray(arm['points'],float)[:,::-1]
                distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
                reach=min(arm['length'],max(4*width,.65*max(norms)))
                if reach<2.5*width:
                    shaft_ok=False;break
                sample=np.array([np.interp(reach,distances,points[:,j]) for j in (0,1)])
                vector=sample-anchor; norm=np.linalg.norm(vector)
                if norm<.65*reach:
                    shaft_ok=False;break
                shaft_vectors.append(vector/norm)
            if not shaft_ok:continue
            align=[float(v@direction) for v in shaft_vectors]
            if min(abs(a) for a in align)<.88 or align[0]*align[1]>=0:continue
            if float(shaft_vectors[0]@shaft_vectors[1])>-.90:continue
            # A wing cannot include/reach any other branch or form a cycle.
            removing=set().union(*(set(a['points'][1:]) for a in wings))
            if any(len(graph[p])>2 for p in removing) or removing&group['junction']:continue
            deleting.update(removing)
            evidence.append({'point':anchor.tolist(),'direction':direction.tolist(),
                'wing_endpoints':[np.asarray(a['points'][-1][::-1],float).tolist() for a in wings],
                'wing_lengths':[float(a['length']) for a in wings],
                'removed_pixels':len(removing)})
            break
    for p in deleting:
        for q in graph.pop(p,[]):
            if q in graph and p in graph[q]:graph[q].remove(p)
    return evidence
