"""Separate paired arrow wings from photographic strand masks.

Only two short, backward-facing leaves on a continuing shaft qualify. The
source remains unchanged; the returned copy is a tracing hypothesis and the
arrow direction is retained as source evidence, in mask coordinates.
"""
from itertools import combinations

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .knotfolio_backend import _adjacency, _trace, _remove_pixel_cycles, _prune_spurs
from .bundle_contacts import _distance_to_path


def paired_shaft_arrows(ink):
    """Return shared pixel evidence for two wings on a continuing shaft.

    Both label protection and photo tracing use this same geometry: two
    terminal wings point backward on opposite sides of a shaft that continues
    through the arrow in both directions. The component's overall aspect ratio
    is irrelevant. The returned paths use (x, y) mask coordinates; input ink is
    never changed.
    """
    ink = np.asarray(ink, dtype=bool)
    skeleton = skeletonize(ink)
    radii = ndimage.distance_transform_edt(np.pad(ink, 1))[1:-1, 1:-1][skeleton]
    width = max(1.5, float(np.median(radii)*2)) if len(radii) else 1.5
    graph = _adjacency(skeleton)
    _prune_spurs(graph, 1.5*width)
    _remove_pixel_cycles(graph, width)
    _prune_spurs(graph, 1.5*width)
    chains = _trace(graph)
    parent = {p: p for p in graph if len(graph[p]) > 2}
    def root(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    lengths = [sum(np.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(c,c[1:])) for c in chains]
    for chain, length in zip(chains, lengths):
        a,b=chain[0],chain[-1]
        if a in parent and b in parent and length <= 6*width:
            parent[root(a)] = root(b)
    groups = {}
    for i,(chain,length) in enumerate(zip(chains,lengths)):
        for end in (0,1):
            ordered = chain if not end else chain[::-1]
            a,b = ordered[0],ordered[-1]
            if a not in parent or (b in parent and root(a)==root(b) and length<=6*width):
                continue
            groups.setdefault(root(a), []).append((i,ordered,length,len(graph[b])==1))
    evidence=[]
    for group in groups.values():
        leaves=[r for r in group if r[3] and 1.5*width<=r[2]<=18*width]
        for first,second in combinations(leaves,2):
            # Tiny raster burrs are already removed by the ordinary
            # tracer; they do not turn a paired arrow into a many-arm fork.
            rest=[r for r in group if r[0] not in (first[0],second[0])
                  and not (r[3] and r[2]<1.5*width)]
            if len(rest)!=2 or min(r[2] for r in rest)<5*width:
                continue
            pa,pb=[np.array(r[1],float)[:,::-1] for r in (first,second)]
            anchor=(pa[0]+pb[0])/2
            va,vb=pa[-1]-anchor,pb[-1]-anchor
            if min(np.linalg.norm(va)/first[2],np.linalg.norm(vb)/second[2])<.65:
                continue
            ua,ub=va/np.linalg.norm(va),vb/np.linalg.norm(vb)
            direction=-(ua+ub); norm=np.linalg.norm(direction)
            if norm<.5 or max(first[2],second[2])/min(first[2],second[2])>3.5:
                continue
            direction/=norm
            normal=np.array([-direction[1],direction[0]])
            sideways=[float(v@normal) for v in (va,vb)]
            backwards=[-float(u@direction) for u in (ua,ub)]
            if (sideways[0]*sideways[1]>=0 or min(abs(v) for v in sideways)<.6*width
                    or min(backwards)<.22 or max(backwards)>.96):
                continue
            shaft=[]; tangents=[]
            for _,chain,_,_ in rest:
                p=np.array(chain,float)[:,::-1]
                arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
                sample=np.array([np.interp(min(8*width,arc[-1]),arc,p[:,a]) for a in (0,1)])
                tangent=sample-anchor; tangent/=max(1e-9,np.linalg.norm(tangent))
                shaft.append(p); tangents.append(tangent)
            if tangents[0]@tangents[1]>-.75 or min(abs(t@direction) for t in tangents)<.8:
                continue
            own = {first[0], second[0], rest[0][0], rest[1][0]}
            evidence.append(dict(point=anchor.tolist(), direction=direction.tolist(),
                                 wing_lengths=[first[2],second[2]], stroke_width=width,
                                 wings=[pa,pb], shaft=shaft,
                                 other_paths=[np.array(c,float)[:,::-1]
                                              for i,c in enumerate(chains)
                                              if i not in own and len(c)>1]))
            break
    return evidence


def separate_photo_arrows(rgb):
    ink = np.asarray(rgb).min(2) < 205
    output = np.asarray(rgb).copy()
    evidence = []
    for record in paired_shaft_arrows(ink):
        pa, pb = record['wings']
        shaft = record['shaft']
        width = record['stroke_width']
        lo=np.floor(np.minimum(pa.min(0),pb.min(0))-width-1).astype(int)
        hi=np.ceil(np.maximum(pa.max(0),pb.max(0))+width+2).astype(int)
        x1,y1=np.maximum(lo,[0,0]);x2,y2=np.minimum(hi,[ink.shape[1],ink.shape[0]])
        yy,xx=np.mgrid[y1:y2,x1:x2]; points=np.column_stack([xx.ravel(),yy.ravel()])
        wing_distance=np.minimum(_distance_to_path(points,pa),_distance_to_path(points,pb))
        protected=np.minimum(_distance_to_path(points,shaft[0]),_distance_to_path(points,shaft[1]))<=.8*width+1
        for other in record['other_paths']:
            if np.any(other.max(0)<[x1-width,y1-width]) or np.any(other.min(0)>[x2+width,y2+width]):
                continue
            protected|=_distance_to_path(points,other)<=.8*width+1
        erase=((wing_distance<=.9*width+1)&~protected).reshape(yy.shape)
        view=output[y1:y2,x1:x2]
        if np.any(erase & (view.min(2)<205)):
            view[erase]=255
            evidence.append({key:record[key] for key in
                             ('point','direction','wing_lengths','stroke_width')})
    return (output if evidence else None), evidence
