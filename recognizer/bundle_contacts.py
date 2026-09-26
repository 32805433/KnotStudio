"""Conservative repair of an under-tip touching visible over-strands.

Only an isolated three-arm junction with one straight-through pair is eligible.
The third arm must point at a unique existing endpoint across parallel visible
strands. The photo caller may explicitly allow one over-strand. Four-arm solid
crossings never supply an over/under choice here.
"""
from __future__ import annotations

import numpy as np
import cv2

from .bundled_crossings import bundle_evidence, stable_tangent
from .knotfolio_backend import _intersections


def _point_at(points, distance):
    lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if lengths[-1]<distance:
        return None
    return np.array([np.interp(distance,lengths,points[:,axis]) for axis in (0,1)])


def _distance_to_path(points, path):
    starts,ends=path[:-1],path[1:]
    v=ends-starts
    delta=points[:,None,:]-starts
    t=np.sum(delta*v,axis=2)/np.maximum(np.sum(v*v,axis=1),1e-12)
    closest=starts+np.clip(t,0,1)[...,None]*v
    return np.min(np.linalg.norm(points[:,None,:]-closest,axis=2),axis=1)


def repair_bundle_contacts(rgb, trace, *, allow_single=False):
    """Return (cleaned copy or None, diagnostics), without changing source ink.

    A repair only deletes part of the short third arm, outside a protected
    corridor around the observed continuous over-pair. The original pixels in
    that corridor are retained exactly; no new stroke is painted.
    ``allow_single`` retains every geometric certificate except bundle spacing,
    replacing that condition with a bounded, unique facing continuation.
    """
    diagnostic={'bundle_contact_repairs':[]}
    branches=np.asarray(trace.get('diagnostics',{}).get('branch_points',[]),float).reshape(-1,2)
    width=float(trace.get('diagnostics',{}).get('stroke_width',1.5))
    if (not len(branches) or len(branches)>12 or not np.isfinite(width)
            or width<=0 or not trace.get('paths')):
        return None,diagnostic
    paths=[np.asarray(p['points'],float) for p in trace['paths']]
    if any(p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all() for p in paths):
        return None,diagnostic
    starts=np.concatenate([p[:-1] for p in paths])
    ends=np.concatenate([p[1:] for p in paths])
    owners=np.concatenate([np.full(len(p)-1,i) for i,p in enumerate(paths)])
    # Parallel under-arms arrive in order. An endpoint already consumed by a
    # strong, crossing-supported continuation cannot also receive the T arm.
    occupied=set()
    for match in trace.get('matches',[]):
        if match.get('crossings') and min(match.get('alignment',[0.,0.]))>=.85:
            occupied.update((tuple(match['a']),tuple(match['b'])))
    endpoints=[]
    for i,(record,path) in enumerate(zip(trace['paths'],paths)):
        if record.get('closed'):
            continue
        for end in (0,1):
            if (i,end) in occupied:
                continue
            point=path[-1] if end else path[0]
            if np.min(np.linalg.norm(branches-point,axis=1))<1.5:
                continue
            endpoints.append((i,end,point,stable_tangent(path,end,3*width)))
    output=np.asarray(rgb).copy()
    h,w=output.shape[:2]
    _,ink_components=cv2.connectedComponents((output.min(axis=2)<245).astype(np.uint8),connectivity=8)
    for branch in branches:
        # Nearby junctions do not describe a single resolvable T.
        if np.count_nonzero(np.linalg.norm(branches-branch,axis=1)<5*width)>1:
            continue
        incident=[]
        for i,path in enumerate(paths):
            if trace['paths'][i].get('closed'):
                # Polygon simplification may rotate a closed path's start and
                # omit the collinear T vertex. Recover its two local arms by
                # projecting the known junction onto that closed polyline.
                vec=np.diff(path,axis=0)
                t=np.sum((branch-path[:-1])*vec,axis=1)/np.maximum(np.sum(vec*vec,axis=1),1e-12)
                projected=path[:-1]+np.clip(t,0,1)[:,None]*vec
                k=int(np.argmin(np.linalg.norm(projected-branch,axis=1)))
                if np.linalg.norm(projected[k]-branch)<1.5:
                    loop=np.vstack([projected[k],path[k+1:],path[1:k+1],projected[k]])
                    for direction,away in enumerate((loop,loop[::-1])):
                        if _point_at(away,3*width) is not None:
                            incident.append((i,direction,away,-stable_tangent(away,0,3*width)))
                continue
            for end in (0,1):
                if np.linalg.norm(path[-1 if end else 0]-branch)<1.5:
                    away=path[::-1] if end else path
                    sample=_point_at(away,3*width)
                    if sample is not None:
                        incident.append((i,end,away,-stable_tangent(away,0,3*width)))
        if len(incident)!=3:
            continue
        pairs=sorted([(-float(incident[a][3]@incident[b][3]),a,b)
                      for a in range(3) for b in range(a)],reverse=True)
        local_over_pair=False
        if pairs[0][0]<.88 or pairs[0][0]-pairs[1][0]<.25:
            if not allow_single:
                continue
            # On a curved over-strand, a long chord measures the bend rather
            # than the tangent at the junction. Require the same unambiguous
            # continuous pair at two shorter, stroke-relative scales. Keep
            # the third arm's longer direction for the facing-end test.
            local=[]
            for reach in (1.5*width,2*width):
                directions=[-stable_tangent(item[2],0,reach) for item in incident]
                ranked=sorted([(-float(directions[i]@directions[j]),i,j)
                               for i in range(3) for j in range(i)],reverse=True)
                if ranked[0][0]<.88 or ranked[0][0]-ranked[1][0]<.25:
                    break
                local.append((ranked,directions))
            if len(local)!=2 or local[0][0][0][1:]!=local[1][0][0][1:]:
                continue
            pairs,directions=local[-1]
            for index in pairs[0][1:]:
                incident[index]=(*incident[index][:3],directions[index])
            local_over_pair=True
        _,a,b=pairs[0]
        third=next(i for i in range(3) if i not in (a,b))
        arm=incident[third]
        if abs(float(arm[3]@incident[a][3]))>.65:
            continue
        cut_length=3*width
        cut=_point_at(arm[2],cut_length)
        if cut is None:
            continue
        proposals=[]
        for pid,end,point,tangent in endpoints:
            delta=point-cut
            length=float(np.linalg.norm(delta))
            if length<4*width or length>.28*np.hypot(w,h):
                continue
            direction=delta/length
            alignment=[float(direction@(-arm[3])),float(-direction@tangent)]
            if min(alignment)<.8:
                continue
            hits=_intersections(cut,point,starts,ends,owners,width)
            # At the T itself, a pixel skeleton can detour around a filled
            # junction and intersect its own short tail. Use the observed
            # straight-through over-pair to locate this first crossing; retain
            # actual visible intersections everywhere outside that tiny disk.
            over_direction=incident[a][3]-incident[b][3]
            over_direction/=np.linalg.norm(over_direction)
            denominator=direction[0]*over_direction[1]-direction[1]*over_direction[0]
            if abs(denominator)<.65:
                continue
            offset=branch-cut
            along=(offset[0]*over_direction[1]-offset[1]*over_direction[0])/denominator
            first_hit=cut+along*direction
            if not 0<along<length or np.linalg.norm(first_hit-branch)>1.5*width:
                continue
            hits=[hit for hit in hits if np.linalg.norm(hit['point']-branch)>2*width
                  and not(hit['path']==arm[0] and hit['gap_t']<along/length)]
            hits.insert(0,dict(point=first_hit,path=len(paths),gap_t=along/length,
                               sine_angle=abs(float(denominator))))
            if len(hits)<(1 if allow_single else 2):
                continue
            if local_over_pair and len(hits)!=1:
                continue
            if any(hit['path']==arm[0] for hit in hits):
                continue
            positions=np.array([hit['gap_t']*length for hit in hits])
            if len(hits)==1:
                # A photographed under-tip may touch just one uninterrupted
                # over-strand. Keep the same three-arm, straight-through,
                # unique-facing-end and protected-ink certificate as bundles.
                # The single crossing supplies no bundle spacing calibration,
                # so explicitly bound the unsupported continuation instead.
                # Only the missing continuation needs this bound. The part
                # from the cut to the T is already visible ink, trimmed by us
                # to expose the crossing; counting it penalizes thick strokes
                # and small crops even when the observed gap is short.
                unsupported_length=length-along
                if unsupported_length>min(85*width,.15*np.hypot(w,h)):
                    continue
                evidence={'single_contact':True,'unsupported_length':unsupported_length}
            else:
                spacing=float(np.median(np.diff(positions)))
                if max(positions[0],length-positions[-1])>max(4*width,2.5*spacing):
                    continue
                # Check bundle parallelism/spacing on its occupied footprint.
                lo=positions[0]-.5*spacing
                hi=positions[-1]+.5*spacing
                clipped_hits=[{**hit,'gap_t':float((pos-lo)/(hi-lo))}
                              for hit,pos in zip(hits,positions)]
                evidence=bundle_evidence(cut+direction*lo,cut+direction*hi,clipped_hits,
                                         np.vstack([starts,branch-4*width*over_direction]),
                                         np.vstack([ends,branch+4*width*over_direction]),
                                         np.r_[owners,len(paths)],width)
                if evidence is None:
                    continue
            proposals.append((pid,end,point,hits,evidence))
        # A T with two geometrically plausible destinations is ambiguous.
        if len(proposals)!=1:
            continue
        pid,end,point,hits,evidence=proposals[0]
        # Restrict pixel work to a small box around the short under-arm tail.
        count=max(3,int(cut_length)+1)
        tail=np.array([_point_at(arm[2],s) for s in np.linspace(0,cut_length,count)])
        lo=np.floor(tail.min(axis=0)-width).astype(int)
        hi=np.ceil(tail.max(axis=0)+width+1).astype(int)
        x1,y1=np.maximum(lo,[0,0]);x2,y2=np.minimum(hi,[w,h])
        yy,xx=np.mgrid[y1:y2,x1:x2]
        pixels=np.column_stack([xx.ravel(),yy.ravel()]).astype(float)
        over=[]
        for k in (a,b):
            samples=[_point_at(incident[k][2],s)
                     for s in np.linspace(0,4*width,20)]
            if any(p is None for p in samples):
                over=[];break
            over.append(np.array(samples))
        if len(over)!=2:
            continue
        protected=np.minimum(_distance_to_path(pixels,over[0]),_distance_to_path(pixels,over[1]))<=.65*width+1
        # Protect every unrelated visible strand in the edit box, including an
        # unrelated arc of the same component. The cut plane already protects
        # the selected arm's continuation beyond its new endpoint.
        for other_id,other in enumerate(paths):
            if other_id==arm[0]:
                continue
            if (np.any(other.max(axis=0)<[x1-width,y1-width])
                    or np.any(other.min(axis=0)>[x2+width,y2+width])):
                continue
            protected|=_distance_to_path(pixels,other)<=.9*width+1
        remove=(_distance_to_path(pixels,tail)<.9*width)&~protected
        # Keep the endpoint's final pixels beyond the cut plane; the removed
        # segment reaches the cut, but never erases the continuing under-arm.
        remove&=((pixels-cut)@arm[3] <= .1*width)
        view=output[y1:y2,x1:x2]
        erase=remove.reshape(view.shape[:2])
        target_component=int(ink_components[int(round(branch[1])),int(round(branch[0]))])
        if not target_component:
            continue
        # Pixel components also protect small untraced loops or extra marks;
        # geometric corridors alone cannot protect something omitted by tracing.
        erase&=ink_components[y1:y2,x1:x2]==target_component
        erased_pixels=int(np.count_nonzero(erase&(view.min(axis=2)<245)))
        if not erased_pixels:
            continue
        view[erase]=255
        diagnostic['bundle_contact_repairs'].append({
            'junction':branch.tolist(),'trimmed_under_tip':cut.tolist(),
            'opposite_tip':point.tolist(),'crossings':[hit['point'].tolist() for hit in hits],
            'crossing_count':len(hits),'bbox':[int(x1),int(y1),int(x2),int(y2)],
            'erased_pixels':erased_pixels,'reason':('unique T-tip continuation under a visible strand'
                if len(hits)==1 else 'unique T-tip continuation under a parallel bundle')})
        if evidence.get('single_contact'):
            diagnostic['bundle_contact_repairs'][-1]['unsupported_gap_length']=evidence['unsupported_length']
            diagnostic['bundle_contact_repairs'][-1]['local_over_tangents']=local_over_pair
    return (output if diagnostic['bundle_contact_repairs'] else None),diagnostic
