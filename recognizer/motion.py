"""Smooth regular-isotopy dragging of oriented classical link diagrams.

Curves have a material parameter and an auxiliary height. A compact C2 bump
translates a neighborhood of the picked point. The height separates crossings;
we never choose crossing types from a planar redraw. Swept spatial collision
checks reject unresolved contacts, and polynomial tangent checks forbid the
projection cusps needed for Reidemeister I. Explicit curls are handled elsewhere.

Geometry is numerical: adaptive chords approximate the smooth curves with a
bounded interpolation error, included in the collision clearance. Nongeneric
projected targets (tangencies/triple points) are not committed as PD diagrams.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import chain
import hashlib
import json
import math

import networkx as nx
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment, linprog
from scipy.spatial import cKDTree

from .diagram import component_walks, pd_code, validate, copy_diagram as _copy_diagram
from .geometry import intersection


class MoveRejected(ValueError):
    """An ambiguous or non-isotopic proposal; the last accepted diagram survives."""


def _branch_costs(old,new,lengths):
    """Squared periodic material distances, with both self-crossing assignments."""
    a=np.array([[b['parameter'] for b in pair] for pair in old])
    b=np.array([[b['parameter'] for b in pair] for pair in new])
    periods=np.array([lengths[x['curve']] for x in old[0]])
    def costs(points):
        delta=(a[:,None,:]-points[None,:,:]+periods/2)%periods-periods/2
        return np.sum(delta*delta,axis=2)
    result=costs(b)
    if old[0][0]['curve']==old[0][1]['curve']:result=np.minimum(result,costs(b[:,::-1]))
    return result


def _legacy_fingerprint(diagram):
    """Verify saved lifts written before the versioned binary point digest."""
    values={k:diagram.get(k,{}) for k in ('crossings','edges','component_orientations')}
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def _fingerprint(diagram):
    """Hash all previous projection metadata and every point without JSON floats.

    Shapes and little-endian float64 bytes make the point encoding unambiguous
    and portable. The remaining metadata is unchanged in scope. XYZ lift
    coordinates themselves remain outside the projection cache key, as before.
    """
    values={k:diagram.get(k,{}) for k in ('crossings','component_orientations')}
    values['edges']=[{key:value for key,value in edge.items() if key!='points'}
                     for edge in diagram.get('edges',[])]
    metadata=json.dumps(values,sort_keys=True,separators=(',',':')).encode()
    digest=hashlib.sha256(b'Knot Studio projection fingerprint v2\0')
    digest.update(len(metadata).to_bytes(8,'little'));digest.update(metadata)
    for edge in diagram.get('edges',[]):
        points=np.asarray(edge['points'],dtype='<f8',order='C')
        digest.update(np.asarray((points.ndim,*points.shape),dtype='<u8').tobytes())
        digest.update(points.tobytes(order='C'))
    return 'v2:'+digest.hexdigest()


def _projection_bijection(a,b):
    """Fast positive certificate for the oriented crossing rotation system.

    Once one crossing and its port rotation are paired, every neighboring
    pairing is forced. Check that bijection explicitly instead of solving an
    unrestricted graph-isomorphism search on every small drag. Geometry only
    orders candidate roots; it never substitutes for an incidence check.
    Failure falls back to the general isomorphism routine below.
    """
    ca={c['id']:c for c in a['crossings']};cb={c['id']:c for c in b['crossings']}
    ea={e['id']:e for e in a['edges']};eb={e['id']:e for e in b['edges']}
    unseen=set(ca);available=set(cb)
    while unseen:
        root=min(unseen);point=ca[root]['point']
        candidates=sorted(available,key=lambda cid:math.dist(point,cb[cid]['point']))[:3]
        for target in candidates:
            parity=(cb[target]['over']-ca[root]['over'])%2
            for initial_shift in (parity,parity+2):
                forward={};backward={};rotations={};pending=[(root,target,initial_shift)]
                valid=True
                while pending:
                    left,right,shift=pending.pop()
                    if left in forward:
                        if forward[left]!=right or rotations[left]!=shift:valid=False;break
                        continue
                    if left not in unseen or right not in available or right in backward:
                        valid=False;break
                    source,dest=ca[left],cb[right]
                    if (source['over']+shift)%2!=dest['over']:
                        valid=False;break
                    forward[left]=right;backward[right]=left;rotations[left]=shift
                    for port,ref in enumerate(source['ports']):
                        other=dest['ports'][(port+shift)%4]
                        end=ea[ref['edge']]['start' if ref['end'] else 'end']
                        next_end=eb[other['edge']]['start' if other['end'] else 'end']
                        pending.append((end[0],next_end[0],(next_end[1]-end[1])%4))
                if valid:break
            if valid:break
        else:return False
        unseen.difference_update(forward);available.difference_update(backward)
    return not available


def _same_projection(a,b):
    if len(a['crossings'])!=len(b['crossings']):return False
    if len(component_walks(a))!=len(component_walks(b)):return False
    left,right=pd_code(a),pd_code(b)
    if left==right:return True
    if _projection_bijection(a,b):return True
    from .projection_equivalence import same_pd_projection
    return same_pd_projection(left,right)


def to_curves(diagram):
    """Return physically oriented closed XYZ curves, including trivial circles."""
    stored=diagram.get('spatial_curves')
    fingerprint=diagram.get('spatial_fingerprint')
    checker=_fingerprint if isinstance(fingerprint,str) and fingerprint.startswith('v2:') else _legacy_fingerprint
    if stored and fingerprint==checker(diagram):
        return _copy_diagram({'spatial_curves':stored})['spatial_curves']
    edges={e['id']:e for e in diagram['edges']}
    crossings={c['id']:c for c in diagram['crossings']}
    orientations=diagram.get('component_orientations',{})
    height=max(10.,math.sqrt(diagram.get('width',800)*diagram.get('height',600))/35)
    curves=[]
    for ci,walk in enumerate(component_walks(diagram)):
        if orientations.get(str(ci),orientations.get(ci,1))==-1:
            walk=[(eid,1-direction) for eid,direction in reversed(walk)]
        points=[];marks=[]
        for eid,direction in walk:
            edge=edges[eid]
            path=np.asarray(edge['points'],float)
            if direction:path=path[::-1]
            start=edge['end' if direction else 'start']
            end=edge['start' if direction else 'end']
            if start is not None:
                path=path.copy();path[0]=crossings[start[0]]['point'];path[-1]=crossings[end[0]]['point']
                marks.append((len(points)-1 if points else 0,
                              height if start[1]%2==crossings[start[0]]['over'] else -height))
            points.extend(path.tolist() if not points else path[1:].tolist())
        pts=np.asarray(points,float)
        if np.linalg.norm(pts[0]-pts[-1])>1e-7:pts=np.vstack([pts,pts[0]])
        s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
        if marks:
            xs=np.array([s[i] for i,_ in marks]);zs=np.array([z for _,z in marks])
            z=np.interp(s,xs,zs,period=s[-1]);z[-1]=z[0]
        else:z=np.zeros(len(pts))
        curves.append({'points':np.column_stack([pts,z]).tolist()})
    return curves


def _box_pairs(low,high,active=None):
    """Exact box overlaps, using short contiguous blocks on dense curves.

    Each parent contains every child box, so disjoint parents certify that
    none of their segment pairs can meet. The unchanged tree search handles
    sparse arrays and arrays whose ordering does not give useful blocks.
    """
    low=np.asarray(low,float);high=np.asarray(high,float)
    if len(low)>=512:
        result=_blocked_box_pairs(low,high,active)
        if result is not None:return result
    return _tree_box_pairs(low,high,active)


def _blocked_box_pairs(low,high,active):
    size=8
    n=len(low);starts=np.arange(0,n,size);ends=np.minimum(starts+size,n)
    parent_low=np.minimum.reduceat(low,starts,axis=0)
    parent_high=np.maximum.reduceat(high,starts,axis=0)
    # Shuffled tiny boxes can give almost every parent the entire scene's
    # bounds. Detect that before a tree query materializes quadratically many
    # parent pairs. Consecutive polyline segments have bounded parent spread.
    parent_spread=np.linalg.norm(parent_high-parent_low,axis=1)
    child_spread=np.maximum.reduceat(np.linalg.norm(high-low,axis=1),starts)
    if np.median(parent_spread/np.maximum(child_spread,1e-12))>size*1.5:
        return None
    active=None if active is None else np.asarray(active,dtype=bool)
    parent_active=None if active is None else np.logical_or.reduceat(active,starts)
    parents=_tree_box_pairs(parent_low,parent_high,parent_active)
    self_ids=np.arange(len(starts)) if active is None else np.flatnonzero(parent_active)
    # Unordered/random boxes can give unhelpfully overlapping parent bounds.
    # Bound this path's work and retain the old exact search for those inputs.
    if len(parents)+len(self_ids)>n:return None
    parents=np.vstack((parents,np.column_stack((self_ids,self_ids))))
    parts=[];offsets=np.arange(size)
    for begin in range(0,len(parents),256):
        first,second=parents[begin:begin+256].T
        ii=starts[first,None,None]+offsets[None,:,None]
        jj=starts[second,None,None]+offsets[None,None,:]
        valid=(ii<ends[first,None,None])&(jj<ends[second,None,None])&(ii<jj)
        i=np.broadcast_to(ii,valid.shape)[valid]
        j=np.broadcast_to(jj,valid.shape)[valid]
        if active is not None:
            moving=active[i]|active[j];i,j=i[moving],j[moving]
        overlap=np.ones(len(i),dtype=bool)
        for dimension in range(low.shape[1]):
            overlap&=((high[i,dimension]>=low[j,dimension])
                      &(high[j,dimension]>=low[i,dimension]))
        if np.any(overlap):parts.append(np.column_stack((i[overlap],j[overlap])))
    if not parts:return np.empty((0,2),dtype=int)
    result=np.concatenate(parts)
    return result[np.lexsort((result[:,0],result[:,1]))]


def _tree_box_pairs(low,high,active=None):
    """Exact intersecting AABB candidates, without scanning every short chord.

    Radius buckets keep one unusually long segment from enlarging every tree
    query. Bounding balls only prune candidates; the final box comparisons use
    the original coordinates. No geometric clearance is relaxed here.
    """
    low=np.asarray(low,float);high=np.asarray(high,float)
    if len(low)<2:return np.empty((0,2),dtype=int)
    if active is not None:
        active=np.asarray(active,dtype=bool)
        if not np.any(active):return np.empty((0,2),dtype=int)
        # Stationary/stationary pairs are irrelevant to a swept certificate.
        # First discard boxes outside the entire moving support's swept box,
        # then build trees only for this local neighborhood. The broad box is
        # conservative even when a long stationary strand enters the region.
        near=np.all(high>=low[active].min(axis=0),axis=1)&np.all(low<=high[active].max(axis=0),axis=1)
        original_ids=np.flatnonzero(near)
        low,high,active=low[original_ids],high[original_ids],active[original_ids]
    else:
        original_ids=None
    centers=(low+high)*.5
    radii=np.linalg.norm((high-low)*.5,axis=1)
    levels=np.floor(np.log2(np.maximum(radii,1e-12))).astype(int)
    groups=[]
    for level in np.unique(levels):
        ids=np.flatnonzero(levels==level)
        moving=active is None or bool(np.any(active[ids]))
        groups.append((ids,cKDTree(centers[ids]),float(radii[ids].max()),moving))
    pairs=[]
    for i,(ids,tree,radius,moving) in enumerate(groups):
        for j in range(i,len(groups)):
            others,other_tree,other_radius,other_moving=groups[j]
            # No swept/projection certificate needs stationary/stationary
            # pairs. Skip their radius-bucket tree queries altogether.
            if not moving and not other_moving:continue
            # A small outward rounding guard prevents boundary misses.
            distance=radius+other_radius+1e-10*max(1.,radius+other_radius)
            if i==j:
                found=tree.query_pairs(distance,output_type='ndarray')
                if not len(found):continue
                a,b=ids[found[:,0]],ids[found[:,1]]
            else:
                found=tree.sparse_distance_matrix(other_tree,distance,output_type='ndarray')
                if not len(found):continue
                a,b=ids[found['i']],others[found['j']]
            keep=np.all(high[a]>=low[b],axis=1)&np.all(high[b]>=low[a],axis=1)
            if active is not None:keep&=active[a]|active[b]
            if np.any(keep):pairs.append(np.column_stack((a[keep],b[keep])))
    if not pairs:return np.empty((0,2),dtype=int)
    result=np.concatenate(pairs)
    if original_ids is not None:result=original_ids[result]
    result.sort(axis=1)
    return result[np.lexsort((result[:,0],result[:,1]))]


def from_curves(curves,width=800,height=600,*,_projection_only=False,_export=True,_array_geometry=False,
                _pair_candidates=None,_crossing_segments=None,_curve_cache=None):
    """Rebuild a PD graph from oriented XYZ polylines and their true heights.

    The optional private collector receives the actual finite-segment pairs
    realizing crossings, including duplicate hits at shared vertices. Keep
    these indices for a stationary cache: interpolated material parameters
    can round onto a neighboring segment which does not actually intersect.
    """
    array_geometry=_array_geometry and not _export
    # The optional gesture-local cache is for immutable private XYZ inputs.
    # Retain at most the latest input for each physical component. A moving
    # or newly refined curve has a different array and is checked in full;
    # unchanged curves need not repeat their finite/tangent/arclength scans.
    arrays=[];arclength=[]
    for ci,curve in enumerate(curves):
        cache_key=curve.get('_cache_key',ci)
        cached=None if _curve_cache is None else _curve_cache.get(cache_key)
        if cached is not None and cached[0] is curve['points']:
            _,p,s,_=cached
            arrays.append(p);arclength.append(s)
            continue
        p=np.asarray(curve['points'],dtype=float)
        if p.ndim!=2 or p.shape[1]!=3 or len(p)<4 or not np.isfinite(p).all():
            raise MoveRejected('A spatial component needs a finite closed XYZ curve.')
        if np.linalg.norm(p[0]-p[-1])>1e-7:p=np.vstack([p,p[0]])
        # A depth-only preparation keeps the planar tangent and arclength
        # exactly fixed. Still check all fresh XYZ values and closure above.
        # The minimum XY spacing guard prevents a changed Z from affecting
        # the separate 3D duplicate-removal threshold below.
        if (cached is not None and cached[3]>2e-8 and p.shape==cached[1].shape
                and np.array_equal(p[:,:2],cached[1][:,:2])):
            s=cached[2]
            _curve_cache[cache_key]=(curve['points'],p,s,cached[3])
            arrays.append(p);arclength.append(s)
            continue
        keep=np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-8]
        if not np.all(keep):p=p[keep]
        if len(p)<4:raise MoveRejected('A component collapsed.')
        directions=np.diff(p[:,:2],axis=0)
        lengths=np.sqrt(directions[:,0]*directions[:,0]+directions[:,1]*directions[:,1])
        following=np.roll(directions,-1,axis=0)
        dot=directions[:,0]*following[:,0]+directions[:,1]*following[:,1]
        if np.any(dot<-.999999*lengths*np.roll(lengths,-1)):
            raise MoveRejected('A projected strand backtracks into a cusp.')
        if np.any(lengths<1e-8):
            raise MoveRejected('A strand has a degenerate projected tangent.')
        s=np.r_[0.,np.cumsum(lengths)]
        if _curve_cache is not None:_curve_cache[cache_key]=(curve['points'],p,s,float(lengths.min()))
        arrays.append(p)
        arclength.append(s)
    if not arrays:
        result={'width':width,'height':height,'crossings':[],'edges':[],'component_orientations':{},'spatial_curves':[]}
        result['spatial_fingerprint']=_fingerprint(result)
        return result
    starts=np.concatenate([p[:-1] for p in arrays]);ends=np.concatenate([p[1:] for p in arrays])
    lengths=np.array([len(p)-1 for p in arrays])
    component=np.repeat(np.arange(len(arrays)),lengths)
    local=np.concatenate([np.arange(n) for n in lengths])
    if _pair_candidates is not None and np.array_equal(lengths,[len(c['points'])-1 for c in curves]):
        pairs=_pair_candidates
    else:
        pairs=_box_pairs(np.minimum(starts[:,:2],ends[:,:2])-.0025,
                         np.maximum(starts[:,:2],ends[:,:2])+.0025)
    if len(pairs):
        first,second=pairs.T
        adjacent=(component[first]==component[second])&((abs(local[first]-local[second])<=1)
                    |(abs(local[first]-local[second])==lengths[component[first]]-1))
        pairs=pairs[~adjacent]
    events=[[] for _ in arrays];crossings=[];hits=[];hit_grid=defaultdict(list)
    endpoint_contacts=[];crossing_segments=[]
    for old,index in pairs:
        ci,si=int(component[index]),int(local[index]);cj,sj=int(component[old]),int(local[old])
        aa,bb=starts[index],ends[index];c,d=starts[old],ends[old]
        a,b=aa[:2],bb[:2]
        hit=intersection(a,b,c[:2],d[:2])
        # A spatial curve needs actual finite-segment incidence. The shared
        # recognition helper also tolerates slight supporting-line extensions;
        # handle those as proximity contacts, never as invented crossings.
        if hit is None or not (0.<=hit[0]<=1. and 0.<=hit[1]<=1.):
            if _segment_distance(np.r_[a,0.],np.r_[b,0.],np.r_[c[:2],0.],np.r_[d[:2],0.])<1e-7:
                if ci==cj:
                    first,last=sorted((si,sj));s=arclength[ci]
                    gap=min(s[last]-s[first+1],s[-1]-s[last+1]+s[first])
                    if gap<1e-7:
                        # Two pieces joined by a tiny real edge are locally
                        # consecutive, not separate tangential strands. Keep
                        # that edge and its endpoints exactly as supplied.
                        continue
                v=b-a;w=d[:2]-c[:2]
                sine=abs(v[0]*w[1]-v[1]*w[0])/(np.linalg.norm(v)*np.linalg.norm(w))
                if sine<1e-4:
                    raise MoveRejected('Projected strands are tangent or overlap; move beyond this singular target.')
                # An actual transverse crossing can be numerically just beyond
                # this chord's end, on its material neighbor. Check against
                # the complete crossing list before treating the near miss
                # as a tangency. This also handles a closed curve's seam.
                # Never extend a segment or invent a crossing here.
                endpoint_contacts.append(((ci,si),(cj,sj)))
            continue
        t,u,point=hit
        v=b-a;w=d[:2]-c[:2]
        sine=abs(v[0]*w[1]-v[1]*w[0])/(np.linalg.norm(v)*np.linalg.norm(w))
        if sine<1e-4:raise MoveRejected('Finish the drag beyond the tangency before committing a diagram.')
        first=(ci,(si+t)%lengths[ci]);second=(cj,(sj+u)%lengths[cj])
        pair=sorted([first,second])
        duplicate=False
        cell=tuple(np.floor(point/.025).astype(int))
        nearby=sorted(index for dx in (-1,0,1) for dy in (-1,0,1)
                      for index in hit_grid.get((cell[0]+dx,cell[1]+dy),()))
        for index in nearby:
            oldpoint,oldpair=hits[index]
            separation=np.linalg.norm(point-oldpoint)
            if separation>.025:continue
            if all(x[0]==y[0] and abs(x[1]-y[1])<1e-5 for x,y in zip(pair,oldpair)):
                duplicate=True;break
            # Just before/after an RII tangency, two distinct crossings of the
            # same two short material arcs can be arbitrarily close in XY.
            # They are not a three-sheet cluster. Keep the cluster guard for
            # a third material branch, and never excuse coincident points.
            def same_local_branch(x,y):
                if x[0]!=y[0]:return False
                s=arclength[x[0]]
                def material(t):
                    i=int(t)
                    return s[i]+(t-i)*(s[i+1]-s[i])
                distance=abs(material(x[1])-material(y[1]))
                return min(distance,s[-1]-distance)<.1
            if separation>1e-7 and any(all(same_local_branch(x,y) for x,y in zip(pair,other))
                                      for other in (oldpair,oldpair[::-1])):
                continue
            raise MoveRejected('The projected target is a triple point or unresolved crossing cluster; move a little farther.')
        if _crossing_segments is not None:
            crossing_segments.append(((ci,si),(cj,sj)))
        if duplicate:continue
        z1=aa[2]+t*(bb[2]-aa[2]);z2=c[2]+u*(d[2]-c[2])
        if abs(z1-z2)<.015 and not _projection_only:raise MoveRejected('Two spatial strands would touch. Try a different path or smaller neighborhood.')
        cid=len(crossings);crossings.append({'id':cid,'point':point.tolist(),'ports':[],'over':1})
        if 'weights' in curves[ci] and 'weights' in curves[cj]:
            wa=curves[ci]['weights'];wb=curves[cj]['weights']
            crossings[-1]['motion_branches']=[
                {'weight':float(wa[si]+t*(wa[si+1]-wa[si])),'over':bool(z1>z2)},
                {'weight':float(wb[sj]+u*(wb[sj+1]-wb[sj])),'over':bool(z2>z1)}]
            if 'parameters' in curves[ci] and 'parameters' in curves[cj]:
                for branch,owner,segment,fraction,z in zip(crossings[-1]['motion_branches'],
                        (ci,cj),(si,sj),(t,u),(z1,z2)):
                    parameters=curves[owner]['parameters']
                    branch.update(curve=owner,parameter=float(parameters[segment]+
                        fraction*(parameters[segment+1]-parameters[segment])),height=float(z))
        # Internal projection probes must still have an alternating crossing
        # incidence, even when two heights coincide. They never leave the drag
        # engine: committed diagrams always use the strict height check above.
        if _projection_only and abs(z1-z2)<.015:z1=z2+.03
        events[ci].append((first[1],cid,z1>z2))
        events[cj].append((second[1],cid,z2>z1))
        hit_grid[cell].append(len(hits));hits.append((point,pair))
    if endpoint_contacts:
        material_hits=[]
        for _,pair in hits:
            branches=[]
            for ci,parameter in pair:
                si=int(parameter);s=arclength[ci]
                branches.append((ci,s[si]+(parameter-si)*(s[si+1]-s[si])))
            material_hits.append(branches)

        def touches_segment(branch,segment):
            ci,position=branch;cj,si=segment
            if ci!=cj:return False
            s=arclength[ci];left,right=s[si:si+2]
            return any(left-1e-7<=position+shift<=right+1e-7
                       for shift in (-s[-1],0.,s[-1]))

        for first,second in endpoint_contacts:
            if not any((touches_segment(a,first) and touches_segment(b,second)) or
                       (touches_segment(b,first) and touches_segment(a,second))
                       for a,b in material_hits):
                raise MoveRejected('Projected strands are tangent or overlap; move beyond this singular target.')
    edges=[];incidence=defaultdict(list)
    def section(points,left,right):
        def point(t):
            i=min(int(t),len(points)-2)
            return points[i]+(t-i)*(points[i+1]-points[i])
        return np.vstack((point(left),points[int(left)+1:math.ceil(right)],point(right)))
    for ci,points in enumerate(arrays):
        ev=sorted(events[ci]);xy=points[:,:2]
        if not ev:
            edges.append({'id':len(edges),'points':xy.copy(),'start':None,'end':None,'closed':True,'component':ci,'motion_curve':ci})
            continue
        for i,(t,cid,over) in enumerate(ev):
            u,other,endover=ev[(i+1)%len(ev)]
            if i+1<len(ev):path_points=section(xy,t,u)
            else:
                first,last=section(xy,t,len(xy)-1),section(xy,0,u)[1:]
                path_points=np.vstack((first,last))
            path_points[0]=crossings[cid]['point'];path_points[-1]=crossings[other]['point']
            eid=len(edges)
            edge={'id':eid,'points':path_points,'start':[cid,-1],'end':[other,-1],'closed':False,'component':ci,'motion_curve':ci}
            edges.append(edge)
            for end,at,flag in [(0,cid,over),(1,other,endover)]:
                path=path_points if not end else path_points[::-1]
                direction=next((np.asarray(p)-path[0] for p in path[1:] if np.linalg.norm(np.asarray(p)-path[0])>1e-6),None)
                if direction is None:raise MoveRejected('A projected arc collapsed.')
                incidence[at].append((math.atan2(-direction[1],direction[0]),eid,end,flag))
    for crossing in crossings:
        refs=sorted(incidence[crossing['id']]);flags=[]
        if len(refs)!=4:raise MoveRejected('A projected crossing is not four-valent.')
        for port,(_,eid,end,over) in enumerate(refs):
            crossing['ports'].append({'edge':eid,'end':end})
            edges[eid]['end' if end else 'start']=[crossing['id'],port];flags.append(over)
        if flags[0]!=flags[2] or flags[1]!=flags[3] or flags[0]==flags[1]:
            raise MoveRejected('The projected crossing is not transverse.')
        crossing['over']=0 if flags[0] else 1
    result={'width':width,'height':height,'crossings':crossings,'edges':edges,'component_orientations':{}}
    verdict=validate(result)
    if not verdict['valid']:raise MoveRejected('; '.join(verdict['errors']))
    # Validate finite geometry in contiguous arrays even for public callers;
    # Python-list serialization belongs only at the final output boundary.
    if not array_geometry:
        for edge in edges:edge['points']=edge['points'].tolist()
    if _export:
        result['spatial_curves']=[{'points':a.tolist()} for a in arrays]
        result['spatial_fingerprint']=_fingerprint(result)
    else:
        # Private probes skip the full coordinate fingerprint; it is computed
        # once for the committed public frame, which owns its geometry.
        result['spatial_curves']=[{'points':a.copy() if array_geometry else a.tolist()} for a in arrays]
    if _crossing_segments is not None:
        _crossing_segments.extend(crossing_segments)
    return result


def _segment_distance(a,b,c,d):
    """Minimum distance of two finite 3D segments, including parallel ones."""
    u=b-a;v=d-c;w=a-c
    aa=float(u@u);bb=float(u@v);cc=float(v@v);dd=float(u@w);ee=float(v@w)
    denom=aa*cc-bb*bb
    candidates=[(0.,np.clip(ee/max(cc,1e-16),0.,1.)),
                (1.,np.clip((bb+ee)/max(cc,1e-16),0.,1.)),
                (np.clip(-dd/max(aa,1e-16),0.,1.),0.),
                (np.clip((bb-dd)/max(aa,1e-16),0.,1.),1.)]
    if denom>1e-16:
        s=(bb*ee-cc*dd)/denom;t=(aa*ee-bb*dd)/denom
        if 0<=s<=1 and 0<=t<=1:candidates.append((s,t))
    return min(float(np.linalg.norm(w+s*u-t*v)) for s,t in candidates)


def _segment_distances(a,b,c,d):
    """Vector form of the same finite-segment minimum used by the fallback."""
    u=b-a;v=d-c;w=a-c
    dot=lambda x,y:np.einsum('ij,ij->i',x,y)
    aa=dot(u,u);bb=dot(u,v);cc=dot(v,v);dd=dot(u,w);ee=dot(v,w)
    denominator=aa*cc-bb*bb
    minimum=np.full(len(a),np.inf)
    for ss,tt in ((np.zeros(len(a)),np.clip(ee/np.maximum(cc,1e-16),0.,1.)),
                  (np.ones(len(a)),np.clip((bb+ee)/np.maximum(cc,1e-16),0.,1.)),
                  (np.clip(-dd/np.maximum(aa,1e-16),0.,1.),np.zeros(len(a))),
                  (np.clip((bb-dd)/np.maximum(aa,1e-16),0.,1.),np.ones(len(a)))):
        vector=w+ss[:,None]*u-tt[:,None]*v
        minimum=np.minimum(minimum,dot(vector,vector))
    nonparallel=denominator>1e-16
    safe=np.where(nonparallel,denominator,1.)
    ss=(bb*ee-cc*dd)/safe;tt=(aa*ee-bb*dd)/safe
    interior=nonparallel&(ss>=0)&(ss<=1)&(tt>=0)&(tt<=1)
    vector=w+ss[:,None]*u-tt[:,None]*v
    minimum=np.minimum(minimum,np.where(interior,dot(vector,vector),np.inf))
    return np.sqrt(minimum)


def _swept_clear(before,after,tolerance=.04):
    """Conservative all-time separation using Lipschitz distance bounds."""
    if not before:return 0
    starts=np.concatenate([p[:-1] for p in before]);ends=np.concatenate([p[1:] for p in before])
    final_starts=np.concatenate([p[:-1] for p in after]);final_ends=np.concatenate([p[1:] for p in after])
    das=final_starts-starts;dbs=final_ends-ends
    moving=(np.linalg.norm(das,axis=1)>=1e-10)|(np.linalg.norm(dbs,axis=1)>=1e-10)
    if not np.any(moving):return 0
    lengths=np.array([len(p)-1 for p in before])
    component=np.repeat(np.arange(len(before)),lengths)
    local=np.concatenate([np.arange(n) for n in lengths])
    # Adaptive sampling can put several vertices inside the clearance radius.
    # Index adjacency alone then mistakes consecutive parts of one arc for
    # separate strands. Only excuse such a pair when its connecting arc has
    # a strictly monotone projection throughout the entire linear sweep.
    offsets=np.r_[0,np.cumsum(lengths)]
    # Auxiliary height preparation can stretch a tiny connecting arc greatly
    # in Z. It remains a local part of one strand, not a new obstacle. Use its
    # projected arclength only to nominate local pairs; the strict 3-D
    # monotonicity test below still certifies their entire connecting arc.
    segment_lengths=np.maximum(np.linalg.norm((ends-starts)[:,:2],axis=1),
                              np.linalg.norm((final_ends-final_starts)[:,:2],axis=1))
    material_lengths=[np.r_[0,np.cumsum(segment_lengths[a:b])]
                      for a,b in zip(offsets[:-1],offsets[1:])]
    def local_monotone(first,second):
        ci=component[first]
        if ci!=component[second]:return False
        lo,hi=sorted((int(local[first]),int(local[second])))
        s=material_lengths[ci];n=lengths[ci]
        direct=s[hi]-s[lo+1];wrapped=s[n]-s[hi+1]+s[lo]
        if min(direct,wrapped)>2*tolerance:return False
        ids=(np.arange(lo,hi+1) if direct<=wrapped else
             np.r_[np.arange(hi,n),np.arange(lo+1)])+offsets[ci]
        old_vectors=ends[ids]-starts[ids]
        new_vectors=final_ends[ids]-final_starts[ids]
        axis=old_vectors.sum(axis=0)+new_vectors.sum(axis=0)
        norm=np.linalg.norm(axis)
        if norm<1e-12:return False
        axis/=norm
        return bool(np.all(old_vectors@axis>1e-10) and
                    np.all(new_vectors@axis>1e-10))
    low=np.minimum.reduce([starts,ends,final_starts,final_ends])-tolerance
    high=np.maximum.reduce([starts,ends,final_starts,final_ends])+tolerance
    pairs=_box_pairs(low,high,active=moving)
    if len(pairs):
        first,second=pairs.T
        adjacent=(component[first]==component[second])&((abs(local[first]-local[second])<=1)
                    |(abs(local[first]-local[second])==lengths[component[first]]-1))
        pairs=pairs[(moving[first]|moving[second])&~adjacent]
    checks=0
    if len(pairs):
        # Certify the first Lipschitz interval in a single numerical batch.
        # Only ambiguous neighbors need the scalar subdivision/monotonicity
        # fallback. A rounding guard makes this shortcut more conservative.
        first,second=pairs.T
        speeds=np.maximum.reduce([np.linalg.norm(x[first]-y[second],axis=1)
                                   for x in (das,dbs) for y in (das,dbs)])
        distances=_segment_distances(starts[first]+.5*das[first],ends[first]+.5*dbs[first],
                                     starts[second]+.5*das[second],ends[second]+.5*dbs[second])
        checks=len(pairs)
        pairs=pairs[distances-speeds*.5<=tolerance+1e-10]
    for old,index in pairs:
        if local_monotone(old,index):continue
        a,b,da,db=starts[index],ends[index],das[index],dbs[index]
        c,d,dc,dd=starts[old],ends[old],das[old],dbs[old]
        speed=max(np.linalg.norm(x-y) for x in (da,db) for y in (dc,dd))
        stack=[(0.,1.,0)]
        while stack:
            left,right,depth=stack.pop();mid=(left+right)/2
            distance=_segment_distance(a+mid*da,b+mid*db,c+mid*dc,d+mid*dd)
            checks+=1
            if distance-speed*(right-left)/2>tolerance:continue
            if distance<=tolerance or depth>=18:
                raise MoveRejected('The swept strand would touch another strand in space; this motion was stopped.')
            stack.extend([(left,mid,depth+1),(mid,right,depth+1)])
    return checks


def _projected_rii_times(before,after):
    """Find RII events of the linearly moving, sampled planar curves.

    A polygonal pair is born/dies when a vertex crosses a nonincident segment
    and its two neighboring vertices lie on the same side of that segment.
    Vertex/line incidence is a quadratic in time. Enumerating these roots
    prevents a birth and a death between display frames from canceling out of
    the endpoint crossing counts. Ordinary crossing/vertex passages are not
    events and do not require an extra diagram rebuild.
    """
    if not before:return []
    starts=np.concatenate([p[:-1,:2] for p in before]);ends=np.concatenate([p[1:,:2] for p in before])
    ds=np.concatenate([q[:-1,:2]-p[:-1,:2] for p,q in zip(before,after)])
    de=np.concatenate([q[1:,:2]-p[1:,:2] for p,q in zip(before,after)])
    sizes=np.array([len(p)-1 for p in before])
    component=np.repeat(np.arange(len(sizes)),sizes)
    local=np.concatenate([np.arange(n) for n in sizes])
    moving=(np.linalg.norm(ds,axis=1)>1e-10)|(np.linalg.norm(de,axis=1)>1e-10)
    low=np.minimum.reduce([starts,ends,starts+ds,ends+de])-1e-9
    high=np.maximum.reduce([starts,ends,starts+ds,ends+de])+1e-9
    candidates=_box_pairs(low,high,active=moving)
    if len(candidates):
        first,second=candidates.T
        adjacent=(component[first]==component[second])&((abs(local[first]-local[second])<=1)
                    |(abs(local[first]-local[second])==sizes[component[first]]-1))
        candidates=candidates[(moving[first]|moving[second])&~adjacent]
    if not len(candidates):return []
    # Enumerate each vertex/segment incidence once, then solve its quadratic
    # in a NumPy batch. Only actual event roots enter the chronological scan.
    # The previous scalar equations and tolerances are preserved verbatim.
    offsets=np.r_[0,np.cumsum(sizes)]
    first,second=candidates.T
    vertex_segments=np.r_[first,first,second,second]
    lines=np.r_[second,second,first,first]
    endpoint_flags=np.r_[np.zeros(len(first),int),np.ones(len(first),int),
               np.zeros(len(first),int),np.ones(len(first),int)]
    vertices=offsets[component[vertex_segments]]+(local[vertex_segments]+endpoint_flags)%sizes[component[vertex_segments]]
    # Integer lexsort gives the same ordered unique pairs as np.unique(axis=0)
    # without its structured-row comparisons. A swept RII neighborhood can
    # contain many candidate incidences even with a compact collision grid.
    order=np.lexsort((lines,vertices))
    vertices,lines=vertices[order],lines[order]
    distinct=np.r_[True,(vertices[1:]!=vertices[:-1])|(lines[1:]!=lines[:-1])]
    vertices,lines=vertices[distinct],lines[distinct]
    a=starts[vertices]-starts[lines];da=ds[vertices]-ds[lines]
    v=ends[lines]-starts[lines]
    dv=de[lines]-ds[lines]
    cross=lambda x,y:x[:,0]*y[:,1]-x[:,1]*y[:,0]
    c0=cross(a,v);c1=cross(da,v)+cross(a,dv);c2=cross(da,dv)
    scale=np.maximum.reduce((abs(c0),abs(c1),abs(c2),np.full(len(c0),1e-30)))
    linear=abs(c2)<1e-12*scale
    indices=np.flatnonzero(linear&(abs(c1)>=1e-12*scale))
    root_indices=[indices];root_times=[-c0[indices]/c1[indices]]
    discriminant=c1*c1-4*c2*c0
    indices=np.flatnonzero(~linear&(discriminant>=0))
    q=-.5*(c1[indices]+np.copysign(np.sqrt(discriminant[indices]),c1[indices]))
    repeated=abs(q)<1e-30
    root_indices.append(indices[repeated]);root_times.append(-c1[indices[repeated]]/(2*c2[indices[repeated]]))
    indices=indices[~repeated];q=q[~repeated]
    root_indices.extend((indices,indices));root_times.extend((q/c2[indices],c0[indices]/q))
    indices=np.concatenate(root_indices);times=np.concatenate(root_times)
    interior=(times>1e-10)&(times<1-1e-10)
    indices,times=indices[interior],times[interior]
    direction=v[indices]+times[:,None]*dv[indices]
    denominator=np.einsum('ij,ij->i',direction,direction)
    parameter=np.einsum('ij,ij->i',a[indices]+times[:,None]*da[indices],direction)/np.maximum(denominator,1e-30)
    inside=(denominator>=1e-20)&(parameter>=-1e-9)&(parameter<=1+1e-9)
    indices,times,direction=indices[inside],times[inside],direction[inside]
    vertices,lines=vertices[indices],lines[indices]
    points=starts[vertices]+times[:,None]*ds[vertices]
    owner=component[vertices];material=local[vertices]
    neighbors=[offsets[owner]+(material+shift)%sizes[owner] for shift in (-1,1)]
    side=[cross(starts[index]+times[:,None]*ds[index]-points,direction) for index in neighbors]
    product=side[0]*side[1]
    if np.any((product<=0)&(abs(product)<1e-18)):
        raise MoveRejected('The projected path contains a degenerate tangency. Choose a slightly different direction.')
    events={}
    for index in np.flatnonzero(product>0):
        key=(int(owner[index]),int(material[index]),int(lines[index]),round(float(times[index]),12))
        events[key]=(float(times[index]),points[index])
    result=[];locations=[]
    for time,point in sorted(events.values(),key=lambda item:item[0]):
        if not result or time-result[-1]>1e-9:
            result.append(time);locations.append(point)
        elif np.linalg.norm(point-locations[-1])>1e-7:
            # Independent events at the same instant cannot be ordered by an
            # interval sample; in particular a birth must not disappear into
            # an unrelated death in the same component-pair crossing count.
            raise MoveRejected('The projected path has simultaneous crossing events. Move more gradually or choose a slightly different direction.')
    return result


class _SmoothCurve:
    def __init__(self,points):
        points=np.asarray(points,float)
        kept=[0]
        for i in range(1,len(points)-1):
            delta=points[i,:2]-points[kept[-1],:2]
            if math.hypot(float(delta[0]),float(delta[1]))>=.8:kept.append(i)
        # A saved edge can include its crossing endpoint twice. The linear
        # thinning above can retain one copy of the first point at the end;
        # adding the periodic endpoint would then invent a zero-length arc.
        # Remove only redundant XYZ closure samples, not a distinct spatial
        # branch whose projection has genuinely collapsed.
        while len(kept)>1 and np.linalg.norm(points[kept[-1]]-points[0])<1e-7:
            kept.pop()
        points=np.vstack([points[kept],points[0]])
        if len(points)<4:raise MoveRejected('This component is too small to drag smoothly.')
        self.s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1))]
        if np.any(np.diff(self.s)<1e-7):raise MoveRejected('A component has a collapsed arc.')
        self.length=self.s[-1]
        self.spline=CubicSpline(self.s,points,bc_type='periodic')
        # The tangent of a cubic is a quadratic Bezier curve. A positive
        # projection of all three control vectors bounds its norm away from
        # zero, avoiding thousands of unnecessary quartic root solves.
        h=np.diff(self.s)
        A=3*self.spline.c[0,:,:2];B=2*self.spline.c[1,:,:2];C=self.spline.c[2,:,:2]
        controls=np.stack([C,C+B*h[:,None]/2,C+B*h[:,None]+A*h[:,None]**2])
        chord=np.diff(points[:,:2],axis=0)/h[:,None]
        regular=np.min(np.sum(controls*chord[None,:,:],axis=2),axis=0)>1e-4
        for i in np.flatnonzero(~regular):
            dx=Polynomial([C[i,0],B[i,0],A[i,0]])
            dy=Polynomial([C[i,1],B[i,1],A[i,1]])
            norm2=dx*dx+dy*dy
            candidates=[0.,h[i]]+[r.real for r in norm2.deriv().roots() if abs(r.imag)<1e-7 and 0<r.real<h[i]]
            if min(norm2(x) for x in candidates)<1e-8:
                raise MoveRejected('The input has a cusp or collapsed tangent; repair that region before smooth dragging.')
        second=np.linalg.norm(self.spline(self.s,2),axis=1)
        self.second_bound=np.maximum(second[:-1],second[1:])
        self._sample_cache={}

    def reset_depth(self,marks):
        """Recondition auxiliary heights without altering the XY spline.

        Only crossing heights constrain a lift of a fixed planar diagram.
        Two lifts with the same crossing inequalities are joined by vertical
        interpolation through embeddings. This is independent of a drag's
        later spatial collision certificate.
        """
        if not marks:return
        marks=sorted(marks)
        parameters,heights=np.asarray(marks).T
        z=np.interp(self.s,parameters,heights,period=self.length)
        z[-1]=z[0]
        self.spline.c[:,:,2]=CubicSpline(self.s,z,bc_type='periodic').c
        second=np.linalg.norm(self.spline(self.s,2),axis=1)
        self.second_bound=np.maximum(second[:-1],second[1:])
        self._sample_cache.clear()

    def samples(self,extra_second_bound=0.,error=.005,*,patches=()):
        # Each patch is (periodic material center, support half-width, maximum
        # second derivative). Only spans intersecting a C2 patch need denser
        # chords. Elsewhere the old spline bound remains authoritative.
        key=(float(extra_second_bound),float(error),tuple(patches))
        if key in self._sample_cache:return self._sample_cache[key]
        h=np.diff(self.s)
        bound=self.second_bound+extra_second_bound
        if patches:
            bound=bound.copy();middle=(self.s[:-1]+self.s[1:])*.5
            for center,radius,maximum in patches:
                distance=np.abs((middle-center+self.length/2)%self.length-self.length/2)
                bound+=maximum*(distance<=radius+h/2)
        step=np.minimum(4.,np.sqrt(8*error/np.maximum(bound,1e-8)))
        counts=np.maximum(1,np.ceil(h/step).astype(int))
        previous=getattr(self,'_latest_sample_counts',None)
        if previous is not None and np.array_equal(counts,previous[0]):
            # A changed displacement bound often chooses the same integers.
            # Its actual sample coordinates are then exactly unchanged; reuse
            # their identity so spline/projection caches can reuse them too.
            result=previous[1]
        else:
            offsets=np.r_[0,np.cumsum(counts)]
            samples=np.repeat(self.s[:-1],counts)+(np.arange(offsets[-1])-np.repeat(offsets[:-1],counts))*np.repeat(h/counts,counts)
            result=np.r_[samples,self.length]
            self._latest_sample_counts=(counts,result)
        # Only keep the baseline plus the latest adaptive grid during a drag.
        self._sample_cache={k:v for k,v in self._sample_cache.items() if k[0]==0. and not k[2]}
        self._sample_cache[key]=result
        return result


class SmoothDrag:
    """One gesture; move targets use the original picked point's coordinates."""
    def __init__(self,diagram,edge_id,point,radius=65.,under=False):
        if not validate(diagram)['valid']:raise MoveRejected('Repair the diagram before dragging.')
        self.original=_copy_diagram(diagram);self.last_diagram=self.original
        self.anchor=np.asarray(point,float);self.current=np.zeros(2);self.last_moves=[]
        self.under=bool(under)
        self._array_probes=False
        self._cancellation_regions=None
        self.width=diagram.get('width',800);self.height=diagram.get('height',600)
        self.curves=[_SmoothCurve(c['points']) for c in to_curves(diagram)]
        edge=next((e for e in diagram['edges'] if e['id']==edge_id),None)
        if edge is None:raise MoveRejected('Pick a strand first.')
        # Component deletion and other graph edits may leave obsolete cached
        # geometry tags. Current oriented traversal is the authoritative order.
        self.ci=next(i for i,walk in enumerate(component_walks(diagram)) if any(eid==edge_id for eid,_ in walk))
        curve=self.curves[self.ci]
        # Restrict picking to the selected component, preserving its orientation.
        samples=curve.samples();xy=curve.spline(samples)[:,:2]
        nearest=int(np.argmin(np.linalg.norm(xy-self.anchor,axis=1)))
        self.center=float(samples[nearest])%curve.length
        self.radius=min(max(12.,float(radius)),curve.length*.42)
        self._prepare_regular()
        if any(math.dist(self.anchor,c['point'])<1. for c in diagram['crossings']):
            raise MoveRejected('Pick the strand beside the crossing, rather than the crossing center.')
        self.lift=0.
        self.depth_bumps=[]
        initial=[c.spline(c.samples()) for c in self.curves]
        rebuilt=self._diagram([c.samples() for c in self.curves],initial)
        if not _same_projection(diagram,rebuilt):
            raise MoveRejected('This crowded drawing cannot yet be smoothed without changing its diagram. Enlarge or repair that region first.')
        # A previous gesture can leave an old crossing barely separated in
        # auxiliary Z. That is a valid diagram, but the next ordinary planar
        # drag would immediately hit the clearance threshold. Refresh the
        # lift while keeping the XY splines and every crossing choice fixed.
        margin=max(2.,math.sqrt(self.width*self.height)/120)
        self.depth_refreshed=any(abs(c['motion_branches'][0]['height']-
                                    c['motion_branches'][1]['height'])<margin
                                 for c in rebuilt['crossings'])
        if self.depth_refreshed:
            marks=defaultdict(list)
            height=max(10.,math.sqrt(self.width*self.height)/35)
            for crossing in rebuilt['crossings']:
                for branch in crossing['motion_branches']:
                    marks[branch['curve']].append((branch['parameter'],height if branch['over'] else -height))
            for i,curve in enumerate(self.curves):curve.reset_depth(marks[i])
            grids=[c.samples() for c in self.curves]
            refreshed=self._diagram(grids,[c.spline(s) for c,s in zip(self.curves,grids)])
            if not _same_projection(rebuilt,refreshed):
                raise MoveRejected('The internal crossing spacing could not be refreshed safely. Use a smaller neighborhood.')
            # Check each branch, not just a potentially symmetric PD graph.
            for crossing in rebuilt['crossings']:
                a,b=crossing['motion_branches']
                za=self.curves[a['curve']].spline(a['parameter'])[2]
                zb=self.curves[b['curve']].spline(b['parameter'])[2]
                if (za-zb)*(1 if a['over'] else -1)<margin:
                    raise MoveRejected('The internal crossing spacing could not be refreshed safely. Use a smaller neighborhood.')
            rebuilt=refreshed
        # Shift chooses a *new* RII pair, not the height of the entire grabbed
        # neighborhood. Existing crossings therefore remain untouched at grab.
        self.last_projection=rebuilt
        self.checks=0
        self._prepare_static(rebuilt)

    def _prepare_static(self,diagram):
        """Retain independent projection components throughout one gesture.

        Whole spatial components remain closed. A projected component joined
        by an existing crossing is always included as a unit; new neighbors
        are added conservatively by bounding boxes before any reconstruction.
        Thus no omitted strand can acquire an unexamined crossing.
        """
        parents=list(range(len(self.curves)))
        def find(i):
            while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
            return i
        for crossing in diagram['crossings']:
            a,b=[branch['curve'] for branch in crossing['motion_branches']]
            parents[find(a)]=find(b)
        groups=defaultdict(set)
        for i in range(len(parents)):groups[find(i)].add(i)
        self._projection_groups={i:groups[find(i)] for i in range(len(parents))}
        self._active_components=set(self._projection_groups[self.ci])
        self._static_diagram=diagram
        self._static_grids=[c.samples() for c in self.curves]
        self._static_arrays=[c.spline(g) for c,g in zip(self.curves,self._static_grids)]
        self._position_cache={i:(g,a) for i,(g,a) in
                              enumerate(zip(self._static_grids,self._static_arrays))}
        self._weight_cache=None
        self._static_low=np.array([a[:,:2].min(axis=0) for a in self._static_arrays])
        self._static_high=np.array([a[:,:2].max(axis=0) for a in self._static_arrays])
        self._static_crossing_segments=self._initial_crossing_segments

    def _weight(self,s,derivative=0):
        length=self.curves[self.ci].length
        u=((np.asarray(s)-self.center+length/2)%length-length/2)/self.radius
        inside=np.abs(u)<1
        if derivative==0:return np.where(inside,(1-u*u)**3,0.)
        return np.where(inside,-6*u*(1-u*u)**2/self.radius,0.)

    def _grids(self,target,bumps=None):
        bound=6*math.sqrt(max(np.linalg.norm(self.current),np.linalg.norm(target))**2+self.lift**2)/self.radius**2
        patches=[[] for _ in self.curves]
        if bound:patches[self.ci].append((self.center,self.radius,bound))
        for bump in self.depth_bumps if bumps is None else bumps:
            ci,center,a,b,c,d,amplitude=self._bump_parts(bump)
            patches[ci].append(((center+(a+d)/2)%self.curves[ci].length,
                               (d-a)/2,6*abs(amplitude)/min(b-a,d-c)**2))
        return [c.samples(patches=tuple(patches[i])) for i,c in enumerate(self.curves)]

    def _grid_weight(self,grid):
        """Reuse the exact material bump values on one adaptive grid."""
        cached=getattr(self,'_weight_cache',None)
        if cached is None or cached[0] is not grid:
            cached=(grid,self._weight(grid));self._weight_cache=cached
        return cached[1]

    def _bump_parts(self,bump):
        # Original RII patches are relative to the selected material point;
        # RIII can also prepare stationary sheets without moving their XY.
        return (self.ci,self.center,*bump) if len(bump)==5 else bump

    def _grid_depth_weight(self,grid,shape,ci,center):
        """Reuse exact vertical patch weights on one immutable material grid."""
        cache=getattr(self,'_depth_grid_cache',{})
        record=cache.get(ci)
        if record is None or record[0] is not grid:
            record=(grid,{});cache[ci]=record
        key=(center,*shape)
        if key not in record[1]:
            # A rejected trial must not accumulate arbitrarily many patches.
            if len(record[1])>=32:record[1].clear()
            record[1][key]=self._depth_weight(grid,shape,ci,center)
        self._depth_grid_cache=cache
        return record[1][key]

    def _positions(self,grids,delta,lift=None,bumps=None):
        # All probes for one event share their spline samples. Heights and XY
        # offsets are still evaluated independently below; only the immutable
        # base curve is cached, with at most one adaptive grid per component.
        arrays=[]
        cache=getattr(self,'_position_cache',{})
        for i,(c,s) in enumerate(zip(self.curves,grids)):
            if hasattr(self,'_static_arrays') and s is self._static_grids[i]:
                base=self._static_arrays[i]
            else:
                stored=cache.get(i)
                if stored is None or stored[0] is not s:
                    stored=(s,c.spline(s));cache[i]=stored
                base=stored[1]
            arrays.append(base)
        self._position_cache=cache
        offset=np.r_[delta,self.lift if lift is None else lift]
        arrays[self.ci]=arrays[self.ci]+self._grid_weight(grids[self.ci])[:,None]*offset
        copied={self.ci}
        for bump in self.depth_bumps if bumps is None else bumps:
            ci,center,*shape,amplitude=self._bump_parts(bump)
            if ci not in copied:arrays[ci]=arrays[ci].copy();copied.add(ci)
            arrays[ci][:,2]+=amplitude*self._grid_depth_weight(grids[ci],shape,ci,center)
        return arrays

    def _material_offset(self,s):
        length=self.curves[self.ci].length
        return (np.asarray(s)-self.center+length/2)%length-length/2

    def _depth_weight(self,s,shape,curve=None,center=None):
        """C2 plateau for a local sheet, including a whole newborn bigon."""
        a,b,c,d=shape
        if curve is None:x=self._material_offset(s)
        else:
            length=self.curves[curve].length
            x=(np.asarray(s)-center+length/2)%length-length/2
        def smooth(t):
            t=np.clip(t,0.,1.)
            return t*t*t*(10+t*(-15+6*t))
        return smooth((x-a)/(b-a))*smooth((d-x)/(d-c))

    def _diagram(self,grids,arrays,projection_only=False):
        if not hasattr(self,'_projection_curve_cache'):self._projection_curve_cache={}
        if hasattr(self,'_static_diagram'):
            return self._local_diagram(grids,arrays,projection_only)
        segments=[]
        result=from_curves([{'points':p,'parameters':grids[i],
            'weights':self._grid_weight(grids[i]) if i==self.ci else np.zeros(len(p))}
            for i,p in enumerate(arrays)],self.width,self.height,_projection_only=projection_only,
            _export=False,_crossing_segments=segments,_curve_cache=self._projection_curve_cache)
        # If reconstruction removed duplicate vertices, its segment indices
        # do not address the original sample grid. Use the full pair search
        # in that exceptional case instead of retaining a misaligned cache.
        self._initial_crossing_segments=(segments if all(
            len(c['points'])==len(grid) for c,grid in zip(result['spatial_curves'],grids)) else None)
        return result

    def _local_diagram(self,grids,arrays,projection_only):
        # This is deliberately a superset: a distant component's unusually
        # long strand entering the moving component's box is included too.
        xy=arrays[self.ci][:,:2]
        low=np.array([xy[:,0].min(),xy[:,1].min()])-.005
        high=np.array([xy[:,0].max(),xy[:,1].max()])+.005
        neighbors=np.flatnonzero(np.all(self._static_high>=low,axis=1)&
                                 np.all(self._static_low<=high,axis=1))
        for ci in neighbors:self._active_components.update(self._projection_groups[int(ci)])
        active=sorted(self._active_components)
        candidates=self._projection_pairs(grids,arrays,active)
        if len(active)==len(arrays):
            return from_curves([{'points':p,'parameters':grids[i],
                'weights':self._grid_weight(grids[i]) if i==self.ci else np.zeros(len(p))}
                for i,p in enumerate(arrays)],self.width,self.height,
                _projection_only=projection_only,_export=False,_array_geometry=self._array_probes,
                _pair_candidates=candidates,_curve_cache=self._projection_curve_cache)
        changed=from_curves([{'points':arrays[i],'parameters':grids[i],'_cache_key':i,
            'weights':self._grid_weight(grids[i]) if i==self.ci else np.zeros(len(arrays[i]))}
            for i in active],self.width,self.height,_projection_only=projection_only,_export=False,
            _array_geometry=self._array_probes,_pair_candidates=candidates,
            _curve_cache=self._projection_curve_cache)
        # Re-index incidence without revisiting stationary geometric vertices.
        # Curve-major edge order keeps component_walks and spatial_curves in
        # exactly the same physical orientation/order as the global rebuild.
        crossings=[];edge_records=[];cross_maps=[]
        for block,drawing in enumerate((self._static_diagram,changed)):
            cmap={};cross_maps.append(cmap)
            for cross in drawing['crossings']:
                if block==0 and cross['motion_branches'][0]['curve'] in self._active_components:continue
                cid=len(crossings);cmap[cross['id']]=cid
                branches=[dict(b) for b in cross['motion_branches']]
                if block:
                    for branch in branches:branch['curve']=active[branch['curve']]
                crossings.append({**cross,'id':cid,'ports':[],'motion_branches':branches})
            for edge in drawing['edges']:
                ci=edge['motion_curve'] if block==0 else active[edge['motion_curve']]
                if block==0 and ci in self._active_components:continue
                edge_records.append((ci,edge['id'],block,edge))
        edges=[]
        for ci,_,block,old in sorted(edge_records,key=lambda record:record[:2]):
            edge={**old,'id':len(edges),'component':ci,'motion_curve':ci}
            for end,key in enumerate(('start','end')):
                if old[key] is None:continue
                cid,port=old[key];cid=cross_maps[block][cid]
                edge[key]=[cid,port]
                crossings[cid]['ports'].append((port,{'edge':edge['id'],'end':end}))
            edges.append(edge)
        for cross in crossings:cross['ports']=[ref for _,ref in sorted(cross['ports'])]
        spatial=list(self._static_diagram['spatial_curves'])
        for local,ci in enumerate(active):spatial[ci]=changed['spatial_curves'][local]
        result={'width':self.width,'height':self.height,'crossings':crossings,'edges':edges,
                'component_orientations':{},'spatial_curves':spatial}
        verdict=validate(result)
        if not verdict['valid']:raise MoveRejected('; '.join(verdict['errors']))
        return result

    def _projection_pairs(self,grids,arrays,active):
        """Reuse the validated stationary projection, including within a curve.

        A segment is reused only when both material endpoints AND both XY
        endpoints match the initial segment exactly. Refinement, displacement,
        and new neighbors all force fresh local box queries. Cached crossings
        are still intersected using current XYZ heights by from_curves.
        """
        if self._static_crossing_segments is None:return None
        cache=getattr(self,'_projection_segment_cache',{})
        maps={};changed=[];offset=0
        for ci in active:
            grid=grids[ci];base=self._static_grids[ci];points=arrays[ci]
            record=cache.get(ci)
            if record is None or record['grid'] is not grid:
                if grid is base:
                    old=np.arange(len(base)-1);material=np.ones(len(old),dtype=bool)
                else:
                    old=np.clip(np.searchsorted(base,grid[:-1]),0,len(base)-2)
                    material=(base[old]==grid[:-1])&(base[old+1]==grid[1:])
                record={'grid':grid,'old':old,'material':material,'points':None,
                        'starts':self._static_arrays[ci][old,:2],
                        'ends':self._static_arrays[ci][old+1,:2]}
                cache[ci]=record
            if record['points'] is not points:
                old=record['old']
                stable=(record['material']
                        &(points[:-1,0]==record['starts'][:,0])
                        &(points[:-1,1]==record['starts'][:,1])
                        &(points[1:,0]==record['ends'][:,0])
                        &(points[1:,1]==record['ends'][:,1]))
                mapping=np.full(len(base)-1,-1,dtype=int)
                mapping[old[stable]]=np.flatnonzero(stable)
                record.update(points=points,mapping=mapping,changed=~stable)
            mapping=record['mapping']
            maps[ci]=np.where(mapping>=0,mapping+offset,-1)
            changed.append(record['changed']);offset+=len(grid)-1
        self._projection_segment_cache=cache
        starts=np.concatenate([arrays[i][:-1,:2] for i in active])
        ends=np.concatenate([arrays[i][1:,:2] for i in active])
        pairs=_box_pairs(np.minimum(starts,ends)-.0025,np.maximum(starts,ends)+.0025,
                         active=np.concatenate(changed))
        stationary=[]
        for (ci,si),(cj,sj) in self._static_crossing_segments:
            if ci not in maps or cj not in maps:continue
            a,b=maps[ci][si],maps[cj][sj]
            if a>=0 and b>=0:stationary.append((a,b))
        if stationary:pairs=np.vstack((pairs,np.asarray(stationary,dtype=int)))
        if len(pairs):
            pairs.sort(axis=1)
            pairs=pairs[np.lexsort((pairs[:,0],pairs[:,1]))]
        return pairs

    def _public_diagram(self,diagram):
        """Serialize only the accepted public frame, never internal probes."""
        # Give deepcopy the final point lists directly. Converting the engine
        # arrays first used to allocate every row twice and degraded the next
        # frame's internal array representation. The public copy still owns
        # every list, including geometry retained from stationary components.
        memo={}
        for item in diagram.get('edges',[])+diagram.get('spatial_curves',[]):
            points=item['points']
            if isinstance(points,np.ndarray) and id(points) not in memo:
                memo[id(points)]=points.tolist()
        result=_copy_diagram(diagram,memo)
        if 'spatial_curves' in result:result['spatial_fingerprint']=_fingerprint(result)
        return result

    def _new_crossings(self,diagram):
        """Track crossing branches by material position, never PD numbering.

        RII births occur within a pair of components. Matching the older
        crossings first keeps a modifier change from acting on those crossings.
        The subsequent spatial certificate is authoritative if a crowded step
        makes this numerical continuation ambiguous.
        """
        old_groups=defaultdict(list);new_groups=defaultdict(list)
        def branches(crossing):
            return sorted(crossing['motion_branches'],key=lambda b:(b['curve'],b['parameter']))
        for crossing in self.last_projection['crossings']:
            old_groups[tuple(b['curve'] for b in branches(crossing))].append(crossing)
        for crossing in diagram['crossings']:
            new_groups[tuple(b['curve'] for b in branches(crossing))].append(crossing)
        born=[]
        for pair,new in new_groups.items():
            old=old_groups[pair]
            if (len(new)-len(old))%2:
                raise MoveRejected('The proposed drag has an unresolved projection singularity; move more gradually.')
            if len(new)<=len(old):continue
            if not old:
                born.extend(new);continue
            cost=_branch_costs([branches(c) for c in old],[branches(c) for c in new],
                               [c.length for c in self.curves])
            _,matched=linear_sum_assignment(cost)
            matched=set(matched)
            born.extend(c for i,c in enumerate(new) if i not in matched)
        return born

    def _birth_choices(self,born,requested,displacement):
        """Preserve the sheets of a shrinking, initially empty bigon.

        Small wiggles can create temporary pairs while those same two arcs
        cancel. Treat them as continuation of the existing over/under sheets.
        A genuine birth outside those material arcs still uses Shift. Once
        the old lens has disappeared, there are no continuing pairs to inherit.
        This chooses a candidate lift; the full swept certificate is unchanged.
        """
        choices=[bool(requested)]*len(born)
        if not born:return choices
        if self._cancellation_regions is None:
            from .reidemeister_ii import removable_bigons
            drawing=self._static_diagram
            cmap={c['id']:c for c in drawing['crossings']}
            emap={e['id']:e for e in drawing['edges']}
            self._cancellation_regions=[]
            # Only a bigon whose actual boundary contains the grabbed material
            # point can provide continuation choices. Discover those few
            # candidates from incidence before testing any polygon geometry.
            # Scanning every remote face here made one RII birth spend seconds
            # certifying irrelevant, densely sampled lenses.
            grouped=defaultdict(list);selected=set()
            for edge in drawing['edges']:
                start,end=edge.get('start'),edge.get('end')
                if start is None or end is None or start[0]==end[0]:continue
                pair=tuple(sorted((start[0],end[0])))
                grouped[pair].append(edge['id'])
                if edge['motion_curve']!=self.ci:continue
                branches=[]
                for cid,port in (start,end):
                    crossing=cmap[cid];over=port%2==crossing['over']
                    branches.append(next(b for b in crossing['motion_branches'] if b['over']==over))
                a,b=branches;length=self.curves[self.ci].length
                if ((self.center-a['parameter'])%length<=
                        (b['parameter']-a['parameter'])%length):
                    selected.add(edge['id'])
            identifiers=set()
            for (c1,c2),ids in grouped.items():
                for first in ids:
                    if first not in selected:continue
                    for second in ids:
                        if second==first:continue
                        a,b=sorted((first,second));identifiers.add(f'{c1}:{c2}:{a}:{b}')
            regions=chain.from_iterable(removable_bigons(drawing,bigon_id=identifier)
                                         for identifier in sorted(identifiers))
            for region in regions:
                spans=[];paths=[]
                for eid in region['edge_ids']:
                    edge=emap[eid];branches=[]
                    for cid,port in (edge['start'],edge['end']):
                        crossing=cmap[cid];over=port%2==crossing['over']
                        branches.append(next(b for b in crossing['motion_branches'] if b['over']==over))
                    a,b=branches;ci=a['curve'];length=self.curves[ci].length
                    if b['curve']!=ci:break
                    spans.append((ci,a['parameter'],(b['parameter']-a['parameter'])%length,a['over']))
                    paths.append(edge['points'])
                if len(spans)!=2:continue
                for i,(ci,start,length,over) in enumerate(spans):
                    if ci!=self.ci or (self.center-start)%self.curves[ci].length>length:continue
                    other=np.asarray(paths[1-i]);s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(other,axis=0),axis=1))]
                    middle=other[np.searchsorted(s,s[-1]/2)]
                    self._cancellation_regions.append((spans,i,middle-self.anchor))
        def contains(span,branch):
            ci,start,length,_=span
            return branch['curve']==ci and (branch['parameter']-start)%self.curves[ci].length<=length+1e-7
        def ordered(region,crossing):
            spans,selected,_=region
            a,b=crossing['motion_branches']
            for pair in ((a,b),(b,a)):
                if all(contains(span,branch) for span,branch in zip(spans,pair)):return pair
            return None
        for region in self._cancellation_regions:
            spans,selected,toward=region
            if float(displacement@toward)<=1e-8:continue
            continuing=[pair for crossing in self.last_projection['crossings']
                        if (pair:=ordered(region,crossing)) is not None]
            if len(continuing)<2 or any(pair[selected]['over']!=spans[selected][3] for pair in continuing):continue
            for i,crossing in enumerate(born):
                pair=ordered(region,crossing)
                if pair is None:continue
                moving=pair[selected];other=pair[1-selected]
                if moving['weight']-other['weight']>1e-8:choices[i]=not spans[selected][3]
        return choices

    def _birth_depth(self,born,under):
        """Prepare smooth depth changes on free arcs before a new RII event.

        Each compact C2 bump vanishes in a neighborhood of every current
        crossing preimage. Both height preparation and planar translation are
        later checked as swept spatial motions, so no crossing can be flipped.
        """
        wanted=np.where(np.broadcast_to(under,(len(born),)),-1.,1.)
        margin=max(2.,math.sqrt(self.width*self.height)/120)
        pairs=[]
        for crossing in born:
            fixed,moving=sorted(crossing['motion_branches'],key=lambda b:b['weight'])
            if moving['curve']!=self.ci or moving['weight']-fixed['weight']<1e-8:
                raise MoveRejected('The moving branch of this new pair is ambiguous; use a smaller move.')
            pairs.append((moving,fixed))
        gaps=np.array([a['height']-b['height'] for a,b in pairs])
        if np.all(wanted*gaps>=margin):return list(self.depth_bumps)
        protected=[float(self._material_offset(b['parameter'])) for c in self.last_projection['crossings']
                   for b in c['motion_branches'] if b['curve']==self.ci]
        cuts=sorted(set([-self.radius,self.radius]+[x for x in protected if abs(x)<self.radius]))
        intervals=defaultdict(list)
        for moving,_ in pairs:
            position=float(self._material_offset(moving['parameter']))
            index=int(np.searchsorted(cuts,position))-1
            if index<0 or index>=len(cuts)-1:
                raise MoveRejected('The new pair is outside the moving neighborhood; use a smaller move.')
            intervals[index].append(position)
        shapes=[]
        for index,positions in intervals.items():
            left,right=cuts[index:index+2]
            # Keep a small unchanged collar beside every older crossing.
            collar=min(.25,(right-left)*.01)
            a,d=left+collar,right-collar
            b=min(min(positions),a+.15*(d-a))
            c=max(max(positions),d-.15*(d-a))
            if min(b-a,d-c)<1.:
                raise MoveRejected('This new pair is too close to an existing crossing for a safe depth choice. Move more gradually or use a smaller neighborhood.')
            # Include the intervening material arc, where the two crossings
            # actually coalesce. Isolated bumps only at their final positions
            # would allow birth at the wrong height and then a forbidden flip.
            shapes.append((a,b,c,d))
        matrix=np.array([[float(self._depth_weight(a['parameter'],shape))-
            (float(self._depth_weight(b['parameter'],shape)) if b['curve']==self.ci else 0.)
            for shape in shapes] for a,b in pairs])
        desired=margin-wanted*gaps
        if np.all(wanted==wanted[0]):
            # This is the original nonnegative solve, but one free interval
            # only needs scalar bounds, not initialization of an LP solver.
            if len(shapes)==1:
                coefficients=matrix[:,0];lower=0.;upper=math.inf
                for coefficient,required in zip(coefficients,desired):
                    if abs(coefficient)<1e-12:
                        if required>1e-8:lower=math.inf
                    elif coefficient>0:lower=max(lower,required/coefficient)
                    else:upper=min(upper,required/coefficient)
                if not math.isfinite(lower) or lower>upper+1e-8:
                    raise MoveRejected('These simultaneous new crossings need different depth neighborhoods. Move more gradually.')
                amplitudes=np.array([lower])*wanted[0]
            else:
                solution=linprog(np.ones(len(shapes)),A_ub=-matrix,b_ub=-desired,bounds=(0,None),method='highs')
                if not solution.success:raise MoveRejected('These simultaneous new crossings need different depth neighborhoods. Move more gradually.')
                amplitudes=wanted[0]*solution.x
        else:
            signed=matrix*wanted[:,None]
            solution=linprog(np.ones(2*len(shapes)),A_ub=np.hstack((-signed,signed)),
                             b_ub=-desired,bounds=(0,None),method='highs')
            if not solution.success:raise MoveRejected('These simultaneous new crossings need different depth neighborhoods. Move more gradually.')
            amplitudes=solution.x[:len(shapes)]-solution.x[len(shapes):]
        return self.depth_bumps+[(*shape,float(amplitude))
            for shape,amplitude in zip(shapes,amplitudes) if abs(amplitude)>1e-10]

    def _continued_crossings(self,diagram):
        """Match persisting crossing sheets through an RIII slide.

        Material parameters identify sheets even when their planar order
        changes. This preparation is only used in steps without births/deaths.
        """
        def branches(c):
            return sorted(c['motion_branches'],key=lambda b:(b['curve'],b['parameter']))
        groups=[]
        for drawing in (self.last_projection,diagram):
            group=defaultdict(list)
            for c in drawing['crossings']:
                bs=branches(c);group[tuple(b['curve'] for b in bs)].append(bs)
            groups.append(group)
        old,new=groups
        if set(old)!=set(new) or any(len(old[k])!=len(new[k]) for k in old):return None
        result=[]
        def cost(a,b):
            return sum(((x['parameter']-y['parameter']+self.curves[x['curve']].length/2)
                        %self.curves[x['curve']].length-self.curves[x['curve']].length/2)**2
                       for x,y in zip(a,b))
        for pair,aa in old.items():
            bb=new[pair]
            costs=_branch_costs(aa,bb,[c.length for c in self.curves])
            rows,columns=linear_sum_assignment(costs)
            for i,j in zip(rows,columns):
                a,b=aa[i],bb[j]
                if pair[0]==pair[1] and cost(a,b[::-1])<cost(a,b):b=b[::-1]
                result.append((a,b))
        return result

    def _slide_depth_candidates(self,projected):
        """Re-space a moving sheet, preserving all existing crossing choices.

        Canonical lifts assign just two crossing heights. Near a triple point,
        a top sheet can therefore meet the middle sheet's height at the fixed
        crossing. A local vertical preparation gives the three sheets room.
        Every candidate still requires both vertical and planar swept checks;
        cyclic/forbidden crossing orders cannot pass that certificate.
        """
        matched=self._continued_crossings(projected)
        if not matched:return
        margin=max(2.,math.sqrt(self.width*self.height)/120)
        r=self.radius
        shape=(-r,-.7*r,.7*r,r)
        lower,upper=-math.inf,math.inf
        for old,new in matched:
            sign=1. if old[0]['over'] else -1.
            for a,b in (old,new):
                if not any(x['curve']==self.ci and x['weight']>1e-8 for x in (a,b)):continue
                weights=[float(self._depth_weight(x['parameter'],shape)) if x['curve']==self.ci else 0.
                         for x in (a,b)]
                coefficient=sign*(weights[0]-weights[1])
                required=margin-sign*(a['height']-b['height'])
                if abs(coefficient)<1e-10:
                    if required>1e-8:return
                elif coefficient>0:lower=max(lower,required/coefficient)
                else:upper=min(upper,required/coefficient)
        if lower>upper:return
        best=float(np.clip(0.,lower,upper))
        candidates=[best]
        for amplitude in (margin,-margin,2*margin,-2*margin,4*margin,-4*margin):
            candidates.append(float(np.clip(amplitude,lower,upper)))
        seen=set()
        for amplitude in candidates:
            if abs(amplitude)<1e-8 or abs(amplitude)>1e4*margin or amplitude in seen:continue
            seen.add(amplitude)
            yield self.depth_bumps+[(*shape,amplitude)]

    def _triangle_depth_candidates(self,projected):
        """Give all three sheets of a nearby triangular face distinct room.

        The middle sheet may need the two stationary sheets to separate in Z.
        These compact patches leave every planar coordinate untouched during
        preparation. Old and continued crossing inequalities constrain an L1
        minimum-height solve; the all-time spatial certificate remains final.
        """
        matched=self._continued_crossings(projected)
        if not matched:return
        drawing=self.last_projection
        crossings={c['id']:c for c in drawing['crossings']}
        edges={e['id']:e for e in drawing['edges']}
        seen=set();triangles=[]
        for cid in crossings:
            for port in range(4):
                key=(cid,port);face=[]
                while key not in seen:
                    seen.add(key);at,p=key
                    ref=crossings[at]['ports'][p];edge=edges[ref['edge']]
                    face.append(edge)
                    end=edge['start' if ref['end'] else 'end']
                    key=(end[0],(end[1]+1)%4)
                if len(face)!=3 or len({e['id'] for e in face})!=3:continue
                patches=[];active=False
                for edge in face:
                    branches=[]
                    for at,p in (edge['start'],edge['end']):
                        crossing=crossings[at];over=p%2==crossing['over']
                        branches.append(next(b for b in crossing['motion_branches'] if b['over']==over))
                    a,b=branches;ci=a['curve'];length=self.curves[ci].length
                    if ci!=b['curve']:break
                    delta=(b['parameter']-a['parameter']+length/2)%length-length/2
                    # Only short local triangular sheets, never a long exterior
                    # arc that happens to have the same combinatorial face size.
                    arc=sum(math.dist(x,y) for x,y in zip(edge['points'],edge['points'][1:]))
                    if arc>self.radius*2 or abs(delta)>self.radius*2:break
                    active|=any(x['curve']==self.ci and x['weight']>.05 for x in branches)
                    center=(a['parameter']+delta/2)%length
                    half=abs(delta)/2+1.
                    tail=max(4.,min(self.radius*.3,15.))
                    support=half+tail
                    if support>=length*.45:break
                    patches.append((ci,center,-support,-half,half,support))
                if len(patches)==3 and active:triangles.append(patches)
        margin=max(2.,math.sqrt(self.width*self.height)/120)
        for patches in triangles:
            rows=[];gaps=[]
            for old,new in matched:
                sign=1. if old[0]['over'] else -1.
                for a,b in (old,new):
                    row=[]
                    for ci,center,*shape in patches:
                        weights=[float(self._depth_weight(x['parameter'],shape,ci,center))
                                 if x['curve']==ci else 0. for x in (a,b)]
                        row.append(sign*(weights[0]-weights[1]))
                    if max(abs(x) for x in row)<1e-10:continue
                    rows.append(row);gaps.append(sign*(a['height']-b['height']))
            if not rows:continue
            matrix=np.asarray(rows)
            for extra in (1.,2.,4.):
                desired=margin*extra-np.asarray(gaps)
                solution=linprog(np.ones(6),A_ub=np.hstack((-matrix,matrix)),
                                 b_ub=-desired,bounds=(0,None),method='highs')
                if not solution.success:continue
                values=solution.x[:3]-solution.x[3:]
                if max(abs(values))>1e4*margin:continue
                candidate=self.depth_bumps+[(*patch,float(value)) for patch,value in zip(patches,values)
                                            if abs(value)>1e-8]
                if candidate!=self.depth_bumps:yield candidate

    def _prepare_regular(self):
        """Cache the fixed material polynomials for this gesture's C2 support."""
        curve=self.curves[self.ci];length=curve.length
        breaks=list(curve.s)
        breaks.extend((self.center+x)%length for x in (-self.radius,self.radius,-length/2,length/2))
        breaks=sorted(set(breaks));pieces=[]
        for a,b in zip(breaks,breaks[1:]):
            if b-a<1e-9:continue
            mid=(a+b)/2
            wrapped=(mid-self.center+length/2)%length-length/2
            if abs(wrapped)>=self.radius:continue
            ci=min(np.searchsorted(curve.s,mid,side='right')-1,len(curve.s)-2)
            shift=a-curve.s[ci]
            A=3*curve.spline.c[0,ci,:2];B=2*curve.spline.c[1,ci,:2];C=curve.spline.c[2,ci,:2]
            tangent=np.zeros((6,2));tangent[:3]=[C+B*shift+A*shift**2,B+2*A*shift,A]
            offset=wrapped-(mid-a)
            wp=np.zeros(6)
            for power,coefficient in ((1,-6.),(3,12.),(5,-6.)):
                for j in range(power+1):
                    wp[j]+=coefficient*math.comb(power,j)*offset**(power-j)/self.radius**(power+1)
            pieces.append((b-a,tangent,wp))
        self._regular_pieces=pieces
        # Bernstein control vectors give a cheap sufficient test for every
        # material point and every time in a step. Ambiguous spans still use
        # the exact same polynomial-root cusp test as before.
        transform=np.array([[math.comb(i,j)/math.comb(5,j) if j<=i else 0.
                             for j in range(6)] for i in range(6)])
        powers=np.array([h**np.arange(6) for h,_,_ in pieces])
        tangents=np.array([v for _,v,_ in pieces]);weights=np.array([w for _,_,w in pieces])
        self._tangent_controls=np.einsum('ij,pj,pjk->pik',transform,powers,tangents)
        self._weight_controls=np.einsum('ij,pj,pj->pi',transform,powers,weights)
        middle=self._tangent_controls.mean(axis=1)
        self._tangent_axes=middle/np.maximum(np.linalg.norm(middle,axis=1),1e-16)[:,None]

    def _regular(self,target):
        """Find projected-tangent zeros over the whole straight mouse step.

        The tangent is polynomial in material position and affine in time.
        Eliminating time with a 2D determinant leaves degree <=5 polynomials.
        """
        delta=target-self.current
        if np.linalg.norm(delta)<1e-9:return
        poly=np.polynomial.polynomial
        current_controls=self._tangent_controls+self._weight_controls[:,:,None]*self.current
        target_controls=self._tangent_controls+self._weight_controls[:,:,None]*target
        start_projection=np.sum(current_controls*self._tangent_axes[:,None,:],axis=2)
        end_projection=np.sum(target_controls*self._tangent_axes[:,None,:],axis=2)
        safe=(start_projection.min(axis=1)>1e-4)&(end_projection.min(axis=1)>1e-4)
        for index in np.flatnonzero(~safe):
            width,tangent,wp=self._regular_pieces[index]
            vx=tangent[:,0]+self.current[0]*wp;vy=tangent[:,1]+self.current[1]*wp
            equation=vx*delta[1]-vy*delta[0]
            if max(abs(equation),default=0)<1e-9:
                f=vx*delta[0]+vy*delta[1]
                g=f+float(delta@delta)*wp
                product=np.convolve(f,g)
                candidates=[0.,width]+[r.real for r in poly.polyroots(np.arange(1,len(product))*product[1:])
                                      if abs(r.imag)<1e-6 and 0<r.real<width]
                if min(poly.polyval(x,product) for x in candidates)<=1e-10:
                    raise MoveRejected('This drag would form a cusp. Use Add curl for a Reidemeister I move.')
                continue
            roots=[r.real for r in poly.polyroots(equation) if abs(r.imag)<1e-6 and -1e-8<=r.real<=width+1e-8]
            for x in roots:
                weight=float(poly.polyval(x,wp))
                if abs(weight)<1e-12:continue
                v=np.array([poly.polyval(x,vx),poly.polyval(x,vy)])
                t=-float(v@delta)/(float(delta@delta)*weight)
                if -1e-8<=t<=1+1e-8 and np.linalg.norm(v+t*weight*delta)<1e-5:
                    raise MoveRejected('This drag would form a cusp. Use Add curl for a Reidemeister I move.')

    def move(self,target,under=None,*,_export=True):
        """Advance chronologically so separate RII events get their own depth.

        ``_export=False`` is a private read-only view for consumers that will
        immediately replace this graph (such as compact-box recompression).
        It must not be mutated or handed to application/history state. Normal
        callers receive an independent JSON-compatible value as before.

        Endpoint crossing counts alone miss a pair dying at one location while
        another is born farther along the mouse path. Mandatory event intervals
        resolve births and deaths separately; certified coarse steps use the
        previous fine chronology as fallback. The public request is atomic.
        """
        target=np.asarray(target,float)
        if target.shape!=(2,) or not np.isfinite(target).all():raise MoveRejected('Choose a finite position.')
        requested=self.under if under is None else bool(under)
        start=self.anchor+self.current
        distance=float(np.linalg.norm(target-start))
        if distance<1e-7:return self._public_diagram(self.last_diagram) if _export else self.last_diagram
        self._regular(target-self.anchor)
        step=min(8.,self.radius/24.,min(c.length for c in self.curves)/64.)
        count=int(math.ceil(distance/step))
        if count>256:raise MoveRejected('This mouse step is too large to resolve its crossing events. Move more gradually.')
        grids=self._grids(target-self.anchor)
        events=_projected_rii_times(self._positions(grids,self.current),
                                   self._positions(grids,target-self.anchor))
        fractions={i/count:False for i in range(1,count+1)}
        for left,right in zip(events,events[1:]+[1.]):
            # These intervals are mandatory: skipping one can silently erase
            # a tightly spaced birth/death pair from the observed chronology.
            if right-left<1e-9:
                raise MoveRejected('The projected crossing events are too close to resolve. Move more gradually.')
            fractions[(left+right)/2]=True
        if len(fractions)>256:
            raise MoveRejected('This mouse step contains too many crossing events. Move more gradually.')
        saved={name:getattr(self,name) for name in ('current','depth_bumps','last_diagram',
               'last_projection','checks','last_moves','under','lift')}
        mandatory={fraction:True for fraction,required in fractions.items() if required}
        # The all-time cusp/spatial certificates, rather than short display
        # frames, certify a move. Keep every interval that the exact RII scan
        # requires; a midpoint additionally bounds continuation ambiguity for
        # long RIII slides. Retry the previous fine chronology when needed.
        coarse={1.:False,**mandatory}
        if count>1:coarse.setdefault(.5,False)
        if len(coarse)>=len(fractions):coarse=fractions
        def restore():
            for name,value in saved.items():setattr(self,name,value)
        def advance(schedule):
            moves=[];accepted=[]
            for fraction,required in sorted(schedule.items()):
                point=start+(target-start)*fraction
                try:diagram=self._move_once(point,requested)
                except MoveRejected as exc:
                    # A nongeneric projected instant is not a displayable
                    # diagram. The next accepted frame still certifies the
                    # entire skipped spatial interval.
                    if fraction<1 and not required and any(word in str(exc) for word in ('projected','projection','transverse')):
                        continue
                    raise
                moves.extend(self.last_moves);accepted.append(float(fraction))
            return diagram,moves,accepted
        retried=False
        self._array_probes=True
        try:
            try:diagram,moves,accepted=advance(coarse)
            except MoveRejected as exc:
                restore()
                # Only geometric ambiguity/clearance failures warrant a finer
                # attempt. Unexpected/injected failures remain atomic errors.
                geometric=any(word in str(exc).lower() for word in
                    ('projected','projection','crossing','depth','spatial','space','strand','pair','neighborhood'))
                if coarse is fractions or not geometric:raise
                retried=True
                diagram,moves,accepted=advance(fractions)
        except Exception:
            restore()
            raise
        finally:
            self._array_probes=False
        self.last_moves=moves
        self.last_step_stats={'rii_event_times':events,'mandatory_fractions':sorted(mandatory),
                              'accepted_fractions':accepted,'fine_retry':retried,
                              'previous_uniform_steps':count}
        diagram['motion']['moves']=moves
        self.last_diagram=diagram
        self.last_projection=diagram
        return self._public_diagram(diagram) if _export else diagram

    def _move_once(self,target,requested):
        displacement=target-self.anchor
        if np.linalg.norm(displacement-self.current)<1e-7:return self.last_diagram
        self._regular(displacement)
        grids=self._grids(displacement)
        projected=self._diagram(grids,self._positions(grids,displacement),projection_only=True)
        born=self._new_crossings(projected)
        choices=self._birth_choices(born,requested,displacement)
        bumps=self._birth_depth(born,choices)
        grids=self._grids(displacement,bumps)
        before=self._positions(grids,self.current)
        prepared=before if bumps==self.depth_bumps else self._positions(grids,self.current,bumps=bumps)
        after=self._positions(grids,displacement,bumps=bumps)
        try:
            checks=(0 if prepared is before else _swept_clear(before,prepared))+_swept_clear(prepared,after)
        except MoveRejected as collision:
            if born:raise
            for candidate in chain(self._slide_depth_candidates(projected),self._triangle_depth_candidates(projected)):
                trial_grids=self._grids(displacement,candidate)
                trial_before=self._positions(trial_grids,self.current)
                trial_prepared=self._positions(trial_grids,self.current,bumps=candidate)
                trial_after=self._positions(trial_grids,displacement,bumps=candidate)
                try:
                    checks=_swept_clear(trial_before,trial_prepared)+_swept_clear(trial_prepared,trial_after)
                except MoveRejected:
                    continue
                grids,before,prepared,after,bumps=trial_grids,trial_before,trial_prepared,trial_after,candidate
                break
            else:
                raise collision
        if bumps==self.depth_bumps:
            # With identical depths the projection probe is already exactly
            # this sampled graph. Certify its strict height condition before
            # reuse instead of rebuilding a dense diagram twice per frame.
            for crossing in projected['crossings']:
                a,b=crossing['motion_branches']
                if abs(a['height']-b['height'])<.015:
                    raise MoveRejected('Two spatial strands would touch. Try a different path or smaller neighborhood.')
            diagram=projected
        else:
            diagram=self._diagram(grids,after)
        before_count=len(self.last_diagram['crossings']);after_count=len(diagram['crossings'])
        final_born=born if diagram is projected else self._new_crossings(diagram)
        final_choices=choices if diagram is projected else self._birth_choices(final_born,requested,displacement)
        for crossing,under in zip(final_born,final_choices):
            moving=max(crossing['motion_branches'],key=lambda b:b['weight'])
            if moving['over']==under:
                raise MoveRejected('This new pair cannot safely use the requested depth. Move more gradually or use a smaller neighborhood.')
        if (after_count-before_count)%2:
            raise MoveRejected('The proposed drag has an unresolved projection singularity; move more gradually.')
        self.last_moves=[]
        if before_count!=after_count:self.last_moves.append(f'RII/III motion: {after_count-before_count:+d} crossings')
        elif not _same_projection(self.last_diagram,diagram):self.last_moves.append('RII/III motion: crossing order changed')
        diagram['motion']={'kind':'smooth_regular_isotopy','radius':self.radius,
            'under_requested':requested,
            'depth_refreshed':self.depth_refreshed,
            'moves':self.last_moves,'swept_distance_checks':checks,'chord_error_bound':.005,
            'collision_clearance':.04,'cusp_check':'polynomial tangent roots',
            'note':'Numerical C2 curve motion with conservative swept spatial separation; singular targets are rejected.'}
        self.current=displacement;self.depth_bumps=bumps
        self.last_diagram=diagram;self.last_projection=diagram;self.checks+=checks
        return diagram
