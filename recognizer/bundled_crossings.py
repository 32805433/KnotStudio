"""Geometric evidence for a strand omitted underneath a parallel bundle.

The scale of an occluded region is the spacing of its visible strands, not just
pen width. A long connector is admitted only when both ends face the region,
its visible intersections are parallel, and the region accounts for its length.
This module receives pixels-derived geometry only, never a sample identity.
"""
from __future__ import annotations
import numpy as np


def bundle_evidence(a, b, hits, starts, ends, owners, width):
    """Describe a compact parallel over-strand bundle, or return None.

    Visible pieces may belong to the same component; path IDs are not component
    identities. Requiring distinct paths would incorrectly reject nested loops.
    """
    if len(hits) < 2:
        return None
    length = float(np.linalg.norm(b-a))
    positions = np.array([h['gap_t'] * length for h in hits])
    spacing = np.diff(positions)
    if spacing.min() < 1.4*width or spacing.max() > 4*spacing.min():
        return None
    directions=[]
    for hit in hits:
        indices=np.flatnonzero(owners == hit['path'])
        vec=ends[indices]-starts[indices]
        norms=np.linalg.norm(vec,axis=1)
        t=np.sum((hit['point']-starts[indices])*vec,axis=1)/np.maximum(norms**2,1e-12)
        closest=starts[indices]+np.clip(t,0,1)[:,None]*vec
        k=int(np.argmin(np.linalg.norm(closest-hit['point'],axis=1)))
        directions.append(vec[k]/max(norms[k],1e-12))
    directions=np.array(directions)
    parallel=float(np.min(np.abs(directions@directions.T)))
    if parallel < .65 or min(h['sine_angle'] for h in hits) < .3:
        return None
    # A gap must stop reasonably close to the outermost strands. Otherwise a
    # distant bundle merely happens to lie somewhere along an unsupported join.
    margin=max(3*width, 1.3*float(np.median(spacing)))
    if max(positions[0], length-positions[-1]) > margin:
        return None
    return dict(crossing_count=len(hits), spacing=float(np.median(spacing)),
                parallelism=parallel, margins=[float(positions[0]),float(length-positions[-1])])


def stable_tangent(points, end, distance):
    """Line-fit endpoint direction over an arc-length window (pixel-noise safe)."""
    points=np.asarray(points,float)
    if end:
        points=points[::-1]
    lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    return _sampled_tangent(points, lengths, distance)


def _sampled_tangent(points, lengths, distance):
    extent=min(float(distance),float(lengths[-1]))
    if extent < 1e-6:
        return np.array([1.,0.])
    s=np.linspace(0,extent,9)
    samples=np.column_stack([np.interp(s,lengths,points[:,axis]) for axis in (0,1)])
    # A linear least-squares fit averages the stair steps of thin raster ink.
    slope=(s-s.mean()) @ (samples-samples.mean(axis=0))
    return -slope/max(float(np.linalg.norm(slope)),1e-12)


class EndpointTangents:
    """Reuse arc lengths and identical reach windows within one matching pass."""
    def __init__(self, arrays, endpoints):
        self.arrays, self.endpoints = arrays, endpoints
        self.prepared, self.values = {}, {}

    def get(self, index, distance):
        if index not in self.prepared:
            endpoint = self.endpoints[index]
            points = self.arrays[endpoint['path']]
            if endpoint['end']:
                points = points[::-1]
            lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            self.prepared[index] = points, lengths
        points, lengths = self.prepared[index]
        extent = min(float(distance), float(lengths[-1]))
        key = index, extent
        if key not in self.values:
            self.values[key] = _sampled_tangent(points, lengths, extent)
        return self.values[key]


def overlaps_visible_ink(a,b,starts,ends,width):
    """A missing segment cannot run along an already visible intermediate arc.

    Intersection tests alone miss collinear ink. Without this check a connector
    can skip several genuine gap endpoints in a row of ordinary crossings.
    """
    delta=b-a
    length=float(np.linalg.norm(delta))
    direction=delta/max(length,1e-12)
    normal=np.array([-direction[1],direction[0]])
    vec=ends-starts
    norms=np.linalg.norm(vec,axis=1)
    aligned=np.abs(vec@direction)> .95*norms
    near=(np.abs((starts-a)@normal)<.8*width)&(np.abs((ends-a)@normal)<.8*width)
    t0=(starts-a)@direction;t1=(ends-a)@direction
    overlap=np.minimum(np.maximum(t0,t1),length-2*width)-np.maximum(np.minimum(t0,t1),2*width)
    return bool(np.sum(np.maximum(overlap[aligned&near],0))>max(2*width,.04*length))


def calibrated_single_evidence(a, b, hits, alignment, width, bundle_margin):
    """Admit a single crossing using gap scale measured from visible bundles.

    A fine-line illustration may leave much more whitespace at crossings than
    its pen width suggests. A confident bundle supplies an independent local
    drawing convention: its two external blank margins. The caller computes
    their median from well-aligned multi-crossing candidates. With no such
    calibration, an arbitrarily long single-crossing join stays unsupported.
    """
    if (bundle_margin is None or not np.isfinite(bundle_margin)
            or bundle_margin <= 0 or len(hits) != 1
            or min(alignment) < .9 or hits[0]['sine_angle'] < .65):
        return None
    fraction=float(hits[0]['gap_t'])
    if not .15 <= fraction <= .85:
        return None
    length=float(np.linalg.norm(np.asarray(b)-np.asarray(a)))
    margins=[fraction*length,(1-fraction)*length]
    if max(margins) > max(3*width,1.6*bundle_margin):
        return None
    return dict(kind='single_crossing_calibrated_from_bundle', crossing_count=1,
                margins=margins, calibration_margin=float(bundle_margin))
