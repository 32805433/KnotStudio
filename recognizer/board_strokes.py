"""Conservative cleanup of enclosed pores in photographed chalk bands.

Chalk texture can leave compact background islands inside a single broad
stroke. Their skeletons become little cycles, which the tracer correctly
reports as branches. This repair needs both small scale and two substantial
continuations; small standalone circles and open gaps are not pores.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def fill_chalk_pores(mask):
    """Return a foreground copy and source-coordinate review records.

Only enclosed, compact background regions are added. Existing foreground is
never removed, and an island that would join distinct ink components is left
alone. Callers should treat this as a reviewed photo hypothesis and require
agreement between independent contrast masks before accepting reconstruction.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 3:
        raise ValueError('Expected a two-dimensional foreground mask.')
    result = mask.copy()
    holes = ndimage.binary_fill_holes(mask) & ~mask
    if not np.any(holes):
        return result, []
    labels, count = ndimage.label(holes)
    # Dense photographic noise is not reliable local stroke evidence.
    if count > 2048:
        return result, []
    radii = ndimage.distance_transform_edt(mask)[skeletonize(mask)]
    width = max(1.5, float(np.median(radii) * 2)) if len(radii) else 1.5
    ink_labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    candidates, provisional = [], mask.copy()
    provisional_owners = ink_labels.copy()
    for label, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        local = labels[region] == label
        area = int(local.sum())
        if max(local.shape) > 1.25 * width or area > .5 * width * width:
            continue
        radius = float(ndimage.distance_transform_edt(np.pad(local, 1)).max())
        if radius > .3 * width:
            continue
        expanded = tuple(slice(max(0, r.start-1), min(size, r.stop+1))
                         for r, size in zip(region, mask.shape))
        local_hole = labels[expanded] == label
        neighbors = ndimage.binary_dilation(local_hole, structure=np.ones((3, 3)))
        owners = np.unique(ink_labels[expanded][neighbors & mask[expanded]])
        if len(owners) != 1:
            continue
        owner = int(owners[0])
        candidates.append((region, local, radius, area, owner))
        provisional[region] |= local
        provisional_owners[region][local] = owner
    if not candidates:
        return result, []

    # Provisional fills prevent adjacent pores in the same continuing band
    # from hiding each other's support. No large knot/loop interior enters it.
    distance = ndimage.distance_transform_edt(provisional)
    reach = int(np.ceil(3 * width))
    evidence = []
    for region, local, radius, area, owner in candidates:
        yy, xx = np.nonzero(local)
        center = np.array([region[1].start + xx.mean(), region[0].start + yy.mean()])
        x, y = np.rint(center).astype(int)
        x0, x1 = max(0, x-reach), min(mask.shape[1], x+reach+1)
        y0, y1 = max(0, y-reach), min(mask.shape[0], y+reach+1)
        py, px = np.nonzero(provisional_owners[y0:y1, x0:x1] == owner)
        points = np.column_stack((px+x0, py+y0))
        points = points[np.linalg.norm(points-center, axis=1) <= 3*width]
        if len(points) < 10:
            continue
        middle = points.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov(points.T))
        axis, normal = vectors[:, -1], vectors[:, 0]
        offset = float((center-middle) @ normal)
        # A round ring or a junction lacks the elongated neighborhood of a
        # continuing chalk band. Permit pores near either edge of that band.
        if values[-1] < 5*values[0] or abs(offset) > width:
            continue
        middle = center-offset*normal
        samples = middle + np.array([-2.5, -1.5, 1.5, 2.5])[:, None]*width*axis
        samples = samples[:, None, :] + np.linspace(-.25, .25, 5)[None, :, None]*width*normal
        support = ndimage.map_coordinates(
            distance, [samples[..., 1].ravel(), samples[..., 0].ravel()],
            order=1, mode='constant').reshape(4, 5)
        support_owners = ndimage.map_coordinates(
            provisional_owners, [samples[..., 1].ravel(), samples[..., 0].ravel()],
            order=0, mode='constant').reshape(4, 5)
        support[support_owners != owner] = 0
        support = support.max(axis=1)
        if np.any(support < .2*width):
            continue
        result[region] |= local
        evidence.append(dict(
            bbox=[region[1].start, region[0].start, region[1].stop, region[0].stop],
            center=center.tolist(), filled_pixels=area, inradius=radius,
            stroke_width=width, band_axis=axis.tolist(), continuation_radii=support.tolist()))
    return result, evidence
