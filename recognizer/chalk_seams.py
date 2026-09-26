"""Visible-ink evidence for the transverse join of overlapping chalk tips.

This gate does not identify a seam on its own. The caller must first establish
oppositely directed, overlapping ends, then check the entire trimmed edit for
other visible strands and previously inferred crossings. It is intended only
for reviewed photo recovery with agreement between distinct contrast masks.
"""

import numpy as np
from scipy import ndimage


def chalk_overlap_evidence(mask, first, second, stroke_width, *, center_widths=2., blank_widths=1.):
    """Return support for a proposed trimmed-tip bridge, or None.

    Centerline separation alone overestimates the missing ink: each chalk band
    already occupies part of the connector. The default allows two stroke
    widths between centers and one width of unsupported background. The
    pre-matching photo check can use the ordinary short-gap scale (three and
    two widths), with isolation and distinct-mask agreement checked by its
    callers. Both centers must lie on source ink; an intermediate ink run is
    an obstacle, never evidence supporting the seam.
    """
    mask = np.asarray(mask, dtype=bool)
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    width = float(stroke_width)
    if (mask.ndim != 2 or first.shape != (2,) or second.shape != (2,)
            or not np.all(np.isfinite(np.r_[first, second, width, center_widths, blank_widths]))
            or width <= 0 or not 0 < center_widths <= 3 or not 0 <= blank_widths <= 2):
        return None
    length = float(np.linalg.norm(second-first))
    if not 0 < length <= center_widths*width:
        return None
    points = np.linspace(first, second, max(2, 1+int(np.ceil(4*length))))
    ink = ndimage.map_coordinates(
        mask.astype(np.float32), [points[:, 1], points[:, 0]],
        order=1, mode='constant', cval=0.) >= .5
    if not ink[0] or not ink[-1]:
        return None
    boundaries = np.flatnonzero(ink[1:] != ink[:-1])+1
    # Starting and finishing in ink gives exactly two transitions for a gap.
    # Four or more would cross another piece of visible ink.
    if len(boundaries) > 2:
        return None
    spacing = length/(len(points)-1)
    unsupported = (float(boundaries[1]-boundaries[0])*spacing
                   if len(boundaries) == 2 else 0.)
    supported_fraction = float(ink.mean())
    # Local chalk thickness can vary sharply even along one band. The actual
    # blank span is the relevant bound; a fractional occupancy cutoff would
    # reject thinner tips despite the same short, fully isolated gap.
    if unsupported > blank_widths*width:
        return None
    return dict(bridge_ends=[first.tolist(), second.tolist()],
                length=length, unsupported_length=unsupported,
                supported_fraction=supported_fraction, stroke_width=width)
