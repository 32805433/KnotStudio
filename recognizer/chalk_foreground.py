"""Local stroke evidence for chalk, shared by tracing and label review.

Broad eraser smears and illumination changes are not chalk strokes.  Measure
contrast at the observed stroke scale as well as at the background scale;
never fill gaps or infer a strand from the rejected pixels.
"""
import cv2
import numpy as np


class ChalkEvidence:
    def __init__(self, smooth, contrast, radius):
        peak = float(np.percentile(contrast, 99.8))
        core = (contrast > max(25., peak * .55)).astype(np.uint8)
        distance = cv2.distanceTransform(core, cv2.DIST_L2, 5)
        maxima = (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8))) & (distance > 1.)
        widths = distance[maxima]
        self.width = float(np.clip(2 * np.median(widths) if widths.size else 3.,
                                   2., max(3., radius)))
        background = cv2.GaussianBlur(smooth, (0, 0), max(1.2, self.width))
        ridge = (smooth-background).max(axis=2)
        # Noise is estimated away from the visible chalk. A floor handles
        # compression artifacts on otherwise smooth photographs.
        residual = smooth-cv2.GaussianBlur(smooth, (0, 0), 1.)
        quiet = contrast < max(12., peak*.15)
        noise = float(np.median(np.abs(residual[quiet]))) if quiet.any() else 1.
        support = (ridge > max(5., 4.*noise)) & (ridge > .18*contrast)
        # Evidence belongs to a stroke's core. Retain its original shoulders
        # and tips too: trimming a single tip pixel can change an arrow or a
        # narrow crossing. This grows eligibility, never the foreground mask.
        pad = max(1, round(self.width*.5))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*pad+1,)*2)
        self.support = cv2.dilate(support.astype(np.uint8), kernel).astype(bool)

    def filter(self, mask, seed):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        cores = np.bincount(labels[seed & mask], minlength=n)
        sharp = np.bincount(labels[self.support & mask], minlength=n)
        area = stats[:, cv2.CC_STAT_AREA]
        # A handful of bright flecks must not validate a large dusty island.
        # Keep the original pixels of supported components intact. Cutting
        # their low-contrast edges can turn rough chalk into false endpoints
        # or alter the three-arm contacts used to reconstruct crossings.
        allowed = ((sharp >= .65*area)
                   & ((cores >= np.maximum(3., .08*area)) | (sharp >= .90*area)))
        allowed[0] = False
        return allowed[labels]


def photographic_chalk(rgb, diagnostics=None):
    """Return local chalk masks in original coordinates, or None for other media.

    Detection is bounded at the same resolution used by board tracing. The
    original RGB array is never altered; labels still own image-coordinate
    pixels and require the user's explicit deletion.
    """
    from .board_photo import BoardContrast, looks_photographic
    if not looks_photographic(rgb)[0]:
        return None
    contrast = BoardContrast(rgb)
    if contrast.polarity != 'dark':
        return None
    peak = float(np.percentile(contrast.mean, 99.8))
    weak, strong = max(12., peak*.275), max(25., peak*.55)
    mask = contrast.chalk.filter(contrast.mean > weak, contrast.mean > strong)
    # Reject isolated sub-stroke flecks, not small closed components. Unlike
    # tracing, label review must retain little minus signs and dot accents.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    allowed = stats[:, cv2.CC_STAT_AREA] >= max(3., .35*contrast.chalk.width**2)
    allowed[0] = False
    mask = allowed[labels]
    fringe = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    fringe &= contrast.mean > max(5., weak*.4)
    shape = (rgb.shape[1], rgb.shape[0])
    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), shape, interpolation=cv2.INTER_NEAREST).astype(bool)
        fringe = cv2.resize(fringe.astype(np.uint8), shape, interpolation=cv2.INTER_NEAREST).astype(bool)
    if diagnostics is not None:
        diagnostics.update(foreground_method='local_chalk', foreground_scale=contrast.scale,
                           chalk_stroke_width=contrast.chalk.width/contrast.scale)
    return mask, mask | fringe, True
