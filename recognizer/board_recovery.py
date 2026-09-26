"""Reviewed, failure-only reconstruction of rough photographed chalk.

Pores and an overlapping pen seam are geometric hypotheses. Two distinct
contrast masks must agree, every source component must remain supported, and
the reconstructed embedding must be valid. No labels are silently deleted.
"""
import hashlib

import numpy as np
from scipy import ndimage

from .board_photo import prepared_board_contrast, looks_photographic, recognize_board
from .board_strokes import fill_chalk_pores
from .recognition_runtime import checkpoint
from .style_recovery import diagram_supported_by_ink


def recognize_board_strokes(rgb, *, mode='auto'):
    if mode == 'off' or (mode == 'auto' and not looks_photographic(rgb)[0]):
        return None
    contrast = prepared_board_contrast(rgb, 'board' if mode == 'auto' else mode)
    if contrast.polarity != 'dark':
        return None
    masks = {}

    class PoreContrast:
        scale = contrast.scale
        polarity = contrast.polarity

        def candidates(self):
            seen = set()
            for pixels, preparation in contrast.candidates():
                checkpoint('repairing chalk texture')
                original = pixels.min(2) < 205
                filled, pores = fill_chalk_pores(original)
                digest = hashlib.sha256(np.packbits(filled).tobytes()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                cleaned = pixels.copy()
                if pores:
                    added = filled & ~original
                    indices = ndimage.distance_transform_edt(~original, return_distances=False,
                                                            return_indices=True)
                    cleaned[added] = pixels[indices[0][added], indices[1][added]]
                key = tuple(preparation.get(k) for k in ('method', 'strong', 'weak'))
                masks[key] = (original, pores)
                yield cleaned, {**preparation, 'chalk_pores_filled': len(pores),
                                'foreground_fraction': float(filled.mean())}

    result = recognize_board(rgb, mode, options={'same_path_seams': True},
                             _contrast=PoreContrast(), _minimum_agreement=2,
                             _require_embedding=True)
    if result['diagram'] is None:
        return None
    preparation = result['diagnostics']['preprocessing']
    source, pores = masks[tuple(preparation.get(k) for k in ('method', 'strong', 'weak'))]
    scale = contrast.scale
    local_diagram = {'edges': [{'points': (np.asarray(e['points'])*scale).tolist()}
                               for e in result['diagram']['edges']]}
    if not diagram_supported_by_ink(local_diagram, source,
                                   result['diagnostics']['stroke_width']*scale):
        return None
    reviews = []
    for pore in pores:
        review = dict(pore)
        for key in ('center', 'bbox', 'continuation_radii'):
            review[key] = [v/scale for v in pore[key]]
        for key in ('inradius', 'stroke_width'):
            review[key] /= scale
        reviews.append(review)
    result['diagnostics']['board_recovery'] = dict(
        method='chalk_pores_and_seams', minimum_agreement=2,
        distinct_masks=len(masks), pores=reviews,
        same_path_seam=bool(result['diagnostics'].get('same_path_seam_repair')),
        semantic_label_removal=False)
    message = ('Chalk texture or overlapping pen tips were repaired using agreeing '
               'contrast masks. Inspect the small loops, seams, and crossings.')
    result['warnings'].insert(0, message)
    result['diagram']['recognition']['warnings'] = result['warnings']
    return result
