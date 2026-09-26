"""Opt-in experimental ink classifier for the classical photo recognizer.

A small forest predicts ink cores; local contrast determines boundaries.
Text and arrows are foreground training examples, not semantic deletions.
Two confidence thresholds must reconstruct the same embedded projection.
Sparse pixel validation does not establish full-diagram accuracy.
"""
from pathlib import Path
from functools import lru_cache
import json
import cv2
import numpy as np

FEATURE_NAMES = ['morph', 'local_mean', 'red_residual', 'green_residual',
    'blue_residual', 'residual_chroma', 'gray', 'background_gray',
    'contrast_relative', 'morph_relative', 'local_2', 'local_4', 'local_8',
    'local_16', 'ridge_1', 'ridge_2', 'ridge_4', 'gradient',
    'mean_2', 'mean_4', 'mean_8', 'color_chroma']


def feature_maps(contrast):
    """Return HxWxF float32 features from a BoardContrast object."""
    rgb=contrast.rgb.astype(np.float32)
    smooth=contrast.smooth
    sign=-1. if contrast.polarity=='light' else 1.
    gray=cv2.cvtColor(smooth,cv2.COLOR_RGB2GRAY)
    background=cv2.GaussianBlur(gray,(0,0),contrast.radius)
    peak=max(float(np.percentile(contrast.mean,99.8)),15.)
    mpeak=max(float(np.percentile(contrast.morph,99.8)),15.)
    f=[contrast.morph/255.,contrast.mean/255.]
    f += [contrast.color_residual[:,:,c]/255. for c in range(3)]
    f += [np.ptp(contrast.color_residual,axis=2)/255.,gray/255.,background/255.,
          np.clip(contrast.mean/peak,-2,3),np.clip(contrast.morph/mpeak,0,3)]
    for sigma in (2,4,8,16):
        f.append(sign*(gray-cv2.GaussianBlur(gray,(0,0),sigma))/255.)
    for sigma in (1,2,4):
        g=cv2.GaussianBlur(gray,(0,0),sigma)
        xx=cv2.Sobel(g,cv2.CV_32F,2,0,ksize=3)/4
        yy=cv2.Sobel(g,cv2.CV_32F,0,2,ksize=3)/4
        xy=cv2.Sobel(g,cv2.CV_32F,1,1,ksize=3)/4
        # Signed second derivative perpendicular to a thin ink ridge.
        disc=np.sqrt((xx-yy)**2+4*xy**2)
        eig=(-sign*(xx+yy)+disc)/2
        f.append(eig*sigma*sigma/255.)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8
    f.append(np.hypot(gx,gy)/255.)
    for sigma in (2,4,8):
        f.append(cv2.GaussianBlur(contrast.mean,(0,0),sigma)/255.)
    f.append(np.ptp(rgb,axis=2)/255.)
    return np.stack(f,axis=-1).astype(np.float32)


@lru_cache(maxsize=2)
def load_model(path=None):
    path=Path(path) if path else Path(__file__).with_name('models')/'ink_forest_v1.json'
    model=json.loads(path.read_text())
    if model['features']!=FEATURE_NAMES:
        raise ValueError('Ink feature model/version mismatch.')
    return model


def predict_features(features, model=None):
    model=model or load_model()
    features=np.asarray(features,dtype=np.float32)
    shape=features.shape[:-1]
    flat=features.reshape(-1,len(FEATURE_NAMES))
    out=np.zeros(len(flat),np.float32)
    for tree in model['trees']:
        feature=np.asarray(tree['feature'],np.int32)
        # Preserve the fitted double-precision split: rounding a threshold to
        # float32 can move a training observation across an exact midpoint.
        threshold=np.asarray(tree['threshold'],np.float64)
        left=np.asarray(tree['left'],np.int32)
        right=np.asarray(tree['right'],np.int32)
        values=np.asarray(tree['positive_probability'],np.float32)
        node=np.zeros(len(flat),np.int32)
        for _ in range(model['max_depth']+1):
            active=feature[node]>=0
            if not active.any():break
            i=np.flatnonzero(active)
            n=node[i]
            node[i]=np.where(flat[i,feature[n]]<=threshold[n],left[n],right[n])
        out+=values[node]
    return (out/len(model['trees'])).reshape(shape)


def learned_candidates(contrast, model=None):
    """Yield two learned-seed/local-contrast masks for topology comparison.

    No semantic text or arrow classifier runs. Tiny or unsupported contrast
    components can be discarded as segmentation noise and are disclosed.
    Candidate thresholds were selected on development only.
    """
    probability=predict_features(feature_maps(contrast),model)
    base=contrast.rgb if contrast.polarity=='light' else 255-contrast.rgb
    ink=base.astype(float)-base.min(2,keepdims=True)
    span=ink.max(2,keepdims=True)
    ink=np.where(span>35,ink/np.maximum(span,1)*160,0).astype(np.uint8)
    peak=max(float(np.percentile(contrast.morph,99.8)),20.)
    # Sparse core annotations do not teach antialiased boundaries. A local
    # contrast envelope therefore defines weak pixel support, while the
    # classifier supplies high-confidence seeds. This avoids the broad halo
    # produced by thresholding core probabilities as if they were boundaries.
    weak=(contrast.morph>max(7.,peak*.12)) & (contrast.mean>3.)
    count,labels,stats,_=cv2.connectedComponentsWithStats(weak.astype(np.uint8),8)
    for threshold in (.60,.80):
        seed=weak & (probability>=threshold) & (contrast.morph>max(12.,peak*.2))
        cores=np.bincount(labels[seed],minlength=count)
        good=(cores>=2) & (stats[:,4]>=8)
        good[0]=False
        keep=good[labels]
        out=np.full_like(contrast.rgb,255)
        out[keep]=ink[keep]
        yield out,dict(method='learned_ink_forest',threshold=threshold,
            foreground_fraction=float(keep.mean()),discarded_regions=[stats[i,:4].tolist() for i in range(1,count)
                if not good[i] and stats[i,4]>=8],
            model_version='ink-core-development-v1',experimental=True)


def recognize_learned_ink(rgb, mode='auto'):
    """Run the opt-in learner and classical graph reconstruction.

    Returns the ordinary recognition result (including failures). Callers
    should keep their earlier result when this returns no diagram. This
    helper does not infer orientation; the public pipeline does that once
    against the original source.
    """
    from .board_photo import BoardContrast, recognize_board

    class LearnedContrast(BoardContrast):
        def candidates(self):
            yield from learned_candidates(self)

    contrast=LearnedContrast(rgb, 'board' if mode=='auto' else mode)
    result=recognize_board(rgb,mode,_contrast=contrast,
        _minimum_agreement=2,_require_embedding=True)
    result['diagnostics']['learned_ink']={
        'model':'ink-core-development-v1', 'experimental':True,
        'thresholds':[.60,.80], 'minimum_agreement':2,
        'semantic_label_removal':False,
        'training_annotation_sha256':load_model()['annotation_sha256']}
    if result['diagram'] is not None:
        result['status']='needs_review'
        warning=('An experimental ink model supplied the photo masks; '
            'inspect faint strokes, inferred crossings and discarded small marks.')
        result['warnings'].insert(0,warning)
        result['diagram']['recognition']['warnings']=result['warnings']
    return result
