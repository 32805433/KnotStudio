"""Conservative image-derived removal of regular diagram grids.

These transforms use pixel geometry and color only. No dataset names, source
SVGs, table identities, or reference labels enter the recognition pipeline.
"""
import cv2
import numpy as np
from scipy import ndimage


def _bands(values, threshold):
    indices=np.flatnonzero(values>threshold)
    if not len(indices):return []
    return np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)


def _regular(bands):
    if len(bands)<4:return False
    centers=np.array([np.mean(x) for x in bands]);spacings=np.diff(centers)
    return np.std(spacings)/max(1,np.mean(spacings))<.12


def prepare(rgb):
    rgb=np.asarray(rgb,dtype=np.uint8)[:,:,:3]
    border=np.concatenate([rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]])
    if (border.max(axis=1)<45).mean()>.9 and (rgb.max(axis=2)<45).mean()>.5:
        return 255-rgb,{'transform':'dark_background_inverted'}
    # A border-to-border frame distinguishes these backgrounds from parallel
    # strands of a rectangular knot drawing. Never erase such strands as a grid.
    dark_frame=rgb.min(axis=2)<175
    framed=all(float(side.mean())>.8 for side in (dark_frame[0],dark_frame[-1],dark_frame[:,0],dark_frame[:,-1]))
    if not framed:
        return rgb,{'transform':'none'}
    r,g,b=[rgb[:,:,i].astype(float) for i in range(3)]
    blue=(b>r+12)&(b>g+8)
    blue_rows=_bands(blue.mean(axis=1),.45)
    blue_cols=_bands(blue.mean(axis=0),.45)
    if _regular(blue_rows) and _regular(blue_cols):
        green=(g>r+15)&(g>b+15)&(r<200)
        if green.sum()>100:
            # Retain dark vertex dots only when adjoining the green strand.
            black=rgb.max(axis=2)<80
            labels,n=ndimage.label(black)
            distance=ndimage.distance_transform_edt(~green)
            keep=green.copy()
            for sl in ndimage.find_objects(labels):
                if sl is None:continue
                local=labels[sl];values=np.unique(local[local>0])
                for label in values:
                    mask=local==label
                    if mask.sum()>min(rgb.shape[:2])**2*.003:continue
                    if distance[sl][mask].min()<4:
                        keep[sl]|=mask
            out=np.full_like(rgb,255);out[keep]=0
            return out,{'transform':'colored_grid_removed','grid_rows':len(blue_rows),'grid_columns':len(blue_cols)}
    dark=rgb.min(axis=2)<175
    rows=_bands(dark.mean(axis=1),.8);cols=_bands(dark.mean(axis=0),.8)
    if _regular(rows) and _regular(cols):
        thickness=max(np.median([len(x) for x in rows]),np.median([len(x) for x in cols]))
        # Thin square grid lines and thicker knot strokes are separated by width.
        radius=max(1,int(np.ceil(thickness/2)))
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
        keep=cv2.morphologyEx(dark.astype(np.uint8),cv2.MORPH_OPEN,kernel).astype(bool)
        # Residual frame corners are disconnected small objects, not strands.
        labels,n=ndimage.label(keep)
        sizes=np.bincount(labels.ravel());keep&=sizes[labels]>max(35,radius*radius*12)
        margin=int(thickness)+2
        keep[:margin,:]=keep[-margin:,:]=False
        keep[:,:margin]=keep[:,-margin:]=False
        out=np.full_like(rgb,255);out[keep]=0
        return out,{'transform':'thin_grid_removed','grid_rows':len(rows),'grid_columns':len(cols),'opening_radius':radius}
    return rgb,{'transform':'none'}


def separate_narrow_contacts(rgb, branch_points, stroke_width):
    """Try one-pixel ink erosion only near unresolved skeleton junctions.

    This opens existing thin contacts; it never chooses an overpass at a solid
    X. Preserve every foreground component and reject deletion of thin arms.
    The caller must reconstruct a complete diagram and flag it for review.
    """
    if stroke_width < 4 or not branch_points:
        return None, {}
    radius=max(8,round(3*stroke_width))
    region=np.zeros(rgb.shape[:2],np.uint8)
    for x,y in branch_points:
        cv2.circle(region,(round(x),round(y)),radius,1,-1)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    thinned=cv2.dilate(rgb,kernel)  # Dilating white erodes dark ink.
    cleaned=np.where(region[:,:,None],thinned,rgb).astype(np.uint8)
    ink=rgb.min(axis=2)<150
    retained=cleaned.min(axis=2)<150
    labels,count=ndimage.label(ink,np.ones((3,3)))
    _,remaining_count=ndimage.label(retained,np.ones((3,3)))
    if remaining_count<=count or not retained.any():
        return None, {}
    survivors=set(np.unique(labels[retained]))
    if any(label not in survivors for label in range(1,count+1)):
        return None, {}
    removed=ink & ~retained
    distance=ndimage.distance_transform_edt(~retained)
    if not removed.any() or distance[removed].max()>2.1:
        return None, {}  # A thin arm disappeared, rather than just its border.
    return cleaned,{'radius_pixels':1,'region_radius':radius,
                    'points':[list(map(float,p)) for p in branch_points]}
