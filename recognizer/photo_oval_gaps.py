"""Source-supported continuation of one mostly visible oval in a photograph."""
import cv2
import numpy as np


def oval_gap_evidence(points, width, tangents, hits, starts, ends, owners, own_path,
                      *, rejection_reason='endpoint_direction', fixed_connectors=()):
    """Require a fitted major arc and the same crossing on its missing arc.

    This supplies direction evidence for a curved same-path gap; it cannot
    connect different visible paths or justify a different over-strand order.
    """
    if rejection_reason not in ('endpoint_direction', 'unsupported_single_crossing_length'):
        return None
    points = np.asarray(points, float)
    if len(points) < 8 or len(hits) != 1 or np.linalg.norm(points[-1]-points[0]) > 50*width:
        return None
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if s[-1] < 30*width:
        return None
    sample = np.column_stack([np.interp(np.linspace(0, s[-1], 65), s, points[:, i]) for i in (0, 1)])
    # Uniform arc-length samples avoid pixel-direction bias. Give the two
    # observed endpoints extra weight because they anchor the missing arc.
    fit = np.vstack([sample, np.repeat(sample[[0, -1]], 4, axis=0)]).astype(np.float32)
    try:
        center, axes, degrees = cv2.fitEllipse(fit)
    except (cv2.error, ValueError, FloatingPointError):
        return None
    if not np.isfinite(np.r_[center, axes, degrees]).all():
        return None
    axes = np.array(axes)/2
    if min(axes) < 3*width or max(axes) > 5*min(axes):
        return None
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    unit = (points-center) @ rotation / axes
    radius = np.linalg.norm(unit, axis=1)
    projection = (unit/np.maximum(radius[:, None], 1e-12)*axes) @ rotation.T + center
    errors = np.linalg.norm(projection-points, axis=1)
    if np.median(errors) > width or np.percentile(errors, 95) > 2*width or max(errors[0], errors[-1]) > 3*width:
        return None
    angles = np.unwrap(np.arctan2(unit[:, 1], unit[:, 0]))
    covered = float(angles[-1]-angles[0]); sign = np.sign(covered)
    changes = np.diff(angles)
    if not 1.25*np.pi <= abs(covered) <= 1.95*np.pi or np.sum(abs(changes[changes*sign < 0])) > .05*np.sum(abs(changes)):
        return None
    derivative = np.column_stack([-axes[0]*np.sin(angles[[0, -1]]), axes[1]*np.cos(angles[[0, -1]])]) @ rotation.T
    derivative *= np.array([-sign, sign])[:, None]
    derivative /= np.linalg.norm(derivative, axis=1)[:, None]
    if min(np.sum(np.array(tangents)*derivative, axis=1)) < .8:
        return None
    parameter = np.linspace(0., 1., 65)
    theta = angles[-1] + parameter*(angles[0]+sign*2*np.pi-angles[-1])
    curve = np.column_stack([axes[0]*np.cos(theta), axes[1]*np.sin(theta)]) @ rotation.T + center
    curve += (1-parameter[:, None])*(points[-1]-curve[0]) + parameter[:, None]*(points[0]-curve[-1])
    from .knotfolio_backend import _segments_cross
    if any(_segments_cross(a, b, c, d) for a, b in zip(curve, curve[1:]) for c, d in fixed_connectors):
        return None
    # Check all finite segment intersections, including sampled vertices.
    v, w = np.diff(curve, axis=0), ends-starts
    delta = starts[None, :, :]-curve[:-1, None, :]
    denominator = v[:, None, 0]*w[None, :, 1]-v[:, None, 1]*w[None, :, 0]
    stable = abs(denominator) > 1e-8; safe = np.where(stable, denominator, 1.)
    t = (delta[:, :, 0]*w[None, :, 1]-delta[:, :, 1]*w[None, :, 0])/safe
    u = (delta[:, :, 0]*v[:, None, 1]-delta[:, :, 1]*v[:, None, 0])/safe
    rows, columns = np.nonzero(stable & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1))
    crossings = []
    for row, column in zip(rows, columns):
        point = curve[row]+t[row, column]*v[row]
        if (owners[column] == own_path and
                min(np.linalg.norm(point-points[0]), np.linalg.norm(point-points[-1])) < 1e-4):
            continue
        sine = abs(denominator[row, column])/max(np.linalg.norm(v[row])*np.linalg.norm(w[column]), 1e-12)
        if sine < .3:
            return None
        if not any(np.linalg.norm(point-old[0]) < 1.5 for old in crossings):
            crossings.append((point, int(owners[column])))
    if len(crossings) != 1 or crossings[0][1] != hits[0]['path']:
        return None
    return dict(kind='source_oval_continuation', covered_degrees=float(abs(covered)*180/np.pi),
                fit_error_95=float(np.percentile(errors, 95)), crossing_path=hits[0]['path'])
