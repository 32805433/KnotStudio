"""Display and export a recognized diagram without changing its combinatorics.

This module deliberately has no Tk dependency, so CLI/batch exports also work on
machines without a display. Coordinates remain in the original image frame.
"""
from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .twist_boxes import (box_corners, box_port_points, is_twist_box,
                          transition_port, twist_label)


COLORS = ("#2563eb", "#dc3545", "#16866a", "#9254c8", "#dd861a", "#00879c", "#c44392", "#546577")
DEFAULT_ORIENTATION_COLOR = "#8a929e"
_RASTER_DISPLAY_KEY = "_knot_studio_display"


def _display_width(display):
    if not isinstance(display, dict):
        return None
    width = display.get("stroke_width")
    if isinstance(width, (int, float)) and not isinstance(width, bool) and math.isfinite(width) and width > 0:
        return float(width)
    return None


def inherit_display_style(diagram, source):
    """Keep appearance when an edit rebuilds the graph in the same coordinates."""
    width = _display_width(source.get("display")) if source else None
    if diagram is not None and width is not None:
        diagram["display"] = {"stroke_width": width}


def apply_rendered_appearance(diagram, image):
    """Restore only stroke width after independently recognizing rendered pixels.

    This in-memory hint belongs to the editable image, survives pencil/eraser
    copies, and contains no topology. A resized image invalidates it. External
    images use the ordinary display default; PNG export does not serialize it.
    """
    display = image.info.get(_RASTER_DISPLAY_KEY)
    width = _display_width(display)
    if (diagram is not None and width is not None
            and display.get("size") == image.size
            and (diagram.get("width"), diagram.get("height")) == image.size):
        diagram["display"] = {"stroke_width": width}


def edge_color(edge, colored=True):
    return COLORS[int(edge.get("component", 0)) % len(COLORS)] if colored else "#171717"


def line_width(diagram):
    explicit = _display_width(diagram.get("display"))
    if explicit is not None:
        return explicit
    return max(2.0, min(7.0, math.sqrt(diagram.get("width", 800) * diagram.get("height", 600)) / 220))


def _crossings(diagram):
    return {c["id"]: c for c in diagram.get("crossings", [])}


def edge_points(edge, crossing_map):
    """Connect skeleton endpoints to their explicit crossing vertices."""
    points = [tuple(p) for p in edge.get("points", [])]
    if not points:
        return []
    start, end = edge.get("start"), edge.get("end")
    if start is not None and start[0] in crossing_map:
        vertex = crossing_map[start[0]]
        p = tuple(box_port_points(vertex)[start[1]] if is_twist_box(vertex) else vertex["point"])
        if math.dist(p, points[0]) > 0.01:
            points.insert(0, p)
    if end is not None and end[0] in crossing_map:
        vertex = crossing_map[end[0]]
        p = tuple(box_port_points(vertex)[end[1]] if is_twist_box(vertex) else vertex["point"])
        if math.dist(p, points[-1]) > 0.01:
            points.append(p)
    if edge.get("closed") and points[0] != points[-1]:
        points.append(points[0])
    return points


def _prefix(points, length):
    if not points:
        return []
    result = [points[0]]
    remaining = length
    for a, b in zip(points, points[1:]):
        d = math.dist(a, b)
        if d > remaining:
            if d:
                result.append((a[0] + (b[0] - a[0]) * remaining / d, a[1] + (b[1] - a[1]) * remaining / d))
            break
        result.append(b)
        remaining -= d
    return result


def paths(diagram, colored=True, stroke_width=None, *, _fit_attempt=0, with_ids=False):
    """Return visible strokes, with gaps cut only into the underpassing arcs.

    White overpass masks can both miss an acute-angle contact and erase a
    neighboring strand. Trimming the explicit under-arcs avoids both problems.
    """
    width = stroke_width or line_width(diagram)
    crossing_map = _crossings(diagram)
    edges = {e["id"]: e for e in diagram.get("edges", [])}
    geometry={}
    for eid,edge in edges.items():
        points=edge_points(edge,crossing_map)
        if len(points)>4:
            # Display-only approximation below a small fraction of one stroke
            # width; the editable curve and PD retain all original vertices.
            original = points
            first,last=points[0],points[-1]
            points=cv2.approxPolyDP(np.asarray(points,np.float32),min(.1,width/32),False).reshape(-1,2).tolist()
            points[0]=first
            if math.dist(points[-1],last)>1e-5:points.append(last)
            else:points[-1]=last
            if first == last and len(points) < 4:
                # Even a subpixel isolated circle must survive as geometry so
                # editable-raster clearance can enlarge it or report a limit.
                points = original
        geometry[eid]=points
    trims={eid:[0.,0.] for eid in edges}
    for crossing in crossing_map.values():
        if is_twist_box(crossing):
            continue
        over = int(crossing.get("over", 1)) % 2
        ports = crossing.get("ports", [])
        if len(ports) != 4:
            continue
        over_paths=[]
        for index in (over, over + 2):
            port = ports[index]
            points=geometry[port['edge']][::1 if port['end']==0 else -1]
            over_paths.append(_prefix(points,width*18))
        over_points=np.asarray(over_paths[0][::-1]+over_paths[1][1:],float)
        for index in (1-over,3-over):
            port=ports[index]
            points=geometry[port['edge']][::1 if port['end']==0 else -1]
            trims[port['edge']][port['end']]=_under_trim(points,over_points,width)
    result=[]
    for eid,edge in edges.items():
        points=geometry[eid]
        start,end=trims[eid]
        if start or end:
            length=sum(math.dist(a,b) for a,b in zip(points,points[1:]))
            if start+end>length*.7 and _fit_attempt<6 and length>1e-8:
                # Preserve a visible arc between nearby crossings. Drawing at
                # the old width would consume the whole arc in its two gaps.
                fitted=width*min(.8,.6*length/(start+end))
                return paths(diagram,colored,stroke_width=fitted,_fit_attempt=_fit_attempt+1,
                             with_ids=with_ids)
        if start or end:
            points=_trim_ends(points,start,end)
        if len(points)>1:
            stroke = (points,edge_color(edge,colored),width)
            result.append(stroke+(eid,) if with_ids else stroke)
    return result


def _under_trim(points,over_points,width):
    """Length of an underpass gap giving white clearance at any crossing angle."""
    local=np.asarray(_prefix(points,width*18),float)
    if len(local)<2 or len(over_points)<2:return 0.
    a,b=over_points[:-1],over_points[1:]
    v=b-a;length2=np.maximum(np.sum(v*v,axis=1),1e-20)
    # Work on local segments; caps on each stroke consume one full stroke width
    # of centerline separation, and the remainder is explicit white clearance.
    target=1.8*width
    target2 = target*target
    segments = [(float(p[0]), float(p[1]), float(d[0]), float(d[1]), float(r2))
                for p, d, r2 in zip(a, v, length2)]

    def near(x, y):
        for ax, ay, vx, vy, denominator in segments:
            qx, qy = x-ax, y-ay
            t = max(0., min(1., (qx*vx+qy*vy)/denominator))
            ex, ey = qx-t*vx, qy-t*vy
            if ex*ex+ey*ey < target2:
                return True
        return False

    # Only the first point outside the overpass's clearance capsules matters.
    # Avoid constructing a distance matrix for the rest of the long local arc.
    i = next((j for j, p in enumerate(local) if not near(float(p[0]), float(p[1]))), None)
    cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.diff(local,axis=0),axis=1))]
    if i is None:return float(cumulative[-1])
    if i==0:return 0.
    left,right=0.,1.
    length=float(cumulative[i]-cumulative[i-1])
    iterations=max(1,int(math.ceil(math.log2(max(1.,length/.08)))))
    # Bisection evaluates one point at a time. Tiny NumPy broadcasts here
    # dominated interactive crossing-gap construction on dense diagrams.
    # Use the same capped projection and squared-distance comparison with
    # scalar floats, stopping as soon as any overpassing segment is too close.
    px, py = map(float, local[i-1])
    dx, dy = map(float, local[i]-local[i-1])
    for _ in range(iterations):
        mid=(left+right)/2
        x, y = px+mid*dx, py+mid*dy
        if near(x, y):left=mid
        else:right=mid
    return float(cumulative[i-1]+right*(cumulative[i]-cumulative[i-1]))


def _trim_ends(points,start,end):
    points=np.asarray(points,float)
    cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    length=cumulative[-1]
    if start+end>=length:return []
    def at(s):
        i=min(len(points)-2,max(0,int(np.searchsorted(cumulative,s,side='right')-1)))
        t=(s-cumulative[i])/max(cumulative[i+1]-cumulative[i],1e-20)
        return (points[i]+t*(points[i+1]-points[i])).tolist()
    middle=points[(cumulative>start)&(cumulative<length-end)].tolist()
    return [at(start),*middle,at(length-end)]


def arrows(diagram, colored=True, *, with_ids=False):
    """One direction arrow per component, traversed through opposite ports."""
    edge_map = {e["id"]: e for e in diagram.get("edges", [])}
    crossing_map = _crossings(diagram)
    groups = {}
    for edge in edge_map.values():
        groups.setdefault(edge.get("component", 0), []).append(edge)
    result = []
    orientations = diagram.get("component_orientations", {})
    for component, group in groups.items():
        seed = group[0]  # Match the model's component_walks insertion order.
        sign = orientations.get(str(component), orientations.get(component, 1))
        direction = -1 if sign == -1 else 1
        directions = {}
        edge = seed
        while edge["id"] not in directions:
            directions[edge["id"]] = direction
            endpoint = edge.get("end" if direction == 1 else "start")
            if endpoint is None or endpoint[0] not in crossing_map:
                break
            vertex = crossing_map[endpoint[0]]
            port = vertex["ports"][transition_port(vertex, endpoint[1])]
            edge = edge_map.get(port["edge"])
            if edge is None:
                break
            direction = 1 if port["end"] == 0 else -1
        chosen = max(group, key=lambda e: sum(math.dist(a, b) for a, b in zip(e.get("points", []), e.get("points", [])[1:])))
        points = chosen.get("points", [])
        if len(points) < 2:
            continue
        lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
        half = sum(lengths) / 2
        for i, length in enumerate(lengths):
            if half <= length and length:
                a, b = points[i], points[i + 1]
                point = (a[0] + (b[0] - a[0]) * half / length, a[1] + (b[1] - a[1]) * half / length)
                sign = directions.get(chosen["id"], 1)
                tangent = ((b[0] - a[0]) / length * sign, (b[1] - a[1]) / length * sign)
                # Source recognition may leave some directions unspecified.
                # Keep the chosen/default diagram orientation available for PD
                # and editing, but do not present it as a detected source arrow.
                unknown = ("orientation" in diagram.get("recognition", {})
                           and str(component) not in orientations and component not in orientations)
                color = DEFAULT_ORIENTATION_COLOR if unknown else edge_color(chosen, colored)
                arrow = (point, tangent, color)
                result.append(arrow+(chosen['id'],) if with_ids else arrow)
                break
            half -= length
    return result


def arrow_polygon(point, tangent, size):
    x, y = point
    dx, dy = tangent
    return [(x + dx * size, y + dy * size), (x - dx * size + dy * size * 0.65, y - dy * size - dx * size * 0.65), (x - dx * size - dy * size * 0.65, y - dy * size + dx * size * 0.65)]


def box_shapes(diagram):
    """Compact display records; rendering never expands the stored twists."""
    result = []
    for box in diagram.get('crossings', []):
        if not is_twist_box(box):
            continue
        corners = box_corners(box)
        label = twist_label(box)
        # The bundle enters the two axial ends, irrespective of any additional
        # over/under passages. Text's downward direction points to the lower
        # axial end in screen coordinates. For an exactly horizontal bundle,
        # both ends have equal height; use the right end consistently.
        ux, uy = box['axis']
        if uy < 0 or (uy == 0 and ux < 0):
            ux, uy = -ux, -uy
        angle = math.degrees(math.atan2(-ux, uy))  # clockwise on a y-down canvas
        transverse, axial = box['size']
        font_size = max(1., min(axial*.53,
                              transverse*.72/max(1., len(label)*.60)))
        result.append({'id': box['id'], 'corners': corners, 'point': box['point'],
                       'label': label, 'label_angle': angle,
                       'font_size': font_size, 'stroke_width': line_width(diagram)})
    return result


def _passage_distance(points, *, point=None, parameter=None):
    """Arc distance for a stored passage point or normalized segment parameter."""
    points = np.asarray(points, float)
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    if parameter is not None:
        coordinate = float(parameter)*(len(points)-1)
        index = min(len(points)-2, max(0, int(coordinate)))
        fraction = float(np.clip(coordinate-index, 0., 1.))
    else:
        fractions = np.clip(np.sum((np.asarray(point)-points[:-1])*vectors, axis=1)/
                            np.maximum(lengths*lengths, 1e-20), 0., 1.)
        locations = points[:-1]+fractions[:, None]*vectors
        index = int(np.argmin(np.linalg.norm(locations-np.asarray(point), axis=1)))
        fraction = float(fractions[index])
    return float(cumulative[index]+fraction*lengths[index]), float(cumulative[-1])


def _passage_gap(under, under_distance, over, over_distance, width):
    """Trim one under-branch, accounting for the existing three-width halos."""
    before = _prefix(over, over_distance)
    after = _trim_ends(over, over_distance, 0.)
    local_over = _prefix(before[::-1], width*22.5)[::-1]+_prefix(after, width*22.5)[1:]
    under_before = _prefix(under, under_distance)[::-1]
    under_after = _trim_ends(under, under_distance, 0.)
    # A later under-piece's white halo has radius 1.5 widths. Its endpoint must
    # stay more than two widths from the over-strand centerline, or painter order
    # would erase the top strand again. The same gap works for PIL, SVG and Tk.
    clearance_width = width*1.25
    return (under_distance-_under_trim(under_before, np.asarray(local_over), clearance_width),
            under_distance+_under_trim(under_after, np.asarray(local_over), clearance_width))


def box_passages(diagram, colored=True, *, with_ids=False):
    """Visible passage pieces, trimmed at their stored mutual/self crossings.

    The opaque box already hides below-box passages. For passages above it,
    explicit crossing records determine the gap; list order cannot change which
    branch passes over. The return API is shared by Tk, PIL and SVG consumers.
    No implicit bundle twists are expanded and no graph geometry is changed.
    """
    edges = {e['id']: e for e in diagram.get('edges', [])}
    width = line_width(diagram)
    result = []
    for box in diagram.get('crossings', []):
        if not is_twist_box(box):
            continue
        ports = box_port_points(box)
        passages = box.get('passages', [])
        geometry = {i: passage.get('points') or [ports[passage['ports'][0]], ports[passage['ports'][1]]]
                    for i, passage in enumerate(passages) if passage.get('over')}
        gaps = {i: [] for i in geometry}
        for record in box.get('passage_crossings', []):
            a, b = record['pair']
            if a not in geometry or b not in geometry:
                continue
            if a == b:
                first, second = record['parameters']
                over_parameter = record['over_parameter']
                under_parameter = second if abs(over_parameter-first) < abs(over_parameter-second) else first
                under_distance, _ = _passage_distance(geometry[a], parameter=under_parameter)
                over_distance, _ = _passage_distance(geometry[a], parameter=over_parameter)
                under, over = a, a
            else:
                over = record['over']
                under = b if over == a else a
                under_distance, _ = _passage_distance(geometry[under], point=record['point'])
                over_distance, _ = _passage_distance(geometry[over], point=record['point'])
            gaps[under].append(_passage_gap(geometry[under], under_distance,
                                          geometry[over], over_distance, width))
        for i, points in geometry.items():
            a, _ = passages[i]['ports']
            edge = edges.get(box['ports'][a]['edge'], {})
            color = edge_color(edge, colored)
            length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
            cursor = 0.
            # Sorting and merging overlapping cut intervals avoids drawing the
            # middle of a gap twice when two crossings are close together.
            for start, end in sorted(gaps[i]):
                start, end = max(0., start), min(length, end)
                if start > cursor+1e-8:
                    piece = _trim_ends(points, cursor, length-start)
                    if len(piece)>1:
                        stroke = (piece, color, width)
                        result.append(stroke+(edge.get('id'),) if with_ids else stroke)
                cursor = max(cursor, end)
            if cursor < length-1e-8:
                piece = _trim_ends(points, cursor, 0.)
                if len(piece)>1:
                    stroke = (piece, color, width)
                    result.append(stroke+(edge.get('id'),) if with_ids else stroke)
    return result


def _box_font(size):
    # No network/font installation; Pillow's bundled scalable default is a fallback.
    for name in ('/System/Library/Fonts/Times.ttc', 'DejaVuSerif.ttf'):
        try:
            return ImageFont.truetype(name, max(1, round(size)))
        except OSError:
            pass
    return ImageFont.load_default(size=max(1, round(size)))


def _draw_box_layers(image, diagram, factor, origin, colored):
    draw = ImageDraw.Draw(image)
    convert = lambda p: ((p[0]-origin[0])*factor, (p[1]-origin[1])*factor)
    for shape in box_shapes(diagram):
        corners = [convert(p) for p in shape['corners']]
        draw.polygon(corners, fill='white')
        draw.line(corners+[corners[0]], fill='#171717',
                  width=max(1, round(shape['stroke_width']*factor)), joint='curve')
        font = _box_font(shape['font_size']*factor)
        # A centered transparent tile preserves the text anchor as it rotates.
        # Pillow and Tk use counterclockwise angles; SVG uses clockwise angles
        # in the diagram's screen coordinates.
        left, top, right, bottom = font.getbbox(shape['label'], anchor='mm')
        half_width = math.ceil(max(abs(left), abs(right)))+2
        half_height = math.ceil(max(abs(top), abs(bottom)))+2
        tile = Image.new('RGBA', (2*half_width+1, 2*half_height+1))
        ImageDraw.Draw(tile).text((half_width, half_height), shape['label'],
                                 fill='#171717', font=font, anchor='mm')
        tile = tile.rotate(-shape['label_angle'], resample=Image.Resampling.BICUBIC, expand=True)
        x, y = convert(shape['point'])
        image.paste(tile, (round(x-(tile.width-1)/2), round(y-(tile.height-1)/2)), tile)
    for points, color, width in box_passages(diagram, colored):
        coords = [convert(p) for p in points]
        draw.line(coords, fill='white', width=max(1, round(width*3*factor)), joint='curve')
        draw.line(coords, fill=color, width=max(1, round(width*factor)), joint='curve')


def _stroke_clearance(strokes, *, per_segment=False, limit_factor=1.6,
                      neighbor_factor=4.):
    """Separation of nonlocal visible strands, capped at ``limit_factor`` widths.

    The default minimum protects editable-raster conversion. Curve fitting
    can request one clearance per source segment to preserve tight regions
    without suppressing smoothing everywhere else. Material neighbors and
    the two halves of a continuous overpass are allowed to meet.
    """
    from .motion import _box_pairs

    width = min(stroke for _, _, stroke in strokes)
    limit = limit_factor * width
    starts, ends, owners, positions, lengths = [], [], [], [], []
    path_points, path_lengths, segment_indices, path_limits = [], [], [], []
    feature_limit = limit
    for owner, (points, _, _) in enumerate(strokes):
        p = np.asarray(points, float)
        ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.r_[0, np.cumsum(ds)]
        keep = ds > 1e-10
        starts.extend(p[:-1][keep]); ends.extend(p[1:][keep])
        owners.extend([owner] * int(keep.sum()))
        positions.extend(s[:-1][keep]); lengths.extend(ds[keep])
        path_points.append(p); path_lengths.append(s[-1])
        segment_indices.append(np.flatnonzero(keep))
        path_limit = limit
        if np.linalg.norm(p[0]-p[-1]) < 1e-6:
            # Every pair in a tiny isolated loop may be a material neighbor.
            # An independent interior-size bound prevents filling that loop.
            # 4*area/perimeter is the diameter for a circle or for a triangle's
            # incircle; local open-strand staircase jitter remains exempt.
            centered = p-p[0]
            area = abs(float(np.sum(centered[:-1, 0]*centered[1:, 1]-centered[1:, 0]*centered[:-1, 1]))) / 2
            path_limit = min(limit, 4*area/s[-1] if s[-1] > 1e-10 else 0.)
            feature_limit = min(feature_limit, path_limit)
        path_limits.append(path_limit)

    def result(candidates=None, first=None, second=None):
        if not per_segment:
            if candidates is not None and len(candidates):
                return min(feature_limit, float(candidates.min()))
            return feature_limit
        clearance = np.repeat(path_limits, [len(indices) for indices in segment_indices])
        if candidates is not None:
            np.minimum.at(clearance, first, candidates)
            np.minimum.at(clearance, second, candidates)
        output, offset = [], 0
        for p, indices, cap in zip(path_points, segment_indices, path_limits):
            local = np.full(len(p)-1, cap)
            local[indices] = clearance[offset:offset+len(indices)]
            output.append(local)
            offset += len(indices)
        return output

    a, b = np.asarray(starts), np.asarray(ends)
    if len(a) < 2:
        return result()
    owner = np.asarray(owners)
    s, ds = np.asarray(positions), np.asarray(lengths)
    pairs = _box_pairs(np.minimum(a, b) - limit / 2,
                       np.maximum(a, b) + limit / 2)
    if not len(pairs):
        return result()
    i, j = pairs.T
    # Four endpoint projections give segment distance unless segments cross.
    v, w = b[i] - a[i], b[j] - a[j]
    vv, ww = np.sum(v*v, axis=1), np.sum(w*w, axis=1)
    t0 = np.clip(np.sum((a[i]-a[j])*w, axis=1)/ww, 0, 1)
    t1 = np.clip(np.sum((b[i]-a[j])*w, axis=1)/ww, 0, 1)
    u0 = np.clip(np.sum((a[j]-a[i])*v, axis=1)/vv, 0, 1)
    u1 = np.clip(np.sum((b[j]-a[i])*v, axis=1)/vv, 0, 1)
    distances = np.stack([
        np.linalg.norm(a[i]-a[j]-t0[:, None]*w, axis=1),
        np.linalg.norm(b[i]-a[j]-t1[:, None]*w, axis=1),
        np.linalg.norm(a[j]-a[i]-u0[:, None]*v, axis=1),
        np.linalg.norm(b[j]-a[i]-u1[:, None]*v, axis=1)], axis=1)
    choice = distances.argmin(axis=1)
    rows = np.arange(len(i))
    separation = distances[rows, choice]
    u = np.stack([np.zeros(len(i)), np.ones(len(i)), u0, u1], axis=1)[rows, choice]
    t = np.stack([t0, t1, np.zeros(len(i)), np.ones(len(i))], axis=1)[rows, choice]
    cross = lambda x, y: x[:, 0]*y[:, 1] - x[:, 1]*y[:, 0]
    determinant = cross(v, w)
    safe = np.where(abs(determinant) > 1e-12, determinant, 1.)
    ui, ti = cross(a[j]-a[i], w)/safe, cross(a[j]-a[i], v)/safe
    intersects = (abs(determinant) > 1e-12) & (ui >= 0) & (ui <= 1) & (ti >= 0) & (ti <= 1)
    separation[intersects] = 0.
    u[intersects], t[intersects] = ui[intersects], ti[intersects]
    si, sj = s[i]+u*ds[i], s[j]+t*ds[j]
    neighborhood = neighbor_factor*width
    local = (owner[i] == owner[j]) & (abs(si-sj) < neighborhood)
    for k, p in enumerate(path_points):
        if np.linalg.norm(p[0]-p[-1]) < 1e-6:
            local |= (owner[i] == k) & (owner[j] == k) & (path_lengths[k]-abs(si-sj) < neighborhood)
    # A shared endpoint is the continuous overpass connection. Exempt its
    # immediate neighborhood, but not later near-contacts of the same paths.
    endpoints = {}
    for k, p in enumerate(path_points):
        for end in (0, -1):
            key = tuple(np.round(p[end], 6))
            endpoints.setdefault(key, []).append((k, 0. if end == 0 else path_lengths[k]))
    for group in endpoints.values():
        for left, (ki, xi) in enumerate(group):
            for kj, xj in group[left+1:]:
                local |= ((owner[i] == ki) & (owner[j] == kj) & (abs(si-xi) < neighborhood) & (abs(sj-xj) < neighborhood))
                local |= ((owner[i] == kj) & (owner[j] == ki) & (abs(si-xj) < neighborhood) & (abs(sj-xi) < neighborhood))
    return result(separation[~local], i[~local], j[~local])


def render_image(diagram, scale=2.0, colored=True, show_orientation=False, background="white", min_stroke_pixels=None):
    """Render a clean, antialiased PNG-compatible Pillow image."""
    scale = max(0.1, float(scale))
    strokes=paths(diagram,colored)
    canvas_width, canvas_height = diagram.get('width',800), diagram.get('height',600)
    origin = np.zeros(2)
    boxes = box_shapes(diagram)
    geometry = [np.asarray(points) for points, _, _ in strokes if points]
    geometry.extend(np.asarray(shape['corners']) for shape in boxes)
    if geometry:
        # The diagram canvas is unbounded. Ordinary PNG exports need the same
        # complete geometry as editable-raster conversion, including strands
        # or boxes moved beyond the original image's positive/negative bounds.
        bounds = np.concatenate(geometry)
        widths = [stroke for _, _, stroke in strokes]
        widths.extend(shape['stroke_width'] for shape in boxes)
        margin = 2*max(widths)+2
        origin = np.minimum(0, bounds.min(axis=0)-margin)
        extent = np.maximum([canvas_width, canvas_height], bounds.max(axis=0)+margin)
        canvas_width, canvas_height = extent-origin
    maximum_scale = 6000/max(canvas_width, canvas_height)
    if min_stroke_pixels:
        # Repeated drawing/recognition cycles must not double the image forever.
        scale = min(scale, maximum_scale)
    if min_stroke_pixels and strokes:
        for _ in range(5):
            width = min(stroke for _, _, stroke in strokes)
            clearance = _stroke_clearance(strokes)
            if clearance >= 1.6*width - 1e-6:
                break
            fitted = clearance / 1.8
            if fitted * maximum_scale < 2:
                raise ValueError('Strands are too close to make a clear drawing. Move them slightly apart, then convert again.')
            strokes = paths(diagram, colored, stroke_width=fitted)
        else:
            raise ValueError('Strands are too close to make a clear drawing. Move them slightly apart, then convert again.')
        narrowest=min(stroke for _,_,stroke in strokes)
        if narrowest * maximum_scale < 2:
            raise ValueError('Strands are too close to make a clear drawing. Move them slightly apart, then convert again.')
        scale=max(scale,min(float(min_stroke_pixels)/max(narrowest,1e-6),
                            maximum_scale))
    supersample = 2
    factor = scale * supersample
    image = Image.new("RGB", (max(1, round(canvas_width * factor)), max(1, round(canvas_height * factor))), background)
    draw = ImageDraw.Draw(image)
    for points, color, width in strokes:
        if len(points) < 2:
            continue
        coordinates = [((x-origin[0]) * factor, (y-origin[1]) * factor) for x, y in points]
        stroke = max(1, round(width * factor))
        draw.line(coordinates, fill=color, width=stroke, joint="curve")
        # Rounded caps keep polyline joins and closed arcs continuous.
        if color != "#ffffff":
            radius = stroke / 2
            for x, y in (coordinates[0], coordinates[-1]):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    if show_orientation:
        for point, tangent, color in arrows(diagram, colored):
            draw.polygon([((x-origin[0]) * factor, (y-origin[1]) * factor) for x, y in arrow_polygon(point, tangent, line_width(diagram) * 2)], fill=color)
    _draw_box_layers(image, diagram, factor, origin, colored)
    image = image.resize((max(1, round(image.width / supersample)), max(1, round(image.height / supersample))), Image.Resampling.LANCZOS)
    # Preserve the fitted logical width in the new pixel coordinate system.
    # Re-estimating it from antialiased pixels (or the canvas-size default)
    # causes thickness to drift on every drawing/recognition round trip.
    if strokes:
        image.info[_RASTER_DISPLAY_KEY] = {
            "stroke_width": min(stroke for _, _, stroke in strokes) * scale,
            "size": image.size,
        }
    return image


def render_editable_image(diagram, colored=True):
    """Create drawing pixels with stable thickness and sufficient resolution.

    A recognized internal raster already has sufficient sampling density. Keep
    that resolution instead of repeatedly doubling it: very thick rasterized
    skeletons develop spurious small branches. Clearance fitting may still
    increase the resolution if an edit has brought strands closer together.
    """
    width = _display_width(diagram.get("display"))
    scale = 1. if width is not None else 2.
    return render_image(diagram, scale=scale, colored=colored,
                        show_orientation=False, min_stroke_pixels=4)


def svg_text(diagram, colored=True, show_orientation=False):
    width, height = diagram.get("width", 800), diagram.get("height", 600)
    strokes = paths(diagram, colored)
    points = ([p for strand, _, _ in strokes for p in strand] +
              [p for shape in box_shapes(diagram) for p in shape['corners']])
    left, top = 0., 0.
    if points:
        margin = max(12., line_width(diagram) * 3)
        left = min(0., min(p[0] for p in points) - margin)
        top = min(0., min(p[1] for p in points) - margin)
        width = max(width, max(p[0] for p in points) + margin) - left
        height = max(height, max(p[1] for p in points) + margin) - top
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{left} {top} {width} {height}">', '<title>Editable link diagram export</title>', f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="white"/>']
    for points, color, stroke in strokes:
        if len(points) > 1:
            coords = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
            cap = "butt" if color == "#ffffff" else "round"
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{html.escape(color)}" stroke-width="{stroke:.3f}" stroke-linecap="{cap}" stroke-linejoin="round"/>')
    if show_orientation:
        for point, tangent, color in arrows(diagram, colored):
            points = arrow_polygon(point, tangent, line_width(diagram) * 2)
            coords = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
            parts.append(f'<polygon points="{coords}" fill="{color}"/>')
    for shape in box_shapes(diagram):
        coords = ' '.join(f'{x:.3f},{y:.3f}' for x, y in shape['corners'])
        parts.append(f'<polygon class="twist-box" data-box-id="{shape["id"]}" points="{coords}" fill="white" stroke="#171717" stroke-width="{shape["stroke_width"]:.3f}"/>')
        x, y = shape['point']
        parts.append(f'<text x="{x:.3f}" y="{y:.3f}" transform="rotate({shape["label_angle"]:.3f} {x:.3f} {y:.3f})" text-anchor="middle" dominant-baseline="central" font-family="serif" font-size="{shape["font_size"]:.3f}" fill="#171717">{html.escape(shape["label"])}</text>')
    for points, color, stroke in box_passages(diagram, colored):
        coords = ' '.join(f'{x:.3f},{y:.3f}' for x, y in points)
        parts.append(f'<polyline class="box-overpass-halo" points="{coords}" fill="none" stroke="white" stroke-width="{stroke*3:.3f}" stroke-linecap="round"/>')
        parts.append(f'<polyline class="box-overpass" points="{coords}" fill="none" stroke="{html.escape(color)}" stroke-width="{stroke:.3f}" stroke-linecap="round"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def save_svg(diagram, path, **options):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_text(diagram, **options), encoding="utf-8")


def render_png(diagram, path, **options):
    """Write a PNG; exported for the command-line and evaluation tools."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    render_image(diagram, **options).save(path, format="PNG")
    return path


def render_svg(diagram, path, **options):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_svg(diagram, path, **options)
    return path


def _tex_label(value):
    """Quote literal label text, including TeX commands and comment characters."""
    escapes = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}',
               '$': r'\$', '&': r'\&', '#': r'\#', '%': r'\%',
               '_': r'\_', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(escapes.get(char, ' ' if char.isspace() else char)
                   for char in str(value))


def _tikz_has_intersections(strokes):
    """Detect even small local loops, beyond the stroke-clearance heuristic.

    The clearance check exempts material neighbors, which is appropriate for
    stroke thickness but could hide a tiny new loop. Here only consecutive
    segments and exact endpoint joins may touch; all other contacts reject a fit.
    """
    from .motion import _box_pairs

    starts, ends, owners, indices, counts = [], [], [], [], []
    closed = []
    for owner, (points, _, _) in enumerate(strokes):
        points = np.asarray(points, dtype=float)
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10]]
        closed.append(len(points) > 2 and np.linalg.norm(points[0]-points[-1]) < 1e-8)
        counts.append(len(points)-1)
        starts.extend(points[:-1]); ends.extend(points[1:])
        owners.extend([owner]*(len(points)-1)); indices.extend(range(len(points)-1))
    if len(starts) < 2:
        return False
    a, b = np.asarray(starts), np.asarray(ends)
    pairs = _box_pairs(np.minimum(a, b), np.maximum(a, b))
    if not len(pairs):
        return False
    i, j = pairs.T
    owner, index = np.asarray(owners), np.asarray(indices)
    count = np.asarray(counts)[owner]
    adjacent = (owner[i] == owner[j]) & ((abs(index[i]-index[j]) == 1) |
               (np.asarray(closed)[owner[i]] & (abs(index[i]-index[j]) == count[i]-1)))
    i, j = i[~adjacent], j[~adjacent]
    if not len(i):
        return False
    v, w = b[i]-a[i], b[j]-a[j]
    cross = lambda p, q: p[:, 0]*q[:, 1]-p[:, 1]*q[:, 0]
    denominator = cross(v, w)
    safe = np.where(abs(denominator) > 1e-12, denominator, 1.)
    t, u = cross(a[j]-a[i], w)/safe, cross(a[j]-a[i], v)/safe
    hit = (abs(denominator) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    # Collinear overlap is also a contact; a single shared endpoint is allowed
    # below, but retracing a segment is never a smooth continuation.
    parallel = (abs(denominator) <= 1e-12) & (abs(cross(a[j]-a[i], v)) < 1e-10)
    lengths = np.sum(v*v, axis=1)
    lo = np.sum((a[j]-a[i])*v, axis=1)/lengths
    hi = np.sum((b[j]-a[i])*v, axis=1)/lengths
    overlap = np.minimum(np.maximum(lo, hi), 1.)-np.maximum(np.minimum(lo, hi), 0.)
    if np.any(parallel & (overlap > 1e-8)):
        return True
    hit |= parallel & (overlap >= -1e-8)
    for end_i in (0, 1):
        for end_j in (0, 1):
            pi, pj = (a if end_i == 0 else b)[i], (a if end_j == 0 else b)[j]
            terminal_i = index[i] == (0 if end_i == 0 else count[i]-1)
            terminal_j = index[j] == (0 if end_j == 0 else count[j]-1)
            joined = terminal_i & terminal_j & (np.linalg.norm(pi-pj, axis=1) < 1e-8)
            hit &= ~joined
    return bool(np.any(hit))


def _tikz_curves(strokes):
    """Fit display/export curves, retaining endpoint joins and visible clearance.

    A cubic must stay in a locally bounded tube around its source polyline.
    A second, diagram-wide clearance check protects nearby strands and tiny
    loops; tight geometry uses a smaller tolerance or the original strokes.
    The model is never resampled or edited by this presentation step.
    """
    from .bezier import fit_cubics, flatten_cubics, smooth_polyline

    if not strokes:
        return []
    width = min(stroke for _, _, stroke in strokes)
    original_clearance = _stroke_clearance(strokes)
    if original_clearance <= 1e-5:
        return [None] * len(strokes)
    local_clearances = _stroke_clearance(strokes, per_segment=True, limit_factor=10.,
                                        neighbor_factor=10.)
    # Paths meeting at the same endpoint are the two visible halves of an
    # overpass. Give them a shared tangent rather than fitting a kink there.
    endpoints = {}
    for i, (points, _, _) in enumerate(strokes):
        for end in (0, -1):
            endpoints.setdefault(tuple(np.round(points[end], 6)), []).append((i, end))
    tangents = {}
    for group in endpoints.values():
        if len(group) != 2 or group[0][0] == group[1][0]:
            continue
        directions = []
        for i, end in group:
            points = np.asarray(strokes[i][0][::1 if end == 0 else -1]).tolist()
            inner = np.asarray(_prefix(points, width*3)[-1])-points[0]
            directions.append(inner/max(float(np.linalg.norm(inner)), 1e-12))
        shared = directions[0]-directions[1]
        length = float(np.linalg.norm(shared))
        if length < 1e-8:
            continue
        shared /= length
        for (i, end), outward in zip(group, (shared, -shared)):
            tangents[i, end] = outward if end == 0 else -outward

    flat_error = min(.01, width*.005)
    # Quantization is part of validation: check the coordinates actually sent
    # to TeX, not more accurate control points that never reach the output.
    for factor in (1., .5, .25):
        curves, flattened = [], []
        for i, (points, color, stroke) in enumerate(strokes):
            # Freehand wiggles need more room than the old half-width fit.
            # Restrict each source interval separately near crossing gaps or
            # other strands; a narrow gap elsewhere must not freeze this arc.
            tolerance = stroke*3.*factor
            limits = np.maximum(1e-8, np.minimum(tolerance, local_clearances[i]*.35*factor))
            softened, remaining = smooth_polyline(points, limits, radius=stroke*6.*factor)
            fitted = fit_cubics(softened, tolerance,
                                start_tangent=tangents.get((i, 0)),
                                end_tangent=tangents.get((i, -1)),
                                segment_tolerances=remaining)
            fitted = [np.round(curve, 3) for curve in fitted]
            curves.append(fitted)
            flat = flatten_cubics(fitted, flat_error)
            flattened.append((flat, color, stroke))
        # Retain at least 90% of the previous narrowest clearance, accounting
        # for both flattened curves' deviation from their actual cubics.
        if (not _tikz_has_intersections(flattened) and
                _stroke_clearance(flattened) >= original_clearance*.9 + 2*flat_error):
            return curves
    return [None] * len(strokes)


def _tikz_curve_coordinates(points, curves):
    """Serialize compact cubic paths; polygons and explicit fallback stay linear."""
    def point(p):
        return f'({p[0]:.3f},{p[1]:.3f})'
    if curves is None:
        return '\n  -- '.join(' -- '.join(point(p) for p in points[i:i+8])
                              for i in range(0, len(points), 8))
    if not curves:
        return ''
    parts = [point(curves[0][0])]
    for curve in curves:
        p0, p1, p2, p3 = curve
        chord = p3-p0
        length = float(np.linalg.norm(chord))
        cross = lambda p: abs(float(chord[0]*(p-p0)[1]-chord[1]*(p-p0)[0]))
        straight = (length > 1e-9 and cross(p1) <= 1e-4*length and cross(p2) <= 1e-4*length
                    and 0 <= np.dot(p1-p0, chord) <= length*length
                    and 0 <= np.dot(p2-p0, chord) <= length*length)
        parts.append('  -- '+point(p3) if straight else
                     '  .. controls '+point(p1)+' and '+point(p2)+' .. '+point(p3))
    if np.array_equal(curves[0][0], curves[-1][-1]):
        parts.append('  -- cycle')
    return '\n'.join(parts)


def prepare_display_curves(diagram):
    """Prepare shared canvas/TikZ geometry without importing GUI dependencies."""
    from .display_curves import prepare_display_curves as prepare
    return prepare(diagram)


def render_tikz(diagram, path=None, colored=True, show_orientation=False, smooth=True,
                prepared=None):
    """Return a pasteable TikZ picture, optionally writing a UTF-8 ``.tex`` file.

    Geometry, crossing gaps, component colors and compact twist-box layers use
    the same display helpers as SVG and the canvas; no implicit twists expand.
    Smooth export fits bounded-error cubic Bezier curves to the visible strands
    without changing the editable diagram. Set ``smooth=False`` for polylines.
    One image pixel is 0.75 PDF big points, preserving SVG's natural size. The
    negative y unit preserves the screen frame while labels stay upright.
    """
    if smooth and prepared is None:
        # Batch exports use exactly the same fit and curve attachments as the
        # canvas, while interactive callers can supply its already-ready fit.
        prepared = prepare_display_curves(diagram)
    if prepared is not None:
        # A prepared snapshot owns the precise geometry currently displayed.
        # Reuse its fitted controls without another gap calculation or fit.
        diagram = prepared.diagram
        strokes, passages = prepared.raw_strokes(colored), prepared.raw_passages(colored)
        stroke_curves = prepared.stroke_curves if smooth else [None]*len(strokes)
        passage_curves = prepared.passage_curves if smooth else [None]*len(passages)
        directions = prepared.arrows(colored) if smooth else arrows(diagram, colored)
    else:
        strokes = paths(diagram, colored)
        passages = box_passages(diagram, colored)
        stroke_curves, passage_curves = [None]*len(strokes), [None]*len(passages)
        directions = arrows(diagram, colored) if show_orientation else []
    shapes = box_shapes(diagram)
    arrow_shapes = [(arrow_polygon(point, tangent, line_width(diagram)*2), color)
                    for point, tangent, color in directions] if show_orientation else []
    all_points = ([p for points, _, _ in strokes+passages for p in points] +
                  [p for shape in shapes for p in shape['corners']] +
                  [p for points, _ in arrow_shapes for p in points])
    left, top = 0., 0.
    right, bottom = diagram.get('width', 800), diagram.get('height', 600)
    if all_points:
        margin = max(12., line_width(diagram)*3)
        left = min(left, min(p[0] for p in all_points)-margin)
        top = min(top, min(p[1] for p in all_points)-margin)
        right = max(right, max(p[0] for p in all_points)+margin)
        bottom = max(bottom, max(p[1] for p in all_points)+margin)

    # These definitions are local to the picture, so multiple pasted exports
    # do not alter colors elsewhere in a containing LaTeX document.
    palette = dict.fromkeys(['#171717'] + [color for _, color, _ in strokes+passages]
                            + [color for _, color in arrow_shapes])
    colors = {color: f'knotstudiocolor{i}' for i, color in enumerate(palette)}
    parts = [r'% Add \usepackage{tikz} to your LaTeX preamble.',
             '% Paste this picture inside the document; no extra TikZ libraries are needed.',
             '% Coordinates use image pixels (1 px = 0.75 bp), with y increasing downward.',
             r'\begin{tikzpicture}[x=0.75bp,y=-0.75bp,line cap=round,line join=round]']
    for color, name in colors.items():
        parts.append(rf'\definecolor{{{name}}}{{HTML}}{{{color[1:].upper()}}}')

    def coordinates(points):
        return _tikz_curve_coordinates(points, None)

    frame = f'({left:.3f},{top:.3f}) rectangle ({right:.3f},{bottom:.3f})'
    parts.extend([rf'\path[use as bounding box] {frame};', rf'\fill[white] {frame};'])
    parts.append('% Strands, with explicit gaps at undercrossings.')
    if smooth:
        parts.append('% Adaptive cubic Bezier curves; endpoints and crossing gaps are retained.')
    for (points, color, stroke), curves in zip(strokes, stroke_curves):
        cap = ',line cap=butt' if color == '#ffffff' else ''
        parts.append(rf'\draw[draw={colors[color]},line width={stroke*.75:.3f}bp{cap}] '
                     + _tikz_curve_coordinates(points, curves) + ';')
    if arrow_shapes:
        parts.append('% Component orientation arrows.')
        for points, color in arrow_shapes:
            parts.append(rf'\fill[fill={colors[color]}] ' + coordinates(points) + ' -- cycle;')
    if shapes:
        parts.append('% Compact twist boxes and their literal labels.')
    for shape in shapes:
        parts.append(rf'\path[fill=white,draw={colors["#171717"]},line width={shape["stroke_width"]*.75:.3f}bp] '
                     + coordinates(shape['corners']) + ' -- cycle;')
        x, y = shape['point']
        # TikZ nodes are not mirrored by a negative y coordinate unit. Their
        # rotation is counterclockwise, the opposite of the SVG screen angle.
        # Scale a standard 10pt font instead of requesting arbitrary font sizes:
        # basic LaTeX fonts otherwise substitute a smaller size for large boxes.
        # A TeX point is 1/72.27 inch; a PDF big point is 1/72 inch.
        font_scale = shape['font_size']*.75*72.27/720
        font = r'\rmfamily\fontsize{10pt}{12pt}\selectfont'
        parts.append(rf'\node[anchor=center,inner sep=0pt,outer sep=0pt,text={colors["#171717"]},'
                     rf'rotate={-shape["label_angle"]:.3f},scale={font_scale:.6f},font={{{font}}}] '
                     rf'at ({x:.3f},{y:.3f}) {{{_tex_label(shape["label"])}}};')
    if passages:
        parts.append('% Passages above the boxes, with white clearance halos.')
    for (points, color, stroke), curves in zip(passages, passage_curves):
        geometry = _tikz_curve_coordinates(points, curves)
        parts.append(rf'\draw[draw=white,line width={stroke*3*.75:.3f}bp] ' + geometry + ';')
        parts.append(rf'\draw[draw={colors[color]},line width={stroke*.75:.3f}bp] ' + geometry + ';')
    parts.append(r'\end{tikzpicture}')
    text = '\n'.join(parts)+'\n'
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding='utf-8')
    return text
