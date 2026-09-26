"""Geometry-only typography adjacency, including scripts and signed labels.

No glyph is decoded. These helpers supply conservative spatial constraints to
the label grouper; they neither propose ink nor modify the input image.
"""
from __future__ import annotations


def _script(parent, child):
    """Directional adjacency for a glyph's subscript or upper-right prime.

    A subscript sign can be much shorter than a letter, and a detached minus
    may be below the parent's baseline. Test that geometry independently of
    the letter-height test; a baseline minus still belongs to prefix/word
    grouping. This uses placement and shape only, not decoded characters.
    """
    x,y,r,d = parent['bbox']; xx,yy,rr,dd = child['bbox']
    h,hh = d-y,dd-yy
    if parent.get('bar') or h <= 0 or hh <= 0:
        return False
    if not (xx >= x+.45*(r-x) and -.30*h <= xx-r <= .60*h):
        return False
    if child.get('bar'):
        # A minus subscript is short, distinctly low, and beside the glyph.
        # Keep an underline/fraction bar and an equals sign out of this path.
        return (not child.get('paired_bar') and xx >= r-.12*h
                and .18*h <= rr-xx <= .95*h and hh <= .28*h
                and y+.70*h <= (yy+dd)/2 <= d+.40*h)
    # A subscript may overlap the right edge of an italic capital. Testing
    # this before horizontal-overlap/fraction rules preserves T_0, L_2, etc.
    subscript = (.25*h <= hh <= .82*h and rr-xx <= 1.2*h
                 and y+.28*h <= yy <= d+.32*h
                 and dd >= d-.08*h and dd <= d+.85*h)
    # Apostrophes/primes are small high accents immediately to the right.
    prime = (hh <= .65*h and rr-xx <= .45*h
             and y-.95*h <= yy and y-.20*h <= dd <= y+.55*h)
    return subscript or prime


def typography_neighbors(a, b):
    """Whether two ink pieces can belong to one mathematical text label."""
    if _script(a,b) or _script(b,a):
        return True
    x,y,r,d = a['bbox']; xx,yy,rr,dd = b['bbox']
    h,hh = d-y,dd-yy
    large = max(h,hh)
    horizontal_gap = max(xx-r,x-rr,0)
    overlap = min(r,rr)-max(x,xx)
    vertical_gap = max(yy-d,y-dd,0)
    if overlap > .30*min(r-x,rr-xx):
        if a.get('bar') or b.get('bar'):
            if a.get('paired_bar') or b.get('paired_bar'):
                # Equals-sign strokes belong to their own baseline, not to
                # the caption above/below as a fraction rule would suggest.
                return (a.get('bar') and b.get('bar')
                        and vertical_gap <= .6*max(r-x,rr-xx))
            return vertical_gap <= .85*large and abs((x+r)-(xx+rr)) <= 1.5*max(r-x,rr-xx)
        return (min(h,hh) <= .4*large and vertical_gap <= .65*large
                and abs((x+r)-(xx+rr)) <= .9*large)
    if a.get('bar') or b.get('bar'):
        return horizontal_gap <= .85*large and abs((y+d)-(yy+dd)) <= 1.0*large
    return (horizontal_gap <= .70*large and min(h,hh) >= .14*large
            and vertical_gap <= .18*large and abs(y-yy) <= .85*large)


def paired_horizontal_bars(pieces):
    """IDs of aligned close horizontal strokes forming an equals-like pair."""
    bars = [(i,p['bbox']) for i,p in pieces.items() if p.get('bar')]
    paired = set()
    for k,(i,(x,y,r,d)) in enumerate(bars):
        for j,(xx,yy,rr,dd) in bars[k+1:]:
            short,long = min(r-x,rr-xx),max(r-x,rr-xx)
            gap = max(yy-d,y-dd,0)
            if (short >= .75*long and min(r,rr)-max(x,xx) >= .85*short
                    and 1.2*max(d-y,dd-yy) <= gap <= .60*short
                    and max(d-y,dd-yy) <= .2*short):
                paired.update((i,j))
    return paired


def signed_prefix_owners(pieces):
    """Map sign-sized horizontal bars to their nearest right-hand glyph.

    A fraction bar overlaps its numerator/denominator horizontally and is not
    treated as a prefix. Ownership prevents the minus sign of the next label
    from joining the preceding label simply because both gaps are small.
    """
    owners = {}
    for index,piece in pieces.items():
        if not piece.get('bar') or piece.get('paired_bar'):
            continue
        x,y,r,d = piece['bbox']
        choices = []
        for other, glyph in pieces.items():
            if other == index or glyph.get('bar'):
                continue
            xx,yy,rr,dd = glyph['bbox']
            h = dd-yy
            gap = xx-r
            vertical = ((y+d)/2-yy)/max(1,h)
            if (0 <= gap <= .85*h and .20*h <= r-x <= 1.4*h
                    and d-y <= .28*h and .15 <= vertical <= .85):
                choices.append((gap/max(1,h)+.12*abs(vertical-.55),other))
        if choices:
            owners[index] = min(choices)[1]
    return owners


def blocked_typography_pairs(pieces, owners=None):
    """Unordered component pairs that would attach a prefix to left text."""
    owners = signed_prefix_owners(pieces) if owners is None else owners
    blocked = set()
    for bar,owner in owners.items():
        bx,by,br,bd = pieces[bar]['bbox']
        gx,gy,gr,gd = pieces[owner]['bbox']
        h = gd-gy
        for other,piece in pieces.items():
            if other in (bar,owner) or piece.get('bar'):
                continue
            x,y,r,d = piece['bbox']
            # The prefix of a signed subscript belongs with its parent. The
            # subscript's digit can be as tall as that parent (for example in
            # a small raster), so the low sign itself supplies the evidence.
            if _script(piece, pieces[bar]):
                continue
            if (r <= bx+.15*h and x < bx and bx-r <= .85*h
                    and min(d,gd)>max(y,gy) and .6*h <= d-y <= 1.5*h):
                blocked.add(frozenset((bar,other)))
    return blocked


def can_merge_typography_groups(a, b, pieces, owners=None):
    """Keep adjacent complete signed expressions individually clickable.

    A leading minus sign followed by a similar-baseline expression is a common
    repeated coefficient layout. Two such complete runs should not be merged
    merely by a generic word-spacing threshold. Internal subtraction remains
    eligible when one side is not independently prefixed (e.g. n-1 or -2q).
    """
    owners = signed_prefix_owners(pieces) if owners is None else owners

    def parenthesis_like(index):
        p = pieces[index]
        x,y,r,d = p['bbox']
        return (not p.get('bar') and p.get('ends') == 2
                and p.get('branches',0) == 0 and not p.get('closed')
                and 2.*p.get('stroke_width',1.) <= r-x <= .40*(d-y))

    # A leading sign on a parenthesized expression and its internal subtraction
    # are not two diagram coefficients. Paired tall, narrow open curves at the
    # outer ends are sufficient geometry; the intervening text is not decoded.
    left,right = sorted((a,b),key=lambda g:min(pieces[i]['bbox'][0] for i in g))
    left_body = sorted((i for i in left if not pieces[i].get('bar')),
                       key=lambda i:pieces[i]['bbox'][0])
    right_body = sorted((i for i in right if not pieces[i].get('bar')),
                        key=lambda i:pieces[i]['bbox'][0])
    if (len(left_body) >= 2 and len(right_body) >= 2
            and parenthesis_like(left_body[0]) and parenthesis_like(right_body[-1])):
        return True

    def prefix(group):
        nonbars = [i for i in group if not pieces[i].get('bar')]
        if not nonbars:
            return None
        left = min(pieces[i]['bbox'][0] for i in nonbars)
        for bar,owner in owners.items():
            if bar in group and owner in group and pieces[bar]['bbox'][2] <= left:
                return pieces[owner]['bbox']
        return None

    p,q = prefix(a),prefix(b)
    if p is None or q is None:
        return True
    h,hh = p[3]-p[1],q[3]-q[1]
    large = max(h,hh)
    return not (min(h,hh) >= .65*large
                and abs((p[1]+p[3])-(q[1]+q[3])) <= .55*large)
