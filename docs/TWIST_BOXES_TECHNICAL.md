# Twist boxes: technical reference

For importing and editing boxes in the app, see the [twist-box guide](TWIST_BOXES.md).
This page describes the stored representation, recognition, and Python interfaces.

## Geometry and sign convention

A compact box retains its position, dimensions, axis, coefficient, ordered strand
attachments, and separately stored over/under passages. Storage of the implicit
twists does not grow with their number.

With the box axis directed from top to bottom on the page, a positive elementary
crossing has the top-right strand passing over the top-left strand. Component
orientations are separate; reversing one component can change an oriented
crossing sign without changing box handedness. See the
[sign illustration](twist-signs.svg).

The displayed number rotates with the box. Its baseline follows the lower of
the two bundle-attachment edges, with the bottom of the characters toward that
edge. If the edges have equal height, the right-hand edge is used. Separate
passing strands do not affect the label orientation.

For `m` strands and coefficient `q`, the stored integer is `half_twists = 2q`.
The bundle's explicit expansion has `abs(2q) * m * (m - 1) / 2` crossings before
including crossings with separate passages. Odd half-twist parity reverses
lane order; even parity preserves it.

## Certifying an explicit twist region

The selected strands must admit a planar braid sweep: successive crossings
exchange neighboring lanes, respect the order along each strand, and agree
with the crossing's incoming branches and boundary attachments. Segment
intersections, self-contacts, and cyclic branch order are checked first.

The resulting signed braid word is checked against a power of the whole-bundle
half twist `Delta_m` using its faithful Artin action on the free group. This
is stronger than checking crossing counts, endpoint permutations, or a knot
invariant. Different words related by Reidemeister II/III moves or commutation
of distant crossings can certify the same twist.

The implementation has finite word-growth and crossing limits. It does not
certify every drawing isotopic to a twist: self-crossings, internal closed
components, nested boxes, ambiguous intersections, and insufficient room around
the selection may prevent conversion. A rejection is not a proof that the
selected tangle differs mathematically from a twist.

Combining boxes uses the same algebraic certification without expanding large
coefficients. Candidate frames use each box's axis, their average, and the line
between their centers; smaller valid enclosing regions are preferred. Both
connecting routes are considered when both ends connect the same two boxes.

## Motion and alignment

Dragging uses a geometric proxy controlled by lane count and twist parity.
Dense geometry stays in numeric arrays until a frame is accepted; passage
checks compare nearby segments. PD previews reuse the exact prior result when
incidence and orientation have not changed. Passage changes invalidate that
cache. The proxy is never used as the exported PD.

Automatic alignment runs on opening JSON, finishing recognition, and committing
box operations. It shortens a box axially and uses the freed room for smooth
connectors to matching evenly spaced lanes. It preserves coefficient, lane
order, and orientation, including odd half twists. It leaves already aligned
boxes unchanged. Boxes with separate passages, overlapping geometry, or
insufficient room keep their existing layout.

Alignment changes the in-memory diagram; it does not rewrite a loaded JSON.
Recognition and box operations include it in the same undoable edit.
It runs after pixel tracing and source-arrow recognition, not on every frame.

Energy minimization fixes each whole box rectangle, including stored passages,
and favors perpendicular exterior attachment tangents. It operates without
expanding implicit twists. Exterior curl and bigon edits similarly retain box
geometry and avoid expansion; exterior edge IDs may change when strands split
or join. See [layout energy](ENERGY_TECHNICAL.md) for the solver.

## Python editing operations

The four operations are also available from Python. All return independent
diagrams and leave their input unchanged:

```python
from recognizer.box_insert import add_twists, prepare_twist_insertion
from recognizer.box_operations import (
    combine_twist_boxes, delete_zero_box, expand_twist_box,
    positive_half_twist, negative_half_twist,
    positive_full_twist, negative_full_twist,
)
from recognizer.box_compression import analyze_twist_region, compress_twist_region

changed = add_twists(diagram, [[x0, y0], [x1, y1]], "1/2")
changed = combine_twist_boxes(diagram, first_id, second_id)
changed = delete_zero_box(diagram, box_id)
changed = expand_twist_box(diagram, box_id, amount="-1", side="bottom")
changed = expand_twist_box(diagram, box_id)  # The whole selected box.
proposal = analyze_twist_region(diagram, polygon)
changed = compress_twist_region(diagram, polygon)
```

The four signed half/full-twist helpers extract their named amount from an
existing box, subject to the same sign and room checks. `side="top"` corresponds
to UI end A, and `side="bottom"` to end B. `side="both"` expands the entire box
with symmetric room growth. `analyze_twist_region` returns the
proposed rectangle, lane count, coefficient and exact braid certificate without
changing the diagram. `prepare_twist_insertion` similarly returns a zero-box
preview for a slice; adding the requested number is a separate step.

## Compact representation and PD

Boxes are vertices in the existing `crossings` array with `kind: "twist_box"`.
Their fields include:

| Field | Meaning |
| --- | --- |
| `point` | Box center in diagram coordinates |
| `size` | `[transverse_width, axial_length]` |
| `axis` | Unit vector from the entry end toward the exit end |
| `half_twists` | Exact signed integer, twice the displayed full-twist number |
| `label` | Canonical display string consistent with `half_twists` |
| `strand_count` | Number of lanes in the bundle, at least two |
| `ports` | References to exterior edge endpoints |
| `port_points`, `bundle_ports` | Boundary geometry and ordered entry/exit lanes |
| `passages`, `passage_crossings` | Additional strands passing over/under the box and their mutual crossings, when present |

Storage of the internal bundle does not grow with the twist magnitude. Exterior
curves and any separately drawn passage geometry still require points. Saving
JSON retains this representation; it does not replace the bundle with sampled
internal ink.

Plain PD has no twist-box notation, so `pd_code(diagram)` expands the boxes into
ordinary crossings on a temporary copy. Its default expansion limit is **10,000
crossings**. A larger expansion produces an explicit error instead of truncating
the twists. The compact diagram can still be edited and saved as JSON when its
PD preview exceeds that limit. Actual expansion size also includes ordinary
crossings and crossings introduced by external passages.

Python callers can request an explicit expansion with a chosen finite budget:

```python
from recognizer.twist_boxes import expand_boxes
from recognizer.diagram import pd_code

ordinary_diagram = expand_boxes(diagram, max_crossings=20_000)
pd = pd_code(ordinary_diagram)
```

`expand_boxes(..., proxy=True)` is an internal motion aid. It preserves boundary
connectivity but generally changes the link type. Its PD must never be presented
as the original link's PD.

## Recognition API and algorithm

```python
import numpy as np
from PIL import Image
from recognizer.box_detection import detect_twist_boxes, inspect_twist_box
from recognizer.pipeline import recognize_array

rgb = np.array(Image.open("diagram.png").convert("RGB"))
review = detect_twist_boxes(rgb)
# Inspect review["boxes"] and correct an uncertain coefficient before tracing.
result = recognize_array(rgb, options={"twist_boxes": review})
```

`recognize` and `recognize_array` detect boxes automatically by default.
`options={"twist_boxes": False}` disables this detection explicitly.
`inspect_twist_box(rgb, outline, label="-3/2")` inspects a user-selected convex
quadrilateral and returns one compatible candidate. Its explicit value does not
override attachment uncertainty. Review objects include an image size and pixel
hash; a proposal from a different or edited image is rejected as stale.

`recognizer.diagram.crossing_free_count(diagram, pd=None)` counts components
omitted from PD notation. For example, a closed three-strand zero-twist box has
PD `[]` and three omitted unknot components. Without an already computed PD,
the helper uses compact component walks and never expands a large coefficient.
It returns `None` when optional passage intersections prevent a certified
count; `validate(diagram)["crossing_free_components"]` follows the same rule.
Pass the already computed PD as the second argument to obtain the exact count
without repeating expansion. The helper subtracts the components represented
by the PD from the full compact component count. A missing count must not be
interpreted as zero when exporting a link.

The detector is classical and reads pixels only:

1. Extract dark or light ink and find approximately rectangular bounded regions.
   Small gaps in hand-drawn corners are bridged for proposals only.
2. Rectify each candidate interior, suppress border remnants, and look for a
   coefficient within it.
3. Decode exact signed integers or half integers using local Tesseract and
   independent sign/stem geometry. Conflicting or unsupported readings remain
   unknown; letters are not silently converted to digits.
4. Scan outside opposite sides to locate **every** entering strand. Border
   thickness, nearby curves and small attachment gaps are checked separately.
   Equal opposite-side lane counts determine the bundle axis and strand count.
5. Trace the exterior using temporary untwisted connector strokes, then restore
   the exact discrete twist permutation. The numeric twist is not rasterized
   merely to recognize the surrounding graph. On light backgrounds, a fallback
   for faint or colored strands tries combined-ink extractions and requires at
   least two thresholds to agree on the complete planar crossing diagram.

The detector accepts only image arrays; it never reads census names, image
filenames, collection sidecars, ground truth or reference PDs. If Tesseract is
unavailable, simple signed-one geometry can still be recognized and other
coefficients may require manual entry.
