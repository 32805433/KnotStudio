# Architecture and development

Knot Studio has three main layers: recognition from pixels, an editable planar
link-diagram model, and presentation through Tk or exported files. Recognition
does not use a database or look up known knots by filename. The
[user guide](USER_GUIDE.md) describes the application controls.

Source setup, command-line usage, and test instructions are in the
[developer guide](DEVELOPING.md).

## Dependencies and assets

The runtime dependencies declared in [pyproject.toml](../pyproject.toml) are
NumPy, SciPy, scikit-image, OpenCV headless, NetworkX, and Pillow. Their transitive
dependencies and native libraries are included by the application build.
Node.js and Java are not runtime requirements.

The macOS release build uses Python 3.13 and the exact versions recorded in
`requirements-runtime.txt` and `requirements-build.txt`. Install the latter for
release builds; the broader package requirements support source installation.

| Resource | Role |
| --- | --- |
| `recognizer/models/label_components_v2.json` | Geometry classifier for ordinary-label proposals; it does not decode text |
| `recognizer/models/ink_forest_v1.json` | Experimental photo-ink classifier, enabled explicitly after ordinary recognition fails |
| `examples/` | A small collection for exploring features, with provenance records |
| `docs/` | User and developer documentation |
| `licenses/` | Preserved third-party license notices |
| `tools/` and `packaging/` | Standalone macOS build support |

Models are local package data. Recognition needs neither a download nor a
remote model service. Example names and saved example outputs are not inputs to
the recognition algorithm.

## Recognition API

```python
from recognizer.pipeline import recognize, recognize_array

result = recognize("diagram.png", options={"time_budget": 10})
print(result["status"], result["warnings"])
if result["diagram"] is not None:
    diagram = result["diagram"]
    print(result["pd_code"])
```

The file API decodes pixels with Pillow, applies EXIF orientation, and composites
transparency onto white. `recognize_array` accepts grayscale, RGB, or RGBA arrays.
Results contain a diagram or `None`, PD or `None`, warnings, diagnostics, and a
status of `ok`, `needs_review`, or `failed`. Check diagram availability separately:
an unfinished reconstruction may need review without containing a diagram, and
compact boxes can exceed the PD expansion budget.

`unlinked_unknot_components` counts crossing-free components omitted by ordinary
crossing-list PD. Preserve this field alongside PD or save the complete diagram
as JSON; an unavailable count does not mean zero.

The main stages are:

1. Detect twist-box frames, coefficients, and possible strand attachments.
   Missing coefficients and unresolved attachments remain explicit.
2. Extract visible ink and trace centerlines. Electronic and photograph routes
   use pixel evidence, stroke scale, color, and local contrast.
3. Match supported crossing gaps using endpoint directions and crossing geometry.
   Reconstructed gaps pass under the visible strands they cross.
4. Build and validate the planar graph, then try bounded recovery for supported
   image defects. Ambiguous repairs retain review warnings.
5. Recover component orientation from source arrows when evidence is consistent.

`recognition_runtime.py` shares a cooperative deadline and bounded pixel-keyed
cache across retries. Recognition in the app uses a 10-second budget. Native
calls can run beyond a deadline before returning. Python callers can configure
the budget with `options={"time_budget": seconds}`; for offline work,
`options={"time_budget": None}` removes the deadline explicitly.

Ordinary labels are retained by default. The UI offers separate review/delete
operations. `options={"remove_labels": True}` is an explicit batch-cleanup
option; small genuine components can resemble text, so inspect the result.

## Diagram model and editing

`diagram.py` stores ordinary crossings with four counterclockwise ports, an
over/under pair, and reciprocal edge references. Edges store polyline geometry;
crossing-free components have explicit closed edges. Component orientation is
stored separately from traversal.

`pd_code(diagram)` numbers arcs from 1. Each crossing starts with the incoming
under-strand and continues counterclockwise. Orientation changes may change PD
numbering. `crossing_free_count(diagram, pd=None)` handles omitted components;
passing an already computed PD avoids repeated box expansion.

Compact twist boxes use `kind: "twist_box"` entries in the `crossings` array.
They store integer `half_twists`, ordered bundle ports, geometry, and optional
external passages. PD export expands a temporary copy with a default limit of
10,000 crossings. The internal `proxy=True` expansion is only a motion aid and
must not be exported as the original link's PD.

| Modules | Responsibility |
| --- | --- |
| `pipeline.py`, `knotfolio_backend.py`, `geometry.py` | Pixel recognition, gap reconstruction, and graph assembly |
| `board_photo.py`, `board_recovery.py`, `chalk_foreground.py` | Photograph and chalk recovery |
| `drawing.py`, `label_review.py`, `label_model.py` | Raster editing and label proposals |
| `motion.py`, `reidemeister.py`, `reidemeister_ii.py` | Local strand motion, curls, and bigons |
| `twist_boxes.py`, `box_*.py` | Compact-box recognition, representation, editing, and expansion |
| `relaxation.py` | Layout relaxation with fixed boundaries and box holes |
| `render.py`, `bezier.py`, `display_curves.py` | Raster/vector export and guarded smooth display |
| `ui.py` | Tk controls, selection, background work, and undo/redo |

Expensive recognition and geometry operations run outside the Tk thread.
Accepted graph, PD, and display state are committed together, and stale worker
results are discarded. A completed drag, box operation, or relaxation is one
undo step. Drawing edits need recognition before their PD is updated.

With **Smooth display** enabled, the canvas and TikZ share one cached curve fit.
Arclength-based averaging softens freehand jitter before cubic fitting.
Each source segment has a local
displacement budget based on nearby strand clearance; averaging consumes part
of that budget and fitting uses the remainder. Endpoints and overpass joins are
retained, and rounded curves are checked for intersections and narrow gaps.
Unsafe fits use a smaller budget or the original paths. This changes only the
presentation: picking maps back to the stored model and PD data are unchanged.
The checkbox starts off; in that state, canvas and TikZ use the original visible
polylines and need no fit. PNG and SVG also use those paths. The desktop exports
follow the component-color and orientation-arrow checkboxes.

## Tests and extension boundaries

The release source contains a focused test suite using synthetic and bundled
inputs. Run `python -m pytest` after installing the `test` extra. Add meaningful
regressions for changed behavior and check new UI interactions in native Tk;
test doubles do not establish that a macOS control works visually.

Keep recovery based on source pixels, finite search budgets, and explicit
uncertainty. Keep topology changes separate from layout changes. Preserve graph
incidence, orientation, and compact-box semantics through edits. Rendering must
not change topology. A geometry operation's rejection is not a mathematical
impossibility proof, and a valid reconstruction is not an accuracy certificate.
