# Knot Studio user guide

Turn a picture of a knot or link into a diagram, then draw, move strands, change
crossings, and save your work.

## Open the app

Follow the [installation guide](INSTALLING.md), then open **Knot Studio.app**.
Choose **Open…** to load a picture or a saved diagram. Supported pictures include
PNG, JPEG, WebP, TIFF, and BMP; save a PDF page as an image first. You can return
to this page through **Help → User guide**.

## Drawing mode and diagram mode

- In **drawing mode**, edit the picture with the pencil and eraser.
  **Recognize drawing** turns it into a diagram.
- In **diagram mode**, move strands, change crossings, and use the other diagram
  tools. **Convert to drawing** turns the diagram back into a picture you can
  draw on.

**Recognize drawing** is disabled when you already have a diagram, and
**Convert to drawing** is disabled when you already have a drawing. The mode is
shown below the canvas. The app shows one at a time; it does not keep a separate
original-picture view. Use **Undo** to reverse a conversion, or open the original
picture again.

## Open, review, and recognize

1. Choose **Open…** or press Command+O on Mac or Ctrl+O on Windows/Linux. A picture opens in drawing mode; an
   editable JSON file opens in diagram mode.
2. If possible labels or twist boxes are found, the app marks them and waits
   for you. Ordinary labels appear in amber; boxes appear in blue or orange.
   Review and delete unwanted labels using the controls below, and
   [check any twist boxes](TWIST_BOXES.md#import-and-review).
3. If review was requested, choose **Recognize drawing** when you are ready.
   Otherwise, recognition starts automatically after opening the picture.
4. Check the reconstructed crossings, components, and any orientations or box
   attachments against your picture. **Details** explains any warnings.

**Undo** returns to the picture before recognition, including when recognition
started automatically. You can then erase missed labels or repair a crossing.

### Review labels

Markings identify possible labels; they do not transcribe the text. Check each
marking before deleting it: a small circle could be a label `0` or a real link
component.

- With **Select / move** selected, click an amber marking to delete that label.
- **Shift-click** a marking to keep the ink and dismiss the marking.
- **Delete all marked labels** removes the remaining marked regions in one
  undoable step. First dismiss any markings on strands you want to keep.
- **Mark possible labels** checks again after you draw or erase.
  Rescanning an unchanged picture keeps any twist-box values you have reviewed.

You can recognize without deleting every marking. These edits affect the open
picture; they do not overwrite the original file.

### Recognition time and results

Recognition searches for up to about 10 seconds. Image preparation and some
operations may take longer.

| Message | Meaning and next step |
| --- | --- |
| **Graph validated** | The reconstructed diagram passed internal consistency checks. Check that its crossings and components match the picture. |
| **Needs review** | Some interpretation is uncertain. Read **Details** and inspect the marked regions. |
| **Drawing changed** | The picture has changed since recognition. Choose **Recognize drawing** to update its diagram and PD code. |
| **Enter twist values** | At least one box has no number yet. Supply the missing values before recognition can finish. |
| **Recognition failed** | No diagram was produced. Read **Details**, repair the picture if needed, and try again. |

If recognition fails, the picture stays available for repair. **View → Show
recognition issues** marks unresolved branches and gaps; these marks are not
included in exported images. Repair unclear crossing gaps or erase unrelated
ink, then try again. Touching strands with no visible over/under choice, glare,
faint strokes, and crowded crossings can prevent recognition. Outlined ribbon
diagrams and virtual crossings are not supported.

### Photographs and blackboards

Knot Studio automatically detects drawings and board photographs.

Blackboard recognition reduces eraser dust, but sharp stray chalk marks and
residue touching a strand may still need erasing. Inspect faint strands and
crossing gaps after recognition. To retry the original photograph after it has
become a diagram, open it again.

### Orientations

When clear arrows in the picture agree, recognition keeps the indicated
component direction. Otherwise it uses a default direction. Enable **Show
orientations** to inspect them: gray arrows indicate default directions;
arrows matching the strand color indicate detected or manually chosen directions.
Conflicting source arrows are reported in **Details**. Use **Reverse orientation**
to change a direction yourself.

Editable JSON preserves directions. **Convert to drawing** removes orientation
arrows, so recognizing that picture again may not preserve the previous
directions unless you draw new arrows.

## Draw and erase

Choose **New sketch** for a blank picture, or **Convert to drawing** to draw on
the displayed diagram.

- **Pencil** passes **over** existing strands by default and inserts crossing
  gaps automatically. Hold **Shift** while crossing a strand to pass **under**
  it. Keep Shift unchanged through each crossing; you can change it between
  crossings. The later part of a stroke follows the same rule at a self-crossing.
- **Eraser** removes ink under the brush. Pencil and eraser remember separate
  widths while the app is open.
- **Block eraser** removes the connected patch of ink you click. Crossing gaps
  stop it; touching strokes count as one patch. To delete an entire link
  component, use **Delete component** in diagram mode instead.

On detected dark boards, the pencil is white and the eraser is black. On light
backgrounds, the pencil is dark and the eraser is white. Each stroke or erasing
action can be undone.

Draw closed curves, then choose **Recognize drawing**. Save an editable JSON
before converting if you want to retain the exact diagram: conversion to pixels
can lose detail in tight gaps.

## Move around the view

Use the mouse wheel or zoom buttons to zoom. Hold the right mouse button and
drag to move the view, or hold **Control** while dragging with the left button
(also convenient on a trackpad). **Fit** shows the whole diagram. Scroll the
button panel to reach controls that do not fit in the window.

A diagram can extend beyond the original picture, and exports include the whole
diagram. Drawing mode is limited to the picture's boundary.

## Edit a diagram

Diagram edits update PD code automatically; you do not need to recognize again.

| Action | Control |
| --- | --- |
| Select a component or crossing | **Select / move**, then click |
| Switch a crossing | **Switch crossing**, or select it and press **X** |
| Reverse a component direction | **Reverse orientation**, or select a strand and press **R** |
| Delete a component | **Delete component**, or select a strand and press Delete/Backspace |
| Undo or redo | **Undo / Redo**, or Command+Z and Shift+Command+Z on Mac; Ctrl+Z and Ctrl+Shift+Z (or Ctrl+Y) on Windows/Linux |

Switching crossings and deleting components can change the link type. Component
deletion is unavailable while twist boxes are present.

### Smooth dragging

Choose **Select / move**, then drag a strand away from a crossing. **Move radius**
controls how much nearby strand moves with it; set the radius before starting
the drag.

Dragging can introduce or remove a Reidemeister II crossing pair and perform
Reidemeister III moves. For a new pair, the moving strand passes over by default;
hold **Shift as the pair forms** to pass under. Existing crossings retain their
over/under choices. For a Reidemeister III move, drag the short side of a
triangular region past the opposite crossing.

If a position is rejected, try a different point or radius. The last accepted
position stays visible; releasing the mouse keeps it. Wait for **Smooth drag
saved** before another edit or save. **Escape** cancels the entire drag, and
**Undo** restores the diagram before it. Dragging does not add or remove curls;
use the controls below for those.

### Curls and bigons

**Add curl…** asks you to choose a positive or negative crossing sign, a point on
a strand away from crossings, and a side on which to place the curl.

**Simplify** shades removable curls green and removable Reidemeister II bigons
purple. Click a shaded region to remove its one or two crossings. Remaining
removable regions are shaded for another click. Each removal can be undone
separately; **Escape** leaves this mode and keeps completed edits.

A region containing other strands or a twist box cannot be removed. Curls and
bigons on strands outside twist boxes can still be added or simplified.

## Twist boxes

This feature is currently experimental. A box's number counts full twists of
its whole bundle; `1/2` is one half twist and `-1/2` is the inverse. Boxes can
contain two or more strands.

Click **Twist boxes (experimental)** to expand its controls. Use them to review
recognized boxes, insert twists, combine adjacent boxes, expand a box into
visible crossings, or turn a circled twist region into a box. In **Select / move**,
drag a highlighted box edge to resize it. Drag an external strand over a box
normally, or hold **Shift** while creating the passage to pass under it.

An orange **?** box needs a value before recognition can finish. Use **Review
twist boxes** to click it or choose **Fill missing twist values…**. Check both
the numbers and the points where strands meet the boxes. Imports with boxes
remain **Needs review** even when their structure passes checks.

See the [twist-box guide](TWIST_BOXES.md) for sign conventions, step-by-step
instructions, and current limits. Save editable JSON to preserve boxes for
later editing.

## Smooth a selected region or the whole diagram

1. Leave **Full screen region (visible diagram area)** unchecked and choose
   **Minimize energy…**.
2. Draw a closed boundary around the part you want to smooth, then release.
   Keep each crossing clearly inside or outside the boundary.
3. Let the layout improve, then choose **Stop** when you are satisfied.

To smooth the whole visible canvas, enable **Full screen region (visible diagram
area)** before clicking **Minimize energy…**. This refers to the area you can see
in the canvas, not operating-system fullscreen mode. Choose **Fit** first to include the
entire diagram.

Strands outside the boundary and a narrow margin around it stay fixed. Twist
boxes and strands inside them also stay fixed. Nearby strands are encouraged
to meet box edges at right angles. Smoothing keeps crossings and PD code; use
**Simplify** separately to remove curls or bigons.

- **Stop** keeps the last displayed layout.
- **Escape** restores the layout from the start of the run.
- **Undo** restores a completed run; **Redo** reapplies it.

The process can settle or stop when geometry prevents further movement; it does
not guarantee the best possible shape. See [Diagram smoothing](ENERGY.md) for
more detail.

## Save and export

Once you have a diagram, choose **Save…** or press Command+S on Mac or Ctrl+S on Windows/Linux and select a
format. Choose **Editable diagram JSON** to reopen the diagram and continue
editing later. An unrecognized drawing cannot yet be saved from the app;
recognize it first.

| Format | Use it for |
| --- | --- |
| **Editable diagram JSON** | Save the diagram, its geometry, component directions, and twist boxes for later editing. |
| **PD code text** | Use the diagram's PD code in another program. |
| **PNG image** | Save a picture for sharing or inserting in documents. |
| **SVG vector image** | Save an image that stays sharp when resized. |
| **TikZ code** | Insert the drawing in a LaTeX document. |

Hover over the text in the **PD code** or **TikZ code** tab to reveal its copy
button. For TikZ, add `\usepackage{tikz}` to your LaTeX preamble and paste the
copied code into your document, or save a `.tex` file and include it with
`\input{diagram.tex}`.
Exported drawings follow **Color components** and **Show orientations**.

**Smooth display**, off by default, softens small freehand wiggles on the canvas
without changing the editable geometry or PD code. Smoothing is gentler near
crossings and nearby strands to keep them distinct. Turn it off to inspect the
underlying paths.
TikZ preview, copying, and export follow this setting: smooth curves when enabled,
straight segments when disabled. PNG and SVG exports use the underlying paths
regardless of this setting.

After drawing edits, recognize again to update diagram exports; the code tabs
indicate when they belong to the previous diagram.

PD labels follow component directions. A plain PD crossing list does not record
components with no crossings, so save the editable JSON as well when those
components matter. Twist boxes expand into ordinary crossings for PD output;
very large expansions may exceed the preview limit. The editable diagram can
still be saved.

## Help and further information

If the app will not open, see [Installation](INSTALLING.md). For an error report,
include the message and, if possible, a picture or saved diagram that reproduces
it. The app log is at `~/Library/Logs/Knot Studio/app.log`; remove private file
paths before sharing it.

- [Example pictures and their sources](EXAMPLES.md)
- [Twist boxes](TWIST_BOXES.md)
- [Diagram smoothing](ENERGY.md)
- [Developer guide](DEVELOPING.md): run from source, test, and build releases.
