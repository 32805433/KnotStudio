# Diagram smoothing

**Minimize energy…** improves spacing and smoothness inside a region you choose.
It keeps the crossings, component directions, and PD code unchanged. Use
**Simplify** separately to remove curls or bigons.

For smoother appearance without moving the stored strands, use **Smooth display**
as described in the [user guide](USER_GUIDE.md#save-and-export).

## Smooth part of a diagram

1. Leave **Full screen region (visible diagram area)** unchecked and choose
   **Minimize energy…**.
2. Hold the mouse button and circle the strands you want to smooth. Keep each
   crossing clearly inside or outside this boundary, then release to start.
3. Watch the layout change and choose **Stop** when you are satisfied.

The boundary and a narrow margin around it stay fixed, as do all strands
outside it. Allow enough room inside the selection for the strands to move.

## Smooth the whole diagram

Choose **Fit** so that you can see the entire diagram, enable **Full screen
region (visible diagram area)**, then click **Minimize energy…**.

“Full screen region” means the visible canvas; it does not mean operating-system
fullscreen mode. If you are zoomed in, the selected region may cover only part
of the diagram. Zooming or moving the view after smoothing starts does not change the
selected region.

## Stop, cancel, or undo

| Control | Result |
| --- | --- |
| **Stop** | Keep the last displayed layout, even if preparation is still running. |
| **Escape** | Cancel the run and restore its starting layout. |
| **Undo** after the run | Restore the layout before the completed run. **Redo** reapplies it. |

## Diagrams with twist boxes

Boxes stay fixed, including their numbers, sizes, and the points where strands
meet them. Portions of strands passing over or under a box also stay fixed
inside it. This applies even if your selection boundary cuts through a box.

Strands outside boxes can move. Smoothing favors meeting each box edge at a
right angle, while respecting nearby strands and the fixed boundaries. It does
not force an attachment through another strand to obtain that angle.

## What to expect

Smoothing balances shorter connections, space between strands, and gentler
bends. It may settle into a stable layout or stop because the geometry prevents
further movement. It does not guarantee the best possible shape or remove
crossings automatically. Try selecting a larger region if a small selection
leaves too little room to move.

The reported energy improvement is measured since the most recent internal
sampling refresh. The displayed percentage can reset or change at that point
without the diagram moving; it is not a progress bar toward completion.

The method is based on KnotJob's diagram layout. The
[technical reference](ENERGY_TECHNICAL.md) describes the energy terms, sampling,
and checks used to preserve the diagram.
