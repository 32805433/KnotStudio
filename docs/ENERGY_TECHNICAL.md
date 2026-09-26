# Layout energy: technical reference

For the controls and their effect on a diagram, see [Diagram smoothing](ENERGY.md).

## What the solver optimizes

The Python implementation in [relaxation.py](../recognizer/relaxation.py) is
inspired by the attraction and repulsion used in KnotJob's drawing layout.
It combines quartic attraction between neighboring landmarks, inverse-square
repulsion between other landmarks, and a bending penalty along strands.
Coordinates are normalized by a typical sampling distance.

A triangulated planar map extends landmark movement to the drawn strands.
Boundary and box constraints fix the corresponding parts of this map. A
backtracking line search checks both energy decrease and positive triangle area
throughout each accepted step, subject to floating-point tolerances. Strand
resolution checks prevent unresolved bends from stretching into folds.

The solver refreshes the triangulation and redistributes samples as the drawing
changes. These refreshes retain the current drawn curve. A whole-diagram run
also constrains its geometric arclength centroid to prevent drift.

Each accepted step lowers the energy for its current sampling. Resampling changes
the discrete energy baseline without moving the drawing, so the displayed
percentage measures improvement **since sampling refresh**. Percentages from
different baselines are not a single monotone measure of progress.

The flow may stop at a stationary layout or against a geometric constraint. It
does not guarantee a global minimum or perform unrestricted spatial knot
motion. Accepted layouts preserve the planar diagram subject to numerical
tolerances; inspect the result as with other geometry edits.
