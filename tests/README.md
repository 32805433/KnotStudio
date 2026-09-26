# Portable regression checks

Run `python -m pytest tests` from the project folder after installing the test
dependencies. Geometry and recognition tests generate their own drawings;
resource tests also check the bundled example manifest and sample files. They
do not read a database, research corpus, original checkout, or system font file.

The suite checks drawing and crossing clearance on light and dark backgrounds,
raster reconstruction and PD export, orientation, Reidemeister I and II moves,
twist-box expansion, energy reduction with fixed exterior geometry, recognition
deadlines and isolated caches, and both bundled model files. The suite also covers
freehand display smoothing, close-strand clearance, and shared canvas/TikZ geometry
without changing the editable model. Synthetic geometry tests were retained from
the application development suite; regressions requiring
private or large image collections are not part of this source release.

`python tools/check_release.py --import-modules` additionally checks repository
contents and imports every runtime module from an unrelated working directory.
