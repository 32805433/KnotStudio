# Third-party notices

Knot Studio is distributed as a combined work under GNU GPL version 3 or later.
See [LICENSE](../LICENSE). Earlier or upstream source-file grants remain in
place where stated. The software is supplied without warranty.

## KnotFolio

The recognition backend adapts ideas and control flow from **KnotFolio**,
Copyright (C) 2019 Kyle Miller, under GPL version 2 or later. The retained upstream
reference is commit `a2cd207e7be5b3da384765bdfe7a818d7e3eaa4a`.

- [Upstream project](https://github.com/kmill/knotfolio)
- [Upstream README at the reference commit](https://github.com/kmill/knotfolio/blob/a2cd207e7be5b3da384765bdfe7a818d7e3eaa4a/readme.md)
- [Preserved GPL version 2 license text](../licenses/KnotFolio-GPL-2.0.md)

`recognizer/knotfolio_backend.py` is a Python implementation with global endpoint
matching, tangent scoring, and additional image recovery. Its header retains
Kyle Miller's copyright, the upstream commit, and the GPL-2.0-or-later grant.
The permission to use a later GPL version permits inclusion in this GPL-3.0-or-later
distribution. The upstream JavaScript application is not bundled or invoked.

## KnotJob

The layout solver uses an independently implemented attraction/repulsion energy
inspired by KnotJob's drawing layout. Knot Studio adds region-relative motion,
fixed box holes, bending, and dynamic sampling in Python. It does not bundle
KnotJob's Java application or require Java.

## Runtime software

The standalone application includes Python, Tcl/Tk, scientific Python libraries,
Tesseract OCR and its trained data, and their required native libraries. Their
own license notices remain applicable. The build gathers dependency metadata and
full available license texts into `licenses/dependencies/` inside the release
artifacts. Consult those notices for the exact versions and bundled components;
this table is a guide to the principal projects, not a replacement for their
complete terms.

| Project | Principal license |
| --- | --- |
| Python | Python Software Foundation license and included notices |
| Tcl/Tk | Tcl/Tk license notices |
| NumPy | BSD-3-Clause, with notices for bundled numerical libraries |
| SciPy | BSD-3-Clause, with notices for bundled numerical libraries |
| scikit-image | BSD-3-Clause and component-specific notices |
| OpenCV Python headless | Apache-2.0 and bundled-library notices |
| NetworkX | BSD-3-Clause |
| Pillow | MIT-CMU and bundled-library notices |
| ImageIO | BSD-2-Clause |
| tifffile | BSD-3-Clause |
| lazy-loader | BSD-3-Clause |
| packaging | Apache-2.0 OR BSD-2-Clause |
| Tesseract and English trained data | Apache-2.0 |

The release build compiles OpenCV from a checksum-pinned source distribution,
including only `core`, `imgproc`, and the Python bindings. It excludes the video
codecs and camera libraries found in some prebuilt OpenCV wheels. The build
checks the installed modules and native library links before packaging.

PyInstaller packages the macOS app. Its bootloader and runtime hooks have their
own license terms and distribution exception, preserved with the build notices.
Build-only tools are not necessarily application runtime dependencies.

## Models and examples

The project includes its local JSON label and experimental ink classifiers as
application assets. It does not download models at startup.
The release retains inference weights and aggregate training information, while
omitting private corpus entry identifiers and internal annotation-file paths.

Sample image credits and reuse terms are recorded in
[Example image credits](../licenses/IMAGES.md). See the
[example guide](EXAMPLES.md) for the picture list.

## Redistributing a binary

Keep the corresponding source, build scripts, dependency notices, and this file
available with each binary release. Publish the matching source archive alongside
the app archive. The [release guide](BUILDING_MACOS.md) includes the source and
notice checks used when preparing a release.
