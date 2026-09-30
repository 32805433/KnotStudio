# Knot Studio 0.2.0

Knot Studio turns pictures of knots and links into editable diagrams. Draw,
move strands, perform Reidemeister moves, smooth diagrams, and export images,
PD code or LaTeX. Numbered twist boxes are experimental.

## Packages

- **macOS-arm64.zip**: Apple silicon, macOS 15.7.5 or later.
- **Windows-x86_64-Setup.exe**: per-user Windows installer; Windows 11 target.
  The portable ZIP contains the same app. Windows 10 22H2 is a compatibility target.
- **Linux-x86_64.tar.gz**: Ubuntu 24.04 LTS or later with X11/XWayland.

Each package includes Python, OCR, examples and offline help. The separate source
ZIP is for developers. SHA256SUMS.txt covers the downloads; verification.json
records the automated package checks.

Default Mac builds are not notarized and Windows builds are unsigned. See the
[installation guide](https://github.com/32805433/KnotStudio/blob/main/docs/INSTALLING.md)
for opening a trusted download.

## Desktop validation before publication

Record the tested OS versions and results here before publishing the draft.
Windows hosted-runner checks alone do not certify Windows 10 or 11 desktop behavior.
