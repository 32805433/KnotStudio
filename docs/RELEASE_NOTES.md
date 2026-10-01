# Knot Studio 0.1.1

Knot Studio turns pictures of knots and links into editable diagrams. Draw,
move strands, perform Reidemeister moves, smooth diagrams, and export images,
PD code or LaTeX. Numbered twist boxes are experimental.

## What is new

- Windows installer and portable app, plus a Linux package for Ubuntu 24.04 or later.
- Native keyboard shortcuts and scrolling on each platform.
- Correct over/under choice when dragging a new Reidemeister II pair inside a larger bigon.

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

## Verification

Automated package checks cover startup, image recognition, OCR, editing, file
exports, and the graphical interface on macOS ARM64, Windows Server 2022, and
Ubuntu 24.04. The Windows installer also passes installation and uninstall checks.
See `verification.json` for the build environments and detailed results. Windows
10 22H2 compatibility remains a target rather than a result of these checks.
