# Desktop builds and releases

One Python/Tk codebase produces three native packages. The build environment must
match the target OS; PyInstaller does not cross-compile these applications.
Published tags and downloads are immutable. Fixes use a new version.

## Supported build targets

| Runner | Output | Minimum/compatibility target |
| --- | --- | --- |
| macOS 15 ARM64 | App ZIP | macOS 15.7.5, Apple silicon |
| Windows Server 2022 x64 | Per-user installer and portable ZIP | Windows 11, with Windows 10 22H2 compatibility |
| Ubuntu 24.04 x64 | Portable tar.gz | Ubuntu 24.04 desktop with X11/XWayland |

Windows hosted CI is a Server environment. Passing its checks does not prove
Windows 10/11 interactive behavior. Linux Xvfb checks do not establish compositor,
Wayland, or monitor-scaling behavior. Use the desktop checklist below before
publishing the first package for a platform.

## Common build environment

Use Python **3.13.2** with Tk support and native development tools. Create a virtual
environment, install `requirements-build.txt`, then run:

```sh
python tools/prepare_build_inputs.py
python tools/build_desktop.py --gui-test \
  --tessdata build/inputs/tessdata --runtime-notices build/inputs/runtime
```

The input helper verifies pinned SHA-256 hashes for English OCR data and Python
notices. Dependencies are pinned in the requirements files. The builder replaces
the general OpenCV wheel with a cached core/imgproc-only source build, then runs
tests, creates offline help and dependency notices, packages the application,
and verifies the relocated app. Missing notices or failed verification stop the
build. `--skip-tests` skips pytest only, not package verification.

The version comes from `pyproject.toml` and must match `recognizer/__init__.py`.
Build metadata records the source commit. Build with a clean checkout for a
release. Generated files live under `build/`, `.build-assets/`, and `dist/`.

### macOS

Install Apple's Command Line Tools, the official Python framework including Tk,
and Homebrew Tesseract. Existing `tools/build_macos.py` commands remain supported.
See [macOS signing and notarization](BUILDING_MACOS.md#sign-and-notarize-a-public-release).

### Windows

Use the official x64 Python 3.13.2 installer with Tk, Visual Studio 2022 C++ build
tools/Windows SDK, and Inno Setup 6. Run in a developer PowerShell session:

```powershell
python -m pip install -r requirements-build.txt
./tools/prepare_windows_ocr.ps1
python tools/prepare_build_inputs.py
python tools/build_desktop.py --gui-test --installer `
  --tesseract build/vcpkg_installed/x64-windows-static-md/tools/tesseract/tesseract.exe `
  --ocr-root build/vcpkg_installed/x64-windows-static-md `
  --tessdata build/inputs/tessdata --runtime-notices build/inputs/runtime
```

OCR is built from the exact vcpkg baseline in `packaging/vcpkg/vcpkg.json`.
Native OCR libraries are linked statically, using the dynamic Microsoft runtime;
required runtime DLLs and installed dependency notices are bundled. Training
programs are excluded. `--installer` also checks a silent per-user install,
installed application self-test, and uninstall in a temporary directory.

The executable manifest declares Windows 10 compatibility and DPI awareness.
This declaration is not a Windows 10 test. Keep the same x64 download for Windows
10 and 11; verify it on Windows 10 22H2 before claiming compatibility.
Default builds are unsigned. For a signed distribution, sign the executable and
installer, timestamp the signatures, and regenerate checksums after signing.
Do not store signing credentials in source or upload them as build artifacts.

### Ubuntu

Build on Ubuntu **24.04**, not a newer distribution, to preserve the minimum
library baseline. Install native packages on the build machine:

```sh
sudo apt-get install build-essential tesseract-ocr libtesseract-dev \
  libleptonica-dev tcl8.6-dev tk8.6-dev xvfb xauth patchelf
```

Use Python 3.13.2 with Tk (CI provisions it through `actions/setup-python`).
Run the common build command under `xvfb-run -a` on a runner without a display.
Native package versions and notices are recorded; Ubuntu security updates may
change them across builds. System glibc and desktop services are not bundled.

## Automated builds

`.github/workflows/desktop.yml` builds and tests all three platforms from one
commit. It runs on pull requests, main, development branches prefixed `codex/`,
and manual dispatch. Failures on one platform do not cancel the other results.
Artifacts remain separate from Git source and expire after 14 days.

Each package contains runtime dependencies, help, selected examples and notices.
`dist/verification.json` records the packaged recognition, OCR, topology, file
export and GUI checks. `dist/build-info.json` identifies the source and target.
The build writes a matching source ZIP and archive checksums. No private dataset
or dataset-review app is included.

## Preparing a release

1. Update the version in `pyproject.toml` and `recognizer/__init__.py`, the
   changelog, and `docs/RELEASE_NOTES.md`. Commit normally; never rewrite a published tag.
2. Run **Prepare release draft** from GitHub Actions on the desired commit,
   supplying a fresh tag such as `v0.1.1`.
3. The workflow refuses an existing tag/release, builds all platforms, verifies
   checksums and that every source ZIP contains identical source, then creates
   a **draft** release with one combined checksum file and verification report.
4. Download those exact artifacts and complete desktop validation. Document
   the OS versions and results in the draft release notes.
5. Publish the reviewed draft. Keep the previous release available.

The workflow never overwrites published downloads and never publishes a draft
without the maintainer's final action. A failed build cannot produce a complete
release draft. Do not rerun a tag to replace its files; create a new version.
If upload fails after tag creation, the tag is retained. Inspect the failed run
and recover the draft manually from its verified artifacts; never silently move
the tag to a different commit.

## Desktop validation

On macOS, Windows 11, Windows 10 22H2, and Ubuntu 24.04, check the downloaded
package on a machine without developer Python or Tesseract installed:

- Install/extract, launch, open Help, close and reopen; uninstall where applicable.
- Open a clean example and a numbered-box example; verify recognition and OCR.
- Draw, erase, Shift-draw, pan, zoom, move strands, and undo/redo with native keys.
- Start/stop energy minimization; verify normal and fullscreen window layouts.
- Save/reopen JSON in a path containing spaces and non-ASCII characters; export
  PNG, SVG, and TikZ, checking orientation/color/smoothing settings.
- Repeat window sizing and controls at 100% and 200% display scale. On Linux,
  check both X11 and an XWayland session if both are claimed supported.

Automated GUI checks exercise the same editor, native Tk events and temporary
files. They complement this checklist rather than replacing visual review.
