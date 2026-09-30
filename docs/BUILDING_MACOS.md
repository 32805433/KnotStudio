# Build and release for macOS

For the shared build workflow and release process, see
[Desktop builds and releases](BUILDING_DESKTOP.md).

The build creates a movable **Knot Studio.app** with its own Python, Tcl/Tk,
scientific libraries, local models, numeric OCR, documentation, and examples.
The built app does not need a source checkout, Homebrew, or an installed Python.

The v0.2.0 build configuration targets **macOS 15.7.5 or later**. The locally
verified application uses **Apple silicon (arm64)**. The build uses the current
Python interpreter's native architecture. To prepare an Intel version, build
on an Intel environment with matching Python and native dependencies and test
it separately. Both architectures have passed automated packaging checks during
development, but interactive use has not been verified on an Intel Mac.
Revalidate the current source before distributing either build.
The arm64 archive cannot run on Intel Macs.

## Prepare a build environment

Start with a [source checkout](DEVELOPING.md#set-up-a-source-checkout).
Use **Python 3.13** with a working Tkinter module and native libraries matching
the target architecture. The local build uses the official Python 3.13
framework and Tk 8.6.15. The pinned requirements record the release build's
Python dependencies:

```sh
python3.13 -c "import tkinter; print(tkinter.TkVersion)"
python3.13 -m venv .venv-build
source .venv-build/bin/activate
python -m pip install -r requirements-build.txt
```

The build script runs directly from the checkout. For editable source development
in the same environment, also run `python -m pip install --no-deps -e .`.

Install Apple's Command Line Tools (the C++ compiler) on the build machine.
The first build compiles a small OpenCV wheel containing only its core and image
processing modules. It verifies the upstream source checksum and caches the
result in `build/opencv/`. Video codecs, camera support, and OpenCV's own GUI are
excluded. This compilation is only a maintainer step; app users install nothing.

Install Tesseract with English trained data on the build machine. The script
copies the executable, required libraries, and trained data into the app. A
non-default executable can be selected with `--tesseract`; select its trained-data
directory with `--tessdata` if automatic discovery cannot locate it.

## Build

From the repository root:

```sh
python tools/build_macos.py
```

The default local build is **ad-hoc signed**. It is not signed with an identified
developer certificate and is not notarized by Apple. A downloaded copy can be
blocked by Gatekeeper. Include the [installation guide](INSTALLING.md) and
disclose the signing status when sharing an app. The signing and notarization
steps below provide Apple's standard distribution path.

For version 0.2.0 on Apple silicon, the app is written to
`dist/Knot Studio.app` and the app archive to
`dist/KnotStudio-0.2.0-macOS-arm64.zip`. Matching source is written to
`dist/KnotStudio-0.2.0-source.zip` and is also included inside the macOS archive.
`dist/SHA256SUMS.txt` records both archive checksums. Build output stays outside
the repository's tracked files.

The script checks the source release, builds the minimal OpenCV dependency, runs
tests, builds offline help and license notices, packages the app, and verifies
its native libraries and relocated runtime. The verification uses a clean search
path without Homebrew or a Python environment and writes `dist/verification.json`.
Add `--gui-test` on a Mac with a desktop session to initialize the relocated
app's Tk interface as well.
`--skip-tests` skips pytest only; bundle verification still runs.

Run `python tools/build_macos.py --help` for the exact supported options. The
build script and [pyproject.toml](../pyproject.toml) are the authoritative build
configuration.

Offline help and its linked files are generated in `.build-assets/resources/`
and bundled together. The help build checks local links and section anchors;
it includes rendered guides, image credits, examples, and the demo. Run
`python tools/build_docs.py` to regenerate these resources without rebuilding
the application.

## Continuous integration

The desktop workflow (`.github/workflows/desktop.yml`) builds macOS ARM64,
Windows x86-64 and Ubuntu x86-64 packages from the same source. The macOS job uses
the official Python 3.13.2 framework with Tk and Homebrew Tesseract. It runs
packaged GUI checks as well as the runtime and dependency checks.

The workflow does not publish downloads. The separate draft-release workflow
validates every platform, then prepares a new draft for desktop review. Default
Mac artifacts use ad-hoc signing and are not notarized.

## Sign and notarize a public release

A Developer ID signed and notarized public macOS release requires the maintainer's Apple Developer ID
Application certificate and notarization credentials. With those configured on
the build machine, pass their configured names:

```sh
python tools/build_macos.py \
  --codesign-identity 'Developer ID Application: YOUR NAME (TEAM ID)' \
  --notary-profile 'YOUR KEYCHAIN PROFILE'
```

Use actual certificate and Keychain profile names; the example strings are
placeholders. Keep credentials out of the repository. Signing and notarization
are release steps, not claims about a default local build.

Before publishing the binary, verify the signing and notarization results and
test the final downloaded archive on a supported Mac. Test it outside the source
checkout with no reliance on the build environment. Confirm startup, opening
an image with **Open…**, box-number OCR, JSON save/reopen, image export, and
offline Help. The [example pictures](EXAMPLES.md) provide suitable inputs. A
source-only GitHub repository can be published independently of this binary
release gate.

## Source and notices

Run the source audit and tests before creating a release:

```sh
python tools/check_release.py --import-modules
python -m pytest
```

Distribute the matching source archive alongside each application archive. Keep
the application license, upstream notices, and the build-generated dependency
notices with the release. The generated `licenses/dependencies/` directory records
the bundled third-party software. See [third-party notices](THIRD_PARTY.md).

The source repository contains the application, its build scripts, documentation,
tests, and selected samples. App bundles, build environments, and generated
distribution archives belong in release artifacts, not in Git history.
