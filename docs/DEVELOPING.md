# Developer guide

This guide is for running Knot Studio from source, changing its code, or building
a release. Users of **Knot Studio.app** do not need Python, a terminal, or these
steps. See the [user guide](USER_GUIDE.md) to use the packaged app.

## Set up a source checkout

Clone the repository and enter its folder:

```sh
git clone https://github.com/32805433/KnotStudio.git
cd KnotStudio
```

Alternatively, download a source archive from
[Releases](https://github.com/32805433/KnotStudio/releases), extract it, and open a
terminal in the extracted directory containing `pyproject.toml`.

Use Python 3.11 or later with Tkinter available. These commands use a macOS or
Linux shell:

```sh
python3 -c "import tkinter; print(tkinter.TkVersion)"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m recognizer --gui
```

The virtual environment keeps dependencies separate from other Python projects.
The editable installation makes source changes available without reinstalling
the package and includes the test dependencies. In a new terminal, return to the
repository and activate `.venv` again before running the program.

To run from source without editing or installing test dependencies, use
`python -m pip install .` instead of the editable-install command.

Numeric OCR uses a local Tesseract executable when running from source. If OCR
is unavailable, unread twist coefficients can be entered manually. The packaged
macOS app already includes Tesseract, Python, and Tk.

## Command-line recognition

The recognizer can run without opening the interface:

```sh
python -m recognizer input.png \
  --output result.json --png diagram.png --svg diagram.svg
python -m recognizer photo.png --board-photo board \
  --output photo-result.json
```

The default recognition budget is 10 seconds. Command-line callers can configure
it with `--time-budget SECONDS`. This is a cooperative budget: a native operation
may take longer to return. Without `--output`, the result is printed as JSON.
Image exports are written only when a diagram is returned. Inspect `status`,
`warnings`, and `diagram`, not only whether the command finished.

`--board-photo` accepts `auto`, `off`, `board`, `light`, or `dark`. The desktop app
uses `auto`. Recognition keeps ordinary labels by default; `--keep-labels` is
retained for compatibility.
`--experimental-ink` enables an optional photo-ink model after normal recognition
fails; its recovered diagrams require review. Run
`python -m recognizer --help` for the current command-line options.

## Export details for Python callers

The Python function `render_tikz` in `recognizer.render` defaults to
`smooth=True`, using fitted cubic Bézier curves. Pass `smooth=False` to export
the original visible polylines. The app follows its **Smooth display** checkbox,
which starts off; preview, copying, and saving all use that setting.

Use `colored` and `show_orientation` to control component colors and arrows.
The app takes these values from **Color components** and **Show orientations**.
PNG and SVG exports use the underlying paths. Display and TikZ smoothing do
not change the editable diagram geometry or PD.

PD labels start at 1 and follow component orientations. Plain crossing-list PD
does not encode components with no crossings. Preserve the editable diagram
JSON, or retain the recognition result's `unlinked_unknot_components` count
alongside the PD code. An unavailable count must not be treated as zero.

## Validate changes

From the activated environment in the repository root:

```sh
python -m pytest
python tools/check_release.py --import-modules
```

Manually exercise affected desktop controls as well. See
[Contributing](../CONTRIBUTING.md) for bug reports and change requirements.

## Implementation and releases

- [Architecture](ARCHITECTURE.md): dependencies, recognition APIs, and module boundaries.
- [Twist-box implementation](TWIST_BOXES_TECHNICAL.md): storage, topology, and editing algorithms.
- [Layout-energy implementation](ENERGY_TECHNICAL.md): forces, sampling, and boundary constraints.
- [macOS build and release](BUILDING_MACOS.md): the dedicated build environment,
  standalone packaging, signing, notarization, and release verification.
- [Third-party notices](THIRD_PARTY.md): attribution and dependency licensing.

Release builds use their own pinned dependencies and build instructions; the
source environment above does not create a standalone app. Keep build outputs,
virtual environments, and private datasets out of Git history.

A Python wheel built from this source includes rendered offline help, linked
files, and examples beside the installed package. This also works with an
installation in a custom target directory. Markdown rendering happens at build
time; running the installed wheel does not require the Markdown package.
