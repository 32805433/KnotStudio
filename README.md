# Knot Studio

Knot Studio is a Mac app that helps you turn pictures of knots and links into
diagrams you can edit. Start with a photo, a screenshot, or your own drawing,
then move strands and modify the diagram. Save the result as an image, PD code,
or export it for LaTeX.

## Demo

[![Knot Studio demo preview](demo/KnotStudio-demo-poster.jpg)](demo/KnotStudio-demo.mp4)

[Watch the demo](demo/KnotStudio-demo.mp4) (2 min 48 sec): image recognition,
drawing, Reidemeister moves, simplification, energy minimization, and PD/TikZ export.
This demonstration recreates the interface using actual program results;
timings have been shortened.
See [demo notes](demo/README.md), [chapters](demo/KnotStudio-demo-chapters.md),
and [captions](demo/KnotStudio-demo.srt).

## Install and use the macOS app

Download [Knot Studio for Mac](https://github.com/32805433/KnotStudio/releases/download/v0.1.0/KnotStudio-0.1.0-macOS-arm64.zip)
from the [v0.1.0 release](https://github.com/32805433/KnotStudio/releases/tag/v0.1.0).

1. Unzip the download and move **Knot Studio.app** to Applications.
2. Open the app and choose **Open…** to load your picture.
3. If labels or twist boxes are marked, review them, delete unwanted labels, and choose **Recognize drawing**. Otherwise, recognition starts automatically.
4. Check the reconstructed crossings and components before using the result.

The version 0.1.0 Mac app requires an **Apple silicon Mac (M1 or later)** running
**macOS 15.7.5 or later**. Intel Macs are not currently supported by the packaged
app. See [installation help](docs/INSTALLING.md) for first-launch instructions.

> **macOS warning:** This release is not notarized by Apple. If macOS says it
> cannot check Knot Studio for malicious software or cannot verify its developer,
> and you trust this download, try opening the app once, then go to
> **System Settings → Privacy & Security → Open Anyway**.
>
> If the warning instead says the app **contains malware, will damage your
> computer, or is damaged**, please [report the exact message](https://github.com/32805433/KnotStudio/issues)
> before proceeding. See [Apple's instructions](https://support.apple.com/102445).

## What it does

- Recognizes knot and link diagrams from screenshots, iPad drawings, and
  photographs of whiteboards and blackboards.
- Supports sketching, erasing, crossing changes, reversing component
  orientations, Reidemeister moves, and diagram smoothing.
- Recognizes and edits numbered twist boxes with integer or half-integer values.
  This is currently experimental.
- Saves diagrams for later editing and exports images, PD code, and LaTeX drawings.

## User guides

| Guide | Contents |
| --- | --- |
| [Installation](docs/INSTALLING.md) | Download, install, and first-launch help |
| [User guide](docs/USER_GUIDE.md) | Open, recognize, draw, edit, save, and export |
| [Examples](docs/EXAMPLES.md) | Example pictures and their sources |
| [Twist boxes](docs/TWIST_BOXES.md) | Recognize, add, combine, and expand twist boxes |
| [Diagram smoothing](docs/ENERGY.md) | Smooth a selected region or the whole diagram |
| [Developer guide](docs/DEVELOPING.md) | Run from source, test, and build releases |

## Acknowledgments

Knot Studio was written by **Codex using GPT-6 Astra**, under the project
author's direction and review.

Knot Studio's main image-recognition and drawing features are based on
[KnotFolio](https://github.com/kmill/knotfolio), created by Kyle Miller.
Its energy minimizer is based on the diagram-layout method in
[KnotJob](https://www.maths.dur.ac.uk/users/dirk.schuetz/knotjob.html),
created by Dirk Schütz. See the [third-party notices](docs/THIRD_PARTY.md)
for implementation details and licensing.

## License

Knot Studio is distributed under [GNU GPL version 3 or later](LICENSE), with no
warranty. Third-party software and example images retain their respective terms. See
[third-party notices](docs/THIRD_PARTY.md) and [example provenance](docs/EXAMPLES.md).
