# Knot Studio 0.1.0

The first version of Knot Studio turns pictures of knots and links into diagrams
you can edit. Draw and erase strands, change crossings, perform Reidemeister
moves, and smooth part or all of a diagram. Save your work for later editing or
export it as an image, PD code, or a LaTeX drawing.

The **Smooth display** checkbox controls whether the canvas and TikZ export use
smooth curves. Exports also follow the component-color and orientation settings.

Numbered twist boxes can be recognized, edited, combined, and expanded into
individual crossings. This is currently experimental.

The app includes an offline user guide and a selection of SnapPy, blackboard,
and iPad [example pictures](EXAMPLES.md) to load with **Open…**.

## Mac app

The packaged app requires an **Apple silicon Mac (M1 or later)** running
**macOS 15.7.5 or later**. Intel Mac, Windows, and Linux app downloads are not
provided.

See the [installation instructions](INSTALLING.md). The app is not
notarized by Apple, so macOS may block its first launch; the installation guide
explains how to open a trusted copy.

## Using the app

Open a picture, review any marked labels or twist boxes, and recognize the
diagram. Check the reconstructed crossings and components against your picture
before using its PD code.

See the [demo](../demo/README.md), [user guide](USER_GUIDE.md), and
[changelog](../CHANGELOG.md).
