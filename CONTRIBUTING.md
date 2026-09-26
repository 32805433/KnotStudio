# Contributing

Start with the [developer guide](docs/DEVELOPING.md) for source setup and tests,
the [architecture guide](docs/ARCHITECTURE.md) for implementation details, and
the [user guide](docs/USER_GUIDE.md) for application controls.

For a bug report, include the application version, macOS version, Mac architecture,
steps to reproduce, expected behavior, and the observed result. A small input
image or saved JSON is helpful when you have permission to share it. Include
recognition warnings and whether the issue also occurs with a bundled example.

Keep changes focused. Recognition must use image pixels rather than example
identities or reference PDs. Preserve explicit ambiguity and warnings. Geometry
edits should retain the previous valid state on rejection and remain undoable.
Add a focused regression for a behavior change and manually exercise affected
desktop controls. Update the relevant guide when a control or file format changes.

Do not commit application bundles, build environments, caches, private images,
or bulk collections. Record new examples in `examples/index.json`, add them to
[the example guide](docs/EXAMPLES.md), and include source attribution and reuse
terms in [the image credits](licenses/IMAGES.md).

Contributions are distributed under the project's GPL-3.0-or-later terms.
Retain upstream copyright and license notices and identify modifications to
derived code. See [third-party notices](docs/THIRD_PARTY.md).
