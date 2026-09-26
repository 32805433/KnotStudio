# Knot Studio icon

The approved icon depicts the four-crossing figure-eight knot in blue on a pale
rounded square. `figure-eight.svg` is the vector artwork; `figure-eight.png` is
the approved 1024-pixel rendering used by `tools/make_icon.py` to generate the
macOS icon at build time. Keep these two assets in sync when changing the design.

The curve was generated within this project from
`((2 + cos(2t)) cos(3t), (2 + cos(2t)) sin(3t), sin(4t))`, projected onto the first
two coordinates with over/under gaps determined by the third. Its PD was checked
against the bundled figure-eight reference diagram. This artwork was not copied
from an external logo or icon library.
