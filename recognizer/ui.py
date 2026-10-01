"""Native Python editor for recognized planar link diagrams.

Run with ``python -m recognizer.ui [image-or-diagram.json]`` from ``knots``.
Recognition runs in a worker; all Tk operations stay on the main thread.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import queue
import tempfile
import threading
import time
import traceback

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
except ImportError as exc:
    raise SystemExit("This Python installation does not include Tkinter. Use a Python build with Tk support (python.org), or install your distribution's python-tk package.") from exc

from PIL import Image, ImageDraw, ImageOps, ImageTk

from recognizer import __version__
from .render import (apply_rendered_appearance, arrow_polygon, arrows,
                     box_shapes, box_passages,
                     inherit_display_style, line_width, paths,
                     render_editable_image, render_png, render_svg, render_tikz)
from .diagram import copy_diagram
from .box_alignment import align_twist_boxes
from .display_cache import DisplayCurveCache
from .twist_boxes import is_twist_box, parse_twist_label, twist_label
from .pd_preview import PDPreviewCache, pd_preview_crossings, prepare_compact_pd_preview
from .desktop_platform import (initial_window_size, open_local_document,
                               shortcut_accelerator, shortcut_modifiers, wheel_steps)

SAVE_FORMATS = (("Editable diagram JSON", ".json"), ("PD code text", ".txt"),
                ("TikZ code", ".tex"),
                ("PNG image", ".png"), ("SVG vector image", ".svg"))


def protect_twist_box_labels(labels, boxes):
    """Never offer a twist coefficient/border for ordinary label deletion."""
    result = copy.deepcopy(labels)
    outlines = [box['bbox'] for box in (boxes or {}).get('boxes', [])]
    def overlaps(candidate):
        a = candidate['bbox']
        return any(a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1] for b in outlines)
    result['candidates'] = [c for c in result.get('candidates', []) if not overlaps(c)]
    return result


def prepare_pd_preview(diagram):
    """Prepare encoding for an already accepted graph, outside Tk's thread.

    Compact boxes can require a substantial temporary expansion. Keep the
    resulting PD (or preview-limit error) tied to this exact graph so painting,
    text refresh and mouse-up never repeat that expansion on the UI thread.
    """
    return prepare_compact_pd_preview(diagram)


def point_in_polygon(point, polygon):
    """Return whether a click is inside a highlighted complementary region."""
    inside = False
    if len(polygon) < 3:
        return False
    x, y = point
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > y) != (b[1] > y):
            boundary_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < boundary_x:
                inside = not inside
    return inside


def center_window(window, parent=None, size=None):
    """Place a window over its parent, or at the center of its screen."""
    window.update_idletasks()
    width, height = size or (window.winfo_width(), window.winfo_height())
    if width <= 1 or height <= 1:
        width, height = window.winfo_reqwidth(), window.winfo_reqheight()
    if parent is not None and parent.winfo_ismapped():
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
    else:
        x = (window.winfo_screenwidth() - width) // 2
        y = (window.winfo_screenheight() - height) // 2
    # Keep the dialog reachable if the parent is partly outside the screen.
    x = max(window.winfo_vrootx(), min(x, window.winfo_vrootx() + window.winfo_vrootwidth() - width))
    y = max(window.winfo_vrooty() + 28, min(y, window.winfo_vrooty() + window.winfo_vrootheight() - height))
    # Geometry positions include the native title bar; root coordinates refer
    # to the client area. Correct the difference once the window is mapped.
    if window.winfo_ismapped():
        x -= window.winfo_rootx() - window.winfo_x()
        y -= window.winfo_rooty() - window.winfo_y()
    window.geometry(f"{width}x{height}{x:+d}{y:+d}")


class HoverCopyButton(ttk.Button):
    """Copy affordance inside a text box, without changing its layout."""

    def __init__(self, text_widget, command):
        super().__init__(text_widget, text='⧉', width=2, padding=(3, 1),
                         command=self._copy, takefocus=True)
        self.text_widget = text_widget
        self.copy_command = command
        self.hide_job = self.reset_job = None
        for widget in (text_widget, self):
            widget.bind('<Enter>', self._show, add='+')
            widget.bind('<Leave>', self._schedule_hide, add='+')
            widget.bind('<FocusIn>', self._show, add='+')
            widget.bind('<FocusOut>', self._schedule_hide, add='+')
        self.bind('<Destroy>', self._destroy, add='+')

    def _show(self, event=None):
        if self.hide_job is not None:
            self.after_cancel(self.hide_job)
            self.hide_job = None
        self.place(relx=1, x=-5, y=5, anchor='ne')
        self.lift()

    def _schedule_hide(self, event=None):
        if self.hide_job is not None:
            self.after_cancel(self.hide_job)
        self.hide_job = self.after(80, self._hide_if_outside)

    def _hide_if_outside(self):
        self.hide_job = None
        pointed = self.winfo_containing(*self.winfo_pointerxy())
        if pointed not in (self, self.text_widget) and self.focus_get() not in (self, self.text_widget):
            self.place_forget()

    def _copy(self):
        if not self.copy_command():
            return
        self.configure(text='✓')
        if self.reset_job is not None:
            self.after_cancel(self.reset_job)
        self.reset_job = self.after(1200, self._reset_icon)

    def _reset_icon(self):
        self.reset_job = None
        self.configure(text='⧉')

    def _destroy(self, event):
        if event.widget is self:
            for job in (self.hide_job, self.reset_job):
                if job is not None:
                    self.after_cancel(job)


class ScrollableControls(ttk.Frame):
    """A vertically scrollable panel, including wheel events over its controls."""

    def __init__(self, master):
        super().__init__(master, padding=(14, 0, 0, 0))
        self.viewport = tk.Canvas(self, width=340, height=400, background="#f4f6fa",
                                  highlightthickness=0, yscrollincrement=1)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._scrollbar_command)
        self.scrollbar.pack(side="right", fill="y")
        self.viewport.pack(side="left", fill="both", expand=True)
        self.viewport.configure(yscrollcommand=self.scrollbar.set)
        self.content = ttk.Frame(self.viewport)
        self.window = self.viewport.create_window(0, 0, window=self.content, anchor="nw")
        self.content.bind("<Configure>", self._layout)
        self.viewport.bind("<Configure>", self._layout)
        self.event_tag = f"ControlsScroll:{self}"
        self.pointer_focus = None
        self.wheel_remainder = 0.
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_class(self.event_tag, sequence, self._wheel)
        self.precise_scrolling = False
        try:
            # Tk 9 uses this event for precise macOS trackpad scrolling.
            self.bind_class(self.event_tag, "<TouchpadScroll>", self._touchpad_scroll)
            self.precise_scrolling = True
        except tk.TclError:
            pass  # Tk 8 still delivers the MouseWheel events handled above.
        self.bind_class(self.event_tag, "<FocusIn>", self._show_focus)
        self.bind_class(self.event_tag, "<ButtonPress-1>", self._pointer_press)
        self.bind_class(self.event_tag, "<ButtonRelease-1>", self._pointer_release)

    def _layout(self, event=None):
        width = self.viewport.winfo_width()
        height = max(self.viewport.winfo_height(), self.content.winfo_reqheight())
        # Leave the embedded frame's height under Tk's geometry management.
        # Pinning it to the last requested height suppresses Configure events
        # when controls appear/disappear (notably on the first image import),
        # leaving the scroll range stale until the outer window is resized.
        self.viewport.itemconfigure(self.window, width=width, height=0)
        self.viewport.configure(scrollregion=(0, 0, width, height))

    def bind_controls(self):
        def attach(widget):
            widget.bindtags((self.event_tag,) + widget.bindtags())
            for child in widget.winfo_children():
                attach(child)
        attach(self)  # Include the scrollbar and empty panel space as well.

    def _scrollbar_command(self, *args):
        if args[0] == 'scroll' and args[2] == 'units':
            self.viewport.yview_scroll(24*int(args[1]), 'units')
        else:
            self.viewport.yview(*args)

    def _touchpad_scroll(self, event):
        _, dy = self.tk.splitlist(self.tk.call('tk::PreciseScrollDeltas', event.delta))
        dy = int(dy)
        if not dy:
            return
        if isinstance(event.widget, tk.Text):
            first, last = event.widget.yview()
            if (dy > 0 and first > 0) or (dy < 0 and last < 1):
                return  # Let Text's own precise scrolling consume the event.
        pixels = int(self.tk.call('tk::ScaleNum', -dy))
        self.viewport.yview_scroll(pixels, 'units')
        return 'break'

    def _wheel(self, event):
        steps = -wheel_steps(event, windowing_system=self.tk.call('tk', 'windowingsystem'),
                             precise_scrolling=self.precise_scrolling)
        if not steps:
            return
        # Let the PD/Details text scroll internally until it reaches its end.
        if isinstance(event.widget, tk.Text):
            first, last = event.widget.yview()
            if (steps < 0 and first > 0) or (steps > 0 and last < 1):
                return
        pixels = 24*steps + self.wheel_remainder
        whole_pixels = math.trunc(pixels)
        self.wheel_remainder = pixels - whole_pixels
        self.viewport.yview_scroll(whole_pixels, "units")
        return "break"

    def _show_focus(self, event):
        widget = event.widget
        # Tk also sends FocusIn to ancestors of the focused widget. Scrolling
        # those frames can jump the entire sidebar back to the top. A mouse
        # press must never scroll its target out from under the pointer before
        # release; automatic reveal is for keyboard/programmatic focus only.
        if widget is not self.focus_get() or widget is self.pointer_focus:
            return
        top = widget.winfo_rooty() - self.content.winfo_rooty()
        bottom = top + widget.winfo_height()
        visible_top = self.viewport.canvasy(0)
        visible_height = self.viewport.winfo_height()
        target = top if top < visible_top else bottom - visible_height if bottom > visible_top + visible_height else None
        if target is not None:
            # Canvas rounds to its scroll increment. Round toward revealing
            # the whole control, so a newly taller panel cannot leave its
            # focused bottom button clipped by a few pixels.
            increment = float(self.viewport.cget('yscrollincrement'))
            if increment > 0:
                rounding = math.floor if top < visible_top else math.ceil
                target = rounding(target/increment)*increment
            self.viewport.yview_moveto(max(0, target) / max(1, self.content.winfo_height()))

    def _pointer_press(self, event):
        self.pointer_focus = event.widget

    def _pointer_release(self, event):
        # Focus events from the press can still be queued behind the release.
        self.after_idle(lambda: setattr(self, "pointer_focus", None))


class TwistExpansionSideDialog(simpledialog.Dialog):
    """Choose a marked box end with one click, without a text-entry prompt."""
    def __init__(self, parent, *, full_expansion):
        self.full_expansion = full_expansion
        super().__init__(parent, title='Side for explicit twists')

    def body(self, master):
        text = ('Choose where to seek extra room if needed. Both sides keeps the expansion centered on the box.'
                if self.full_expansion else
                'Choose the end for the explicit twists. The remaining box stays at the other end.')
        ttk.Label(master, text='A and B are marked beside the selected box.\n'+text,
                  wraplength=390).pack(padx=8, pady=8)
        self.bind('<Map>', self._center_on_editor, add='+')

    def _center_on_editor(self, event):
        if event.widget is self:
            self.after_idle(lambda: center_window(self, self.parent))

    def buttonbox(self):
        buttons = ttk.Frame(self)
        buttons.pack(padx=12, pady=(0, 12))
        choices = [('Side A', 'top'), ('Side B', 'bottom')]
        if self.full_expansion:
            choices.append(('Both sides', 'both'))
        for label, side in choices:
            ttk.Button(buttons, text=label, command=lambda s=side: self.choose(s)).pack(side='left', padx=4)
        ttk.Button(buttons, text='Cancel', command=self.cancel).pack(side='left', padx=4)
        self.bind('<Escape>', self.cancel)

    def choose(self, side):
        self.result = side
        self.cancel()


class CurlSignDialog(simpledialog.Dialog):
    def body(self, master):
        self.bind("<Map>", self._center_on_editor, add="+")
        ttk.Label(master, text="Choose the new crossing's oriented sign.").pack(anchor="w", pady=(0, 8))
        self.sign = tk.IntVar(master, value=1)
        positive = ttk.Radiobutton(master, text="+   Positive curl", variable=self.sign, value=1)
        positive.pack(anchor="w", pady=4)
        ttk.Radiobutton(master, text="−   Negative curl", variable=self.sign, value=-1).pack(anchor="w", pady=4)
        ttk.Label(master, text="Next, click a point on a strand and then its desired side.",
                  wraplength=330).pack(anchor="w", pady=(8, 0))
        return positive

    def _center_on_editor(self, event):
        if event.widget is self:
            # Native window managers may adjust a dialog while mapping it.
            self.after_idle(lambda: center_window(self, self.parent))

    def apply(self):
        self.result = self.sign.get()


def _record(value):
    """Optional diagnostic objects from older/imported files may be null."""
    return value if isinstance(value, dict) else {}


def _records(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def imported_result(data):
    """Check presentation metadata before replacing the active document."""
    result = dict(data)
    for name, fallback, expected in (('status', 'needs_review', str),
                                     ('warnings', [], list), ('diagnostics', {}, dict)):
        value = result.get(name)
        if value is None:
            value = fallback
        if name == 'warnings' and isinstance(value, str):
            value = [value]
        if not isinstance(value, expected):
            raise ValueError(f'Diagram {name} must be {expected.__name__} metadata.')
        result[name] = value
    return result


def recognition_issue_points(result):
    """Normalize original-image diagnostic coordinates for display overlays."""
    if result.get("status") not in ("failed", "needs_review"):
        return []
    diagnostics = _record(result.get("diagnostics"))
    preprocessing = _record(diagnostics.get('preprocessing'))
    issues = []
    groups=[("branch",diagnostics.get("branch_points")),("endpoint",diagnostics.get("unmatched_endpoints")),
            ("board_gap",_record(diagnostics.get("board_photo")).get("review_points")),
            ("seam",diagnostics.get("seam_repairs")),
            ("gap_repair",_record(preprocessing.get('narrow_gap_repair')).get("points"))]
    groups.append(('bundle_gap',_record(diagnostics.get('bundled_crossings')).get('review_points')))
    groups.extend(('bundle_gap', repair.get('crossings'))
                  for repair in _records(preprocessing.get('bundle_contact_repairs')))
    groups.append(('gap_repair',[repair.get('joined_endpoint') for repair in _records(diagnostics.get('endpoint_hook_repairs'))]))
    groups.append(('gap_repair',[repair.get('crossing') for repair in _records(diagnostics.get('long_underpasses'))]))
    groups.append(('gap_repair',[[(r['bbox'][0]+r['bbox'][2])/2, (r['bbox'][1]+r['bbox'][3])/2]
                                for r in _records(diagnostics.get('stroke_hole_repairs'))
                                if isinstance(r.get('bbox'), (list, tuple)) and len(r['bbox']) == 4
                                and all(isinstance(v, (int, float)) for v in r['bbox'])]))
    for kind, points in groups:
        for item in points if isinstance(points, (list, tuple)) else []:
            point = item.get("point") if isinstance(item, dict) else item
            try:
                if len(point) != 2:
                    continue
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                issues.append((kind, (x, y)))
    return issues


class DiagramEditor:
    def __init__(self, root):
        self.root = root
        root.withdraw()
        root.title("Knot Studio · Image to PD")
        initial_size = initial_window_size(root.winfo_screenwidth(), root.winfo_screenheight())
        self.initial_size = initial_size
        root.geometry(f"{initial_size[0]}x{initial_size[1]}")
        root.minsize(min(920, initial_size[0]), min(620, initial_size[1]))
        self.diagram = None
        self.raster_image = None
        self.raster_changed = False
        self.label_detection = None
        self.box_detection = None
        self.source_path = None
        self.result = {}
        self.selected_crossing = None
        self.selected_component = None
        self.selected_box = None
        self.box_combine_first = None
        self.box_edit_pending = None
        self.history, self.future = [], []
        self.events = queue.Queue()
        self.motion_events = queue.Queue()
        self.motion_serial = 0
        self.task_id = 0
        self.busy = False
        self.scale, self.offset = 1.0, (0.0, 0.0)
        self.fit_pending = True
        self.drag = None
        self.drag_timer = None
        self.motion_poll_timer = None
        self.curl_add = None
        self.monogons = []
        self.bigons = []
        self.pan = None
        self.photo = None
        self.display_paths = None
        self._display_curve_cache = DisplayCurveCache()
        self._display_actions = []
        self._display_fit_due = 0.0
        self.layer = tk.StringVar(value="diagram")
        self.tool = tk.StringVar(value="Select")
        self.colored = tk.BooleanVar(value=True)
        self.oriented = tk.BooleanVar(value=False)
        self.smooth_display = tk.BooleanVar(value=False)
        self.show_issues = tk.BooleanVar(value=True)
        self.brush_size = tk.DoubleVar(value=5)
        self.pencil_size = self.brush_size
        self.eraser_size = tk.DoubleVar(value=25)
        self.move_radius = tk.DoubleVar(value=90)
        self.energy_speed = tk.DoubleVar(value=4.0)
        self.energy_full_screen = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Open an image to recognize a link diagram, or start a sketch.")
        self.counts = tk.StringVar(value="No diagram loaded")
        self.confidence = tk.StringVar(value="READY")
        self.zoom_text = tk.StringVar(value="100%")
        self._build()
        self.layer.trace_add("write", lambda *_: self._refresh_conversion_buttons())
        self.tool.trace_add("write", lambda *_: self._refresh_brush())
        self._bindings()
        center_window(root, size=initial_size)
        def center_after_mapping(event):
            if event.widget is root:
                root.unbind("<Map>", startup_binding)
                root.after_idle(lambda: center_window(root))
        startup_binding = root.bind("<Map>", center_after_mapping, add="+")
        root.deiconify()
        root.after(80, self._poll)

    def _build(self):
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f4f6fa")
        style.configure("TLabel", background="#f4f6fa", foreground="#1f2c40", font=("Helvetica", 12))
        style.configure("TButton", font=("Helvetica", 11), padding=(10, 7))
        style.configure("Title.TLabel", font=("Helvetica", 20, "bold"))
        style.configure("Caption.TLabel", font=("Helvetica", 10), foreground="#63718a")
        style.configure("Accent.TButton", background="#245fd4", foreground="white")
        style.configure("TLabelframe", background="#f4f6fa")
        style.configure("TLabelframe.Label", background="#f4f6fa", foreground="#63718a", font=("Helvetica", 10, "bold"))
        # Keep the window manager's preferred size independent of changing
        # labels/status text. Actual allocation still follows manual resizing
        # and native fullscreen, and the child layout fills that allocation.
        root_frame = ttk.Frame(self.root, padding=16,
                               width=self.initial_size[0], height=self.initial_size[1])
        root_frame.pack_propagate(False)
        root_frame.pack(fill="both", expand=True)
        header = ttk.Frame(root_frame)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="Knot Studio", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="IMAGE  →  DIAGRAM  →  PD", style="Caption.TLabel").pack(side="left", padx=18)
        ttk.Button(header, text="Save…", command=self.save_as).pack(side="right")
        ttk.Button(header, text="Open…", command=self.open_file, style="Accent.TButton").pack(side="right", padx=(8, 0))
        ttk.Button(header, text="New sketch", command=self.new_sketch).pack(side="right", padx=8)

        main = ttk.Panedwindow(root_frame, orient="horizontal")
        main.pack(fill="both", expand=True)
        center = ttk.Frame(main)
        main.add(center, weight=4)
        self.canvas = tk.Canvas(center, background="#e5eaf2", highlightthickness=1, highlightbackground="#d4dce8", cursor="arrow")
        self.canvas.pack(fill="both", expand=True)
        lower = ttk.Frame(center)
        lower.pack(fill="x", pady=(7, 0))
        zoom_controls = ttk.Frame(lower)
        zoom_controls.pack(side="right", padx=(8, 0))
        ttk.Button(zoom_controls, text="−", command=lambda: self.zoom(0.8), width=2, padding=(4, 1)).pack(side="left")
        ttk.Label(zoom_controls, textvariable=self.zoom_text, width=6, anchor="center").pack(side="left")
        ttk.Button(zoom_controls, text="+", command=lambda: self.zoom(1.25), width=2, padding=(4, 1)).pack(side="left")
        ttk.Button(zoom_controls, text="Fit", command=self.fit_view, width=4, padding=(4, 1)).pack(side="left", padx=(6, 0))
        ttk.Label(lower, textvariable=self.counts, style="Caption.TLabel", anchor="w").pack(side="left", fill="x", expand=True)
        ttk.Label(lower, text="Right/Ctrl-drag: pan", style="Caption.TLabel",
                  anchor="e").pack(side="right", padx=(8, 0))

        self.controls = ScrollableControls(main)
        main.add(self.controls, weight=1)
        sidebar = self.controls.content
        self.badge = tk.Label(sidebar, textvariable=self.confidence, anchor="w", padx=10, pady=8, background="#e6edf7", foreground="#345072", font=("Helvetica", 10, "bold"))
        self.badge.pack(fill="x", pady=(0, 10))
        tools_frame = ttk.LabelFrame(sidebar, text="EDIT DIAGRAM", padding=10)
        tools_frame.pack(fill="x", pady=(0, 10))
        for name, label in (("Select", "Select / move"), ("Switch crossing", "Switch crossing  ·  X"), ("Reverse component", "Reverse orientation  ·  R"), ("Delete component", "Delete component  ·  ⌫")):
            ttk.Radiobutton(tools_frame, text=label, variable=self.tool, value=name, command=self._tool_changed).pack(anchor="w", pady=3)
        radius = self.move_radius_row = ttk.Frame(tools_frame)
        ttk.Label(radius, text="Move radius", style="Caption.TLabel").pack(side="left", padx=(0, 6))
        ttk.Scale(radius, from_=30, to=220, variable=self.move_radius).pack(side="left", fill="x", expand=True)
        curl_buttons = self.move_radius_after = ttk.Frame(tools_frame)
        curl_buttons.pack(fill="x", pady=(0, 5))
        ttk.Button(curl_buttons, text="Add curl…", command=self.start_add_curl, padding=(6, 4)).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(curl_buttons, text="Simplify", command=self.start_simplify, padding=(6, 4)).pack(side="left", expand=True, fill="x", padx=(3, 0))
        energy_buttons = ttk.Frame(tools_frame)
        energy_buttons.pack(fill="x", pady=(0, 5))
        ttk.Button(energy_buttons, text="Minimize energy…", command=self.start_energy, padding=(6, 4)).pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.stop_button = ttk.Button(energy_buttons, text="Stop", command=self.stop_energy, padding=(6, 4), width=5)
        self.stop_button.pack(side="left")
        self.stop_button.bind("<ButtonPress-1>", self._stop_energy_press)
        ttk.Checkbutton(tools_frame, text="Full screen region (visible diagram area)",
                        variable=self.energy_full_screen).pack(anchor="w", pady=(0, 5))
        appearance_row = ttk.Frame(tools_frame)
        appearance_row.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(appearance_row, text="Color components", variable=self.colored, command=self._appearance_changed).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(appearance_row, text="Show orientations", variable=self.oriented, command=self._appearance_changed).pack(side="left")
        ttk.Checkbutton(appearance_row, text="Smooth display", variable=self.smooth_display,
                        command=self._appearance_changed).pack(side="left", padx=(8, 0))
        ttk.Label(tools_frame, text="Gray arrows: direction not detected in the image.",
                  foreground="#657184").pack(anchor="w", pady=(3, 0))
        buttons = ttk.Frame(tools_frame)
        buttons.pack(fill="x", pady=(10, 0))
        self.undo_button = ttk.Button(buttons, text="Undo", command=self.undo)
        self.undo_button.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.redo_button = ttk.Button(buttons, text="Redo", command=self.redo)
        self.redo_button.pack(side="left", expand=True, fill="x", padx=(3, 0))

        sketch = ttk.LabelFrame(sidebar, text="DRAW / RECOGNIZE", padding=10)
        sketch.pack(fill="x", pady=(0, 10))
        draw_buttons = ttk.Frame(sketch)
        draw_buttons.pack(fill="x")
        for name in ("Pencil", "Eraser", "Block eraser"):
            ttk.Radiobutton(draw_buttons, text=name, variable=self.tool, value=name, command=self._tool_changed).pack(side="left", padx=(0, 10))
        brush = self.brush_width_row = ttk.Frame(sketch)
        self.brush_label = ttk.Label(brush, text="Pencil width", style="Caption.TLabel")
        self.brush_label.pack(side="left", padx=(0, 6))
        self.brush_scale = ttk.Scale(brush, from_=2, to=28, variable=self.brush_size)
        self.brush_scale.pack(side="left", fill="x", expand=True)
        self.brush_width_after = ttk.Label(sketch, text="Pencil: over · Shift + Pencil: under", style="Caption.TLabel")
        self.brush_width_after.pack(anchor="w", pady=(0, 7))
        conversion_buttons = ttk.Frame(sketch)
        conversion_buttons.pack(fill="x", pady=(0, 7))
        conversion_buttons.columnconfigure((0, 1), weight=1, uniform="conversion")
        self.convert_button = ttk.Button(conversion_buttons, text="Convert to drawing", command=self.diagram_to_raster, padding=(6, 7))
        self.convert_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.recognize_button = ttk.Button(conversion_buttons, text="Recognize drawing", command=self.recognize_raster, style="Accent.TButton", padding=(6, 7))
        self.recognize_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        label_buttons = ttk.Frame(sketch)
        label_buttons.pack(fill="x", pady=(0, 5))
        label_buttons.columnconfigure((0, 1), weight=1, uniform="labels")
        self.find_labels_button = ttk.Button(label_buttons, text="Mark possible labels", command=self.find_labels, padding=(6, 7))
        self.find_labels_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.delete_labels_button = ttk.Button(label_buttons, text="Delete all marked labels", command=self.delete_marked_labels, padding=(6, 7))
        self.delete_labels_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.label_count = tk.StringVar(value="No labels marked")
        ttk.Label(sketch, textvariable=self.label_count, style="Caption.TLabel").pack(anchor="w")

        box_section = ttk.Frame(sidebar)
        box_section.pack(fill='x', pady=(0, 10))
        box_header = ttk.Frame(box_section)
        box_header.pack(fill='x')
        self.twist_boxes_expanded = False
        self.twist_boxes_toggle = ttk.Button(box_header, text='▸ Twist boxes (experimental)',
                                             command=self._toggle_twist_boxes)
        self.twist_boxes_toggle.pack(side='left', fill='x', expand=True)
        self.box_count = tk.StringVar(value='No twist boxes')
        ttk.Label(box_header, textvariable=self.box_count, style='Caption.TLabel').pack(side='right', padx=(8, 0))
        boxes = self.twist_boxes_body = ttk.Frame(box_section, padding=10)
        self.edit_box_button = ttk.Button(boxes, text='Edit selected twist value…', command=self.edit_twist_box)
        self.edit_box_button.pack(fill='x', pady=(0, 5))
        # Both modes share value editing; the remaining controls belong only
        # to source-image review or only to the recognized diagram.
        boxes = self.twist_boxes_drawing_controls = ttk.Frame(self.twist_boxes_body)
        self.review_boxes_button = ttk.Button(boxes, text='Review twist boxes', command=self.review_twist_boxes)
        self.review_boxes_button.pack(fill='x', pady=(0, 5))
        self.fill_box_values_button = ttk.Button(boxes, text='Fill missing twist values…',
                                                 command=self.fill_missing_twist_values)
        self.fill_box_values_button.pack(fill='x', pady=(0, 5))
        self.mark_box_button = ttk.Button(boxes, text='Mark a missed twist box…', command=self.mark_twist_box)
        self.mark_box_button.pack(fill='x', pady=(5, 0))
        self.unmark_box_button = ttk.Button(boxes, text='Unmark selected box', command=self.unmark_twist_box)
        self.unmark_box_button.pack(fill='x', pady=(5, 0))
        boxes = self.twist_boxes_diagram_controls = ttk.Frame(self.twist_boxes_body)
        self.add_twists_button = ttk.Button(boxes, text='Add twists…', command=self.start_add_twists)
        self.add_twists_button.pack(fill='x', pady=(0, 5))
        self.combine_boxes_button = ttk.Button(boxes, text='Combine adjacent boxes…', command=self.start_combine_boxes)
        self.combine_boxes_button.pack(fill='x', pady=(0, 5))
        self.delete_zero_box_button = ttk.Button(boxes, text='Delete selected zero box', command=self.delete_selected_zero_box)
        self.delete_zero_box_button.pack(fill='x', pady=(0, 5))
        self.expand_box_button = ttk.Button(boxes, text='Expand selected box…', command=self.expand_selected_box)
        self.expand_box_button.pack(fill='x', pady=(0, 5))
        self.compress_box_button = ttk.Button(boxes, text='Box selected tangle…', command=self.start_box_tangle)
        self.compress_box_button.pack(fill='x')
        ttk.Label(boxes, text='Blue boxes store full twists: 1/2 is one half twist.\nClick a box to review its number. In Select / move, drag a box edge to resize it. Shift-drag a strand to pass below a box; normal dragging passes above.',
                  style='Caption.TLabel', wraplength=290).pack(anchor='w', pady=(5, 0))

        notebook = self.code_notebook = ttk.Notebook(sidebar)
        notebook.pack(fill="both", expand=True)
        pd_frame = ttk.Frame(notebook, padding=8)
        tikz_frame = self.tikz_frame = ttk.Frame(notebook, padding=8)
        details_frame = ttk.Frame(notebook, padding=8)
        notebook.add(pd_frame, text="PD code")
        notebook.add(tikz_frame, text="TikZ code")
        notebook.add(details_frame, text="Details")
        self.pd_text = tk.Text(pd_frame, wrap="word", height=8, width=29, padx=8, pady=8, font=("Menlo", 11), background="white", foreground="#20344f", borderwidth=0, highlightthickness=1, highlightbackground="#dae1ec")
        self.pd_text.pack(fill="both", expand=True)
        self.pd_text.tag_configure('copy_icon_margin', rmargin=32)
        self.pd_copy_button = HoverCopyButton(self.pd_text, self.copy_pd)
        tikz_frame.columnconfigure(0, weight=1)
        tikz_frame.rowconfigure(0, weight=1)
        self.tikz_text = tk.Text(tikz_frame, wrap="none", height=8, width=29, padx=8, pady=8,
                                font=("Menlo", 11), background="white", foreground="#20344f",
                                borderwidth=0, highlightthickness=1, highlightbackground="#dae1ec")
        self.tikz_text.grid(row=0, column=0, sticky="nsew")
        tikz_y = ttk.Scrollbar(tikz_frame, orient="vertical", command=self.tikz_text.yview)
        tikz_y.grid(row=0, column=1, sticky="ns")
        tikz_x = ttk.Scrollbar(tikz_frame, orient="horizontal", command=self.tikz_text.xview)
        tikz_x.grid(row=1, column=0, sticky="ew")
        self.tikz_text.configure(yscrollcommand=tikz_y.set, xscrollcommand=tikz_x.set)
        self.tikz_copy_button = HoverCopyButton(self.tikz_text, self.copy_tikz)
        ttk.Label(tikz_frame, text="Use with \\usepackage{tikz}. Save… → TikZ code exports a .tex file.",
                  style="Caption.TLabel", wraplength=260).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self._put_text(self.tikz_text, "A recognized diagram's TikZ code appears here.")
        notebook.bind("<<NotebookTabChanged>>", self._queue_tikz_refresh)
        self.details_text = tk.Text(details_frame, wrap="word", height=8, width=29, padx=8, pady=8, font=("Helvetica", 11), background="white", foreground="#40516b", borderwidth=0)
        self.details_text.pack(fill="both", expand=True)
        self.controls.bind_controls()
        # A fixed-height, wrapping status area prevents messages from resizing
        # the canvas/sidebar during a click. Long errors remain scrollable and
        # selectable rather than forcing the application to request more room.
        self.status_text = tk.Text(root_frame, height=3, width=1, wrap="word",
                                   font=("Helvetica", 10), background="#f4f6fa",
                                   foreground="#63718a", borderwidth=0,
                                   highlightthickness=0, takefocus=False)
        self.status_text.pack(fill="x", pady=(13, 0))
        self.status.trace_add("write", lambda *_: self._put_text(self.status_text, self.status.get()))
        self._put_text(self.status_text, self.status.get())
        self._menu()
        self._refresh_text()

    def _toggle_twist_boxes(self):
        self.twist_boxes_expanded = not self.twist_boxes_expanded
        if self.twist_boxes_expanded:
            self.twist_boxes_body.pack(fill='x')
        else:
            self.twist_boxes_body.pack_forget()
        self.twist_boxes_toggle.configure(text='▾ Twist boxes (experimental)' if self.twist_boxes_expanded else '▸ Twist boxes (experimental)')

    def _menu(self):
        windowing_system = self.root.tk.call('tk', 'windowingsystem')
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        for label, command, key in (("Open…", self.open_file, 'O'), ("New sketch", self.new_sketch, None), ("Save…", self.save_as, 'S')):
            file_menu.add_command(label=label, command=command,
                                  accelerator=shortcut_accelerator(key, windowing_system) if key else '')
        menu.add_cascade(label="File", menu=file_menu)
        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo,
                              accelerator=shortcut_accelerator('Z', windowing_system))
        edit_menu.add_command(label="Redo", command=self.redo,
                              accelerator=shortcut_accelerator('Z', windowing_system, shift=True))
        edit_menu.add_separator()
        edit_menu.add_command(label="Add curl (Reidemeister I)…", command=self.start_add_curl)
        edit_menu.add_command(label="Simplify", command=self.start_simplify)
        menu.add_cascade(label="Edit", menu=edit_menu)
        view_menu = tk.Menu(menu, tearoff=False)
        view_menu.add_checkbutton(label="Show recognition issues", variable=self.show_issues, command=self.redraw)
        menu.add_cascade(label="View", menu=view_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label='User guide', command=self.open_user_guide)
        help_menu.add_command(label='About Knot Studio', command=lambda: messagebox.showinfo(
            'Knot Studio', f'Knot Studio {__version__}\nLocal knot and link recognition and editing.\n\n'
            'Distributed under GPL-3.0-or-later.\nIncludes work adapted from Kyle Miller’s KnotFolio.\n'
            'Layout minimization is inspired by Dirk Schütz’s KnotJob.\n\n'
            'Images are processed on this computer. No account or internet connection is required.\n'
            'Recognition results need visual review.', parent=self.root))
        help_menu.add_command(label="Controls and limitations", command=lambda: messagebox.showinfo("Knot Studio", "Open an image to recognize it. Select a crossing or component, then press X to switch a crossing or R to reverse orientation. Delete removes the selected component.\n\nSelect / move: click to select, or drag a strand to deform a nearby part smoothly. Move radius controls the size of that neighborhood. Hold Shift as a Reidemeister II pair forms to create undercrossings; otherwise the moving strand passes over. You can change Shift during a drag; existing crossings keep their types. The graph and PD update through accepted Reidemeister II and III moves. A rejected position leaves the last accepted diagram in place; try a smaller move or a different radius. Dragging cannot add or remove a curl.\n\nAdd curl asks for a +/− crossing sign, a point away from crossings, and then a side. Simplify shades removable curls in green and removable bigons in purple. Click a shaded region to remove its one or two crossings. Escape cancels a gesture. Undo restores an entire edit.\n\nMinimize energy asks you to circle the tangle to relax, keeping its boundary fixed. Check Full screen region to use the visible diagram area immediately; Fit first to include the whole diagram. Stop keeps the displayed layout; Escape restores its start.\n\nPencil and Eraser edit the active drawing. Use Convert to drawing to draw on a clean diagram. Drawing normally passes over existing strands; hold Shift while drawing to pass underneath. Crossing gaps are inserted automatically. Then choose Recognize drawing.\n\nNeeds review means recognition is uncertain. A Drawing changed badge means you need to recognize the drawing to generate its diagram and PD. Inspect the diagram before using its PD code."))
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu)

    def open_user_guide(self):
        from .resources import resource_root
        folder = resource_root()/'docs'
        path = folder/'USER_GUIDE.html'
        if not path.exists():
            path = folder/'USER_GUIDE.md'
        try:
            open_local_document(path)
        except OSError as exc:
            messagebox.showerror('Cannot open user guide', str(exc))

    def _bindings(self):
        self.canvas.bind("<Configure>", lambda event: self.fit_view() if self.fit_pending else self.redraw())
        self.canvas.bind("<Button-1>", self._mouse_down)
        self.canvas.bind("<B1-Motion>", self._mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_up)
        self.canvas.bind("<Motion>", self._hover_box_side)
        # macOS Tk can report secondary clicks as button 2; support both
        # mappings, plus Control-drag for trackpads without secondary drag.
        for button in (2, 3):
            self.canvas.bind(f"<Button-{button}>", self._start_pan)
            self.canvas.bind(f"<B{button}-Motion>", self._pan)
            self.canvas.bind(f"<ButtonRelease-{button}>", self._end_pan)
        self.canvas.bind("<Control-Button-1>", self._start_pan)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", self._wheel)
        self.canvas.bind("<Button-5>", self._wheel)
        self.canvas.bind("<Key-x>", lambda event: self.switch_selected())
        self.canvas.bind("<Key-r>", lambda event: self.reverse_selected())
        self.canvas.bind("<Delete>", lambda event: self.delete_selected())
        self.canvas.bind("<BackSpace>", lambda event: self.delete_selected())
        self.root.bind("<Escape>", self.cancel_gesture)
        for key in ("Shift_L", "Shift_R"):
            self.root.bind(f"<KeyPress-{key}>", lambda event: self._shift_during_drag(event, True), add="+")
            self.root.bind(f"<KeyRelease-{key}>", lambda event: self._shift_during_drag(event, False), add="+")
        windowing_system = self.root.tk.call('tk', 'windowingsystem')
        for modifier in shortcut_modifiers(windowing_system):
            for key, command in (('o', self.open_file), ('s', self.save_as),
                                 ('z', self.undo), ('Shift-Z', self.redo)):
                def invoke(event, callback=command):
                    callback()
                    return 'break'
                self.root.bind(f'<{modifier}-{key}>', invoke)
        if windowing_system != 'aqua':
            self.root.bind('<Control-y>', lambda event: (self.redo(), 'break')[1])

    def _snapshot(self):
        tool = self.tool.get()
        # The result normally refers to the same diagram. One deepcopy shares
        # its memo, avoiding a second full geometry copy at every mouse-down.
        memo = {}
        if self.diagram is not None:
            copy_diagram(self.diagram, memo)
        state = copy.deepcopy({"diagram": self.diagram, "result": self.result}, memo)
        prepared = getattr(self, '_prepared_box_pd', None)
        if prepared is not None and prepared[0] is self.diagram:
            state['prepared_pd'] = copy.deepcopy(prepared[1:], memo)
        state.update(raster_image=self.raster_image, raster_changed=self.raster_changed,
                     label_detection=copy.deepcopy(getattr(self, 'label_detection', None)),
                     box_detection=copy.deepcopy(getattr(self, 'box_detection', None)),
                     layer=self.layer.get(), tool="Select" if tool in ("Add curl", "Remove curl", "Remove bigon", "Simplify", "Energy region", "Add twists", "Combine boxes", "Box tangle") else tool)
        return state

    def _remember(self):
        self.history.append(self._snapshot())
        self.history = self.history[-20:]
        self.future.clear()

    def _restore(self, state):
        self._clear_interaction()
        memo = {}
        if state["diagram"] is not None:
            copy_diagram(state["diagram"], memo)
        model_state = copy.deepcopy({"diagram": state["diagram"], "result": state["result"]}, memo)
        self.diagram = model_state["diagram"]
        self.raster_image = state["raster_image"]
        self.raster_changed = state.get("raster_changed", False)
        self.label_detection = copy.deepcopy(state.get("label_detection"))
        self.box_detection = copy.deepcopy(state.get('box_detection'))
        self.selected_box = None
        self.result = model_state["result"]
        prepared = state.get('prepared_pd')
        self._prepared_box_pd = ((self.diagram, *copy.deepcopy(prepared, memo))
                                 if prepared is not None else None)
        self.layer.set(state.get("layer", "diagram" if self.diagram else "source"))
        self.tool.set(state.get("tool", "Select"))
        self.selected_crossing = self.selected_component = None
        self._refresh_text()
        self.redraw()

    def undo(self):
        if self.busy:
            self.status.set("Recognition is running; wait before undoing.")
            return "break"
        if not self.drag and self.tool.get() == 'Simplify' and getattr(self, 'simplify_continuation', False):
            self._clear_interaction()
            self.canvas.configure(cursor='arrow')
        if self.drag or self.curl_add or self.tool.get() in ("Remove curl", "Remove bigon", "Simplify"):
            return self.cancel_gesture()
        if self.history:
            self.future.append(self._snapshot())
            self._restore(self.history.pop())
            self.status.set("Undid the last change.")
        return "break"

    def redo(self):
        if self.busy:
            return "break"
        if self.drag or self.curl_add or self.tool.get() in ("Remove curl", "Remove bigon", "Simplify"):
            return self.cancel_gesture()
        if self.future:
            self.history.append(self._snapshot())
            self._restore(self.future.pop())
            self.status.set("Redid the change.")
        return "break"

    def _reset(self):
        self._clear_interaction()
        self.task_id += 1
        self.busy = False
        self.diagram = None
        self.display_paths = None
        self.raster_changed = False
        self.label_detection = None
        self.box_detection = None
        self.selected_box = None
        self.result = {}
        self.history.clear()
        self.future.clear()
        self.selected_crossing = self.selected_component = None

    def open_file(self, path=None):
        if self._motion_pending():
            return
        if path is None:
            images = '*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp'
            path = filedialog.askopenfilename(parent=self.root, title="Open image or diagram",
                filetypes=[("All supported files", images+' *.json'),
                           ("Images", images), ("Editable diagram JSON", '*.json'), ("All files", '*')])
        if not path:
            return
        if Path(path).suffix.lower() == '.json':
            self.open_json(path)
        else:
            self.open_image(path)

    def open_image(self, path=None):
        if self._motion_pending():
            return
        if path is None:
            path = filedialog.askopenfilename(title="Open a knot or link image", filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp"), ("All files", "*")])
        if not path:
            return
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGBA")
                background = Image.new("RGBA", image.size, "white")
                image = Image.alpha_composite(background, image).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Cannot open image", str(exc))
            return
        self._reset()
        self.raster_image = image
        self.source_path = str(Path(path).resolve())
        self.root.title(f"{Path(path).name} · Knot Studio")
        self.layer.set("source")
        self.fit_view()
        self._refresh_text()
        self.find_labels(recognize_if_empty=True)

    def find_labels(self, recognize_if_empty=False):
        """Propose source labels on a worker; discovery never changes pixels."""
        if self.busy or self._motion_pending() or self.raster_image is None:
            return
        self.cancel_gesture(quiet=True)
        self.layer.set('source')
        self.tool.set('Select')
        image = self.raster_image.copy()
        # A label rescan must not erase box values/attachments explicitly
        # reviewed by the user. Reuse them only for these exact source pixels.
        reviewed_boxes = copy.deepcopy(self.box_detection)
        self.task_id += 1
        task_id = self.task_id
        self.busy = True
        self._refresh_conversion_buttons()
        self.confidence.set('CHECKING LABELS…')
        self.status.set('Finding possible labels for review. The image remains unchanged.')
        self.redraw()

        def worker():
            try:
                import numpy as np
                from .label_review import detect_labels
                from .box_detection import detect_twist_boxes
                from .box_recognition import checked_detection
                detection = detect_labels(np.asarray(image))
                try:
                    boxes = checked_detection(np.asarray(image), reviewed_boxes)
                except ValueError:
                    boxes = detect_twist_boxes(np.asarray(image))
                detection = protect_twist_box_labels(detection, boxes)
                self.events.put((task_id, 'labels', (detection, recognize_if_empty, boxes)))
            except Exception:
                self.events.put((task_id, 'label_error', traceback.format_exc()))
        threading.Thread(target=worker, daemon=True, name='label-detection').start()

    def _label_candidates(self):
        return (getattr(self, 'label_detection', None) or {}).get('candidates', [])

    def _box_candidates(self):
        return (getattr(self, 'box_detection', None) or {}).get('boxes', [])

    def _has_twist_boxes(self):
        return bool(self.diagram and any(is_twist_box(v) for v in self.diagram.get('crossings', [])))

    def review_twist_boxes(self):
        if self.busy or self._motion_pending():
            return
        if self.layer.get() == 'source':
            if not self._box_candidates():
                self.find_labels()
                return
            self.tool.set('Review twist boxes')
            self.status.set('Click a blue box to review its full-twist number. Other labels remain separate.')
        else:
            self.tool.set('Select')
            self.status.set('Click a twist box, then Edit selected twist value. Changing the number can change the link.')
        self.redraw()

    def _source_box_at(self, point):
        matches = [b for b in self._box_candidates()
                   if b['bbox'][0] <= point[0] <= b['bbox'][2] and b['bbox'][1] <= point[1] <= b['bbox'][3]]
        return min(matches, key=lambda b: (b['bbox'][2]-b['bbox'][0])*(b['bbox'][3]-b['bbox'][1]), default=None)

    def _missing_twist_values(self):
        return [b for b in self._box_candidates() if b.get('half_twists') is None]

    def _offer_missing_twist_values(self):
        """Show explicit unknown values without a guessed diagram or modal prompt."""
        missing = self._missing_twist_values()
        if not missing or self.raster_image is None:
            return False
        self.layer.set('source')
        self.tool.set('Review twist boxes')
        self.selected_box = missing[0]['id']
        if hasattr(self, 'twist_boxes_body') and not self.twist_boxes_expanded:
            self._toggle_twist_boxes()
        self.status.set(f'{len(missing)} twist value(s) need your input. Click an orange box, or Fill missing twist values. '
                        'No value is guessed; a complete PD waits for these values.')
        return True

    def fill_missing_twist_values(self):
        """Collect exact user values, retaining completed edits if cancelled."""
        if self.busy or self._motion_pending() or self.layer.get() != 'source':
            return
        missing = self._missing_twist_values()
        if not missing:
            return
        for box in missing:
            self.selected_box = box['id']
            self.redraw()
            if not self.edit_twist_box(box['id']):
                self._offer_missing_twist_values()
                self._refresh_text()
                self.redraw()
                return
        self.recognize_raster()

    def edit_twist_box(self, box_id=None):
        if self.busy or self._motion_pending():
            return
        source = self.layer.get() == 'source'
        box_id = box_id if box_id is not None else getattr(self, 'selected_box', None)
        records = self._box_candidates() if source else [v for v in (self.diagram or {}).get('crossings', []) if is_twist_box(v)]
        chosen = next((b for b in records if b['id'] == box_id), None)
        if chosen is None:
            self.review_twist_boxes()
            return
        value = chosen.get('numeric_label') if source else twist_label(chosen)
        number = next(i for i, b in enumerate(records, 1) if b['id'] == box_id)
        answer = simpledialog.askstring(f'Twist box {number} · full twists',
                f'Enter the value for highlighted box {number}.\nAn integer or half-integer number of full twists:\n1, -2, 1/2, -3/2. Cancel leaves it unchanged.\nChanging this value can change the link and its components.',
                initialvalue=value or '', parent=self.root)
        if answer is None:
            return
        try:
            count = parse_twist_label(answer)
        except ValueError as exc:
            messagebox.showerror('Invalid twist value', str(exc), parent=self.root)
            return
        self._remember()
        if source:
            self.box_detection = copy.deepcopy(self.box_detection)
            chosen = next(b for b in self._box_candidates() if b['id'] == box_id)
            chosen.update(numeric_label=twist_label(count), half_twists=count, number_reviewed=True)
            # A number correction cannot repair missing or ambiguous attachments.
            self.raster_changed = self.diagram is not None
            self.status.set('Twist value recorded. Recognition will use this reviewed value; the source pixels are unchanged.')
        else:
            from .diagram import assign_components
            changed = copy_diagram(self.diagram)
            chosen = next(v for v in changed['crossings'] if v['id'] == box_id)
            chosen.update(half_twists=count, label=twist_label(count))
            try:
                assign_components(changed)
                changed = align_twist_boxes(changed)
                self._edited_diagram(changed)
            except Exception as exc:
                self._restore(self.history.pop())
                messagebox.showerror('Cannot edit twist box', str(exc), parent=self.root)
                return
            self.status.set('Updated the twist value and component connectivity. The link may have changed. Undo restores it.')
        self.selected_box = box_id
        self._refresh_text()
        self.redraw()
        return True

    def _box_diagram_action_ready(self):
        if self.busy or self._motion_pending():
            return False
        if self.layer.get() != 'diagram' or self.diagram is None:
            self.status.set('Recognize the drawing, or open a diagram JSON, before editing twist boxes.')
            return False
        return True

    def _selected_diagram_box(self):
        return next((b for b in (self.diagram or {}).get('crossings', [])
                     if is_twist_box(b) and b['id'] == getattr(self, 'selected_box', None)), None)

    def start_add_twists(self):
        if not self._box_diagram_action_ready():
            return
        self.cancel_gesture(quiet=True)
        self.selected_box = self.selected_crossing = self.selected_component = None
        self.tool.set('Add twists')
        self.canvas.configure(cursor='crosshair')
        self.status.set('Draw a straight slicing segment across parallel strands, away from crossings and boxes. Leave a small clear collar on each side for the box; release to enter its number. Escape cancels.')
        self.redraw()

    def _finish_twist_slice(self, segment):
        if math.dist(*segment)*self.scale < 8:
            self.status.set('Draw a longer segment across at least two parallel strands.')
            self.redraw()
            return
        # Keep the slice visible while the modal value prompt is open.
        self.drag = {'kind': 'twist_slice_preview', 'anchor': segment[0], 'last': segment[1]}
        self.redraw()
        answer = simpledialog.askstring('Add twists · full twists',
                    'Number of full twists to insert on all the sliced strands:\n'
                    'Enter an integer or half-integer, for example 1, -2, 1/2 or -3/2.\n'
                    'This changes the link. The slice must avoid crossings and leave room for a box.',
                    parent=self.root)
        self.drag = None
        if answer is None:
            self.redraw()
            return
        try:
            label = twist_label(parse_twist_label(answer))
        except ValueError as exc:
            messagebox.showerror('Invalid twist value', str(exc), parent=self.root)
            self.redraw()
            return
        self._run_box_edit('insert', (segment, label), 'Adding twists',
                           f'Inserted {label} full twist(s) on the sliced bundle. The link may have changed. Undo restores it.')

    def start_combine_boxes(self):
        if not self._box_diagram_action_ready():
            return
        self.cancel_gesture(quiet=True)
        self.box_combine_first = None
        self.selected_box = self.selected_crossing = self.selected_component = None
        self.tool.set('Combine boxes')
        self.status.set('Click the first twist box, then the adjacent second box on the same bundle. Their numbers will be added. Escape cancels.')
        self.redraw()

    def _combine_box_click(self, point):
        box = next((b for b in box_shapes(self.diagram) if point_in_polygon(point, b['corners'])), None)
        if box is None:
            self.status.set('Click inside a twist box. The two boxes must be joined directly by the same ordered bundle.')
            return
        first = getattr(self, 'box_combine_first', None)
        if first is None:
            self.box_combine_first = self.selected_box = box['id']
            self.status.set('First box selected. Click the adjacent second box on the same bundle; Escape cancels.')
            self.redraw()
            return
        if first == box['id']:
            self.status.set('Select a different adjacent box, or press Escape to cancel.')
            return
        self._run_box_edit('combine', (first, box['id']), 'Combining boxes',
                           'Combined the adjacent boxes by adding their full-twist numbers. Undo restores both boxes.')

    def delete_selected_zero_box(self):
        if not self._box_diagram_action_ready():
            return
        box = self._selected_diagram_box()
        if box is None:
            self.status.set('Click a zero-labeled box in Select mode first.')
            return
        if box['half_twists'] != 0:
            self.status.set('Only a zero-labeled box can be deleted. Use Expand selected box for a nonzero box.')
            return
        self._run_box_edit('delete_zero', (box['id'],), 'Deleting zero box',
                           'Deleted the zero box and joined its untwisted strands. Undo restores the box.')

    def expand_selected_box(self):
        if not self._box_diagram_action_ready():
            return
        box = self._selected_diagram_box()
        if box is None:
            self.status.set('Click a twist box in Select mode, then choose Expand selected box.')
            return
        answer = simpledialog.askstring('Expand twist box',
                    f'This box contains {twist_label(box)} full twists.\n'
                    'Enter "all" to replace it with explicit crossings, or a signed integer/half-integer part.\n'
                    'For example, expanding -1 out of -2 leaves a -1 box.',
                    initialvalue='all', parent=self.root)
        if answer is None:
            return
        amount, side = None, 'bottom'
        if answer.strip().lower() != 'all':
            try:
                amount = twist_label(parse_twist_label(answer))
            except ValueError as exc:
                messagebox.showerror('Invalid twist value', str(exc), parent=self.root)
                return
        # A temporary overlay gives geometric meaning to top/bottom even
        # when the imported box is rotated or has a reversed axis. All-expansion
        # also needs this choice when only one collar has room for growth.
        self.box_expansion_sides = box['id']
        self.redraw()
        try:
            side = TwistExpansionSideDialog(self.root, full_expansion=(
                amount is None or parse_twist_label(amount) == box['half_twists'])).result
        finally:
            self.box_expansion_sides = None
            self.redraw()
        if side is None:
            return
        self._run_box_edit('expand', (box['id'], amount, side), 'Expanding twist box',
                           'Expanded the chosen twists into ordinary crossings. The remaining coefficient, if any, stays compact. Undo restores the box.')

    def start_box_tangle(self):
        if not self._box_diagram_action_ready():
            return
        self.cancel_gesture(quiet=True)
        self.selected_box = self.selected_crossing = self.selected_component = None
        self.tool.set('Box tangle')
        self.canvas.configure(cursor='crosshair')
        self.status.set('Circle a bundle of full or half twists. Cross each entering/leaving strand once, avoid crossings on the boundary, and leave a clear collar. Curved strands are allowed; the two end bundles must leave room for a box. Escape cancels.')
        self.redraw()

    def _finish_box_tangle(self, polygon):
        if len(polygon) < 3:
            self.status.set('Circle a region with at least three distinct boundary points.')
            self.redraw()
            return
        self._run_box_edit('compress', (polygon,), 'Checking selected tangle',
                           'Certified the selected tangle and replaced it with an equivalent twist box. Undo restores the explicit crossings.')

    def _run_box_edit(self, operation, arguments, title, success):
        """Compute on an isolated snapshot; only a current successful job commits."""
        if not self._box_diagram_action_ready():
            return
        self.cancel_gesture(quiet=True)
        snapshot = self._snapshot()
        snapshot['tool'] = 'Select'
        self.task_id += 1
        token = self.task_id
        self.box_edit_pending = {'task_id': token, 'title': title, 'snapshot': snapshot}
        self.busy = True
        self.tool.set('Select')
        self.status.set(title + '… Escape cancels; the displayed diagram is unchanged until validation finishes.')
        self._refresh_conversion_buttons()
        self.redraw()
        original = copy_diagram(snapshot['diagram'])
        arguments = copy.deepcopy(arguments)

        def worker():
            try:
                from .diagram import crossing_free_count, pd_code, validate
                if operation == 'insert':
                    from .box_insert import add_twists
                    changed = add_twists(original, *arguments)
                elif operation == 'compress':
                    from .box_compression import compress_twist_region
                    changed = compress_twist_region(original, *arguments)
                else:
                    from .box_operations import combine_twist_boxes, delete_zero_box, expand_twist_box
                    if operation == 'combine':
                        changed = combine_twist_boxes(original, *arguments)
                    elif operation == 'delete_zero':
                        changed = delete_zero_box(original, *arguments)
                    elif operation == 'expand':
                        ident, amount, side = arguments
                        changed = expand_twist_box(original, ident, amount=amount, side=side)
                    else:
                        raise ValueError('Unknown twist-box operation.')
                changed = align_twist_boxes(changed)
                checked = validate(changed)
                if not checked['valid']:
                    raise ValueError('; '.join(checked.get('errors', ['Invalid resulting diagram.'])))
                pd, trivial, pd_error = None, None, None
                try:
                    if pd_preview_crossings(changed) <= 10000:
                        pd = pd_code(changed)
                        trivial = crossing_free_count(changed, pd)
                except ValueError as exc:
                    # A compact result remains usable when an expanded PD is
                    # over budget, including crossings of external passages.
                    if 'expansion needs' not in str(exc):
                        raise
                    pd_error = str(exc)
                # Deep copies in topology-preserving operations can carry an
                # older insertion's provenance. Only this insertion produced
                # a new orientation decision; do not announce it again later.
                warning = changed.get('twist_box_edit', {}).get('warning') if operation == 'insert' else None
                self.events.put((token, 'box_edit', (changed, snapshot, pd, trivial, pd_error, success, warning)))
            except Exception as exc:
                self.events.put((token, 'box_edit_error', str(exc)))

        threading.Thread(target=worker, daemon=True, name='twist-box-edit').start()

    def _accept_box_edit(self, kind, payload):
        self.box_edit_pending = None
        if kind == 'box_edit_error':
            self.status.set('No change: ' + payload)
        else:
            changed, snapshot, pd, trivial, pd_error, success, warning = payload
            old_ids = {b['id'] for b in snapshot['diagram']['crossings'] if is_twist_box(b)}
            inherit_display_style(changed, self.diagram)
            self.diagram = changed
            self.display_paths = self._pick_geometry = None
            self.result['diagram'] = changed
            self.result['pd_code'] = pd
            self._prepared_box_pd = (changed, pd, trivial, pd_error)
            self.selected_crossing = self.selected_component = None
            boxes = [b for b in changed['crossings'] if is_twist_box(b)]
            added = [b for b in boxes if b['id'] not in old_ids]
            chosen = added[0] if len(added) == 1 else next((b for b in boxes if b['id'] == getattr(self, 'selected_box', None)), None)
            self.selected_box = chosen['id'] if chosen else None
            self.history.append(snapshot)
            self.history = self.history[-20:]
            self.future.clear()
            if warning and warning not in self.result.get('warnings', []):
                self.result.setdefault('warnings', []).append(warning)
            self.status.set(success + (f' Selected box: {twist_label(chosen)}.' if chosen else '') +
                            (' ' + warning if warning else ''))
        self._refresh_text()
        self.redraw()

    def _rehash_box_detection(self):
        if getattr(self, 'box_detection', None) is not None and self.raster_image is not None:
            import hashlib
            import numpy as np
            self.box_detection = copy.deepcopy(self.box_detection)
            digest = hashlib.sha256(np.asarray(self.raster_image.convert('RGB')).tobytes()).hexdigest()
            self.box_detection.update(source_sha256=digest, image_sha256=digest,
                                      image_size=list(self.raster_image.size))

    def unmark_twist_box(self):
        """Dismiss a source proposal without deleting or interpreting its ink."""
        if self.busy or self._motion_pending() or self.layer.get() != 'source':
            return
        ident = getattr(self, 'selected_box', None)
        if not any(b['id'] == ident for b in self._box_candidates()):
            self.status.set('Select a blue source box first. Unmarking keeps all of its pixels.')
            return
        self._remember()
        self.box_detection = copy.deepcopy(self.box_detection)
        self.box_detection['boxes'] = [b for b in self._box_candidates() if b['id'] != ident]
        self.box_detection.setdefault('dismissed_ids', []).append(ident)
        self.selected_box = None
        self._refresh_text()
        self.redraw()
        self.status.set('Unmarked the box; no pixels were erased. Use Mark a missed twist box to replace its rectangle, or recognize without this box. Undo restores the marking.')

    def mark_twist_box(self):
        if self.busy or self._motion_pending() or self.raster_image is None:
            return
        self.cancel_gesture(quiet=True)
        self.layer.set('source')
        self.tool.set('Mark twist box')
        self.status.set('Drag a rectangle around the entire missed twist box, including its border. Then enter its full-twist number. Pixels remain unchanged.')

    def _finish_manual_box(self, rectangle):
        x0, y0, x1, y1 = rectangle
        if x1-x0 < 6/self.scale or y1-y0 < 6/self.scale:
            self.status.set('The marked box is too small. Drag around its entire border.')
            self.redraw()
            return
        answer = simpledialog.askstring('Twist box · full twists',
                'Enter the full-twist number for this box (integer or half-integer):', parent=self.root)
        if answer is None:
            self.redraw()
            return
        try:
            label = twist_label(parse_twist_label(answer))
        except ValueError as exc:
            messagebox.showerror('Invalid twist value', str(exc), parent=self.root)
            self.redraw()
            return
        image = self.raster_image.copy()
        snapshot = self._snapshot()
        self.task_id += 1
        task_id = self.task_id
        self.busy = True
        self.status.set('Checking strand attachments around the marked box…')
        self._refresh_conversion_buttons()
        def worker():
            try:
                import numpy as np
                from .box_detection import inspect_twist_box
                box = inspect_twist_box(np.asarray(image), [[x0,y0],[x1,y0],[x1,y1],[x0,y1]], label=label, use_ocr=False)
                self.events.put((task_id, 'manual_box', (box, snapshot)))
            except Exception:
                self.events.put((task_id, 'manual_box_error', traceback.format_exc()))
        threading.Thread(target=worker, daemon=True, name='box-inspection').start()

    def delete_marked_labels(self, candidate_ids=None):
        if self.busy or self._motion_pending() or self.raster_image is None or self.layer.get() != 'source':
            return
        candidates = self._label_candidates()
        if not candidates:
            self.status.set('No labels are marked. Use Mark possible labels to scan the drawing.')
            return
        if candidate_ids is None:
            candidate_ids = [candidate['id'] for candidate in candidates]
        import numpy as np
        from .label_review import delete_labels, retain_labels_after_deletion
        try:
            cleaned = delete_labels(np.asarray(self.raster_image), self.label_detection, candidate_ids)
            remaining = retain_labels_after_deletion(cleaned, self.label_detection, candidate_ids)
        except ValueError as exc:
            self.label_detection = None
            self._refresh_text()
            self.redraw()
            self.status.set(f'Label markings no longer match the drawing: {exc} Use Mark possible labels again.')
            return
        self._remember()
        # Copy/paste keeps the calibrated stroke-width hint from a converted
        # diagram. The original source file and history images stay unchanged.
        self.raster_image = self.raster_image.copy()
        self.raster_image.paste(Image.fromarray(cleaned))
        # Keep the same geometric nib check that followed the old explicit
        # label-cleanup retry. This enables a pixels-only check; it supplies no
        # crossing information, reference diagram or interpretation of text.
        self.raster_image.info['_knot_studio_label_cleanup'] = True
        self.label_detection = remaining
        self._rehash_box_detection()
        self.raster_changed = True
        self.tool.set('Select')
        self._refresh_text()
        self.redraw()
        self.status.set(f'Deleted {len(candidate_ids)} marked label region(s). Undo restores them. Choose Recognize drawing when ready.')

    def _label_click(self, point, keep=False):
        pad = 4 / self.scale
        matches = [c for c in self._label_candidates()
                   if c['bbox'][0]-pad <= point[0] <= c['bbox'][2]+pad
                   and c['bbox'][1]-pad <= point[1] <= c['bbox'][3]+pad]
        if not matches:
            return
        chosen = min(matches, key=lambda c: (c['bbox'][2]-c['bbox'][0]) * (c['bbox'][3]-c['bbox'][1]))
        if keep:
            self._remember()
            self.label_detection = copy.deepcopy(self.label_detection)
            self.label_detection['candidates'] = [c for c in self._label_candidates() if c['id'] != chosen['id']]
            self._refresh_text()
            self.redraw()
            self.status.set('Kept this region. Its ink is unchanged and Delete all marked labels will leave it alone.')
        else:
            self.delete_marked_labels([chosen['id']])

    def new_sketch(self):
        if self._motion_pending():
            return
        self._reset()
        self.source_path = None
        self.raster_image = Image.new("RGB", (1200, 900), "white")
        self.root.title("Untitled sketch · Knot Studio")
        self.layer.set("source")
        self.tool.set("Pencil")
        self.fit_view()
        self._refresh_text()
        self.status.set("Draw closed curves. Normal drawing passes over; hold Shift to pass under. Then recognize the drawing.")

    def recognize_raster(self, remember=True):
        if self._motion_pending():
            return
        if self.raster_image is None:
            self.status.set("Open an image or create a sketch first.")
            return
        if self.busy:
            return
        self.cancel_gesture(quiet=True)
        if remember:
            self._remember()
        image = self.raster_image.copy()
        from .recognition_runtime import DEFAULT_SECONDS
        recognize_options = {'board_photo':'auto', 'remove_labels':False,
                             'time_budget': DEFAULT_SECONDS}
        if self._box_candidates() or (getattr(self, 'box_detection', None) or {}).get('dismissed_ids'):
            recognize_options['twist_boxes'] = copy.deepcopy(self.box_detection)
        if image.info.get('_knot_studio_label_cleanup'):
            recognize_options['endpoint_hook_retry'] = True
        self.task_id += 1
        task_id = self.task_id
        self.busy = True
        self._refresh_conversion_buttons()
        self.confidence.set("RECOGNIZING…")
        self.badge.configure(background="#e5edff", foreground="#245fd4")
        self.status.set("Recognizing… Using a 10-second search budget.")

        def worker():
            try:
                from .pipeline import recognize
                with tempfile.TemporaryDirectory(prefix="knot-studio-") as directory:
                    input_path = Path(directory) / "input.png"
                    image.save(input_path)
                    result = recognize(str(input_path), options=recognize_options)
                apply_rendered_appearance(result.get("diagram"), image)
                # Normalize the editable layout off the UI thread, after all
                # pixel-based recognition and orientation evidence is read.
                result['diagram'] = align_twist_boxes(result.get('diagram'))
                self.events.put((task_id, "result", result))
            except Exception:
                self.events.put((task_id, "error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True, name="knot-recognition").start()

    def _poll(self):
        try:
            while True:
                task_id, kind, payload = self.events.get_nowait()
                if task_id != self.task_id:
                    continue
                self.busy = False
                if kind in ('box_edit', 'box_edit_error'):
                    self._accept_box_edit(kind, payload)
                    continue
                if kind in ('manual_box', 'manual_box_error'):
                    if kind == 'manual_box_error':
                        self.status.set('Cannot inspect this twist box: ' + payload.strip().splitlines()[-1])
                    else:
                        box, snapshot = payload
                        self.history.append(snapshot)
                        self.history = self.history[-20:]
                        self.future.clear()
                        self.box_detection = copy.deepcopy(getattr(self, 'box_detection', None) or {'boxes': []})
                        used = {str(b['id']) for b in self._box_candidates()}
                        index = 1
                        while f'manual_{index}' in used:
                            index += 1
                        box['id'] = f'manual_{index}'
                        box['number_reviewed'] = True
                        self.box_detection['boxes'].append(box)
                        self._rehash_box_detection()
                        if self.label_detection:
                            self.label_detection = protect_twist_box_labels(self.label_detection, self.box_detection)
                        self.tool.set('Review twist boxes')
                        self.selected_box = box['id']
                        self.status.set('Marked the twist box. Review its attachments before recognition.' if box.get('status') == 'ready' else
                                        'Box number saved, but strand attachments need review. Recognition will report this uncertainty.')
                    self._refresh_text()
                    self.redraw()
                    continue
                if kind in ('labels', 'label_error'):
                    if kind == 'labels':
                        self.label_detection, recognize_if_empty = payload[:2]
                        self.box_detection = copy.deepcopy(payload[2]) if len(payload) > 2 else None
                        self._rehash_box_detection()
                        count = len(self._label_candidates())
                        box_count = len(self._box_candidates())
                        if count or box_count:
                            self.layer.set('source')
                            self.tool.set('Select')
                        self._refresh_text()
                        self.fit_view()
                        if count:
                            self.status.set(f'{count} possible label region(s) marked. Click to erase, Shift-click to keep, or delete all marked labels. Recognize drawing keeps the remaining ink.')
                        elif box_count:
                            self.status.set(f'{box_count} twist box(es) marked in blue. Click a box to review its full-twist number, then Recognize drawing.')
                        elif recognize_if_empty:
                            self.recognize_raster()
                        else:
                            self.status.set('No possible labels found. You can use the eraser for missed labels, then Recognize drawing.')
                        if box_count and self._offer_missing_twist_values():
                            self._refresh_text()
                            self.redraw()
                    else:
                        # Discovery is read-only: a failed rescan must not
                        # discard the user's existing review decisions.
                        self._refresh_text()
                        self.redraw()
                        self.status.set('Label detection could not finish: ' + payload.strip().splitlines()[-1] + '. You can edit with the eraser or recognize the unchanged drawing.')
                    continue
                self.raster_changed = False
                if kind == "error":
                    self.result = {"status": "failed", "warnings": [payload.strip().splitlines()[-1]], "diagnostics": {"traceback": payload}}
                    self.diagram = None
                    self.status.set("Recognition failed. Details contains the error; the source image remains editable.")
                else:
                    self.result = payload
                    self.diagram = payload.get("diagram")
                    if self.diagram:
                        # The recognized graph replaces the active drawing.
                        # Undo retains the pre-recognition editing state.
                        self.raster_image = None
                        self.label_detection = None
                        self.box_detection = None
                        self.layer.set("diagram")
                        self.tool.set("Select")
                    self.status.set("Recognition complete. Inspect crossing gaps before using the PD code." if payload.get("status") == "ok" else "Recognition needs review. Inspect the displayed diagram and the Details tab.")
                    if payload.get('diagnostics', {}).get('budget_exhausted'):
                        self.status.set('Recognition reached its time limit. Inspect the marked regions, repair unclear crossings or unwanted ink, and try again.')
                    if not self.diagram and self.raster_image is not None:
                        detection = payload.get('diagnostics', {}).get('twist_boxes')
                        if detection and detection.get('boxes'):
                            import numpy as np
                            from .box_recognition import checked_detection
                            try:
                                self.box_detection = checked_detection(np.asarray(self.raster_image.convert('RGB')), detection)
                            except ValueError:
                                # Never rebind stale detector coordinates to different ink.
                                pass
                            else:
                                self._offer_missing_twist_values()
                self.selected_crossing = self.selected_component = None
                self._refresh_text()
                self.fit_view()
        except queue.Empty:
            pass
        # Active drags have their own short poll interval. Keep this slower
        # recognition poll to discard any late result from a cancelled drag.
        if not self.drag or self.drag.get("kind") not in ("smooth", "energy", "bigon_scan", "bigon_remove"):
            self._poll_motion()
        self._poll_display_curves()
        self.root.after(80, self._poll)

    def _schedule_motion_poll(self):
        if self.motion_poll_timer is None:
            self.motion_poll_timer = self.root.after(8, self._motion_tick)

    def _motion_tick(self):
        """Present accepted worker results promptly, without polling when idle."""
        self.motion_poll_timer = None
        self._poll_motion()
        if self.drag and self.drag.get("kind") in ("smooth", "energy", "bigon_scan", "bigon_remove"):
            self._schedule_motion_poll()

    def _stop_motion_poll(self):
        if self.motion_poll_timer is not None:
            self.root.after_cancel(self.motion_poll_timer)
            self.motion_poll_timer = None

    def _poll_motion(self):
        """Apply worker results on Tk's thread, then submit only the newest target."""
        try:
            while True:
                token, kind, payload = self.motion_events.get_nowait()
                drag = self.drag
                if not drag or drag.get("token") != token:
                    continue
                drag["inflight"] = False
                if drag.get("kind") == "energy":
                    self._energy_result(kind, payload)
                    continue
                if drag.get("kind") in ("bigon_scan", "bigon_remove"):
                    self._bigon_result(kind, payload)
                    continue
                if kind == "ready":
                    drag["engine"] = payload
                    self.status.set("Slide the box edge to resize it. The opposite edge stays fixed; nearby strands follow. Release to keep it, or Escape to cancel."
                                    if drag.get('box_resize') else
                                    "Drag the strand. Hold Shift as new crossings form to pass underneath; otherwise pass over. Escape cancels this drag.")
                elif kind == "initialization_error":
                    self.cancel_gesture(quiet=True)
                    self.status.set("Cannot start this drag: " + payload)
                    continue
                elif kind == "rejected":
                    drag["rejection"] = payload
                    self.status.set("Position rejected; keeping the last accepted diagram. " + payload)
                else:
                    diagram, moves, target, prepared_paths, prepared_pd = payload
                    self._edited_diagram(diagram, prepared_pd=prepared_pd)
                    if prepared_paths is not None:
                        colored, strokes = prepared_paths
                        self.display_paths = (diagram, colored, strokes)
                    drag["last_target"] = target
                    drag["moved"] = True
                    drag["rejection"] = None
                    drag["moves"].extend(moves)
                    detail = "; ".join(map(str, moves))
                    encoding = "The diagram is updated; PD preview is unavailable." if prepared_pd[2] else "The diagram and PD are updated."
                    self.status.set(("Resizing box" if drag.get('box_resize') else "Moving smoothly")
                                    + (": " + detail if detail else "") + ". " + encoding + " Escape cancels this drag.")
                    self._refresh_text()
                    self.redraw()
                self._apply_smooth_target()
        except queue.Empty:
            pass

    def _pd(self):
        self._pd_crossing_free_count = None
        if self.diagram and pd_preview_crossings(self.diagram) > 10000:
            raise ValueError('This compact diagram exceeds the 10,000-crossing PD preview limit. Its twist boxes remain editable and can be saved as JSON.')
        if not self.diagram:
            return None
        prepared = getattr(self, '_prepared_box_pd', None)
        if prepared is not None and prepared[0] is self.diagram:
            if prepared[3]:
                raise ValueError(prepared[3])
            self._pd_crossing_free_count = prepared[2]
            return prepared[1]
        from .diagram import crossing_free_count, pd_code
        pd = pd_code(self.diagram)
        self._pd_crossing_free_count = crossing_free_count(self.diagram, pd)
        self._prepared_box_pd = (self.diagram, pd, self._pd_crossing_free_count, None)
        return pd

    def _pd_string(self):
        pd = self._pd()
        if pd is None:
            return ""
        return "[" + ",\n ".join("[" + ", ".join(map(str, crossing)) + "]" for crossing in pd) + "]"

    def _tikz_string(self):
        if not self.diagram:
            return ""
        if not self.smooth_display.get():
            return render_tikz(self.diagram, colored=self.colored.get(),
                               show_orientation=self.oriented.get(), smooth=False)
        cache = self._curve_cache()
        if cache is not None:
            if cache.error is not None:
                raise cache.error
            if cache.prepared is None:
                raise RuntimeError("Smooth curves are still being prepared.")
            return render_tikz(self.diagram, colored=self.colored.get(),
                               show_orientation=self.oriented.get(), prepared=cache.prepared)
        return render_tikz(self.diagram, colored=self.colored.get(),
                           show_orientation=self.oriented.get())

    def _curve_cache(self):
        cache = getattr(self, '_display_curve_cache', None)
        if cache is not None and cache.sync(self.diagram):
            self._display_fit_due = time.monotonic() + .12
        return cache

    def _invalidate_display_curves(self):
        cache = self._curve_cache()
        if cache is not None:
            cache.invalidate()
            self._display_fit_due = time.monotonic() + .12

    def _display_prepared(self):
        cache = self._curve_cache()
        enabled = getattr(self, 'smooth_display', None)
        return cache.prepared if cache is not None and enabled is not None and enabled.get() else None

    def _tikz_visible(self):
        return (hasattr(self, 'tikz_text') and
                self.code_notebook.select() == str(self.tikz_frame))

    @staticmethod
    def _run_display_worker(function):
        threading.Thread(target=function, daemon=True, name='knot-display').start()

    def _poll_display_curves(self):
        """Share one background fit between painting and export; never fit in Tk."""
        cache = self._curve_cache()
        if cache is None:
            return
        changed = cache.poll(self.diagram)
        actions = getattr(self, '_display_actions', [])
        current_actions = [(revision, action) for revision, action in actions
                           if revision == cache.revision]
        self._display_actions = current_actions
        if len(current_actions) != len(actions):
            self.status.set('Pending TikZ export cancelled because the diagram changed.')
        if changed:
            self.redraw()
            if self._tikz_visible():
                self._queue_tikz_refresh()
        if cache.prepared is not None or cache.error is not None or not self.smooth_display.get():
            # A worker may finish during a drag; defer exports until it ends.
            if not self.busy and not self.drag:
                self._display_actions = []
                for _, action in current_actions:
                    action()
            return
        enabled = getattr(self, 'smooth_display', None)
        needed = ((enabled is not None and enabled.get() and self.layer.get() == 'diagram')
                  or self._tikz_visible() or bool(current_actions))
        if (needed and not self.busy and not self.drag and self.diagram is not None
                and time.monotonic() >= self._display_fit_due):
            cache.request(self.diagram, self._run_display_worker)
            if cache.error is not None and self._tikz_visible():
                self._queue_tikz_refresh()

    def _defer_tikz_action(self, action):
        if not self.smooth_display.get():
            return False
        cache = self._curve_cache()
        if cache is None or cache.prepared is not None or cache.error is not None or not self.diagram:
            return False
        self._display_actions.append((cache.revision, action))
        self.status.set('Preparing smooth curves for TikZ…')
        return True

    def _queue_tikz_refresh(self, event=None):
        """Generate only the visible preview, after a burst of edits settles."""
        if not hasattr(self, 'tikz_text'):
            return
        job = getattr(self, '_tikz_refresh_job', None)
        if job is not None:
            self.root.after_cancel(job)
            self._tikz_refresh_job = None
        if self.code_notebook.select() != str(self.tikz_frame):
            return
        self._put_text(self.tikz_text, "Updating TikZ code…" if self.diagram else
                       "A recognized diagram's TikZ code appears here.")
        if self.diagram:
            self._tikz_refresh_job = self.root.after(150, self._refresh_tikz)

    def _refresh_tikz(self):
        self._tikz_refresh_job = None
        cache = self._curve_cache()
        if (self.smooth_display.get() and self.diagram and cache is not None
                and cache.prepared is None and cache.error is None):
            self._put_text(self.tikz_text, 'Preparing smooth curves for TikZ…')
            return
        try:
            text = self._tikz_string()
        except Exception as exc:
            text = "% TikZ is unavailable: " + str(exc)
        if self.raster_changed and text:
            text = "% Previous diagram: recognize the changed drawing to update this code.\n\n" + text
        self._put_text(self.tikz_text, text or "A recognized diagram's TikZ code appears here.")

    def _appearance_changed(self):
        self.redraw()
        self._queue_tikz_refresh()

    @staticmethod
    def _put_text(widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _refresh_brush(self):
        erasing = self.tool.get() == "Eraser"
        block = self.tool.get() == 'Block eraser'
        self.brush_size = self.eraser_size if erasing else self.pencil_size
        self.brush_label.configure(text="Click connected ink" if block else "Eraser width" if erasing else "Pencil width")
        self.brush_scale.configure(variable=self.brush_size, to=140 if erasing else 28,
                                   state='disabled' if block else 'normal')
        self._refresh_parameter_rows()

    def _refresh_parameter_rows(self):
        if not hasattr(self, 'move_radius_row'):
            return
        drawing = self.layer.get() == 'source'
        for row, visible, following, padding in (
                (self.move_radius_row, not drawing and self.tool.get() in ('Select', 'Move geometry'), self.move_radius_after, (4, 6)),
                (self.brush_width_row, drawing and self.tool.get() in ('Pencil', 'Eraser'), self.brush_width_after, 7)):
            if visible:
                if not row.winfo_manager():
                    row.pack(fill='x', pady=padding, before=following)
            elif row.winfo_manager():
                row.pack_forget()

    def _refresh_conversion_buttons(self):
        self._refresh_parameter_rows()
        drawing = self.layer.get() == "source"
        if hasattr(self, 'twist_boxes_drawing_controls'):
            for frame, visible in ((self.twist_boxes_drawing_controls, drawing),
                                   (self.twist_boxes_diagram_controls, not drawing)):
                if visible and not frame.winfo_manager():
                    frame.pack(fill='x')
                elif not visible and frame.winfo_manager():
                    frame.pack_forget()
        self.recognize_button.configure(state="normal" if drawing and self.raster_image is not None and not self.busy else "disabled")
        self.convert_button.configure(state="normal" if not drawing and self.diagram is not None and not self.busy else "disabled")
        if hasattr(self, 'find_labels_button'):
            self.find_labels_button.configure(state='normal' if drawing and self.raster_image is not None and not self.busy else 'disabled')
            self.delete_labels_button.configure(state='normal' if drawing and self._label_candidates() and not self.busy else 'disabled')
            count = len(self._label_candidates())
            self.label_count.set(f'{count} possible label region(s)' if count else 'No labels marked')
        if hasattr(self, 'review_boxes_button'):
            count = (len(self._box_candidates()) if drawing else
                     sum(is_twist_box(v) for v in (self.diagram or {}).get('crossings', [])))
            self.review_boxes_button.configure(state='normal' if not self.busy and (self.raster_image is not None or self.diagram) else 'disabled')
            self.edit_box_button.configure(state='normal' if count and not self.busy else 'disabled')
            self.box_count.set(f'{count} twist box(es)' if count else 'No twist boxes')
            self.mark_box_button.configure(state='normal' if self.raster_image is not None and not self.busy else 'disabled')
            self.unmark_box_button.configure(state='normal' if drawing and not self.busy and
                                            any(b['id'] == getattr(self, 'selected_box', None) for b in self._box_candidates()) else 'disabled')
        if hasattr(self, 'fill_box_values_button'):
            missing = len(self._missing_twist_values()) if drawing else 0
            self.fill_box_values_button.configure(state='normal' if missing and not self.busy else 'disabled')
            if missing:
                self.box_count.set(f'{missing} value(s) needed')
        if hasattr(self, 'add_twists_button'):
            available = not drawing and self.diagram is not None and not self.busy and not self.drag
            chosen = self._selected_diagram_box()
            count = sum(is_twist_box(b) for b in (self.diagram or {}).get('crossings', []))
            for button in (self.add_twists_button, self.compress_box_button):
                button.configure(state='normal' if available else 'disabled')
            self.combine_boxes_button.configure(state='normal' if available and count >= 2 else 'disabled')
            self.expand_box_button.configure(state='normal' if available and chosen is not None else 'disabled')
            self.delete_zero_box_button.configure(state='normal' if available and chosen is not None and chosen['half_twists'] == 0 else 'disabled')

    def _refresh_text(self):
        self._refresh_conversion_buttons()
        status = self.result.get("status", "ready")
        label, background, foreground = {
            "ok": ("GRAPH VALIDATED · INSPECT VISUALLY", "#e2f3eb", "#146744"),
            "needs_review": ("NEEDS REVIEW · PD IS PROVISIONAL", "#fff0d5", "#8f5706"),
            "failed": ("RECOGNITION FAILED", "#fde8e8", "#a33131"),
            "ready": ("READY", "#e6edf7", "#345072"),
        }.get(status, (str(status).upper(), "#e6edf7", "#345072"))
        if self.raster_changed:
            label, background, foreground = "DRAWING CHANGED · RECOGNIZE AGAIN", "#fff0d5", "#8f5706"
        elif self.diagram is None and self._missing_twist_values():
            label, background, foreground = "ENTER TWIST VALUES · NO PD YET", "#fff0d5", "#8f5706"
        self.confidence.set(label)
        self.badge.configure(background=background, foreground=foreground)
        pd_error = None
        try:
            text = self._pd_string()
        except Exception as exc:
            text = "PD is unavailable: " + str(exc)
            pd_error = str(exc)
        if self.raster_changed and text:
            text = "Previous diagram's PD:\n\n" + text
        self._put_text(self.pd_text, text or "A recognized diagram's PD code appears here.")
        if hasattr(self, 'pd_copy_button'):
            self.pd_text.tag_add('copy_icon_margin', '1.0', 'end')
        self._queue_tikz_refresh()
        warnings = self.result.get("warnings", [])
        details = "\n\n".join(str(warning) for warning in warnings) if warnings else "No recognition warnings."
        orientation = _record(self.result.get('diagnostics', {}).get('orientation'))
        records = _records(orientation.get('components'))
        if records:
            preserved = sum(c.get('status') == 'consistent' for c in records)
            details += f"\n\nAt recognition, source arrows supplied consistent orientations for {preserved} of {len(records)} component(s). Show orientations displays the stored directions. Components without clear arrow evidence use default directions, shown with gray arrows."
        if self.raster_changed:
            details = "The drawing has changed. Recognize it to generate its diagram and PD code.\n\n" + details
        if recognition_issue_points(self.result):
            details += "\n\nIn drawing mode, red circles mark branching strand contacts and orange circles mark unmatched open ends. Toggle Show recognition issues to inspect the pixels underneath, repair with the eraser or pencil, and recognize again."
        if pd_error:
            details += "\n\nPD validation error: " + pd_error
        details += "\n\nDragging with Select / move updates the graph and PD through allowed local Reidemeister II/III events. Curl creation and removal use the separate Reidemeister I buttons. Rejected drag positions retain the last accepted diagram. Geometric checks use sampled curves and numerical tolerances."
        if self.diagram:
            edges = self.diagram.get("edges", [])
            components = len({edge.get("component", 0) for edge in edges})
            trivial = getattr(self, '_pd_crossing_free_count', None)
            boxes = sum(is_twist_box(v) for v in self.diagram.get('crossings', []))
            ordinary = len(self.diagram.get('crossings', []))-boxes
            self.counts.set("Drawing mode" if self.layer.get() == 'source' else
                            f"Diagram mode  ·  {ordinary} crossings" + (f"  ·  {boxes} twist boxes" if boxes else '') + f"  ·  {components} components" + (f"  ·  {trivial} crossing-free" if trivial else ""))
            if boxes:
                details += '\n\nTwist boxes are stored as compact vertices. Their numbers count full twists; 1/2 means one half twist. PD expands the implicit crossings when within the preview limit. Verify source handedness and strand attachments before relying on the PD.'
            if trivial:
                details += f"\n\n{trivial} crossing-free component(s) are stored in the diagram JSON. Plain PD code cannot encode isolated zero-crossing components."
        else:
            self.counts.set("Drawing mode" if self.layer.get() == 'source' else "No diagram loaded")
        if self.result.get("diagnostics"):
            details += "\n\nDiagnostics\n" + json.dumps(self.result["diagnostics"], indent=2, default=str)
        if self._label_candidates():
            details += '\n\nPossible labels (nothing is erased until you choose):\n' + '\n'.join(
                f"Label {number}: {c.get('reason', 'suspicious detached mark')}"
                for number, c in enumerate(self._label_candidates(), 1))
            diagnostic = self.label_detection.get('diagnostics', {})
            if diagnostic.get('candidate_limit_reached'):
                details += '\nThe marking limit was reached; scan again after reviewing these regions.'
            if diagnostic.get('ocr_budget_exhausted'):
                details += '\nThe label scan reached its time limit; some markings may be missing.'
        if self._box_candidates():
            details += '\n\nTwist-box review:\n' + '\n'.join(
                f"Box {i}: {b.get('numeric_label') or 'number needs review'}; {b.get('status', 'review')}"
                for i, b in enumerate(self._box_candidates(), 1))
            details += '\nCorrecting a number does not resolve uncertain strand attachments.'
        self._put_text(self.details_text, details)
        self.undo_button.configure(state="normal" if self.history else "disabled")
        self.redo_button.configure(state="normal" if self.future else "disabled")

    def _dimensions(self):
        if self.layer.get() == "source" and self.raster_image:
            return self.raster_image.size
        if self.diagram:
            return self.diagram.get("width", 800), self.diagram.get("height", 600)
        if self.raster_image:
            return self.raster_image.size
        return 1200, 900

    def fit_view(self):
        width, height = self._dimensions()
        left, top = 0., 0.
        if self.layer.get() == "diagram" and self.diagram:
            points = ([p for stroke, _, _ in self._diagram_paths() for p in stroke] +
                      [p for shape in box_shapes(self.diagram) for p in shape['corners']])
            if points:
                margin = max(12., line_width(self.diagram) * 3)
                left = min(p[0] for p in points) - margin
                top = min(p[1] for p in points) - margin
                width = max(p[0] for p in points) + margin - left
                height = max(p[1] for p in points) + margin - top
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            self.fit_pending = True
            return
        self.fit_pending = False
        self.scale = max(1e-9, min((cw - 44) / max(width, 1), (ch - 44) / max(height, 1)))
        self.offset = ((cw - width * self.scale) / 2 - left * self.scale,
                       (ch - height * self.scale) / 2 - top * self.scale)
        self.redraw()

    def zoom(self, factor, anchor=None):
        if anchor is None:
            anchor = (self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)
        before = self.to_model(*anchor)
        self.scale = max(1e-9, min(8.0, self.scale * factor))
        self.offset = (anchor[0] - before[0] * self.scale, anchor[1] - before[1] * self.scale)
        self.redraw()

    def _wheel(self, event):
        windowing_system = self.root.tk.call('tk', 'windowingsystem')
        steps = wheel_steps(event, windowing_system=windowing_system)
        if steps:
            if windowing_system == 'aqua':
                steps = math.copysign(1, steps)  # Preserve the macOS canvas zoom speed.
            base = 1.12 if getattr(event, 'num', None) in (4, 5) else 1.1
            self.zoom(base ** max(-10, min(10, steps)), (event.x, event.y))
        return "break"

    def _start_pan(self, event):
        if self.drag and self.drag.get('kind') not in ('energy', 'bigon_scan', 'bigon_remove'):
            return 'break'
        self.canvas.focus_set()
        self.fit_pending = False
        if not self.pan:
            self._pan_cursor = self.canvas.cget('cursor')
        self.pan = (event.x, event.y, self.offset)
        self.canvas.configure(cursor='fleur')
        return 'break'

    def _pan(self, event):
        if self.pan:
            x, y, offset = self.pan
            self.offset = (offset[0] + event.x - x, offset[1] + event.y - y)
            self.redraw()
        return 'break'

    def _end_pan(self, event):
        if self.pan:
            self._pan(event)
            self.pan = None
            self.canvas.configure(cursor=getattr(self, '_pan_cursor', 'arrow'))
        return 'break'

    def to_canvas(self, point):
        return point[0] * self.scale + self.offset[0], point[1] * self.scale + self.offset[1]

    def to_model(self, x, y):
        return (x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale

    def _diagram_paths(self):
        """Reuse model-space strokes until the diagram or color mode changes."""
        colored = self.colored.get()
        prepared = self._display_prepared()
        if prepared is not None:
            return prepared.strokes(colored=colored, scale=self.scale)
        cached = self.display_paths
        if cached is None or cached[0] is not self.diagram or cached[1] != colored:
            cached = self.display_paths = (self.diagram, colored, paths(self.diagram, colored))
        return cached[2]

    def redraw(self):
        canvas = self.canvas
        canvas.delete("all")
        width, height = self._dimensions()
        ox, oy = self.offset
        drawing = self.layer.get() == "source"
        canvas.configure(background="#e5eaf2" if drawing else "white")
        if drawing:
            canvas.create_rectangle(ox + 3, oy + 4, ox + width * self.scale + 3, oy + height * self.scale + 4, fill="#c9d1dd", outline="")
            canvas.create_rectangle(ox, oy, ox + width * self.scale, oy + height * self.scale, fill="white", outline="#cfd7e3")
        self.zoom_text.set(f"{self.scale * 100:.0f}%")
        if self.layer.get() == "source" and self.raster_image is not None:
            # Crop to the viewport before resizing: high zoom never allocates a huge bitmap.
            left = max(0, int(-ox / self.scale))
            top = max(0, int(-oy / self.scale))
            right = min(width, int((canvas.winfo_width() - ox) / self.scale) + 2)
            bottom = min(height, int((canvas.winfo_height() - oy) / self.scale) + 2)
            if right > left and bottom > top:
                image = self.raster_image.crop((left, top, right, bottom))
                image = image.resize((max(1, round((right - left) * self.scale)), max(1, round((bottom - top) * self.scale))), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image)
                canvas.create_image(*self.to_canvas((left, top)), image=self.photo, anchor="nw")
            for number, candidate in enumerate(self._label_candidates(), 1):
                x1, y1 = self.to_canvas(candidate['bbox'][:2])
                x2, y2 = self.to_canvas(candidate['bbox'][2:])
                tags = ('label-candidate', 'label-' + str(candidate['id']))
                canvas.create_rectangle(x1-3, y1-3, x2+3, y2+3, outline='#b45309', width=2,
                                        dash=(5, 3), tags=tags)
                canvas.create_text(x1, y1-5, text=f'Label {number}',
                                   anchor='sw', fill='#92400e', tags=tags)
            for number, box in enumerate(self._box_candidates(), 1):
                x1, y1 = self.to_canvas(box['bbox'][:2])
                x2, y2 = self.to_canvas(box['bbox'][2:])
                tags = ('twist-box-candidate', 'box-' + str(box['id']))
                missing_value = box.get('half_twists') is None
                color = '#c2410c' if missing_value else '#2563eb'
                canvas.create_rectangle(x1-3, y1-3, x2+3, y2+3, outline=color,
                                        width=4 if box['id'] == getattr(self, 'selected_box', None) else 2,
                                        dash=(4, 2) if box.get('status') != 'ready' else (), tags=tags)
                label = '? — click to enter' if missing_value else box.get('numeric_label')
                canvas.create_text(x1, y1-5, text=f"Box {number}: {label}",
                                   anchor='sw', fill=color, tags=tags)
            if self.drag and self.drag.get('kind') == 'box_rectangle':
                x0, y0 = self.to_canvas(self.drag['anchor'])
                x1, y1 = self.to_canvas(self.drag['last'])
                canvas.create_rectangle(x0, y0, x1, y1, outline='#2563eb', dash=(5,3), width=2, tags=('manual-twist-box',))
            if self.show_issues.get() and not self.raster_changed:
                for kind, point in recognition_issue_points(self.result):
                    x, y = self.to_canvas(point)
                    color = {"branch":"#dc2626", "gap_repair":"#078396", "bundle_gap":"#078396"}.get(kind,"#ea7b08")
                    canvas.create_oval(x - 9, y - 9, x + 9, y + 9, outline=color, width=2, tags=("recognition-issue", f"issue-{kind}"))
                removed = _record(self.result.get("diagnostics", {}).get("preprocessing")).get("removed_labels")
                for label in _records(removed):
                    box = label.get("bbox", [])
                    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in box):
                        continue
                    x1, y1 = self.to_canvas(box[:2])
                    x2, y2 = self.to_canvas(box[2:])
                    canvas.create_rectangle(x1-3, y1-3, x2+3, y2+3, outline="#078396", width=2,
                                            dash=(4, 2), tags=("recognition-issue", "issue-label"))
                    canvas.create_text(x2+6, y1, text="label removed", anchor="nw", fill="#078396",
                                       tags=("recognition-issue", "issue-label"))
        elif self.diagram:
            if self.drag and self.drag.get('kind') in ('twist_slice', 'twist_slice_preview'):
                coords = [v for p in (self.drag['anchor'], self.drag['last']) for v in self.to_canvas(p)]
                canvas.create_line(*coords, fill='#c26a09', width=3, dash=(6, 3), tags=('twist-slice',))
            if self.drag and self.drag.get('kind') == 'box_lasso':
                coords = [v for p in self.drag['region'] for v in self.to_canvas(p)]
                if len(coords) >= 6:
                    canvas.create_polygon(*coords, fill='#dbeafe', stipple='gray25', outline='#2563eb',
                                          width=2, dash=(5, 3), tags=('twist-tangle-region',))
                elif len(coords) >= 4:
                    canvas.create_line(*coords, fill='#2563eb', width=2, tags=('twist-tangle-region',))
            if self.drag and self.drag.get('kind') in ('energy', 'energy_lasso'):
                region = self.drag['region']
                coords = [v for p in region for v in self.to_canvas(p)]
                if len(coords) >= 6:
                    canvas.create_polygon(*coords, fill='#d7f0eb', stipple='gray25', outline='#11836f',
                                          width=2, dash=(5, 3), tags=('energy-region',))
                elif len(coords) >= 4:
                    canvas.create_line(*coords, fill='#11836f', width=2, tags=('energy-region',))
            if self.tool.get() in ("Remove curl", "Simplify"):
                for region in self.monogons:
                    polygon = region["polygon"]
                    coords = [coordinate for p in polygon for coordinate in self.to_canvas(p)]
                    if len(coords) >= 6:
                        canvas.create_polygon(*coords, fill="#c8ece0", outline="#389b80", width=1,
                                              tags=("removable-monogon", str(region["id"])))
            if self.tool.get() in ("Remove bigon", "Simplify"):
                for region in self.bigons:
                    coords = [v for p in region['polygon'] for v in self.to_canvas(p)]
                    if len(coords) >= 6:
                        canvas.create_polygon(*coords, fill="#e5dcfa", outline="#8862bd", width=1,
                                              tags=("removable-bigon", str(region['id'])))
            for points, color, width in self._diagram_paths():
                if len(points) > 1:
                    coords = [coordinate for point in points for coordinate in self.to_canvas(point)]
                    canvas.create_line(*coords, fill=color, width=max(0.8, width * self.scale), capstyle="butt" if color == "#ffffff" else "round", joinstyle="round")
            prepared = self._display_prepared()
            if self.oriented.get():
                directions = (prepared.arrows(self.colored.get()) if prepared is not None else
                              arrows(self.diagram, self.colored.get()))
                for point, tangent, color in directions:
                    polygon = arrow_polygon(point, tangent, line_width(self.diagram) * 2)
                    canvas.create_polygon(*[coordinate for p in polygon for coordinate in self.to_canvas(p)], fill=color, outline=color)
            for shape in box_shapes(self.diagram):
                coords = [v for p in shape['corners'] for v in self.to_canvas(p)]
                selected = shape['id'] == getattr(self, 'selected_box', None)
                canvas.create_polygon(*coords, fill='white', outline='#2563eb' if selected else '#171717',
                                      width=max(1., shape['stroke_width']*self.scale),
                                      tags=('twist-box', 'box-'+str(shape['id'])))
                canvas.create_text(*self.to_canvas(shape['point']), text=shape['label'], fill='#171717',
                                   font=('Times', -max(1, round(shape['font_size']*self.scale))),
                                   angle=-shape['label_angle'],
                                   tags=('twist-box-number', 'box-'+str(shape['id'])))
                if selected and self.tool.get() in ('Select', 'Move geometry'):
                    corners = shape['corners']
                    for a, b in zip(corners, corners[1:]+corners[:1]):
                        x, y = self.to_canvas([(a[k]+b[k])/2 for k in (0, 1)])
                        canvas.create_rectangle(x-3, y-3, x+3, y+3, fill='white', outline='#2563eb',
                                                tags=('box-resize-handle',))
                if shape['id'] == getattr(self, 'box_expansion_sides', None):
                    box = next(v for v in self.diagram['crossings'] if v['id'] == shape['id'])
                    distance = box['size'][1]/2 + max(12/self.scale, line_width(self.diagram)*3)
                    for label, factor in (('A', -1), ('B', 1)):
                        point = [box['point'][i]+factor*distance*box['axis'][i] for i in (0, 1)]
                        canvas.create_text(*self.to_canvas(point), text=label, fill='#2563eb',
                                           font=('Helvetica', 13, 'bold'), tags=('twist-expansion-side',))
            passages = (prepared.passages(colored=self.colored.get(), scale=self.scale) if prepared is not None else
                        box_passages(self.diagram, self.colored.get()))
            for points, color, width in passages:
                coords = [v for p in points for v in self.to_canvas(p)]
                canvas.create_line(*coords, fill='white', width=max(.8, width*3*self.scale), capstyle='round', joinstyle='round', tags=('box-overpass-halo',))
                canvas.create_line(*coords, fill=color, width=max(.8, width*self.scale), capstyle='round', joinstyle='round', tags=('box-overpass',))
            if self.selected_component is not None:
                markers = (prepared.markers(self.selected_component) if prepared is not None else
                           [edge['points'][len(edge['points']) // 2]
                            for edge in self.diagram.get('edges', [])
                            if edge.get('component', 0) == self.selected_component and edge.get('points')])
                for point in markers:
                    x, y = self.to_canvas(point)
                    canvas.create_oval(x - 3, y - 3, x + 3, y + 3, outline="#0696aa", width=2)
            for crossing in self.diagram.get("crossings", []):
                if not is_twist_box(crossing) and crossing["id"] == self.selected_crossing:
                    x, y = self.to_canvas(crossing["point"])
                    canvas.create_oval(x - 12, y - 12, x + 12, y + 12, outline="#0598ae", width=2,
                                       tags=('selected-crossing',))
            if self.curl_add and self.curl_add.get("point") is not None:
                x, y = self.to_canvas(self.curl_add["point"])
                canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#d97706", width=2, tags=("curl-anchor",))
                canvas.create_text(x + 11, y - 11, text="Click this strand's desired side", anchor="sw",
                                   fill="#8f5706", tags=("curl-anchor",))
        else:
            canvas.create_text(canvas.winfo_width() / 2, canvas.winfo_height() / 2, text="Open a knot or link image\nor start a new sketch", font=("Helvetica", 20), fill="#8797ad", justify="center")

    def _layer_changed(self):
        if getattr(self, 'box_edit_pending', None):
            self.layer.set('diagram')
            return
        if self._motion_pending():
            self.layer.set("diagram")
            return
        chosen_layer = self.layer.get()
        if chosen_layer == "source" and self.raster_image is None and self.diagram is not None:
            self.layer.set("diagram")
            self.status.set("This diagram has no editable raster. Use Convert to drawing after separating any strands that are too close.")
            return
        self.cancel_gesture(quiet=True)
        self.layer.set(chosen_layer)
        self.selected_crossing = self.selected_component = None
        self.selected_box = None
        if self.layer.get() == "diagram" and self.tool.get() in ("Pencil", "Eraser", "Block eraser", "Review labels", 'Review twist boxes'):
            self.tool.set("Select")
        self.fit_view()

    def _tool_changed(self):
        if getattr(self, 'box_edit_pending', None):
            self.tool.set('Select')
            return
        if self._motion_pending():
            self.tool.set(self.drag.get('tool') or {'energy':'Energy region', 'bigon_scan':'Remove bigon',
                           'bigon_remove':'Remove bigon'}.get(self.drag.get('kind'), 'Move geometry'))
            return
        chosen_tool = self.tool.get()
        self.cancel_gesture(quiet=True)
        self.tool.set(chosen_tool)
        self.selected_crossing = self.selected_component = None
        self.selected_box = None
        if self.tool.get() in ("Pencil", "Eraser", "Block eraser", "Review labels", 'Review twist boxes'):
            if self.raster_image is None and self.diagram is not None:
                self.layer.set("diagram")
                self.tool.set("Select")
                self.status.set("This diagram has no editable raster. Use Convert to drawing after separating any strands that are too close.")
                self.redraw()
                return
            self.layer.set("source")
            self.canvas.configure(cursor="crosshair")
        elif self.diagram is None and self.raster_image is not None:
            self.layer.set("source")
            self.tool.set("Select")
            self.canvas.configure(cursor="arrow")
            self.status.set("Recognize drawing before editing strands in diagram mode.")
        else:
            self.layer.set("diagram")
            self.canvas.configure(cursor="fleur" if self.tool.get() == "Move geometry" else "arrow")
        if self.tool.get() == "Move geometry":
            self.status.set("Drag a strand away from a crossing. Move radius controls the smooth neighborhood. Hold Shift as new crossings form to pass underneath; otherwise pass over. Accepted II/III moves update PD; use the curl buttons for move I. Escape cancels.")
        elif self.tool.get() == "Pencil":
            self.status.set("Draw to pass over strands; hold Shift to pass underneath. Then choose Recognize drawing to update the diagram and PD.")
        elif self.tool.get() == 'Block eraser':
            self.status.set('Click ink to erase its whole connected region. Crossing gaps stop the erasure. Undo restores it.')
        elif self.tool.get() == 'Select' and self.layer.get() == 'diagram':
            self.status.set('Click to select; drag a strand to move it smoothly, or a box edge to resize it. Hold Shift for new undercrossings. Move radius controls the neighborhood.')
        self.redraw()

    def _nearest(self, point, include_crossings=True):
        import numpy as np

        crossing = min((c for c in self.diagram.get("crossings", []) if not is_twist_box(c)), key=lambda c: math.dist(c["point"], point), default=None)
        if include_crossings and crossing is not None and math.dist(crossing["point"], point) <= 14 / self.scale:
            return "crossing", crossing["id"], crossing["point"]
        prepared = self._display_prepared()
        if prepared is not None:
            return prepared.nearest(point, 14 / self.scale)
        # Hit testing needs the original geometry, but only edges whose boxes
        # meet the cursor tolerance can win. Reuse their arrays until an edit
        # replaces the diagram, then project onto nearby segments in one batch.
        cached = getattr(self, '_pick_geometry', None)
        if cached is None or cached[0] is not self.diagram:
            arrays = [(edge['id'], np.asarray(edge.get('points', []), float))
                      for edge in self.diagram.get('edges', []) if len(edge.get('points', [])) > 1]
            # Overpassing portions are stored inside the compact box rather
            # than among external edges; they remain draggable where visible.
            from .twist_boxes import box_port_points
            for box in self.diagram.get('crossings', []):
                if not is_twist_box(box):
                    continue
                ports = box_port_points(box)
                for passage in box.get('passages', []):
                    if passage.get('over'):
                        a, b = passage['ports']
                        points = passage.get('points') or [ports[a], ports[b]]
                        arrays.append((box['ports'][a]['edge'], np.asarray(points, float)))
            low = np.asarray([p.min(axis=0) for _, p in arrays]).reshape(-1, 2)
            high = np.asarray([p.max(axis=0) for _, p in arrays]).reshape(-1, 2)
            cached = self._pick_geometry = (self.diagram, arrays, low, high)
        _, arrays, low, high = cached
        query = np.asarray(point, float)
        limit = 14 / self.scale
        nearby = np.flatnonzero(np.all(low <= query+limit, axis=1) & np.all(high >= query-limit, axis=1))
        best = (float("inf"), None, None)
        for index in nearby:
            eid, points = arrays[index]
            a, v = points[:-1], np.diff(points, axis=0)
            length2 = np.sum(v*v, axis=1)
            t = np.divide(np.sum((query-a)*v, axis=1), length2,
                          out=np.zeros(len(v)), where=length2 > 0.)
            projected = a+np.clip(t, 0., 1.)[:, None]*v
            distances = np.linalg.norm(projected-query, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] < best[0]:
                best = (float(distances[nearest]), eid, tuple(projected[nearest]))
        if best[0] <= 14 / self.scale:
            return "edge", best[1], best[2]
        return None

    def _clear_interaction(self):
        if getattr(self, 'box_edit_pending', None):
            self.task_id += 1
            self.box_edit_pending = None
            self.busy = False
        self.box_combine_first = None
        self.box_expansion_sides = None
        if self.drag and self.drag.get('kind') == 'energy':
            self.drag['stop'].set()
        self.motion_serial += 1
        self._stop_motion_poll()
        if self.drag_timer is not None:
            self.root.after_cancel(self.drag_timer)
            self.drag_timer = None
        self.drag = None
        self.curl_add = None
        self.monogons = []
        self.bigons = []
        self.simplify_continuation = False
        if self.tool.get() in ("Add curl", "Remove curl", "Remove bigon", "Simplify", "Energy region", "Add twists", "Combine boxes", "Box tangle"):
            self.tool.set("Select")

    def cancel_gesture(self, event=None, quiet=False):
        self.pan = None
        drag = self.drag
        completed_simplification = not drag and getattr(self, 'simplify_continuation', False)
        active = bool(drag or self.curl_add or getattr(self, 'box_edit_pending', None) or self.tool.get() in ("Remove curl", "Remove bigon", "Simplify", "Energy region", "Add twists", "Combine boxes", "Box tangle"))
        selection_only = not active and not quiet and (self.selected_crossing is not None or self.selected_component is not None or getattr(self, 'selected_box', None) is not None)
        if not quiet:
            self.selected_crossing = self.selected_component = None
            self.selected_box = None
        active = active or selection_only
        self._clear_interaction()
        if drag and drag.get("snapshot"):
            self._restore(drag["snapshot"])
        elif drag and drag.get("kind") == "raster" and self.history:
            self._restore(self.history.pop())
        self.canvas.configure(cursor="arrow")
        if active:
            if not quiet:
                self.status.set("Selection cleared." if selection_only else
                                "Left Simplify. Completed simplifications are kept." if completed_simplification else
                                "Edit cancelled. The previous diagram is restored.")
            self._refresh_text()
            self.redraw()
        return "break"

    def _edited_diagram(self, diagram, *, prepared_pd=None):
        """Keep the displayed graph and exported PD in the same accepted state."""
        from .diagram import crossing_free_count, pd_code
        if prepared_pd is None:
            pd = pd_code(diagram) if pd_preview_crossings(diagram) <= 10000 else None
            trivial = crossing_free_count(diagram, pd) if pd is not None else None
            prepared_pd = (pd, trivial, None)
        else:
            pd = prepared_pd[0]
        inherit_display_style(diagram, self.diagram)
        self.diagram = diagram
        self._invalidate_display_curves()
        self._prepared_box_pd = (diagram, *prepared_pd)
        self.display_paths = None
        self._pick_geometry = None
        self.result["diagram"] = diagram
        self.result["pd_code"] = pd
        self.selected_crossing = self.selected_component = None

    def _motion_pending(self):
        if self.drag and self.drag.get('kind') in ('bigon_scan', 'bigon_remove'):
            self.status.set('Checking the selected simplification… Escape cancels.')
            return True
        if self.drag and self.drag.get('kind') == 'energy':
            self.status.set('Energy minimization is running. Stop keeps the displayed layout; Escape restores the starting diagram.')
            return True
        if self.drag and self.drag.get("kind") in ("smooth", "select_pending"):
            self.status.set("Finish the current strand drag before this action, or press Escape to cancel it.")
            return True
        return False

    @staticmethod
    def _run_motion_worker(function):
        threading.Thread(target=function, daemon=True, name="knot-motion").start()



    def start_energy(self):
        if self._motion_pending():
            return
        if self.busy or not self.diagram:
            self.status.set('Load a diagram before minimizing energy.')
            return
        self.cancel_gesture(quiet=True)
        self.selected_crossing = self.selected_component = self.selected_box = None
        self.layer.set('diagram')
        self.tool.set('Energy region')
        if self.energy_full_screen.get():
            # Freeze the current viewport in model coordinates. Later zooming,
            # panning or resizing must not move the relaxation boundary.
            width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
            region = [self.to_model(x, y) for x, y in
                      ((0, 0), (width, 0), (width, height), (0, height))]
            self.canvas.configure(cursor='arrow')
            self._begin_energy(region)
            return
        self.canvas.configure(cursor='crosshair')
        self.status.set('Circle a region by dragging around it; release to start. Keep crossings away from the boundary. Twist boxes and their attachments stay fixed; outside strands relax toward perpendicular entry. Stop keeps changes; Escape cancels.')
        self.redraw()

    def _begin_energy(self, region):
        from .relaxation import validate_region
        try:
            region = validate_region(region).tolist()
        except ValueError as exc:
            self.drag = None
            self.status.set(str(exc))
            self.redraw()
            return
        snapshot = self._snapshot()
        self.motion_serial += 1
        token = self.motion_serial
        stop = threading.Event()
        self.drag = dict(kind='energy', region=region, snapshot=snapshot, token=token,
                         stop=stop, engine=None, inflight=True, moved=False)
        original = snapshot['diagram']
        cached_pd = getattr(self, '_prepared_box_pd', None)
        cached_pd = cached_pd[1:] if cached_pd and cached_pd[0] is self.diagram else None
        self.status.set('Preparing energy minimization… Boxes remain fixed. Stop is available immediately.')

        def initialize():
            try:
                from .relaxation import RelativeRelaxation
                engine = RelativeRelaxation(original, region, stop)
                if stop.is_set():
                    return
                # An ambient isotopy keeps incidence and orientation exactly.
                # In particular, do not re-expand a box's implicit crossings
                # on the Tk thread at every animation frame.
                engine.prepared_pd = cached_pd if cached_pd is not None else prepare_pd_preview(original)
                self.motion_events.put((token, 'energy_ready', engine))
            except Exception as exc:
                if not stop.is_set():
                    self.motion_events.put((token, 'energy_error', str(exc)))
        self._schedule_motion_poll()
        self.redraw()
        self._run_motion_worker(initialize)

    def _apply_energy_step(self):
        if self.drag_timer is not None:
            self.root.after_cancel(self.drag_timer)
            self.drag_timer = None
        drag = self.drag
        if not drag or drag.get('kind') != 'energy' or drag['inflight'] or drag['engine'] is None:
            return
        drag['inflight'] = True
        engine, token, stop = drag['engine'], drag['token'], drag['stop']
        colored = self.colored.get()
        appearance = {}
        inherit_display_style(appearance, self.diagram)
        speed = float(self.energy_speed.get())  # Snapshot Tk state on its owning thread.

        def calculate():
            try:
                changed, done = None, False
                started = time.monotonic()
                for _ in range(4):
                    result = engine.step(speed=speed)
                    if result is None:
                        done = True
                        break
                    changed = result
                    if time.monotonic()-started > .025:
                        break
                inherit_display_style(changed, appearance)
                prepared = paths(changed, colored) if changed is not None else None
                if not stop.is_set():
                    self.motion_events.put((token, 'energy_update', (changed, colored, prepared, done, engine.energy, engine.initial_energy, engine.iterations, engine.reason)))
            except Exception as exc:
                if not stop.is_set():
                    self.motion_events.put((token, 'energy_error', str(exc)))
        self._run_motion_worker(calculate)

    def _energy_result(self, kind, payload):
        drag = self.drag
        if kind == 'energy_error':
            self._finish_energy('Energy minimization stopped: '+payload)
        elif kind == 'energy_ready':
            drag['engine'] = payload
            self._apply_energy_step()
        elif kind == 'energy_update':
            changed, colored, strokes, done, energy, initial, count, reason = payload
            if changed is not None:
                self._edited_diagram(changed, prepared_pd=getattr(drag['engine'], 'prepared_pd', None))
                self.display_paths = (changed, colored, strokes)
                drag['moved'] = True
                self._refresh_text()
                self.redraw()
            if done:
                self._finish_energy(reason)
            else:
                reduction = 100*(initial-energy)/max(abs(initial), 1e-20)
                resamples = (self.diagram.get('relaxation') or {}).get('resamples', 0)
                baseline = 'since sampling refresh' if resamples else 'since start'
                sampling = f' · sampling refresh {resamples}' if resamples else ''
                self.status.set(f'Layout energy · relaxing at {self.energy_speed.get():.2g}× · energy {energy:.3f} ({reduction:.1f}% lower {baseline}) · step {count}{sampling}. Stop keeps this layout; Escape cancels.')
                self.drag_timer = self.root.after(16, self._apply_energy_step)

    def _finish_energy(self, message='Energy minimization stopped.'):
        drag = self.drag
        if not drag or drag.get('kind') != 'energy':
            return
        if drag['moved']:
            self.history.append(drag['snapshot'])
            self.history = self.history[-20:]
            self.future.clear()
            message += ' Undo restores the entire relaxation.'
        self._clear_interaction()
        self.canvas.configure(cursor='arrow')
        self._refresh_text()
        self.redraw()
        self.status.set(message)

    def stop_energy(self):
        if self.drag and self.drag.get('kind') == 'energy':
            self._finish_energy()
        elif self.tool.get() == 'Energy region':
            self.cancel_gesture()

    def _stop_energy_press(self, event):
        # Cancel on the first press, before a worker frame or native focus
        # transition can move the button or consume its release. The ordinary
        # command remains available for keyboard activation and accessibility.
        self.stop_energy()
        return "break"

    def start_add_curl(self):
        if self._motion_pending():
            return
        if self.busy or not self.diagram:
            self.status.set("Load a diagram before adding a curl.")
            return
        self.cancel_gesture(quiet=True)
        sign = CurlSignDialog(self.root, "Add curl · Reidemeister I").result
        if sign not in (-1, 1):
            return
        self.layer.set("diagram")
        self.selected_crossing = self.selected_component = self.selected_box = None
        self.tool.set("Add curl")
        self.curl_add = {"sign": sign, "stage": "point"}
        self.canvas.configure(cursor="crosshair")
        self.status.set(f"Add {'positive' if sign > 0 else 'negative'} curl: click a point on a strand away from crossings. Escape cancels.")
        self.redraw()

    def start_remove_curl(self):
        if self._has_twist_boxes():
            return self.start_remove_bigon(curls_only=True)
        if self._motion_pending():
            return
        if self.busy or not self.diagram:
            self.status.set("Load a diagram before removing a curl.")
            return
        self.cancel_gesture(quiet=True)
        self.selected_crossing = self.selected_component = None
        from .reidemeister import removable_monogons
        try:
            self.monogons = removable_monogons(self.diagram)
        except ValueError as exc:
            self.status.set("Cannot find removable curls: " + str(exc))
            return
        self.layer.set("diagram")
        self.tool.set("Remove curl")
        self.canvas.configure(cursor="crosshair")
        if self.monogons:
            self.status.set(f"Remove curl: click inside one of {len(self.monogons)} shaded empty monogon regions. Escape cancels.")
        else:
            self.status.set("No removable empty monogon regions were found. A region containing another strand cannot be removed by Reidemeister I.")
        self.redraw()

    def _curl_click(self, point):
        from .reidemeister import add_curl, remove_curl
        if self.curl_add:
            if self.curl_add["stage"] == "point":
                if self._has_twist_boxes():
                    from .twist_boxes import box_corners
                    if any(is_twist_box(box) and point_in_polygon(point, box_corners(box))
                           for box in self.diagram['crossings']):
                        self.status.set('Choose a strand point outside the twist boxes. Expand a box first to edit its interior.')
                        return
                nearest = self._nearest(point)
                if not nearest or nearest[0] != "edge":
                    self.status.set("Choose a point on a strand away from crossings, then choose its side.")
                    return
                self.curl_add.update(stage="side", edge_id=nearest[1], point=nearest[2])
                self.status.set("Now click on the side of the strand where the curl should appear. Leave room for the loop. Escape cancels.")
                self.redraw()
                return
            edit = self.curl_add
            if self._has_twist_boxes():
                self._add_box_curl_in_background(edit, point)
                return
            try:
                changed = add_curl(self.diagram, edit["edge_id"], edit["point"], sign=edit["sign"], side_point=point)
            except ValueError as exc:
                self.status.set("Cannot add curl here: " + str(exc) + " Choose another side point, or press Escape and pick another strand point.")
                return
            message = f"Added a {'positive' if edit['sign'] > 0 else 'negative'} curl (Reidemeister I). The diagram and PD are updated."
        else:
            region = next((r for r in self.monogons if point_in_polygon(point, r["polygon"])), None)
            if region is None:
                self.status.set("Click inside a shaded monogon region to remove that curl. Escape cancels.")
                return
            if self._has_twist_boxes():
                self._bigon_click(point, monogon=region)
                return
            try:
                changed = remove_curl(self.diagram, region["crossing_id"], edge_id=region["edge_id"])
            except ValueError as exc:
                self.status.set("Cannot remove curl: " + str(exc))
                return
            message = f"Removed a {'positive' if region['sign'] > 0 else 'negative'} curl (Reidemeister I). The diagram and PD are updated."
        self._remember()
        self._edited_diagram(changed)
        self._clear_interaction()
        self.canvas.configure(cursor="arrow")
        self.status.set(message)
        self._refresh_text()
        self.redraw()

    def _add_box_curl_in_background(self, edit, side_point):
        """Keep geometry validation and compact-box PD work off Tk's thread."""
        snapshot = self._snapshot()
        self.motion_serial += 1
        token, colored = self.motion_serial, self.colored.get()
        self.drag = dict(kind='bigon_remove', token=token, inflight=True,
                         snapshot=snapshot, tool='Add curl', removal='add_curl',
                         curl_sign=edit['sign'])
        self.status.set('Adding the curl outside the fixed boxes… Escape cancels.')

        def calculate():
            try:
                from .reidemeister import add_curl
                changed = add_curl(snapshot['diagram'], edit['edge_id'], edit['point'],
                                   sign=edit['sign'], side_point=side_point)
                inherit_display_style(changed, snapshot['diagram'])
                prepared_pd = prepare_pd_preview(changed)
                try: prepared = (colored, paths(changed, colored))
                except Exception: prepared = None
            except Exception as exc:
                self.motion_events.put((token, 'bigon_error', str(exc)))
            else:
                self.motion_events.put((token, 'curl_added', (changed, prepared, prepared_pd)))

        self._schedule_motion_poll()
        self._run_motion_worker(calculate)

    def start_simplify(self):
        self.start_remove_bigon(simplify=True)

    def start_remove_bigon(self, simplify=False, *, curls_only=False):
        if self._motion_pending():
            return
        if self.busy or not self.diagram:
            self.status.set('Load a diagram before simplifying it.')
            return
        self.cancel_gesture(quiet=True)
        self.selected_crossing = self.selected_component = self.selected_box = None
        self.layer.set('diagram')
        self.tool.set('Remove curl' if curls_only else 'Simplify' if simplify else 'Remove bigon')
        self.canvas.configure(cursor='crosshair')
        self.motion_serial += 1
        token = self.motion_serial
        original = copy_diagram(self.diagram)
        self.drag = dict(kind='bigon_scan', token=token, inflight=True, tool=self.tool.get())
        self.status.set('Finding removable curls… Escape cancels.' if curls_only else
                        'Finding removable curls and bigons… Escape cancels.' if simplify else
                        'Finding empty removable bigons… Escape cancels.')
        def calculate():
            try:
                if curls_only:
                    from .reidemeister import removable_monogons
                    regions = removable_monogons(original)
                else:
                    from .reidemeister_ii import removable_bigons
                    regions = removable_bigons(original)
                    if simplify:
                        from .reidemeister import removable_monogons
                        regions = (removable_monogons(original), regions)
            except Exception as exc:
                self.motion_events.put((token, 'bigon_error', str(exc)))
            else:
                self.motion_events.put((token, 'monogons_ready' if curls_only else
                                       'simplify_ready' if simplify else 'bigons_ready', regions))
        self._schedule_motion_poll()
        self.redraw()
        self._run_motion_worker(calculate)

    def _bigon_click(self, point, monogon=None):
        region = monogon or next((r for r in self.bigons if point_in_polygon(point, r['polygon'])), None)
        if region is None:
            self.status.set('Click inside a green curl or purple bigon. Escape cancels.' if self.tool.get() == 'Simplify' else
                            'Click inside a shaded empty bigon to remove its two crossings. Escape cancels.')
            return
        snapshot = self._snapshot()
        self.motion_serial += 1
        token = self.motion_serial
        self.drag = dict(kind='bigon_remove', token=token, inflight=True, snapshot=snapshot,
                         tool=self.tool.get(), removal='curl' if monogon else 'bigon')
        original, region_id = snapshot['diagram'], region['id']
        colored = self.colored.get()
        continue_simplifying = self.tool.get() == 'Simplify'
        self.status.set('Removing the selected curl… Escape cancels.' if monogon else
                        'Removing the selected bigon… Escape cancels.')
        def calculate():
            try:
                if monogon:
                    from .reidemeister import remove_curl
                    changed = remove_curl(original, monogon['crossing_id'], edge_id=monogon['edge_id'])
                else:
                    from .reidemeister_ii import remove_bigon
                    changed = remove_bigon(original, region_id)
            except Exception as exc:
                self.motion_events.put((token, 'bigon_error', str(exc)))
            else:
                inherit_display_style(changed, original)
                prepared_pd = prepare_pd_preview(changed)
                try:prepared = (colored, paths(changed, colored))
                except Exception:prepared = None
                if continue_simplifying:
                    # Discover the new faces on the worker, as part of this
                    # removal's cancellable transaction. Never reuse old IDs
                    # or polygons after the crossing graph has changed.
                    remaining, scan_error = None, None
                    try:
                        from .reidemeister import removable_monogons
                        from .reidemeister_ii import removable_bigons
                        remaining = (removable_monogons(changed), removable_bigons(changed))
                    except Exception as exc:
                        scan_error = str(exc)
                    self.motion_events.put((token, 'simplified', (changed, prepared, remaining, scan_error, prepared_pd)))
                else:
                    self.motion_events.put((token, 'bigon_removed', (changed, prepared, prepared_pd)))
        self._schedule_motion_poll()
        self._run_motion_worker(calculate)

    def _bigon_result(self, kind, payload):
        drag = self.drag
        if kind in ('bigon_removed', 'simplified', 'curl_added'):
            changed, prepared = payload[:2]
            pd_index = 4 if kind == 'simplified' else 2
            prepared_pd = payload[pd_index] if len(payload) > pd_index else None
            self._edited_diagram(changed, prepared_pd=prepared_pd)
            if prepared is not None:
                colored, strokes = prepared
                self.display_paths = (changed, colored, strokes)
            self.history.append(drag['snapshot'])
            self.history = self.history[-20:]
            self.future.clear()
            self._clear_interaction()
            self.canvas.configure(cursor='arrow')
            self._refresh_text()
            action = (f"Added a {'positive' if drag.get('curl_sign', 1) > 0 else 'negative'} curl (Reidemeister I)."
                      if kind == 'curl_added' else 'Removed the curl by Reidemeister I.'
                      if drag.get('removal') == 'curl' else
                      'Removed the bigon: two crossings removed by Reidemeister II.')
            encoding = ('The diagram is updated; PD preview is unavailable.'
                        if prepared_pd is not None and prepared_pd[2] else 'Diagram and PD updated.')
            self.status.set(action + ' ' + encoding + ' Undo restores it.')
            if kind == 'simplified':
                remaining, scan_error = payload[2:4]
                if remaining is not None and any(remaining):
                    self.monogons, self.bigons = remaining
                    self.tool.set('Simplify')
                    self.simplify_continuation = True
                    self.canvas.configure(cursor='crosshair')
                    self.status.set(self.status.get() +
                                    f' Simplify again: click a green curl ({len(self.monogons)}) or purple bigon ({len(self.bigons)}). Escape finishes.')
                elif scan_error:
                    self.status.set(self.status.get() + ' Could not find further removable regions: ' + scan_error)
        else:
            self.drag = None
            self._stop_motion_poll()
            if kind == 'monogons_ready':
                self.monogons = payload
                self.status.set(f'Remove curl: click inside one of {len(payload)} shaded empty monogon regions. Escape cancels.'
                                if payload else 'No removable empty monogon regions were found outside the boxes.')
            elif kind == 'simplify_ready':
                self.monogons, self.bigons = payload
                self.status.set(f'Simplify: click a green curl ({len(self.monogons)}) or purple bigon ({len(self.bigons)}). Escape cancels.'
                                if self.monogons or self.bigons else 'No removable empty curls or bigons were found. Escape leaves Simplify.')
            elif kind == 'bigons_ready':
                self.bigons = payload
                self.status.set(f'Remove bigon: click inside one of {len(payload)} shaded empty bigon regions. Escape cancels.'
                                if payload else 'No removable empty bigons were found. A strand must pass over the other at both crossings, and the region must be empty.')
            elif drag.get('removal') == 'add_curl':
                self.status.set('Cannot add curl here: ' + payload + ' Choose another side point, or press Escape and pick another strand point.')
            elif drag['kind'] == 'bigon_scan':
                self.status.set('Cannot find removable regions: ' + payload + ' Press Escape to leave this tool.')
            else:
                self.status.set('Cannot simplify this region: ' + payload + ' Choose another shaded region or press Escape.')
        self.redraw()

    def _hover_box_side(self, event):
        if self.pan or self.busy or self.drag or self.layer.get() != 'diagram' or self.tool.get() not in ('Select', 'Move geometry'):
            return
        from .box_resize import box_side_hit
        hit = box_side_hit(self.diagram or {}, self.to_model(event.x, event.y), 6/self.scale)
        self.canvas.delete('box-resize-hover')
        cursor = 'fleur' if self.tool.get() == 'Move geometry' else 'arrow'
        if hit:
            _, _, _, _, a, b = hit
            cursor = 'sb_h_double_arrow' if abs(b[1]-a[1]) >= abs(b[0]-a[0]) else 'sb_v_double_arrow'
            self.canvas.create_line(*self.to_canvas(a), *self.to_canvas(b), fill='#2563eb', width=3,
                                    tags=('box-resize-hover',))
        self.canvas.configure(cursor=cursor)

    def _mouse_down(self, event):
        self.canvas.focus_set()
        if self.busy:
            return
        if self.drag:
            self.status.set("An edit is running. Stop keeps energy-minimization changes; Escape cancels the current edit.")
            return
        point = self.to_model(event.x, event.y)
        if self.layer.get() == 'source' and self.tool.get() == 'Block eraser':
            self._erase_ink_component(point)
            return
        if self.layer.get() == 'source' and self.tool.get() == 'Mark twist box':
            self.drag = {'kind': 'box_rectangle', 'anchor': point, 'last': point}
            self.redraw()
            return
        if self.layer.get() == 'source' and self.tool.get() in ('Review labels', 'Review twist boxes', 'Select'):
            box = self._source_box_at(point)
            if box is not None:
                self.selected_box = box['id']
                self.edit_twist_box(box['id'])
                self._refresh_conversion_buttons()
                self.redraw()
                return
            self._label_click(point, keep=bool(event.state & 0x0001))
            return
        if self.tool.get() == 'Energy region' and self.diagram and self.layer.get() == 'diagram':
            self.drag = dict(kind='energy_lasso', region=[point])
            self.redraw()
            return
        if self.diagram and self.layer.get() == 'diagram':
            if self.tool.get() == 'Add twists':
                self.drag = {'kind': 'twist_slice', 'anchor': point, 'last': point}
                self.redraw()
                return
            if self.tool.get() == 'Box tangle':
                self.drag = {'kind': 'box_lasso', 'region': [point]}
                self.redraw()
                return
            if self.tool.get() == 'Combine boxes':
                self._combine_box_click(point)
                return
        if self.layer.get() == "source" and self.tool.get() in ("Pencil", "Eraser"):
            if self.raster_image is None:
                self.new_sketch()
            self._remember()
            self.label_detection = None  # Pencil/eraser invalidate the pixel masks.
            self.box_detection = None
            self.raster_image = self.raster_image.copy()
            self.drag = {"kind": "raster", "last": point, "dark": self._dark_drawing()}
            if self.tool.get() == "Pencil":
                from .drawing import PencilGesture
                self.drag['pencil'] = PencilGesture(self.raster_image, self.brush_size.get(), dark=self.drag['dark'])
            self._draw_stroke(point, point, under=bool(event.state & 0x0001))
            return
        if not self.diagram or self.layer.get() != "diagram":
            return
        if self.tool.get() in ('Select', 'Move geometry'):
            from .box_resize import box_side_hit
            hit = box_side_hit(self.diagram, point, 6/self.scale)
            if hit is not None:
                _, ident, side, anchor, a, b = hit
                length = math.dist(a, b)
                normal = [(b[1]-a[1])/length, (a[0]-b[0])/length]
                self.selected_box = ident
                self.selected_component = self.selected_crossing = None
                self._start_strand_drag(ident, anchor, point, False,
                                        box_resize={'id': ident, 'side': side, 'normal': normal})
                return
        if self.tool.get() == 'Select':
            box = next((b for b in box_shapes(self.diagram) if point_in_polygon(point, b['corners'])), None)
            if box is not None:
                self.selected_box = box['id']
                self.selected_component = self.selected_crossing = None
                self.drag = {'kind': 'select_pending', 'mouse_start': point,
                             'screen_start': (event.x, event.y), 'tool': 'Select', 'box': True}
                self.status.set('Twist box selected. Drag any edge to make it thinner, thicker, shorter or longer. You can also edit its value, expand it, or combine adjacent boxes. Strand alignment is automatic where there is clear room.')
                if hasattr(self, 'add_twists_button'):
                    self._refresh_conversion_buttons()
                self.redraw()
                return
        self.selected_box = None
        if self.tool.get() in ('Remove bigon', 'Simplify'):
            monogon = next((r for r in self.monogons if point_in_polygon(point, r['polygon'])), None) if self.tool.get() == 'Simplify' else None
            self._bigon_click(point, monogon=monogon)
            return
        if self.tool.get() in ("Add curl", "Remove curl"):
            self._curl_click(point)
            return
        # A crossing's generous click target is useful for switching it, but
        # must not hide nearby strands when selecting an arc for an RIII slide.
        nearest = self._nearest(point, include_crossings=self.tool.get() != "Move geometry")
        self.selected_crossing = self.selected_component = None
        if nearest:
            kind, ident, anchor = nearest
            if kind == "crossing":
                self.selected_crossing = ident
            else:
                edge = next(edge for edge in self.diagram["edges"] if edge["id"] == ident)
                self.selected_component = edge.get("component", 0)
            tool = self.tool.get()
            if tool == 'Select':
                # Select immediately, but defer all motion preparation until
                # the pointer has travelled four screen pixels.
                self.drag = {'kind': 'select_pending', 'mouse_start': point,
                             'screen_start': (event.x, event.y), 'tool': 'Select'}
            elif tool == "Move geometry":
                if kind != "edge":
                    self.status.set("Pick a strand away from the crossing so that the moving branch is unambiguous.")
                    return
                self._start_strand_drag(ident, anchor, point, bool(event.state & 0x0001))
                return
            elif tool == "Switch crossing":
                self.switch_selected()
            elif tool == "Reverse component":
                self.reverse_selected()
            elif tool == "Delete component":
                self.delete_selected()
        if hasattr(self, 'add_twists_button'):
            self._refresh_conversion_buttons()
        self.redraw()

    def _promote_selection_drag(self, event):
        pending = self.drag
        if math.dist(pending['screen_start'], (event.x, event.y)) < 4:
            return False
        point = pending['mouse_start']
        nearest = self._nearest(point, include_crossings=False)
        if nearest is None:
            return False
        _, ident, anchor = nearest
        # A click can select a crossing, but its center does not specify
        # which branch to drag. Nearby, clearly separated arcs remain usable.
        ambiguous = any(not is_twist_box(c) and math.dist(anchor, c['point'])*self.scale < 4
                        for c in self.diagram['crossings'])
        if ambiguous or (pending.get('box') and math.dist(point, anchor)*self.scale > 5):
            self.status.set('Pick a strand away from the crossing or box border to move it unambiguously.')
            return False
        self.selected_crossing = self.selected_box = None
        self.selected_component = next(e.get('component', 0) for e in self.diagram['edges'] if e['id'] == ident)
        self._start_strand_drag(ident, anchor, point, bool(event.state & 0x0001))
        return True

    def _start_strand_drag(self, ident, anchor, point, under, box_resize=None):
        self.motion_serial += 1
        token = self.motion_serial
        radius = self.move_radius.get() / self.scale
        snapshot = self._snapshot()
        original = snapshot["diagram"]
        self.drag = {"kind": "smooth", "engine": None, "anchor": anchor, "mouse_start": point,
                     "snapshot": snapshot, "moved": False, "target": anchor, "last_target": anchor,
                     "rejection": None, "moves": [], "token": token, "inflight": True, "ending": False,
                     "attempted_target": None, "attempted_under": None, "under":under,
                     "pending_targets": [], "tool": self.tool.get()}
        self.drag['box_resize'] = box_resize
        cached_pd = getattr(self, '_prepared_box_pd', None)
        cached_pd = cached_pd[1:] if cached_pd and cached_pd[0] is self.diagram else None
        self.status.set("Preparing box resize… Drag its edge; Escape cancels." if box_resize else
                        "Preparing smooth strand motion… Hold Shift as new crossings form to pass underneath. Escape cancels.")

        def initialize():
            try:
                if box_resize:
                    from .box_resize import BoxResize
                    engine = BoxResize(original, ident, box_resize['side'], anchor, radius=radius)
                    engine.prepared_pd = cached_pd if cached_pd is not None else prepare_pd_preview(original)
                elif any(is_twist_box(v) for v in original.get('crossings', [])):
                    from .box_motion import BoxSmoothDrag
                    engine = BoxSmoothDrag(original, ident, anchor, radius=radius, under=under)
                else:
                    from .motion import SmoothDrag
                    engine = SmoothDrag(original, ident, anchor, radius=radius, under=under)
            except Exception as exc:
                self.motion_events.put((token, "initialization_error", str(exc)))
            else:
                self.motion_events.put((token, "ready", engine))
        self._schedule_motion_poll()
        # Paint the selected strand before starting the Python-heavy
        # initializer. Rebuilding hundreds of Tk items after starting
        # that worker repeatedly hands it the GIL at every Tk call,
        # making mouse-down itself stall even though drawing is cheap.
        self.redraw()
        self._run_motion_worker(initialize)
        return

    def _dark_drawing(self):
        """Choose drawing colors from recognition's polarity or the pixels."""
        preparation = (self.result or {}).get('diagnostics', {}).get('preprocessing', {})
        if preparation.get('transform') == 'board_photo' and preparation.get('polarity') in ('dark', 'light'):
            return preparation['polarity'] == 'dark'
        from .drawing import dark_background
        return self.raster_image is not None and dark_background(self.raster_image)

    def _erase_ink_component(self, point):
        if self.raster_image is None:
            return
        from .drawing import erase_connected_ink
        changed, count = erase_connected_ink(self.raster_image, point, dark=self._dark_drawing())
        if changed is None:
            self.status.set('Click directly on ink to erase a connected region.')
            return
        self._remember()
        self.raster_image = changed
        self.raster_changed = True
        self.label_detection = self.box_detection = None
        self.selected_box = None
        self._refresh_text()
        self.redraw()
        self.status.set(f'Erased one connected ink region ({count} pixels). Undo restores it. Recognize drawing when ready.')

    def _draw_stroke(self, a, b, under=False):
        if self.drag and self.drag.get('pencil') is not None:
            self.raster_image = self.drag['pencil'].add(a, b, under=under)
            self.redraw()
            return
        dark = self.drag['dark'] if self.drag and 'dark' in self.drag else self._dark_drawing()
        color = ("black" if dark else "white") if self.tool.get() == "Eraser" else ("white" if dark else "#111111")
        width = max(1, round(self.brush_size.get()))
        draw = ImageDraw.Draw(self.raster_image)
        draw.line([a, b], fill=color, width=width)
        r = width / 2
        draw.ellipse((b[0] - r, b[1] - r, b[0] + r, b[1] + r), fill=color)
        self.redraw()

    def _mouse_drag(self, event):
        if self.pan:
            return self._pan(event)
        if not self.drag:
            return
        if self.drag.get("ending"):
            return
        if self.drag['kind'] == 'select_pending' and not self._promote_selection_drag(event):
            return
        point = self.to_model(event.x, event.y)
        if self.drag['kind'] in ('box_rectangle', 'twist_slice'):
            self.drag['last'] = point
            self.redraw()
            return
        if self.drag['kind'] in ('energy', 'bigon_scan', 'bigon_remove'):
            return
        if self.drag['kind'] in ('energy_lasso', 'box_lasso'):
            if math.dist(point, self.drag['region'][-1])*self.scale >= 3:
                self.drag['region'].append(point)
                self.redraw()
            return
        if self.drag["kind"] == "raster":
            self._draw_stroke(self.drag["last"], point, under=bool(event.state & 0x0001))
            self.drag["last"] = point
            return
        self._queue_smooth_target(point, under=bool(event.state & 0x0001))

    def _shift_during_drag(self, event, pressed):
        drag = self.drag
        if drag and drag.get('box_resize'):
            return
        if not drag or drag.get("kind") != "smooth" or drag.get("ending"):
            return
        if pressed == drag["under"]:
            return
        # Preserve the last position reached under the previous modifier, even
        # when it is still waiting for a worker. A key change affects future
        # crossing births, never the crossing choices of an earlier request.
        point = tuple(drag["target"][i] - drag["anchor"][i] + drag["mouse_start"][i] for i in (0, 1))
        self._queue_smooth_target(point, under=pressed)
        self.status.set("New crossings: " + ("under (Shift held)." if pressed else "over (Shift released).") + " Existing crossings keep their types.")

    def _queue_smooth_target(self, point, under=None):
        drag = self.drag
        if drag.get('box_resize'):
            under = False
        drag["target"] = tuple(drag["anchor"][i] + point[i] - drag["mouse_start"][i] for i in (0, 1))
        if drag.get('box_resize'):
            normal = drag['box_resize']['normal']
            distance = sum((point[i]-drag['mouse_start'][i])*normal[i] for i in (0, 1))
            drag['target'] = tuple(drag['anchor'][i]+distance*normal[i] for i in (0, 1))
        if under is not None:
            drag["under"] = bool(under)
        request = (drag["target"], drag["under"])
        pending = drag["pending_targets"]
        if pending and pending[-1][1] == request[1]:
            pending[-1] = request
        else:
            pending.append(request)
        # Dispatch on the next Tk turn instead of deliberately delaying motion.
        # Coalesce positions only within the same modifier interval. Retaining
        # Shift transitions prevents a busy worker from losing birth choices.
        if self.drag_timer is None:
            self.drag_timer = self.root.after(0, self._apply_smooth_target)

    def _apply_smooth_target(self):
        if self.drag_timer is not None:
            self.root.after_cancel(self.drag_timer)
            self.drag_timer = None
        drag = self.drag
        if not drag or drag.get("kind") != "smooth":
            return
        if drag["inflight"] or drag["engine"] is None:
            return
        while drag["pending_targets"]:
            target, under = drag["pending_targets"].pop(0)
            same_attempt = (drag["attempted_target"] is not None
                            and math.dist(target, drag["attempted_target"]) < 1e-6
                            and under == drag["attempted_under"])
            if same_attempt:
                continue
            drag["attempted_target"], drag["attempted_under"] = target, under
            if math.dist(target, drag["last_target"]) < 1e-6:
                drag["rejection"] = None
                continue
            break
        else:
            if drag["ending"]:
                self._finish_smooth_drag()
            return
        drag["inflight"] = True
        engine, token = drag["engine"], drag["token"]
        colored = self.colored.get()  # Tk variables are read only on Tk's thread.
        appearance = {}
        inherit_display_style(appearance, self.diagram)

        def calculate():
            try:
                changed = engine.move(target, under=under)
                moves = list(getattr(engine, "last_moves", []))
            except Exception as exc:
                self.motion_events.put((token, "rejected", str(exc)))
            else:
                # Crossing gap construction can be significant on a dense
                # diagram. Prepare the exact accepted graph's strokes here;
                # the UI commits graph, PD and its matching paint data together.
                inherit_display_style(changed, appearance)
                try:
                    prepared_paths = (colored, paths(changed, colored))
                except Exception:
                    # Rendering failure must never relabel an accepted geometry
                    # move as rejected: the engine has already advanced.
                    prepared_paths = None
                if drag.get('box_resize'):
                    prepared_pd = engine.prepared_pd
                elif any(is_twist_box(v) for v in changed['crossings']):
                    # One worker owns a gesture's exact cache. Incidence or
                    # passage changes invalidate it; geometry-only frames can
                    # reuse the full encoding without expanding the boxes.
                    if not hasattr(engine, '_pd_preview_cache'):
                        engine._pd_preview_cache = PDPreviewCache()
                    prepared_pd = engine._pd_preview_cache.prepare(changed)
                else:
                    prepared_pd = prepare_pd_preview(changed)
                self.motion_events.put((token, "accepted", (changed, moves, target, prepared_paths, prepared_pd)))
        self._run_motion_worker(calculate)

    def _finish_smooth_drag(self):
        drag = self.drag
        if drag["moved"]:
            self.history.append(drag["snapshot"])
            self.history = self.history[-20:]
            self.future.clear()
            detail = "; ".join(map(str, drag["moves"]))
            message = ("Box resize saved" if drag.get('box_resize') else "Smooth drag saved") + (": " + detail if detail else "") + ". Undo restores the entire drag."
            if drag["rejection"]:
                message += " The final pointer position was rejected; the last accepted position was saved."
            self.status.set(message)
        elif drag["rejection"]:
            self.status.set("No move made: " + drag["rejection"])
        else:
            self.status.set("No geometry change.")
        self.drag = None
        self._stop_motion_poll()
        self._refresh_text()

    def _mouse_up(self, event):
        if self.pan:
            return self._end_pan(event)
        if self.drag and self.drag['kind'] == 'select_pending':
            if not self._promote_selection_drag(event):
                self.drag = None
                self._refresh_text()
                return
        if self.drag:
            if self.drag['kind'] == 'twist_slice':
                segment = [self.drag['anchor'], self.to_model(event.x, event.y)]
                self.drag = None
                self._finish_twist_slice(segment)
                return
            if self.drag['kind'] == 'box_lasso':
                import cv2
                import numpy as np
                points = self.drag['region']+[self.to_model(event.x, event.y)]
                polygon = cv2.approxPolyDP(np.asarray(points, np.float32), 1/self.scale, True).reshape(-1, 2).tolist()
                self.drag = None
                self._finish_box_tangle(polygon)
                return
            if self.drag['kind'] == 'box_rectangle':
                a = self.drag['anchor']
                b = self.to_model(event.x, event.y)
                self.drag = None
                self._finish_manual_box([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])])
                return
            if self.drag['kind'] in ('energy', 'bigon_scan', 'bigon_remove'):
                return
            if self.drag['kind'] == 'energy_lasso':
                import cv2
                import numpy as np
                points = self.drag['region']+[self.to_model(event.x, event.y)]
                region = cv2.approxPolyDP(np.asarray(points, np.float32), 1/self.scale, True).reshape(-1, 2).tolist()
                self.drag = None
                self._begin_energy(region)
                return
            if self.drag.get("ending"):
                return
            if self.drag["kind"] == "raster":
                point = self.to_model(event.x, event.y)
                if math.dist(self.drag['last'], point) > 1e-6:
                    self._draw_stroke(self.drag['last'], point, under=bool(event.state & 0x0001))
                self.raster_changed = True
                self.status.set("Drawing changed. Choose Recognize drawing to update the diagram and PD code.")
            else:
                point = self.to_model(event.x, event.y)
                self._queue_smooth_target(point, under=bool(event.state & 0x0001))
                self.drag["ending"] = True
                if self.drag_timer is not None:
                    self.root.after_cancel(self.drag_timer)
                    self.drag_timer = None
                self._apply_smooth_target()
                if self.drag:
                    self.status.set("Finishing this drag's geometry checks… Escape cancels the entire drag.")
                return
            self._refresh_text()
        self.drag = None

    def _mutate(self, function_name, ident, message):
        if self.diagram is None or self.busy:
            return
        if self._motion_pending():
            return
        self.cancel_gesture(quiet=True)
        from . import diagram as model
        self._remember()
        self._prepared_box_pd = None
        self.display_paths = None  # These commands can mutate the same object.
        self._pick_geometry = None
        self._invalidate_display_curves()
        try:
            changed = getattr(model, function_name)(self.diagram, ident)
            if changed is not None:
                inherit_display_style(changed, self.diagram)
                self.diagram = changed
            self.result["diagram"] = self.diagram
            self.result["pd_code"] = self._pd() if pd_preview_crossings(self.diagram) <= 10000 else None
        except Exception as exc:
            self._restore(self.history.pop())
            messagebox.showerror("Cannot edit diagram", str(exc))
            return
        self.status.set(message)
        self._refresh_text()
        self.redraw()

    def switch_selected(self):
        if self.selected_crossing is not None:
            self._mutate("switch_crossing", self.selected_crossing, "Switched the selected crossing. The PD code has been updated.")
        else:
            self.status.set("Select a crossing first.")

    def reverse_selected(self):
        if self.selected_component is not None:
            self._mutate("reverse_component", self.selected_component, "Reversed the selected component's orientation.")
            self.oriented.set(True)
            self.redraw()
        else:
            self.status.set("Select a strand away from a crossing to choose its component.")

    def delete_selected(self):
        if self.selected_component is not None:
            self._mutate("delete_component", self.selected_component, "Deleted the selected component. Undo restores it.")
            self.selected_component = self.selected_crossing = None
            self.redraw()
        else:
            self.status.set("Select a strand away from a crossing to choose its component.")

    def diagram_to_raster(self):
        if not self.diagram or self.busy:
            return
        if self._motion_pending():
            return
        try:
            image = render_editable_image(self.diagram, colored=self.colored.get())
        except ValueError as exc:
            self.status.set("Cannot convert this diagram to a drawing: " + str(exc))
            return
        self.cancel_gesture(quiet=True)
        self._remember()
        self.raster_image = image
        # Keep the previous graph in Undo, not as an alternate active view.
        self.diagram = None
        self.result = {}
        self.selected_crossing = self.selected_component = None
        self.label_detection = None
        self.box_detection = None
        self.selected_box = None
        self.raster_changed = False
        self.layer.set("source")
        self.tool.set("Pencil")
        self.fit_view()
        self._refresh_text()
        self.status.set("The displayed diagram is now an editable raster. Draw or erase, then recognize it to reconstruct a new graph.")

    def copy_pd(self):
        if self.busy or self._motion_pending():
            return
        try:
            text = self._pd_string()
        except Exception as exc:
            messagebox.showerror("PD unavailable", str(exc))
            return
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status.set("Copied the previous diagram's PD. Recognize the changed drawing to update it." if self.raster_changed else "Copied the PD code." + (" It is provisional until you review the diagram." if self.result.get("status") != "ok" else ""))
            return True

    def save_as(self):
        if self.busy or self._motion_pending():
            return
        if not self.diagram:
            self.status.set("Recognize a diagram before saving. The format menu offers JSON, PD text, TikZ, PNG, and SVG.")
            return
        previous = getattr(self, '_save_format', '.json')
        default = next(label for label, extension in SAVE_FORMATS if extension == previous)
        chosen = tk.StringVar(master=self.root, value=default)
        filename = filedialog.asksaveasfilename(parent=self.root, title="Save diagram",
            initialfile='diagram'+previous, defaultextension='', typevariable=chosen,
            filetypes=[(label, '*'+extension) for label, extension in SAVE_FORMATS])
        if not filename:
            return
        extension = dict(SAVE_FORMATS).get(chosen.get(), previous)
        path = Path(filename)
        # The selected format controls encoding and extension together. Some
        # native dialogs retain the initial filename's old extension.
        if path.suffix.lower() != extension:
            path = (path.with_suffix(extension) if path.suffix.lower() in dict(SAVE_FORMATS).values()
                    else Path(str(path)+extension))
            if path.exists() and not messagebox.askyesno("Replace file?",
                    f"{path.name} already exists. Replace it?", parent=self.root):
                return
        self._save_format = extension
        if extension == '.json':
            self.save_json(str(path))
        elif extension == '.txt':
            self.save_pd(str(path))
        elif extension == '.tex':
            self.save_tikz(str(path))
        else:
            self.export_image(str(path))

    def copy_tikz(self):
        if self.busy or self._motion_pending():
            return
        if self._defer_tikz_action(self.copy_tikz):
            return
        try:
            text = self._tikz_string()
        except Exception as exc:
            messagebox.showerror("TikZ unavailable", str(exc))
            return
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status.set("Copied the previous diagram's TikZ. Recognize the changed drawing to update it."
                            if self.raster_changed else "Copied the TikZ code." +
                            (" Inspect the diagram before using it." if self.result.get("status") != "ok" else ""))
            return True

    def save_tikz(self, path=None):
        if self.busy or self._motion_pending():
            return
        if not self.diagram:
            self.status.set("Recognize a diagram before saving its TikZ code.")
            return
        if path is None:
            path = filedialog.asksaveasfilename(parent=self.root, title="Save TikZ code", defaultextension=".tex",
                                              filetypes=[("TikZ code", "*.tex"), ("All files", "*")])
        if path:
            if self._defer_tikz_action(lambda: self.save_tikz(path)):
                return
            try:
                text = self._tikz_string()
            except Exception as exc:
                messagebox.showerror("TikZ unavailable", str(exc))
                return
            self._write(path, text)

    def save_pd(self, path=None):
        if self.busy or self._motion_pending():
            return
        if not self.diagram:
            self.status.set("Recognize a diagram before saving its PD code.")
            return
        try:
            text = self._pd_string()
        except Exception as exc:
            messagebox.showerror("PD unavailable", str(exc))
            return
        if path is None:
            path = filedialog.asksaveasfilename(title="Save PD code", defaultextension=".txt",
                                              filetypes=[("PD code text", "*.txt"), ("All files", "*")])
        if path:
            self._write(path, text + "\n")

    def save_json(self, path=None):
        if self.busy or self._motion_pending():
            return
        if not self.diagram:
            self.status.set("Recognize a diagram before saving its editable JSON.")
            return
        if path is None:
            path = filedialog.asksaveasfilename(title="Save editable diagram", defaultextension=".json", filetypes=[("Diagram JSON", "*.json")])
        if not path:
            return
        try:
            pd = self._pd()
        except Exception:
            pd = None
        data = {"format": "knot-studio-diagram", "version": 1, "source_file": self.source_path, "diagram": self.diagram, "pd_code": pd, "status": self.result.get("status", "needs_review"), "warnings": self.result.get("warnings", []), "diagnostics": self.result.get("diagnostics", {})}
        self._write(path, json.dumps(data, indent=2, default=str) + "\n")

    def _write(self, path, value):
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(value, encoding="utf-8")
            self.status.set(f"Saved {Path(path).name}.")
        except Exception as exc:
            messagebox.showerror("Cannot save file", str(exc))

    def open_json(self, path=None):
        if self._motion_pending():
            return
        if path is None:
            path = filedialog.askopenfilename(title="Open editable diagram", filetypes=[("Diagram JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            wrapped = isinstance(data, dict) and "prediction" in data
            if wrapped:
                data = data["prediction"]
            if not isinstance(data, dict):
                raise ValueError("This JSON does not contain a usable diagram prediction.")
            data = imported_result(data)
            source_image = None
            source_input = None
            if wrapped and data.get('diagram') is None:
                candidates = sorted(p for p in Path(path).parent.glob("input.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"))
                if candidates:
                    source_input = candidates[0]
                    with Image.open(source_input) as opened:
                        rgba = ImageOps.exif_transpose(opened).convert("RGBA")
                        source_image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
            diagram = data.get("diagram", data)
            if diagram is None and source_image is not None:
                self._reset()
                self.result = data
                self.source_path = str(source_input.resolve())
                self.raster_image = source_image
                self.layer.set("source")
                self.tool.set("Select")
                self.root.title(f"{Path(path).name} · Knot Studio")
                self._refresh_text()
                self.fit_view()
                self.status.set("Loaded the unsuccessful prediction and its source image. Repair the marked issues, then recognize again.")
                return
            if not isinstance(diagram, dict) or not isinstance(diagram.get("edges"), list) or not isinstance(diagram.get("crossings"), list):
                raise ValueError("This JSON does not contain a diagram with edges and crossings.")
            for dimension in ("width", "height"):
                value = diagram.get(dimension, 800 if dimension == "width" else 600)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Diagram {dimension} must be a positive finite number.")
                diagram[dimension] = max(1, round(value))
            from .diagram import assign_components, pd_code, validate
            validation = validate(diagram)
            if not validation.get("valid", False):
                raise ValueError("; ".join(map(str, validation.get("errors", []))))
            if pd_preview_crossings(diagram) <= 10000:
                pd_code(diagram)
            assign_components(diagram)
            diagram = align_twist_boxes(diagram)
        except Exception as exc:
            messagebox.showerror("Cannot open diagram", str(exc))
            return
        self._reset()
        self.diagram = diagram
        self.result = data if "diagram" in data else {"diagram": diagram, "status": "needs_review", "warnings": ["Imported diagram; recognition status is unavailable."]}
        self.result['diagram'] = diagram
        self.source_path = str(source_input.resolve()) if source_input else data.get("source_file")
        self.raster_image = None
        self.layer.set("diagram")
        self.tool.set("Select")
        self.root.title(f"{Path(path).name} · Knot Studio")
        self._refresh_text()
        self.fit_view()
        self.status.set("Loaded editable diagram. Use Convert to drawing to draw or erase.")

    def export_image(self, path=None):
        if self.busy or self._motion_pending():
            return
        if not self.diagram:
            self.status.set("Recognize a diagram before exporting its clean drawing.")
            return
        if path is None:
            path = filedialog.asksaveasfilename(title="Export clean diagram", defaultextension=".png", filetypes=[("PNG image", "*.png"), ("SVG vector image", "*.svg")])
        if not path:
            return
        try:
            options = dict(colored=self.colored.get(), show_orientation=self.oriented.get())
            if Path(path).suffix.lower() == ".svg":
                render_svg(self.diagram, path, **options)
            else:
                render_png(self.diagram, path, scale=2, min_stroke_pixels=4, **options)
            self.status.set(f"Exported {Path(path).name}.")
        except Exception as exc:
            messagebox.showerror("Cannot export image", str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native knot/link image recognition and PD editor")
    parser.add_argument("input", nargs="?", help="Image or editable diagram JSON")
    args = parser.parse_args(argv)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(f"Tk could not open a desktop display: {exc}\nUse the CLI recognizer on a machine without a graphical display.") from exc
    editor = DiagramEditor(root)
    # Finder's Open With / drag onto Dock icon delivers Apple events after Tk
    # starts. Avoid PyInstaller argv emulation, which competes with Tk's loop.
    if root.tk.call('tk', 'windowingsystem') == 'aqua':
        root.createcommand('::tk::mac::OpenDocument',
                           lambda *paths: root.after(0, lambda: editor.open_file(paths[0])) if paths else None)
    if args.input:
        root.after(150, lambda: editor.open_file(args.input))
    root.mainloop()


if __name__ == "__main__":
    main()
