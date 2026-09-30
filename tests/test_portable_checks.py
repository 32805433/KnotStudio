"""Desktop input regressions without requiring a running display server.

The standalone release self-test also dispatches these events through real Tk
widgets on the platform used to build each downloadable application.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip('tkinter')
from recognizer import ui


class Bindings:
    def __init__(self, windowing_system):
        self.windowing_system = windowing_system
        self.callbacks = {}
        self.tk = SimpleNamespace(call=lambda *args: windowing_system)

    def bind(self, sequence, callback, **kwargs):
        if 'Command' in sequence and self.windowing_system != 'aqua':
            raise ui.tk.TclError('Command modifier is only available on Aqua')
        self.callbacks[sequence] = callback


@pytest.mark.parametrize('windowing_system', ['win32', 'x11', 'aqua'])
def test_editor_binds_native_shortcuts_to_the_correct_commands(windowing_system):
    editor = ui.DiagramEditor.__new__(ui.DiagramEditor)
    editor.root = Bindings(windowing_system)
    editor.canvas = Bindings(windowing_system)
    calls = []
    for command in ('open_file', 'save_as', 'undo', 'redo'):
        setattr(editor, command, lambda name=command: calls.append(name))
    editor._bindings()
    modifier = 'Command' if windowing_system == 'aqua' else 'Control'
    for key, expected in [('o', 'open_file'), ('s', 'save_as'), ('z', 'undo'), ('Shift-Z', 'redo')]:
        assert editor.root.callbacks[f'<{modifier}-{key}>'](SimpleNamespace()) == 'break'
        assert calls.pop() == expected
    if windowing_system != 'aqua':
        assert editor.root.callbacks['<Control-y>'](SimpleNamespace()) == 'break'
        assert calls == ['redo']


def zoom_editor(windowing_system):
    editor = ui.DiagramEditor.__new__(ui.DiagramEditor)
    editor.root = Bindings(windowing_system)
    editor.scale, editor.offset = .8, (18., -12.)
    editor.diagram = {'crossings': [], 'edges': []}
    editor.history = []
    editor.redraw = lambda: None
    return editor


@pytest.mark.parametrize('windowing_system,up,down', [
    ('win32', {'delta': 120}, {'delta': -120}),
    ('win32', {'delta': 30}, {'delta': -30}),
    ('x11', {'num': 4}, {'num': 5}),
    ('x11', {'delta': 120}, {'delta': -120}),
    ('aqua', {'delta': 1}, {'delta': -1}),
    ('aqua', {'delta': 120}, {'delta': -120}),
])
def test_native_wheel_zoom_preserves_pointer_anchor_and_document(windowing_system, up, down):
    editor = zoom_editor(windowing_system)
    original = editor.diagram
    anchor = editor.to_model(137, 219)
    for fields, direction in [(up, 1), (down, -1)]:
        before = editor.scale
        assert editor._wheel(SimpleNamespace(x=137, y=219, **fields)) == 'break'
        assert (editor.scale - before) * direction > 0
        assert editor.to_model(137, 219) == pytest.approx(anchor)
        assert editor.diagram is original and editor.history == []
    assert editor.scale == pytest.approx(.8)
    assert editor.offset == pytest.approx((18., -12.))
    editor._wheel(SimpleNamespace(x=137, y=219, delta=0))
    assert editor.scale == pytest.approx(.8)


def test_high_resolution_windows_wheel_accumulates_without_accelerating_zoom():
    full = zoom_editor('win32')
    precise = zoom_editor('win32')
    full._wheel(SimpleNamespace(x=42, y=58, delta=120))
    for _ in range(120):
        precise._wheel(SimpleNamespace(x=42, y=58, delta=1))
    assert precise.scale == pytest.approx(full.scale)
    assert precise.offset == pytest.approx(full.offset)


def sidebar(windowing_system):
    scrolls = []
    controls = SimpleNamespace(tk=Bindings(windowing_system).tk,
                               precise_scrolling=False, wheel_remainder=0.,
                               viewport=SimpleNamespace(yview_scroll=lambda *args: scrolls.append(args)))
    return controls, scrolls


def test_small_windows_wheel_deltas_remain_usable_in_the_sidebar():
    controls, scrolls = sidebar('win32')
    for _ in range(120):
        ui.ScrollableControls._wheel(controls, SimpleNamespace(delta=-1, widget=None))
    assert sum(pixels for pixels, units in scrolls) == 24
    assert all(units == 'units' for pixels, units in scrolls)


@pytest.mark.parametrize('windowing_system,fields', [
    ('win32', {'delta': -120}), ('aqua', {'delta': -1}), ('x11', {'num': 5})])
def test_nested_text_scrolls_before_sidebar_until_it_reaches_the_end(windowing_system, fields):
    controls, scrolls = sidebar(windowing_system)
    text = ui.tk.Text.__new__(ui.tk.Text)
    text.yview = lambda: (.25, .75)
    event = SimpleNamespace(widget=text, **fields)
    assert ui.ScrollableControls._wheel(controls, event) is None
    assert not scrolls
    text.yview = lambda: (.5, 1.)
    assert ui.ScrollableControls._wheel(controls, event) == 'break'
    assert scrolls == [(24, 'units')]
