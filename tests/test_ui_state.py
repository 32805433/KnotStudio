"""Editor state regressions that do not require a desktop display.

The release self-test additionally exercises the real Tk widgets and workers.
"""
import copy
import hashlib
from queue import Queue
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

pytest.importorskip('tkinter')
from recognizer import ui


class Value:
    def __init__(self, value=None):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


@pytest.mark.parametrize('diagnostics', [None, {}, {'orientation': None, 'preprocessing': None},
    {'branch_points': 4, 'endpoint_hook_repairs': [None, {}],
     'stroke_hole_repairs': [{'bbox': None}, {'bbox': ['bad', 0, 1, 2]}],
     'board_photo': None, 'bundled_crossings': None, 'long_underpasses': None}])
def test_old_optional_diagnostics_can_be_displayed(diagnostics):
    result = ui.imported_result({'status': 'needs_review', 'warnings': None,
                                 'diagnostics': diagnostics})
    assert result['warnings'] == []
    assert ui.recognition_issue_points(result) == []


def test_diagnostic_overlay_retains_valid_points_but_ignores_invalid_ones():
    result = {'status': 'failed', 'diagnostics': {
        'branch_points': [{'point': [1, 2]}, None, [float('nan'), 3]],
        'endpoint_hook_repairs': [{'joined_endpoint': [4, 5]}, {}],
        'stroke_hole_repairs': [{'bbox': [10, 20, 12, 22]}]}}
    assert ui.recognition_issue_points(result) == [
        ('branch', (1., 2.)), ('gap_repair', (4., 5.)), ('gap_repair', (11., 21.))]


@pytest.mark.parametrize('metadata', [{'status': []}, {'warnings': 42}, {'diagnostics': ['bad']}])
def test_invalid_import_metadata_is_rejected_before_editor_state_changes(metadata):
    with pytest.raises(ValueError, match='metadata'):
        ui.imported_result(metadata)


@pytest.mark.parametrize('stale', [False, True])
def test_label_rescan_preserves_reviewed_boxes_only_for_unchanged_pixels(monkeypatch, stale):
    from recognizer import box_detection, label_review
    image = Image.new('RGB', (40, 30), 'white')
    boxes = {'boxes': [{'id': 'manual_1', 'bbox': [10, 10, 20, 20],
                        'half_twists': -3, 'number_reviewed': True}],
             'source_sha256': hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
             'image_size': [40, 30], 'dismissed_ids': ['auto_1']}
    original = copy.deepcopy(boxes)
    if stale:
        image.putpixel((0, 0), (0, 0, 0))
    scans = []

    def detect_boxes(rgb):
        scans.append(True)
        return {'boxes': []}

    monkeypatch.setattr(label_review, 'detect_labels', lambda rgb: {'candidates': [
        {'id': 1, 'bbox': [12, 12, 18, 18]}, {'id': 2, 'bbox': [1, 1, 4, 4]}]})
    monkeypatch.setattr(box_detection, 'detect_twist_boxes', detect_boxes)
    # Run this worker synchronously to inspect the queued result without Tk.
    monkeypatch.setattr(ui.threading, 'Thread', lambda **kw: SimpleNamespace(start=kw['target']))
    editor = SimpleNamespace(busy=False, _motion_pending=lambda: False,
        cancel_gesture=lambda **kw: None, layer=Value(), tool=Value(), raster_image=image,
        box_detection=boxes, label_detection=None, task_id=1, events=Queue(),
        _refresh_conversion_buttons=lambda: None, confidence=Value(), status=Value(), redraw=lambda: None)
    ui.DiagramEditor.find_labels(editor)
    _, kind, payload = editor.events.get_nowait()
    assert kind == 'labels', payload
    labels, _, reviewed = payload
    assert boxes == original  # Worker must not mutate the live review.
    if stale:
        assert scans and not reviewed['boxes']
        assert len(labels['candidates']) == 2
    else:
        assert not scans
        assert reviewed == original
        assert [c['id'] for c in labels['candidates']] == [2]


def tikz_editor(smooth=False):
    from test_render import one_crossing
    editor = object.__new__(ui.DiagramEditor)
    editor.diagram = one_crossing()
    editor.smooth_display = Value(smooth)
    editor.colored, editor.oriented = Value(True), Value(True)
    editor.raster_changed = False
    editor._display_actions = []
    editor.status = Value()
    editor.busy, editor.drag = False, None
    editor._motion_pending = lambda: False
    return editor


@pytest.mark.parametrize('error', [None, RuntimeError('fit failed')])
def test_unsmoothed_tikz_does_not_wait_for_or_use_smooth_cache(error):
    from recognizer.render import render_tikz
    editor = tikz_editor()
    cache = SimpleNamespace(prepared=None, error=error)
    editor._curve_cache = lambda: cache
    expected = render_tikz(editor.diagram, colored=True, show_orientation=True, smooth=False)
    assert editor._tikz_string() == expected
    assert '.. controls' not in expected
    assert not editor._defer_tikz_action(lambda: None)
    editor.tikz_text = object()
    shown = []
    editor._put_text = lambda widget, text: shown.append(text)
    editor._refresh_tikz()
    assert shown == [expected]


def test_switching_off_releases_pending_tikz_export_without_starting_fit():
    editor = tikz_editor()
    cache = SimpleNamespace(prepared=None, error=None, revision=3, poll=lambda diagram: False)
    editor._curve_cache = lambda: cache
    exported = []
    editor._display_actions = [(3, lambda: exported.append(editor._tikz_string()))]
    editor._poll_display_curves()
    assert len(exported) == 1 and '.. controls' not in exported[0]
    assert editor._display_actions == []


def test_tikz_copy_and_save_follow_checkbox_with_existing_fit(tmp_path):
    from recognizer.display_curves import prepare_display_curves
    editor = tikz_editor(True)
    prepared = prepare_display_curves(editor.diagram)
    editor._curve_cache = lambda: SimpleNamespace(prepared=prepared, error=None)
    clipboard = []
    editor.root = SimpleNamespace(clipboard_clear=clipboard.clear, clipboard_append=clipboard.append)
    editor.result = {'status': 'ok'}
    written = []
    editor._write = lambda path, text: written.append((path, text))
    for smooth in (True, False, True):
        editor.smooth_display.set(smooth)
        assert editor.copy_tikz()
        editor.save_tikz(str(tmp_path/'diagram.tex'))
        assert clipboard[0] == written[-1][1]
        assert ('.. controls' in clipboard[0]) == smooth
