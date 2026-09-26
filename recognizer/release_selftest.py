"""Offline checks for a relocated source checkout or frozen macOS application.

This is independent of pytest and development fixtures, so the packaged
executable can exercise its own runtime resources and bundled examples.
"""
from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


def _resources():
    from PIL import Image
    from .resources import example_entries, example_path, resource_root
    root = resource_root()
    entries = example_entries()
    assert entries, 'Bundled examples manifest is missing or empty'
    for entry in entries:
        path = example_path(entry)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], f'Sample hash differs: {entry["id"]}'
        with Image.open(path) as image:
            image.verify()
    assert any((root/'docs'/name).is_file() for name in ('USER_GUIDE.md', 'USER_GUIDE.html')), 'User guide is missing'
    assert (root/'LICENSE').is_file(), 'Application license is missing'
    assert (root/'licenses').is_dir() and any((root/'licenses').iterdir()), 'Third-party licenses are missing'
    return f'{len(entries)} bundled images verified; user guide and license files are present.'


def _models():
    import numpy as np
    from . import label_model, learned_ink
    models = (('labels', label_model), ('ink', learned_ink))
    for name, module in models:
        module.load_model.cache_clear()
        model = module.load_model()
        if model is None:
            raise AssertionError(f'{name} model is missing')
        result = module.predict_features(np.zeros((2, len(module.FEATURE_NAMES)), np.float32), model)
        if result.shape != (2,) or not np.isfinite(result).all() or not ((0 <= result) & (result <= 1)).all():
            raise AssertionError(f'{name} model returned invalid probabilities')
    return 'Both bundled inference models loaded and produced finite probabilities.'


def _recognition():
    import numpy as np
    from .diagram import pd_code, validate
    from .motion import from_curves
    from .pipeline import recognize_array
    from .projection_equivalence import same_pd_projection
    from .render import render_image
    t = np.linspace(.017, 2*np.pi+.017, 361)
    points = np.column_stack((320+82*(np.sin(t)+2*np.sin(2*t)),
                              320+82*(np.cos(t)-2*np.cos(2*t)),
                              -40*np.sin(3*t)))
    points[-1] = points[0]
    diagram = from_curves([{'points': points.tolist()}], 640, 640)
    assert len(diagram['crossings']) == 3, 'Synthetic trefoil must have three crossings'
    image = render_image(diagram, scale=1.5, colored=False)
    result = recognize_array(np.asarray(image), options={'remove_labels': False, 'time_budget': 15.})
    assert result['diagram'] is not None, f'Recognition failed: {result["status"]}'
    assert validate(result['diagram'])['valid'], 'Reconstructed graph is invalid'
    assert same_pd_projection(pd_code(diagram), result['pd_code']), 'Reconstructed trefoil PD differs'
    return 'Synthetic trefoil raster reconstructed with the expected three-crossing PD.'


def _twist_boxes():
    from .diagram import component_walks, pd_code, validate
    from .twist_boxes import box_port_points, bundle_ports, compact_component_walks, expand_boxes, make_twist_box
    box = make_twist_box(10, [200, 180], [120, 160], '-3/2', strand_count=3)
    ports = box_port_points(box)
    top, bottom = bundle_ports(box)
    edges = []
    for lane, (a, b) in enumerate(zip(top, bottom)):
        edge_id = lane+30
        box['ports'][a] = {'edge': edge_id, 'end': 0}
        box['ports'][b] = {'edge': edge_id, 'end': 1}
        edges.append(dict(id=edge_id, start=[10, a], end=[10, b], closed=False,
                          points=[ports[a], [50-lane*10, 60-lane*10],
                                  [50-lane*10, 290+lane*10], ports[b]]))
    compact = dict(width=400, height=360, crossings=[box], edges=edges, component_orientations={})
    expanded = expand_boxes(compact)
    assert validate(expanded)['valid'], 'Expanded twist box is invalid'
    assert len(expanded['crossings']) == len(pd_code(expanded)) == 9, 'Incorrect full twist expansion'
    assert len(component_walks(expanded)) == len(compact_component_walks(compact)) == 2
    return 'Three-strand twist box expanded to nine crossings and preserved both components.'


def _reference_samples():
    from .pipeline import recognize
    from .diagram import validate
    from .projection_equivalence import same_pd_projection
    from .resources import example_entries, example_path
    checked = []
    for entry in example_entries():
        reference = entry.get('reference_diagram')
        if not reference:
            continue
        result = recognize(example_path(entry), options={'remove_labels': False, 'time_budget': 15.})
        assert result['diagram'] is not None, f'{entry["id"]}: {result["status"]}'
        assert validate(result['diagram'])['valid'], f'{entry["id"]}: invalid graph'
        assert same_pd_projection(reference['pd_code'], result['pd_code']), f'{entry["id"]}: reference PD differs'
        checked.append(entry['id'])
    assert len(checked) == 3, 'Expected three bundled SnapPy reference diagrams'
    return 'Bundled SnapPy reference PDs verified: '+', '.join(checked)+'.'


def _energy():
    import numpy as np
    from .diagram import pd_code, validate
    from .geometry import check_embedding
    from .motion import from_curves
    from .relaxation import RelativeRelaxation, energy_gradient
    points = np.array([[0., 0.], [.7, .2], [1.9, -.4], [.2, 2.3]])
    adjacency = np.zeros((4, 4), bool)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        adjacency[a, b] = adjacency[b, a] = True
    _, gradient = energy_gradient(points, adjacency)
    for index in range(4):
        for axis in range(2):
            step = np.zeros_like(points)
            step[index, axis] = 1e-6
            derivative = (energy_gradient(points+step, adjacency)[0]-energy_gradient(points-step, adjacency)[0])/2e-6
            assert np.isclose(gradient[index, axis], derivative, rtol=1e-6, atol=1e-6)
    t = np.linspace(.031, 2*np.pi+.031, 121)
    curve = np.column_stack((300+110*np.cos(t), 300+80*np.sin(t), np.zeros(len(t))))
    curve[-1] = curve[0]
    original = from_curves([{'points': curve.tolist()}], 600, 600)
    engine = RelativeRelaxation(original, [[100, 100], [500, 100], [500, 500], [100, 500]])
    before = engine.energy
    for _ in range(5):
        result = engine.step()
        if result is None:
            break
        assert validate(result)['valid'] and check_embedding(result)['valid']
        assert pd_code(result) == []
        assert np.array_equal(engine.positions[~engine.free], engine.rest[~engine.free])
    assert engine.iterations > 0 and engine.energy < before, 'Energy did not decrease'
    return 'Energy derivative verified; relaxation reduced energy while preserving topology and fixed exterior.'


def _ocr():
    import cv2
    import numpy as np
    from PIL import Image
    from .text_labels import _tesseract_executable
    executable = _tesseract_executable()
    assert executable, 'Tesseract OCR executable is missing'
    if getattr(sys, 'frozen', False):
        bundle = Path(sys._MEIPASS).resolve()
        assert Path(executable).resolve().is_relative_to(bundle), 'OCR executable escaped the app bundle'
    rgb = np.full((140, 300, 3), 255, np.uint8)
    cv2.putText(rgb, '-2', (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 3., (0, 0, 0), 5, cv2.LINE_AA)
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format='PNG')
    process = subprocess.run([str(executable), 'stdin', 'stdout', '--psm', '7',
                              '-c', 'tessedit_char_whitelist=-0123456789'],
                             input=stream.getvalue(), capture_output=True, timeout=12, check=True)
    reading = ''.join(process.stdout.decode('utf-8').split())
    assert reading == '-2', f'OCR expected -2, received {reading!r}'
    return 'OCR subprocess read a synthetic negative coefficient using the available runtime.'


def _gui():
    """Exercise the real editor, including its asynchronous recognition path.

    Modal dialogs are treated as failures so the build cannot hang waiting for
    a human. Everything written by this check lives in a temporary directory.
    """
    import copy
    import tempfile
    import threading
    import tkinter as tk
    from PIL import Image
    from . import ui
    from .diagram import pd_code, validate
    from .projection_equivalence import same_pd_projection
    from .resources import example_entries, example_path

    dialogs = ('showerror', 'showwarning', 'showinfo', 'askyesno', 'askokcancel')
    original_dialogs = {name: getattr(ui.messagebox, name) for name in dialogs}
    existing_threads = set(threading.enumerate())
    callback_errors = []

    def reject_dialog(title, message='', **kwargs):
        raise AssertionError(f'Unexpected editor dialog: {title}: {message}')

    def callback_error(exception, value, tb):
        callback_errors.append(''.join(traceback.format_exception(exception, value, tb)))

    root = tk.Tk()

    def pump_until(condition, *, seconds=45.):
        deadline = time.monotonic()+seconds
        while True:
            root.update_idletasks()
            root.update()
            assert not callback_errors, '\n'.join(callback_errors)
            if condition():
                return
            assert time.monotonic() < deadline, 'Editor did not finish within its GUI smoke-test deadline'
            time.sleep(.01)

    try:
        root.report_callback_exception = callback_error
        for name in dialogs:
            setattr(ui.messagebox, name, reject_dialog)
        editor = ui.DiagramEditor(root)
        assert not editor.smooth_display.get(), 'Smooth display should start off'
        pump_until(lambda: editor.canvas.winfo_width() > 1)
        assert editor.recognize_button.instate(['disabled'])
        assert editor.convert_button.instate(['disabled'])

        entry = next(item for item in example_entries() if item['id'] == 'trefoil')
        editor.open_file(example_path(entry))
        assert editor.raster_image is not None and editor.layer.get() == 'source'
        assert editor.busy, 'Opening an image must start label review'
        pump_until(lambda: not editor.busy)
        # The clean bundled trefoil should pass through label review directly
        # into recognition, exactly as opening an image with Open… does.
        assert editor.diagram is not None, f'Bundled trefoil did not reach diagram mode: {editor.status.get()}'
        assert validate(editor.diagram)['valid']
        expected_pd = entry['reference_diagram']['pd_code']
        assert same_pd_projection(expected_pd, pd_code(editor.diagram)), 'Editor trefoil PD differs from the sample reference'
        assert editor.layer.get() == 'diagram' and editor.raster_image is None
        assert editor.recognize_button.instate(['disabled'])
        assert not editor.convert_button.instate(['disabled'])
        assert editor.canvas.find_all(), 'Editor canvas is empty'
        # Exercise Tk's native binding dispatch for both secondary-button
        # mappings and Control-left drag (including releasing Control first).
        pan_original = copy.deepcopy(editor.diagram)
        pan_history = len(editor.history)
        for press, motion, release in (
                ('<ButtonPress-2>', '<B2-Motion>', '<ButtonRelease-2>'),
                ('<ButtonPress-3>', '<B3-Motion>', '<ButtonRelease-3>'),
                ('<Control-ButtonPress-1>', '<B1-Motion>', '<ButtonRelease-1>')):
            before_offset = editor.offset
            editor.canvas.event_generate(press, x=90, y=90)
            editor.canvas.event_generate(motion, x=120, y=110)
            editor.canvas.event_generate(release, x=120, y=110)
            pump_until(lambda: editor.pan is None)
            expected_offset = (before_offset[0]+30, before_offset[1]+20)
            assert all(abs(actual-expected) < 1e-9
                       for actual, expected in zip(editor.offset, expected_offset)), press
            assert editor.diagram == pan_original and len(editor.history) == pan_history
            assert editor.drag is None
        assert 'PD' in editor.pd_text.get('1.0', 'end') or '[' in editor.pd_text.get('1.0', 'end')

        # Automatic recognition must retain the original pixels, just like
        # pressing Recognize drawing. Otherwise missed markings cannot be
        # corrected without reopening (and automatically recognizing) again.
        editor.undo()
        assert editor.diagram is None and editor.raster_image is not None, 'Undo lost the imported source image'
        assert editor.layer.get() == 'source'
        assert not editor.recognize_button.instate(['disabled'])
        with Image.open(example_path(entry)) as original:
            assert editor.raster_image.tobytes() == original.convert('RGB').tobytes()
        editor.redo()
        assert editor.diagram is not None and editor.raster_image is None
        assert same_pd_projection(expected_pd, pd_code(editor.diagram))

        crossing = editor.diagram['crossings'][0]
        ident, over = crossing['id'], crossing['over']
        editor.selected_crossing = ident
        editor.switch_selected()
        assert next(c for c in editor.diagram['crossings'] if c['id'] == ident)['over'] != over
        editor.undo()
        assert next(c for c in editor.diagram['crossings'] if c['id'] == ident)['over'] == over
        assert same_pd_projection(expected_pd, pd_code(editor.diagram))
        before = copy.deepcopy(editor.diagram)
        editor.selected_component = editor.diagram['edges'][0]['component']
        editor.reverse_selected()
        assert editor.diagram['component_orientations'] != before['component_orientations']
        editor.undo()
        assert editor.diagram == before

        with tempfile.TemporaryDirectory(prefix='knot-studio-gui-') as directory:
            directory = Path(directory)
            saved_json, saved_png = directory/'saved diagram.json', directory/'saved image.png'
            editor.save_json(str(saved_json))
            assert saved_json.is_file(), 'Editor did not save JSON'
            saved = json.loads(saved_json.read_text(encoding='utf-8'))
            assert saved['format'] == 'knot-studio-diagram'
            assert same_pd_projection(expected_pd, saved['pd_code'])
            editor.export_image(str(saved_png))
            with Image.open(saved_png) as image:
                assert min(image.size) > 100
                assert image.convert('L').getextrema()[0] < 200, 'Exported image contains no visible drawing'
            saved_svg, saved_tex = directory/'saved image.svg', directory/'saved diagram.tex'
            editor.export_image(str(saved_svg))
            assert '<svg' in saved_svg.read_text()
            editor.save_tikz(str(saved_tex))
            pump_until(saved_tex.is_file)
            assert '\\end{tikzpicture}' in saved_tex.read_text()
            assert '.. controls' not in saved_tex.read_text()
            editor.smooth_display.set(True)
            editor._appearance_changed()
            editor.save_tikz(str(saved_tex))
            pump_until(lambda: '.. controls' in saved_tex.read_text())
            # The checkbox controls the live preview and saved TikZ, including
            # when a cached smooth fit already exists.
            editor.code_notebook.select(editor.tikz_frame)
            editor.smooth_display.set(False)
            editor._appearance_changed()
            pump_until(lambda: '\\end{tikzpicture}' in editor.tikz_text.get('1.0', 'end')
                       and '.. controls' not in editor.tikz_text.get('1.0', 'end'))
            raw_tex = directory/'unsmoothed.tex'
            editor.save_tikz(str(raw_tex))
            assert raw_tex.is_file() and '.. controls' not in raw_tex.read_text()
            editor.smooth_display.set(True)
            editor._appearance_changed()
            pump_until(lambda: '.. controls' in editor.tikz_text.get('1.0', 'end'))

            # Broken metadata must be rejected before replacing a good open
            # diagram. Optional null metadata in older files remains usable.
            broken = directory/'bad metadata.json'
            broken.write_text(json.dumps(dict(saved, diagnostics=['invalid'])))
            errors = []
            ui.messagebox.showerror = lambda *args, **kw: errors.append(args)
            unchanged = editor.diagram
            try:
                editor.open_json(broken)
            finally:
                ui.messagebox.showerror = reject_dialog
            assert errors and editor.diagram is unchanged
            old_json = directory/'older metadata.json'
            old_json.write_text(json.dumps(dict(saved, warnings=None, diagnostics=None)))
            editor.open_json(old_json)
            assert same_pd_projection(expected_pd, pd_code(editor.diagram))
            assert editor.result['warnings'] == [] and editor.result['diagnostics'] == {}

            editor.new_sketch()
            assert editor.diagram is None and editor.layer.get() == 'source'
            assert editor.convert_button.instate(['disabled'])
            assert not editor.recognize_button.instate(['disabled'])
            editor.open_file(saved_json)
            assert editor.diagram is not None and editor.layer.get() == 'diagram'
            assert same_pd_projection(expected_pd, pd_code(editor.diagram)), 'Reopened JSON changed the diagram'

            editor.diagram_to_raster()
            assert editor.diagram is None and editor.raster_image is not None
            assert editor.convert_button.instate(['disabled'])
            assert not editor.recognize_button.instate(['disabled'])
            editor.recognize_raster()
            pump_until(lambda: not editor.busy)
            assert editor.diagram is not None, f'Drawing round trip failed: {editor.status.get()}'
            assert same_pd_projection(expected_pd, pd_code(editor.diagram)), 'Drawing round trip changed the trefoil PD'
            assert editor.recognize_button.instate(['disabled'])
            assert not editor.convert_button.instate(['disabled'])

            # Stop is synchronous even while initialization is on a worker;
            # its late messages cannot restart or alter the stopped diagram.
            unchanged = copy.deepcopy(editor.diagram)
            width, height = editor.diagram['width'], editor.diagram['height']
            region = [[-50, -50], [width+50, -50], [width+50, height+50], [-50, height+50]]
            editor._begin_energy(region)
            assert editor.drag and editor.drag['kind'] == 'energy'
            editor.stop_energy()
            assert editor.drag is None and editor.diagram == unchanged
            editor._begin_energy(region)
            pump_until(lambda: editor.drag is None or editor.drag.get('moved'))
            assert editor.drag is not None and editor.drag['moved'], editor.status.get()
            editor.stop_energy()
            stopped = copy.deepcopy(editor.diagram)
            pump_until(lambda: not any(t.is_alive() for t in threading.enumerate()
                                       if t not in existing_threads), seconds=10.)
            assert editor.drag is None and editor.diagram == stopped
            assert same_pd_projection(expected_pd, pd_code(editor.diagram))
            editor.undo()
            assert editor.diagram == unchanged, 'Undo did not restore the entire relaxation'

            old_task = editor.task_id
            editor.new_sketch()
            pixels = editor.raster_image.tobytes()
            editor.events.put((old_task, 'result', saved))
            editor._poll()
            assert editor.diagram is None and editor.raster_image.tobytes() == pixels, 'A stale worker replaced the new document'

        # Finish any small display-fitting worker before destroying its UI.
        pump_until(lambda: not any(t.is_alive() for t in threading.enumerate()
                                   if t not in existing_threads), seconds=10.)
        return (f'Tk {root.tk.call("info", "patchlevel")}: real editor opened a bundled trefoil, '
                'recognized its reference PD, restored source pixels with Undo/Redo, switched '
                'a crossing and orientation, saved/reopened JSON, exported PNG/SVG/TikZ, and '
                'preserved PD through drawing conversion and energy relaxation. Invalid imports, '
                'immediate Stop, late worker results, Undo, and button states verified.')
    finally:
        for name, original in original_dialogs.items():
            setattr(ui.messagebox, name, original)
        for identifier in root.tk.call('after', 'info'):
            root.after_cancel(identifier)
        root.destroy()


def run(output, gui=False):
    """Write a JSON report and return it. Each check records success separately."""
    checks = []
    tasks = [('bundled_resources', _resources), ('packaged_models', _models), ('raster_recognition_pd', _recognition),
             ('reference_samples', _reference_samples), ('twist_boxes', _twist_boxes), ('energy_relaxation', _energy), ('ocr', _ocr)]
    if gui:
        tasks.append(('tk_gui', _gui))
    for name, function in tasks:
        started = time.monotonic()
        try:
            detail = function()
            row = {'name': name, 'success': True, 'detail': detail}
        except Exception as error:
            row = {'name': name, 'success': False, 'detail': str(error),
                   'traceback': traceback.format_exc()}
        row['seconds'] = round(time.monotonic()-started, 3)
        checks.append(row)
    report = {'schema_version': 1, 'ok': all(row['success'] for row in checks),
              'frozen': bool(getattr(sys, 'frozen', False)), 'platform': platform.platform(),
              'python': platform.python_version(), 'checks': checks}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report
