"""Desktop integration contracts that can be checked on any build host."""
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from recognizer import desktop_platform as desktop, resources


def platform(monkeypatch, name, *, frozen=False):
    # Replace the module reference so host pathlib/import behavior stays intact.
    monkeypatch.setattr(desktop, 'sys', SimpleNamespace(platform=name, frozen=frozen))


@pytest.mark.parametrize('name, relative', [
    ('darwin', 'Library/Logs/Knot Studio'),
    ('win32', 'AppData/Local/Knot Studio/Logs'),
    ('linux', '.local/state/knot-studio/logs'),
])
def test_logs_default_to_a_per_user_platform_directory(tmp_path, monkeypatch, name, relative):
    platform(monkeypatch, name)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.delenv('LOCALAPPDATA', raising=False)
    monkeypatch.delenv('XDG_STATE_HOME', raising=False)
    assert desktop.log_directory() == tmp_path/relative


@pytest.mark.parametrize('name, variable, suffix', [
    ('win32', 'LOCALAPPDATA', 'Knot Studio/Logs'),
    ('linux', 'XDG_STATE_HOME', 'knot-studio/logs'),
])
def test_logs_respect_overridden_user_directories(tmp_path, monkeypatch, name, variable, suffix):
    platform(monkeypatch, name)
    monkeypatch.setenv(variable, str(tmp_path/'Custom User Data'))
    assert desktop.log_directory() == tmp_path/'Custom User Data'/suffix


def test_relative_xdg_override_does_not_write_into_launch_directory(tmp_path, monkeypatch):
    platform(monkeypatch, 'linux')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('XDG_STATE_HOME', 'relative-state')
    assert desktop.log_directory() == tmp_path/'.local/state/knot-studio/logs'


@pytest.mark.parametrize('name, executable', [
    ('win32', 'tesseract.exe'), ('darwin', 'tesseract'), ('linux', 'tesseract'),
])
def test_bundled_ocr_resolves_after_relocation_with_spaces(tmp_path, monkeypatch, name, executable):
    platform(monkeypatch, name)
    bundle = tmp_path/'Moved Knot Studio'
    binary = bundle/'ocr'/'bin'/executable
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b'test executable')
    monkeypatch.setattr(resources, 'resource_root', lambda: bundle)
    monkeypatch.setenv('TESSDATA_PREFIX', 'unrelated-host-data')
    assert resources.bundled_tesseract() == str(binary)
    assert os.environ['TESSDATA_PREFIX'] == str(bundle/'ocr'/'tessdata')


def test_frozen_ocr_never_falls_back_to_a_host_install(monkeypatch):
    import sys
    from recognizer import text_labels
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(resources, 'bundled_tesseract', lambda: None)
    monkeypatch.setattr(text_labels.shutil, 'which', lambda *a, **kw: pytest.fail('host OCR lookup'))
    assert text_labels._tesseract_executable() is None


@pytest.mark.parametrize('name', ['win32', 'darwin', 'linux'])
@pytest.mark.parametrize('kind', ['numeric', 'label', 'box'])
def test_all_ocr_paths_hide_windows_console_and_capture_result(monkeypatch, name, kind):
    from recognizer import box_detection, diagram_labels, text_labels
    platform(monkeypatch, name)
    executable = 'A path with spaces/tesseract.exe'
    calls = []
    recognized = {'numeric': '12', 'label': 'k', 'box': '-3'}[kind]

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = f'level\tconf\ttext\n5\t95\t{recognized}\n'.encode()
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr=b'')

    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(text_labels, '_tesseract_executable', lambda: executable)
    monkeypatch.setattr(diagram_labels, '_tesseract_executable', lambda: executable)
    crop = np.full((20, 20, 3), 255, dtype=np.uint8)
    if kind == 'numeric':
        result = text_labels._ocr_numeric(crop)
    elif kind == 'label':
        result = diagram_labels._ocr_label(crop)
    else:
        result = box_detection._ocr_one(np.ones((20, 20), dtype=bool), executable, 7, 1)
    assert result['text'] == recognized
    assert len(calls) == 1
    command, options = calls[0]
    assert command[0] == executable
    assert options['input'].startswith(b'\x89PNG')
    assert options['capture_output'] and options['check'] and options['timeout'] > 0
    if name == 'win32':
        assert options['creationflags'] & 0x08000000
    else:
        assert 'creationflags' not in options


@pytest.mark.parametrize('name, command', [('darwin', 'open'), ('linux', 'xdg-open')])
@pytest.mark.parametrize('original', [None, '/user/runtime'])
def test_native_document_launcher_preserves_path_and_system_libraries(
        tmp_path, monkeypatch, name, command, original):
    platform(monkeypatch, name, frozen=True)
    document = tmp_path/'Guide with spaces ü.html'
    document.write_text('Guide')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/bundled/libraries')
    monkeypatch.setenv('DYLD_LIBRARY_PATH', '/bundled/mac-libraries')
    monkeypatch.delenv('DYLD_LIBRARY_PATH_ORIG', raising=False)
    if original is None:
        monkeypatch.delenv('LD_LIBRARY_PATH_ORIG', raising=False)
    else:
        monkeypatch.setenv('LD_LIBRARY_PATH_ORIG', original)
    calls = []
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: calls.append((a, kw)))
    desktop.open_local_document(document)
    arguments, options = calls[0]
    assert arguments == ([command, str(document)],)
    assert options['env'].get('LD_LIBRARY_PATH') == original
    assert 'DYLD_LIBRARY_PATH' not in options['env']
    assert os.environ['LD_LIBRARY_PATH'] == '/bundled/libraries'
    assert options['stdin'] == subprocess.DEVNULL


def test_windows_document_launcher_uses_file_association(tmp_path, monkeypatch):
    platform(monkeypatch, 'win32')
    monkeypatch.setattr(desktop, '_open_frozen_windows_document',
                        lambda path: pytest.fail('Unfrozen launch changed DLL search paths'))
    document = tmp_path/'Guide with spaces.html'
    document.write_text('Guide')
    opened = []
    monkeypatch.setattr(os, 'startfile', opened.append, raising=False)
    desktop.open_local_document(document)
    assert opened == [str(document)]
    with pytest.raises(FileNotFoundError):
        desktop.open_local_document(tmp_path/'missing.html')


def windows_dll_api(monkeypatch, original, *, failure=None, grow=False):
    """Fake only Win32 calls, keeping real ctypes Unicode buffer handling."""
    import ctypes
    calls = []
    error = [0]

    def get_directory(size, buffer):
        calls.append(('get', size))
        if failure == ('query' if size == 0 else 'read'):
            error[0] = 5
            return 0
        if size == 0:
            return 2 if grow else len(original) + 1 if original else 0
        if size <= len(original):
            return len(original) + 1
        buffer.value = original
        return len(original)

    def set_directory(value):
        calls.append(('set', value))
        if failure == ('clear' if value is None else 'restore'):
            error[0] = 5
            return 0
        return 1

    def load(name, *, use_last_error):
        assert name == 'kernel32' and use_last_error
        return SimpleNamespace(GetDllDirectoryW=get_directory, SetDllDirectoryW=set_directory)

    monkeypatch.setattr(ctypes, 'WinDLL', load, raising=False)
    monkeypatch.setattr(ctypes, 'set_last_error', lambda value: error.__setitem__(0, value), raising=False)
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: error[0], raising=False)
    monkeypatch.setattr(ctypes, 'WinError', lambda code: OSError(code, 'Win32 failure'), raising=False)
    return calls


@pytest.mark.parametrize('launch_fails', [False, True])
@pytest.mark.parametrize('original', [r'C:\Actual DLL folder ü', ''])
def test_frozen_windows_document_launch_restores_actual_dll_directory(
        tmp_path, monkeypatch, original, launch_fails):
    platform(monkeypatch, 'win32', frozen=True)
    desktop.sys._MEIPASS = r'C:\Different bundle directory'
    document = tmp_path/'Guide with spaces ü.html'
    document.write_text('Guide')
    calls = windows_dll_api(monkeypatch, original)
    launch_error = OSError('No file association')

    def startfile(path):
        calls.append(('open', path))
        if launch_fails:
            raise launch_error

    monkeypatch.setattr(os, 'startfile', startfile, raising=False)
    if launch_fails:
        with pytest.raises(OSError) as error:
            desktop.open_local_document(document)
        assert error.value is launch_error
    else:
        desktop.open_local_document(document)
    changes = [call for call in calls if call[0] != 'get']
    assert changes == ([('set', None), ('open', str(document)), ('set', original)]
                       if original else [('open', str(document))])


@pytest.mark.parametrize('failure', ['query', 'read', 'clear', 'restore'])
def test_frozen_windows_document_launch_checks_win32_failures(tmp_path, monkeypatch, failure):
    platform(monkeypatch, 'win32', frozen=True)
    document = tmp_path/'Guide.html'
    document.write_text('Guide')
    original = r'C:\Original DLL directory'
    calls = windows_dll_api(monkeypatch, original, failure=failure)
    monkeypatch.setattr(os, 'startfile', lambda path: calls.append(('open', path)), raising=False)
    with pytest.raises(OSError, match='Win32 failure') as error:
        desktop.open_local_document(document)
    assert error.value.errno == 5
    changes = [call for call in calls if call[0] != 'get']
    expected = {'query': [], 'read': [], 'clear': [('set', None)],
                'restore': [('set', None), ('open', str(document)), ('set', original)]}
    assert changes == expected[failure]


def test_frozen_windows_document_launch_retries_a_growing_dll_path(tmp_path, monkeypatch):
    platform(monkeypatch, 'win32', frozen=True)
    document = tmp_path/'Guide.html'
    document.write_text('Guide')
    original = r'C:\A longer DLL directory ü'
    calls = windows_dll_api(monkeypatch, original, grow=True)
    monkeypatch.setattr(os, 'startfile', lambda path: calls.append(('open', path)), raising=False)
    desktop.open_local_document(document)
    assert len([call for call in calls if call[0] == 'get']) == 3
    assert calls[-3:] == [('set', None), ('open', str(document)), ('set', original)]


@pytest.mark.parametrize('windowing_system, delta, button, precise, expected', [
    ('win32', 120, None, False, 1), ('win32', -240, None, False, -2),
    ('win32', 30, None, False, .25), ('win32', 0, None, False, 0),
    ('x11', 0, 4, False, 1), ('x11', 0, 5, False, -1),
    ('aqua', -2, None, False, -2), ('aqua', 120, None, True, 1),
])
def test_wheel_event_conventions(windowing_system, delta, button, precise, expected):
    event = SimpleNamespace(delta=delta, num=button)
    assert desktop.wheel_steps(event, windowing_system=windowing_system,
                               precise_scrolling=precise) == expected


@pytest.mark.parametrize('screen', [(1920, 1080), (1280, 720), (960, 540), (800, 600)])
def test_initial_geometry_fits_scaled_desktop(screen):
    width, height = desktop.initial_window_size(*screen)
    assert 0 < width <= min(screen[0] - 80, 1280)
    assert 0 < height <= min(screen[1] - 100, 840)


@pytest.mark.parametrize('windowing_system', ['win32', 'x11', 'aqua'])
def test_editor_shortcuts_use_available_tk_modifiers(windowing_system):
    from recognizer import ui
    editor = object.__new__(ui.DiagramEditor)
    bindings, calls = {}, []
    editor.root = SimpleNamespace(tk=SimpleNamespace(call=lambda *a: windowing_system),
                                  bind=lambda key, callback, **kw: bindings.update({key: callback}))
    editor.canvas = SimpleNamespace(bind=lambda *a, **kw: None)
    for name in ('open_file', 'save_as', 'undo', 'redo'):
        setattr(editor, name, lambda name=name: calls.append(name))
    editor._bindings()
    for key, name in [('o', 'open_file'), ('s', 'save_as'), ('z', 'undo'), ('Shift-Z', 'redo')]:
        assert bindings[f'<Control-{key}>'](None) == 'break'
        assert calls[-1] == name
    command_keys = [key for key in bindings if 'Command' in key]
    if windowing_system == 'aqua':
        assert len(command_keys) == 4
        assert '<Control-y>' not in bindings
        assert desktop.shortcut_accelerator('z', windowing_system, shift=True) == '⇧⌘Z'
    else:
        assert command_keys == []
        assert bindings['<Control-y>'](None) == 'break'
        assert calls[-1] == 'redo'
        assert desktop.shortcut_accelerator('z', windowing_system, shift=True) == 'Ctrl+Shift+Z'


def test_windows_high_resolution_wheel_accumulates_sidebar_pixels():
    from recognizer import ui
    scrolled = []
    controls = SimpleNamespace(tk=SimpleNamespace(call=lambda *a: 'win32'),
        precise_scrolling=False, wheel_remainder=0.,
        viewport=SimpleNamespace(yview_scroll=lambda *a: scrolled.append(a)))
    event = SimpleNamespace(delta=-1, num=None, widget=None)
    for _ in range(5):
        assert ui.ScrollableControls._wheel(controls, event) == 'break'
    assert sum(pixels for pixels, unit in scrolled) == 1


@pytest.mark.parametrize('windowing_system, delta, button, factor', [
    ('win32', 240, None, 1.1**2), ('win32', 30, None, 1.1**.25),
    ('x11', 0, 5, 1/1.12), ('aqua', -3, None, 1/1.1),
])
def test_canvas_zoom_preserves_native_wheel_and_pointer_anchor(windowing_system, delta, button, factor):
    from recognizer import ui
    zoomed = []
    editor = SimpleNamespace(root=SimpleNamespace(tk=SimpleNamespace(call=lambda *a: windowing_system)),
                             zoom=lambda *a: zoomed.append(a))
    event = SimpleNamespace(delta=delta, num=button, x=45, y=90)
    assert ui.DiagramEditor._wheel(editor, event) == 'break'
    assert zoomed[0][0] == pytest.approx(factor)
    assert zoomed[0][1] == (45, 90)
