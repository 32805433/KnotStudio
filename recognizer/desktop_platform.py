"""Small desktop conventions shared by source and bundled releases."""
import os
from pathlib import Path
import subprocess
import sys


def log_directory():
    home = Path.home()
    if sys.platform == 'darwin':
        return home/'Library'/'Logs'/'Knot Studio'
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA') or home/'AppData'/'Local')
        return base/'Knot Studio'/'Logs'
    state = Path(os.environ.get('XDG_STATE_HOME') or home/'.local'/'state')
    # The XDG specification requires absolute paths; ignore relative values.
    if not state.is_absolute():
        state = home/'.local'/'state'
    return state/'knot-studio'/'logs'


def executable_name(name):
    return name + '.exe' if sys.platform == 'win32' else name


def hidden_subprocess_options():
    """OCR is a console executable, even when the editor has no console."""
    if sys.platform == 'win32':
        return {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)}
    return {}


def shortcut_modifiers(windowing_system):
    # Retain the existing Control alternatives on macOS.
    return ('Command', 'Control') if windowing_system == 'aqua' else ('Control',)


def shortcut_accelerator(key, windowing_system, *, shift=False):
    if windowing_system == 'aqua':
        return ('⇧' if shift else '') + '⌘' + key.upper()
    return 'Ctrl+' + ('Shift+' if shift else '') + key.upper()


def wheel_steps(event, *, windowing_system, precise_scrolling=False):
    """Positive means up/zoom in; preserve fractional Windows wheel deltas."""
    number = getattr(event, 'num', None)
    if number in (4, 5):
        return 1. if number == 4 else -1.
    delta = getattr(event, 'delta', 0)
    legacy_aqua = windowing_system == 'aqua' and not precise_scrolling
    return float(delta) / (1 if legacy_aqua else 120)


def initial_window_size(screen_width, screen_height):
    # Screen dimensions are logical pixels on a scaled desktop.
    return min(1280, max(1, screen_width - 80)), min(840, max(1, screen_height - 100))


def _open_frozen_windows_document(path):
    """Keep PyInstaller's DLL directory out of an external app's search path."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    get_directory = kernel32.GetDllDirectoryW
    get_directory.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
    get_directory.restype = wintypes.DWORD
    set_directory = kernel32.SetDllDirectoryW
    set_directory.argtypes = [wintypes.LPCWSTR]
    set_directory.restype = wintypes.BOOL

    size = 0
    while True:
        buffer = ctypes.create_unicode_buffer(size) if size else None
        ctypes.set_last_error(0)
        length = get_directory(size, buffer)
        if not length and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        if not length or length < size:
            original = buffer.value if buffer is not None else ''
            break
        # Retry if another caller changed the directory since the size query.
        size = length + 1
    if not original:
        # Already clear; leave the original empty/default search mode intact.
        os.startfile(path)
        return
    if not set_directory(None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        os.startfile(path)
    finally:
        if not set_directory(original):
            raise ctypes.WinError(ctypes.get_last_error())


def open_local_document(path):
    """Ask the desktop to open a local file, including paths with spaces."""
    path = Path(path).resolve(strict=True)
    if sys.platform == 'win32':
        if getattr(sys, 'frozen', False):
            _open_frozen_windows_document(str(path))
        else:
            os.startfile(str(path))
        return
    environment = os.environ.copy()
    if getattr(sys, 'frozen', False):
        # Bundled libraries are for this process, not the user's browser.
        for variable in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
            original = environment.get(variable + '_ORIG')
            if original is None:
                environment.pop(variable, None)
            else:
                environment[variable] = original
    command = 'open' if sys.platform == 'darwin' else 'xdg-open'
    subprocess.Popen([command, str(path)], env=environment,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
