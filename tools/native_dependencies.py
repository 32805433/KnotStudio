"""Build-time native dependency discovery; never used by the installed app."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


# These libraries are part of the documented Ubuntu 24.04 desktop baseline.
# Other dependencies (including libstdc++, Tcl/Tk, and OCR libraries) must travel
# with the application. The loader and glibc must remain a matched system pair.
LINUX_SYSTEM_LIBRARIES = re.compile(
    r'^(?:ld-linux[^/]*|lib(?:c|m|pthread|dl|rt|util|resolv|nss_[\w]+|'
    r'X11|Xext|Xrender|Xft|Xau|Xdmcp|xcb|fontconfig|freetype|'
    r'GL|GLX|GLdispatch|EGL|drm|gbm)\.so(?:\..*)?)$'
)
WINDOWS_REDISTRIBUTABLE = re.compile(r'^(?:vcruntime|msvcp|concrt|vcomp|python)\d.*\.dll$', re.I)


def windows_system_library(name: str, resolved: Path | None = None) -> bool:
    """Do not mistake an installed MSVC/Python runtime for an OS dependency."""
    if WINDOWS_REDISTRIBUTABLE.match(name):
        return False
    if name.lower().startswith(('api-ms-win-', 'ext-ms-win-')):
        return True
    system = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32'
    if resolved is not None:
        return resolved.resolve().is_relative_to(system.resolve())
    return (system / name).is_file()


def linux_links(binary: Path, env=None) -> tuple[list[Path], list[str]]:
    result = subprocess.run(['ldd', str(binary)], capture_output=True, text=True,
                            env=env, timeout=30)
    if result.returncode:
        return [], [f'{binary.name}: ldd failed: {result.stderr.strip()}']
    paths, missing = [], []
    for line in result.stdout.splitlines():
        if '=> not found' in line:
            missing.append(line.strip())
        match = re.search(r'(?:=>\s+|^\s*)(/.+?)\s+\(0x[0-9a-fA-F]+\)', line)
        if match:
            paths.append(Path(match.group(1)))
    return paths, missing


def windows_imports(binary: Path) -> list[str]:
    import pefile
    with pefile.PE(str(binary), fast_load=True) as pe:
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'],
            pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT'],
        ])
        return sorted({entry.dll.decode('ascii')
                       for key in ('DIRECTORY_ENTRY_IMPORT', 'DIRECTORY_ENTRY_DELAY_IMPORT')
                       for entry in getattr(pe, key, [])})


def windows_dependencies(executable: Path, extra_dirs=()) -> tuple[list[Path], list[str]]:
    """Resolve DLL imports recursively, keeping every non-system dependency."""
    from PyInstaller.depend.bindepend import get_imports
    search = [str(executable.parent), *map(str, extra_dirs)]
    queue, found, errors = [executable.resolve()], set(), []
    while queue:
        binary = queue.pop()
        if binary in found:
            continue
        found.add(binary)
        for name, location in get_imports(str(binary), search_paths=search):
            resolved = Path(location) if location else None
            if windows_system_library(name, resolved):
                continue
            if resolved is None:
                errors.append(f'{binary.name}: missing DLL {name}')
            else:
                queue.append(resolved.resolve())
    return sorted(found), sorted(set(errors))


def ocr_binaries(executable: Path, root: Path | None = None):
    """Additional OCR binaries with their PyInstaller destination directories."""
    if sys.platform == 'win32':
        paths, errors = windows_dependencies(executable, [root / 'bin'] if root else [])
        if errors:
            raise RuntimeError('; '.join(errors))
        # Keep OCR DLLs beside its child process. The app's Python extensions use
        # PyInstaller's separately collected copies in the internal directory.
        return [(str(path), 'ocr/bin') for path in paths]
    return [(str(executable), 'ocr/bin')]
