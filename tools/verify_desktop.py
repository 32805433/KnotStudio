#!/usr/bin/env python3
"""Inspect native dependencies and execute a relocated, isolated desktop bundle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))
from tools.native_dependencies import LINUX_SYSTEM_LIBRARIES, linux_links, windows_imports, windows_system_library


def isolated_environment(empty_path: Path):
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(('PYTHON', 'DYLD_', 'LD_', '_PYI_',
                                          'TESSDATA', 'TESSERACT', 'KNOTSTUDIO'))}
    if sys.platform == 'win32':
        # os.environ is case-insensitive on Windows, but this copied dict is
        # not; its keys commonly contain SYSTEMROOT in uppercase.
        system_root = next((value for key, value in env.items()
                            if key.upper() == 'SYSTEMROOT'), r'C:\Windows')
        env['PATH'] = str(Path(system_root) / 'System32')
    else:
        # Even /usr/bin may contain Python and Tesseract on the build machine.
        # All child executables required by the app must use bundled paths.
        env['PATH'] = str(empty_path)
    env['PYTHONNOUSERSITE'] = '1'
    return env


def self_test(executable: Path, temporary: Path, gui=False):
    report = temporary / 'self-test.json'
    empty = temporary / 'empty-path'
    empty.mkdir(exist_ok=True)
    command = [str(executable), '--self-test', str(report)]
    if gui:
        command.append('--gui')
    process = subprocess.run(command, cwd=temporary, env=isolated_environment(empty),
                             capture_output=True, timeout=240)
    if report.is_file():
        result = json.loads(report.read_text(encoding='utf-8'))
    else:
        result = {'ok': False, 'detail': process.stderr.decode(errors='replace')[-4000:]}
    result['exit_code'] = process.returncode
    result['ok'] = bool(result.get('ok')) and process.returncode == 0
    return result


def inspect_native(app: Path):
    """Reject unresolved/build-machine dependencies and newer Linux ABI needs."""
    errors, count = [], 0
    candidates = list(app.rglob('*'))
    by_name = {p.name.casefold(): p for p in candidates if p.is_file()}
    for candidate in candidates:
        relative = candidate.relative_to(app)
        if candidate.is_symlink():
            if not candidate.exists() or not candidate.resolve().is_relative_to(app):
                errors.append(f'{relative}: missing or external symlink target')
            continue
        if not candidate.is_file():
            continue
        with candidate.open('rb') as stream:
            magic = stream.read(4)
        if sys.platform == 'win32' and magic[:2] == b'MZ':
            count += 1
            try:
                import pefile
                with pefile.PE(str(candidate), fast_load=True) as pe:
                    if pe.FILE_HEADER.Machine != 0x8664:
                        errors.append(f'{relative}: expected an x86_64 PE binary')
                for name in windows_imports(candidate):
                    if name.casefold() not in by_name and not windows_system_library(name):
                        errors.append(f'{relative}: missing bundled DLL {name}')
            except Exception as exc:
                errors.append(f'{relative}: could not inspect PE imports: {exc}')
        elif sys.platform.startswith('linux') and magic == b'\x7fELF':
            count += 1
            try:
                env = dict(os.environ, LD_LIBRARY_PATH=str(app / '_internal'))
                links, missing = linux_links(candidate, env=env)
                errors.extend(f'{relative}: {message}' for message in missing)
                for target in links:
                    if not target.resolve().is_relative_to(app) and not LINUX_SYSTEM_LIBRARIES.fullmatch(target.name):
                        errors.append(f'{relative}: links outside the bundle: {target.name}')
                versions = subprocess.check_output(['readelf', '--version-info', str(candidate)], text=True, timeout=30)
                for major, minor in re.findall(r'Name: GLIBC_(\d+)\.(\d+)', versions):
                    if (int(major), int(minor)) > (2, 39):
                        errors.append(f'{relative}: requires glibc {major}.{minor}, newer than Ubuntu 24.04')
                headers = subprocess.check_output(['readelf', '-h', str(candidate)], text=True, timeout=30)
                if not re.search(r'Machine:\s+Advanced Micro Devices X86-64', headers):
                    errors.append(f'{relative}: expected an x86_64 ELF binary')
                dynamic = subprocess.check_output(['readelf', '-d', str(candidate)], text=True, timeout=30)
                for rpath in re.findall(r'\((?:RPATH|RUNPATH)\).*?\[(.*?)\]', dynamic):
                    if any(part.startswith('/') for part in rpath.split(':')):
                        errors.append(f'{relative}: contains an absolute build-machine library search path')
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f'{relative}: could not inspect ELF dependencies: {exc}')
    if count == 0:
        errors.append('The application contains no native executable or library.')
    return count, sorted(set(errors))


def verify(app: Path, output: Path, gui=False):
    app = app.resolve()
    if sys.platform == 'darwin':
        from tools.verify_macos import verify as verify_macos
        result = verify_macos(app, output, gui=gui)
        info_path = app / 'Contents' / 'Resources' / 'build-info.json'
        if not info_path.is_file():
            info_path = app / 'Contents' / 'Frameworks' / 'build-info.json'
        if info_path.is_file():
            info = json.loads(info_path.read_text(encoding='utf-8'))
            result.update({k: info.get(k) for k in ('version', 'source_commit', 'source_dirty', 'platform', 'architecture')})
            output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        return result
    errors, count, runtime = [], 0, {'ok': False}
    info_path = app / '_internal' / 'build-info.json'
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.is_file() else {}
    with tempfile.TemporaryDirectory(prefix='Knot Studio relocation ') as directory:
        temporary = Path(directory)
        relocated = temporary / 'Moved Knot Studio'
        shutil.copytree(app, relocated, symlinks=True)
        count, errors = inspect_native(relocated)
        executable = relocated / ('KnotStudio.exe' if sys.platform == 'win32' else 'KnotStudio')
        if sys.platform.startswith('linux') and (relocated / 'launch.sh').is_file():
            executable = relocated / 'launch.sh'
        try:
            runtime = self_test(executable, temporary, gui=gui)
            if not runtime.get('ok'):
                errors.append('The relocated application self-test failed.')
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(f'The relocated application self-test failed: {exc}')
    result = {'ok': not errors and runtime.get('ok', False),
              'native_libraries_checked': count, 'external_dependencies': errors,
              'relocated_self_test': runtime, 'isolated_path': True, 'gui_requested': gui,
              **{key: info.get(key) for key in ('version', 'source_commit', 'source_dirty', 'platform', 'architecture')}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if not result['ok']:
        raise RuntimeError(f'Bundle verification failed; inspect {output}')
    return result


def verify_installer(installer: Path, output: Path, gui=False):
    """Install per-user to a fresh path, run the installed app, and uninstall."""
    if sys.platform != 'win32':
        raise RuntimeError('Installer verification requires Windows.')
    result = json.loads(output.read_text(encoding='utf-8'))
    runtime = {'ok': False}
    try:
        with tempfile.TemporaryDirectory(prefix='Knot Studio installer ') as directory:
            temporary = Path(directory)
            installed = temporary / 'Installed Knot Studio'
            verification_id = uuid.uuid4().hex
            try:
                subprocess.run([str(installer.resolve()), '/VERYSILENT', '/SUPPRESSMSGBOXES',
                                '/NORESTART', '/SP-', '/NOICONS', f'/DIR={installed}',
                                f'/KnotStudioVerify={verification_id}',
                                f'/GROUP=Knot Studio Verification {verification_id}',
                                f'/LOG={temporary / "install.log"}'], check=True, timeout=180)
                runtime = self_test(installed / 'KnotStudio.exe', temporary, gui=gui)
            finally:
                uninstaller = installed / 'unins000.exe'
                if uninstaller.is_file():
                    subprocess.run([str(uninstaller), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'],
                                    check=True, timeout=180)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        runtime = {'ok': False, 'detail': str(exc)}
    result['installer_self_test'] = runtime
    result['ok'] = result['ok'] and runtime.get('ok', False)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if not result['ok']:
        raise RuntimeError(f'Installer verification failed; inspect {output}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    parser.add_argument('--output', type=Path, default=Path('dist/verification.json'))
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--installer', type=Path)
    args = parser.parse_args()
    result = verify(args.app, args.output, gui=args.gui)
    if args.installer:
        result = verify_installer(args.installer, args.output, gui=args.gui)
    print(json.dumps(result, indent=2))
