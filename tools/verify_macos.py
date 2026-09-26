"""Verify the bundle's native libraries and run it from a clean, relocated path."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import tempfile


def verify(app: Path, output: Path, gui=False):
    app = app.resolve()
    info = plistlib.loads((app/'Contents'/'Info.plist').read_bytes())
    minimum_allowed = info['LSMinimumSystemVersion']
    def version_tuple(value):
        parts = tuple(int(part) for part in value.split('.'))
        return (parts+(0, 0, 0))[:3]
    native_count, errors = 0, []
    checked = set()
    for candidate in app.rglob('*'):
        if candidate.is_symlink():
            if not candidate.resolve().is_relative_to(app) or not candidate.exists():
                errors.append(f'{candidate.relative_to(app)} has a missing or external symlink target')
            continue
        if not candidate.is_file():
            continue
        with candidate.open('rb') as stream:
            magic = stream.read(4)
        if magic not in (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf',
                         b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'):
            continue
        if candidate.resolve() in checked:
            continue
        checked.add(candidate.resolve())
        native_count += 1
        try:
            linked = subprocess.run(['otool', '-L', str(candidate)], check=True,
                                    capture_output=True, text=True, timeout=30).stdout
            metadata = subprocess.run(['otool', '-l', str(candidate)], check=True,
                                      capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f'{candidate.relative_to(app)} could not be inspected: {error}')
            continue
        for line in linked.splitlines()[1:]:
            name = line.strip().split(' (', 1)[0]
            if name.startswith('/') and not name.startswith(('/usr/lib/', '/System/Library/')):
                errors.append(f'{candidate.relative_to(app)} links outside the app: {name}')
        for block in re.split(r'^Load command \d+\s*$', metadata, flags=re.M):
            field = 'minos' if 'cmd LC_BUILD_VERSION\n' in block else ('version' if 'cmd LC_VERSION_MIN_MACOSX\n' in block else None)
            if field is None:
                continue
            match = re.search(r'^\s+'+field+r'\s+(\d+(?:\.\d+){1,2})\s*$', block, re.M)
            if match and version_tuple(match[1]) > version_tuple(minimum_allowed):
                errors.append(f'{candidate.relative_to(app)} requires macOS {match[1]}, newer than {minimum_allowed}')
    if native_count == 0:
        errors.append('The app contains no native executable or library.')
    try:
        subprocess.run(['codesign', '--verify', '--deep', '--strict', str(app)], check=True,
                       capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, 'stderr', '') or str(error)
        errors.append(f'Code signature verification failed: {str(detail)[-3000:]}')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('PYTHON', 'DYLD_', '_PYI_', 'TESSDATA'))}
    env['PATH'] = '/usr/bin:/bin:/usr/sbin:/sbin'
    env['PYTHONNOUSERSITE'] = '1'
    with tempfile.TemporaryDirectory(prefix='Knot Studio relocation ') as directory:
        temporary = Path(directory)
        relocated = temporary/'Moved Knot Studio.app'
        shutil.copytree(app, relocated, symlinks=True)
        report = temporary/'self-test.json'
        command = [str(relocated/'Contents'/'MacOS'/'KnotStudio'), '--self-test', str(report)]
        if gui:
            command.append('--gui')
        try:
            process = subprocess.run(command, cwd=temporary, env=env, capture_output=True, timeout=180)
            if report.is_file():
                runtime = json.loads(report.read_text())
            else:
                runtime = {'ok': False, 'detail': process.stderr.decode(errors='replace')[-3000:]}
            if process.returncode != 0:
                errors.append(f'Relocated application self-test exited {process.returncode}')
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            runtime = {'ok': False, 'detail': str(error)}
            errors.append(f'Relocated application self-test failed: {error}')
    result = {'ok': not errors and runtime.get('ok', False), 'native_libraries_checked': native_count,
              'external_dependencies': errors, 'relocated_self_test': runtime,
              'isolated_path': True, 'gui_requested': gui}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    if not result['ok']:
        raise RuntimeError(f'Bundle verification failed; inspect {output}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    parser.add_argument('--output', type=Path, default=Path('dist/verification.json'))
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.app, args.output, args.gui), indent=2))
