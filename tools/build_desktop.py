#!/usr/bin/env python3
"""Build, verify, and archive a self-contained native Knot Studio application."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))


def project_version(root=ROOT):
    with (root / 'pyproject.toml').open('rb') as stream:
        return tomllib.load(stream)['project']['version']


def target_platform(system=None, machine=None):
    system = system or sys.platform
    machine = (machine or platform.machine()).lower()
    architecture = {'amd64': 'x86_64', 'aarch64': 'arm64'}.get(machine, machine)
    if system == 'darwin' and architecture in {'arm64', 'x86_64'}:
        return 'macOS', architecture
    if system in {'win32', 'linux'} and architecture == 'x86_64':
        return ('Windows' if system == 'win32' else 'Linux'), architecture
    raise ValueError(f'Unsupported native desktop target: {system}/{machine}')


def run(*command, env=None):
    subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)


def source_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get('GITHUB_SHA', 'unversioned-source')


def find_tessdata(tesseract: Path):
    candidates = [tesseract.parent / 'tessdata', tesseract.parent.parent / 'share' / 'tessdata',
                  Path('/opt/homebrew/share/tessdata'), Path('/usr/local/share/tessdata'),
                  Path('/usr/share/tesseract-ocr/5/tessdata'), Path('/usr/share/tessdata')]
    return next((p for p in candidates if (p / 'eng.traineddata').is_file()), None)


def source_archive(version):
    from tools.check_release import source_files
    source = ROOT / 'dist' / f'KnotStudio-{version}-source.zip'
    with zipfile.ZipFile(source, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in source_files(ROOT):
            if path.is_file() and not path.is_symlink():
                bundle.write(path, Path(f'KnotStudio-{version}') / path.relative_to(ROOT))
    return source


def installation_text(version, target, arch, notarized=False):
    instructions = {
        'macOS': ('Drag Knot Studio.app to Applications and open it.\n'
                  f'Requires macOS 15.7.5 or later, {arch}.\n' +
                  ('This release is signed and notarized.\n' if notarized else
                   'This build is not notarized; macOS may block a downloaded copy.\n')),
        'Windows': ('Extract the entire ZIP, then open KnotStudio/KnotStudio.exe.\n'
                    'Keep the _internal folder beside the executable.\n'
                    'Alternatively, run the Setup.exe installer for a per-user installation.\n'
                    'Requires Windows 10 or Windows 11, 64-bit Intel/AMD.\n'
                    'An unsigned download may show a Windows SmartScreen prompt.\n'),
        'Linux': ('Extract the archive and open KnotStudio/launch.sh.\n'
                  'Run KnotStudio/install-desktop-entry.sh to add a desktop menu icon.\n'
                  'Keep the extracted folder in place after adding the icon.\n'
                  'Baseline: Ubuntu 24.04 x86_64 with a graphical desktop (glibc 2.39).\n'),
    }
    return (f'Knot Studio {version}\n\n' + instructions[target] +
            '\nPython and Tesseract do not need to be installed.\n'
            'The user guide and examples are available from the Help menu.\n'
            'Matching application source is included in the source ZIP.\n'
            'See bundled licenses for dependency notices and source references.\n'
            'Installation help: https://github.com/32805433/KnotStudio/blob/main/docs/INSTALLING.md\n')


def linux_launchers(app):
    for name in ('launch.sh', 'install-desktop-entry.sh'):
        shutil.copy2(ROOT / 'packaging' / 'linux' / name, app / name)
        (app / name).chmod(0o755)
    shutil.copy2(ROOT / '.build-assets' / 'KnotStudio.png', app / 'KnotStudio.png')


def build_installer(version, app):
    compiler = shutil.which('ISCC')
    if compiler is None:
        candidates = [Path(os.environ.get(name, default)) / 'Inno Setup 6' / 'ISCC.exe'
                      for name, default in (('ProgramFiles(x86)', r'C:\Program Files (x86)'),
                                            ('ProgramFiles', r'C:\Program Files'))]
        compiler = next((str(path) for path in candidates if path.is_file()), None)
    if compiler is None:
        raise RuntimeError('Install Inno Setup 6 to produce the Windows installer.')
    run(compiler, f'/DAppVersion={version}', f'/DAppSource={app}',
        f'/DOutputDirectory={ROOT / "dist"}', ROOT / 'packaging' / 'windows' / 'KnotStudio.iss')
    installer = ROOT / 'dist' / f'KnotStudio-{version}-Windows-x86_64-Setup.exe'
    if not installer.is_file():
        raise RuntimeError('Inno Setup did not create the expected installer.')
    return installer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tesseract', type=Path, help='OCR executable to bundle (default: search PATH)')
    parser.add_argument('--tessdata', type=Path, help='Directory containing pinned English eng.traineddata')
    parser.add_argument('--ocr-root', type=Path, help='Windows vcpkg installed target directory')
    parser.add_argument('--runtime-notices', type=Path, help='Prepared Python/Tcl/Tk source notices')
    parser.add_argument('--codesign-identity', help='macOS Developer ID Application identity')
    parser.add_argument('--notary-profile', help='macOS notarytool Keychain profile')
    parser.add_argument('--skip-tests', action='store_true', help='Skip pytest; packaged verification still runs')
    parser.add_argument('--gui-test', action='store_true', help='Initialize Tk/UI from the relocated application')
    parser.add_argument('--installer', action='store_true', help='Also build and verify a Windows Inno Setup installer')
    args = parser.parse_args(argv)
    try:
        target, arch = target_platform()
    except ValueError as exc:
        parser.error(str(exc))
    if args.notary_profile and not args.codesign_identity:
        parser.error('--notary-profile requires --codesign-identity')
    if target != 'macOS' and (args.codesign_identity or args.notary_profile):
        parser.error('Apple signing and notarization options require macOS.')
    if args.installer and target != 'Windows':
        parser.error('--installer is only available on Windows.')
    located = shutil.which('tesseract')
    tesseract = args.tesseract or (Path(located) if located else None)
    if tesseract is None or not tesseract.is_file():
        parser.error('Install Tesseract on the build machine or pass --tesseract.')
    tesseract = tesseract.resolve()
    tessdata = args.tessdata or find_tessdata(tesseract)
    if tessdata is None or not (tessdata / 'eng.traineddata').is_file():
        parser.error('English OCR data is missing; pass --tessdata.')
    tessdata = tessdata.resolve()
    from tools.check_release import check_source
    errors, _ = check_source(ROOT)
    if errors:
        parser.error('\n'.join(errors))
    from tools.build_opencv import build as build_opencv
    build_opencv()
    if not args.skip_tests:
        run(sys.executable, '-m', 'pytest', '-q')
    assets = ROOT / '.build-assets'
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir()
    from tools.collect_licenses import collect
    collect(assets / 'licenses' / 'dependencies', tesseract=tesseract,
            tessdata=tessdata, ocr_root=args.ocr_root,
            runtime_notices=args.runtime_notices)
    from tools.build_docs import build_bundle
    from tools.make_icon import create
    build_bundle(ROOT, assets / 'resources')
    for extension in ('.icns', '.ico', '.png'):
        create(assets / ('KnotStudio' + extension))
    version = project_version()
    build_info = {'version': version, 'source_commit': source_commit(), 'platform': target,
                  'architecture': arch, 'python': platform.python_version(),
                  'build_os': platform.platform(),
                  'minimum_macos': '15.7.5' if target == 'macOS' else None,
                  'minimum_windows': '10 (x64)' if target == 'Windows' else None,
                  'minimum_linux': 'Ubuntu 24.04 x86_64; glibc 2.39' if target == 'Linux' else None,
                  'signing': ('Developer ID' if args.codesign_identity else 'ad-hoc') if target == 'macOS' else 'unsigned',
                  'notarization_requested': bool(args.notary_profile)}
    (assets / 'build-info.json').write_text(json.dumps(build_info, indent=2) + '\n', encoding='utf-8')
    env = dict(os.environ, KNOTSTUDIO_TESSERACT=str(tesseract), KNOTSTUDIO_TESSDATA=str(tessdata),
               KNOTSTUDIO_OCR_ROOT=str(args.ocr_root.resolve()) if args.ocr_root else '',
               KNOTSTUDIO_VERSION=version, KNOTSTUDIO_ARCH=arch,
               KNOTSTUDIO_CODESIGN=args.codesign_identity or '',
               PYINSTALLER_CONFIG_DIR=str(ROOT / 'build' / 'pyinstaller-cache'))
    if target == 'macOS':
        env['MACOSX_DEPLOYMENT_TARGET'] = '15.7.5'
    run(sys.executable, '-m', 'PyInstaller', '--clean', '--noconfirm', ROOT / 'packaging' / 'KnotStudio.spec', env=env)
    dist = ROOT / 'dist'
    app = dist / ('Knot Studio.app' if target == 'macOS' else 'KnotStudio')
    shutil.copy2(assets / 'build-info.json', dist / 'build-info.json')
    if target == 'Linux':
        linux_launchers(app)
    from tools.verify_desktop import verify, verify_installer
    result = verify(app, dist / 'verification.json', gui=args.gui_test)
    source = source_archive(version)
    installation = installation_text(version, target, arch, bool(args.notary_profile))
    installer = None
    if target == 'macOS':
        archive = dist / f'KnotStudio-{version}-macOS-{arch}.zip'
        def archive_app():
            archive.unlink(missing_ok=True)
            run('ditto', '-c', '-k', '--sequesterRsrc', '--keepParent', app, archive)
        archive_app()
        if args.notary_profile:
            run('xcrun', 'notarytool', 'submit', archive, '--keychain-profile', args.notary_profile, '--wait')
            run('xcrun', 'stapler', 'staple', app)
            run('xcrun', 'stapler', 'validate', app)
            run('spctl', '--assess', '--type', 'execute', '--verbose', app)
            archive_app()
        with zipfile.ZipFile(archive, 'a', compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(source, source.name)
            bundle.write(ROOT / 'LICENSE', 'LICENSE')
            bundle.writestr('READ ME.txt', installation)
        shutil.rmtree(dist / 'KnotStudio')
    else:
        # Both installed and portable copies carry the exact application source.
        shutil.copy2(source, app / source.name)
        shutil.copy2(ROOT / 'LICENSE', app / 'LICENSE')
        (app / 'READ ME.txt').write_text(installation, encoding='utf-8')
        if target == 'Windows':
            archive = dist / f'KnotStudio-{version}-Windows-{arch}.zip'
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(app.rglob('*')):
                    if path.is_file():
                        bundle.write(path, Path(app.name) / path.relative_to(app))
            if args.installer:
                installer = build_installer(version, app)
                result = verify_installer(installer, dist / 'verification.json', gui=args.gui_test)
        else:
            archive = dist / f'KnotStudio-{version}-Linux-{arch}.tar.gz'
            with tarfile.open(archive, 'w:gz') as bundle:
                bundle.add(app, arcname=app.name)
    artifacts = [archive, source] + ([installer] if installer else [])
    checksums = []
    for path in artifacts:
        with path.open('rb') as stream:
            checksums.append(f'{hashlib.file_digest(stream, "sha256").hexdigest()}  {path.name}\n')
    (dist / 'SHA256SUMS.txt').write_text(''.join(checksums), encoding='utf-8')
    print(json.dumps({'app': str(app), 'archive': str(archive), 'source': str(source),
                      'installer': str(installer) if installer else None,
                      'archive_megabytes': round(archive.stat().st_size / 1e6, 1),
                      'verification_passed': result['ok'], **build_info}, indent=2))


if __name__ == '__main__':
    main()
