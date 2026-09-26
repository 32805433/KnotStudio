#!/usr/bin/env python3
"""Build a self-contained macOS application, verify it, and archive matching source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

from build_docs import build_bundle as build_docs
from check_release import check_source, source_files
from make_icon import create as make_icon
from verify_macos import verify

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.1.0'


def run(*command, env=None):
    subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tesseract', type=Path, help='OCR executable to bundle (default: search PATH)')
    parser.add_argument('--tessdata', type=Path, help='Directory containing eng.traineddata')
    parser.add_argument('--codesign-identity', help='Developer ID Application identity in Keychain')
    parser.add_argument('--notary-profile', help='notarytool Keychain profile; requires Developer ID signing')
    parser.add_argument('--skip-tests', action='store_true', help='Skip pytest (bundle verification is always performed)')
    parser.add_argument('--gui-test', action='store_true', help='Also initialize the UI in the relocated bundle')
    args = parser.parse_args()
    if sys.platform != 'darwin':
        parser.error('Build the macOS application on macOS.')
    if args.notary_profile and not args.codesign_identity:
        parser.error('--notary-profile requires --codesign-identity')
    tesseract = args.tesseract or (Path(shutil.which('tesseract')) if shutil.which('tesseract') else None)
    if tesseract is None or not tesseract.is_file():
        parser.error('Install Tesseract on the build machine or pass --tesseract.')
    tesseract = tesseract.resolve()
    tessdata = args.tessdata
    if tessdata is None:
        for directory in (tesseract.parent.parent/'share'/'tessdata',
                          Path('/opt/homebrew/share/tessdata'), Path('/usr/local/share/tessdata')):
            if (directory/'eng.traineddata').is_file():
                tessdata = directory
                break
    if tessdata is None or not (tessdata/'eng.traineddata').is_file():
        parser.error('English OCR data is missing; pass --tessdata.')
    errors, _ = check_source(ROOT)
    if errors:
        parser.error('\n'.join(errors))
    # Public binary releases use only the image-processing modules. The standard
    # wheel can contain unused video codecs with additional redistribution terms.
    from build_opencv import build as build_opencv
    build_opencv()
    if not args.skip_tests:
        run(sys.executable, '-m', 'pytest', '-q')
    assets = ROOT/'.build-assets'
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(exist_ok=True)
    from collect_licenses import collect
    collect(assets/'licenses'/'dependencies', tesseract=tesseract, tessdata=tessdata)
    build_docs(ROOT, assets/'resources')
    make_icon(assets/'KnotStudio.icns')
    arch = platform.machine()
    build_info = {'version': VERSION, 'architecture': arch, 'minimum_macos': '15.7.5',
                  'python': platform.python_version(), 'build_macos': platform.mac_ver()[0],
                  'signing': 'Developer ID' if args.codesign_identity else 'ad-hoc',
                  'notarization_requested': bool(args.notary_profile)}
    (assets/'build-info.json').write_text(json.dumps(build_info, indent=2)+'\n')
    env = dict(os.environ, KNOTSTUDIO_TESSERACT=str(tesseract), KNOTSTUDIO_TESSDATA=str(tessdata),
               KNOTSTUDIO_ARCH=arch, KNOTSTUDIO_CODESIGN=args.codesign_identity or '',
               MACOSX_DEPLOYMENT_TARGET='15.7.5',
               PYINSTALLER_CONFIG_DIR=str(ROOT/'build'/'pyinstaller-cache'))
    run(sys.executable, '-m', 'PyInstaller', '--clean', '--noconfirm', ROOT/'packaging'/'KnotStudio.spec', env=env)
    app = ROOT/'dist'/'Knot Studio.app'
    result = verify(app, ROOT/'dist'/'verification.json', gui=args.gui_test)
    archive = ROOT/'dist'/f'KnotStudio-{VERSION}-macOS-{arch}.zip'
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
    source = ROOT/'dist'/f'KnotStudio-{VERSION}-source.zip'
    with zipfile.ZipFile(source, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in source_files(ROOT):
            if path.is_file() and not path.is_symlink():
                bundle.write(path, Path(f'KnotStudio-{VERSION}')/path.relative_to(ROOT))
    installation = ('Knot Studio '+VERSION+'\n\nDrag Knot Studio.app to Applications and open it.\n'
                    'Python and Homebrew are not needed. The user guide is available from the Help menu.\n'
                    'Requires macOS 15.7.5 or later, '+arch+'.\n\n'+
                    ('This release is signed and notarized.\n' if args.notary_profile else
                     'This build is not notarized. macOS may block a downloaded copy.\n'
                     'See https://github.com/32805433/KnotStudio/blob/main/docs/INSTALLING.md for first-launch help.\n')+
                    '\nMatching application source is included in the source ZIP. See bundled licenses for dependencies.\n')
    with zipfile.ZipFile(archive, 'a', compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(source, source.name)
        bundle.write(ROOT/'LICENSE', 'LICENSE')
        bundle.writestr('READ ME.txt', installation)
    checksums = ''.join(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n'
                        for path in (archive, source))
    (ROOT/'dist'/'SHA256SUMS.txt').write_text(checksums)
    # PyInstaller's intermediate onedir tree duplicates the finished app.
    shutil.rmtree(ROOT/'dist'/'KnotStudio')
    print(json.dumps({'app': str(app), 'archive': str(archive), 'source': str(source),
                      'archive_megabytes': round(archive.stat().st_size/1e6, 1),
                      'verification_passed': result['ok'], **build_info}, indent=2))


if __name__ == '__main__':
    main()
