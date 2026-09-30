"""Local Python, Tcl/Tk, Debian, and vcpkg notice evidence for release builds."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import quote


def debian_package(path: Path) -> str:
    candidates = [path, path.resolve()]
    for candidate in list(candidates):
        if str(candidate).startswith(('/lib/', '/bin/')):
            candidates.append(Path('/usr' + str(candidate)))
    for candidate in candidates:
        result = subprocess.run(['dpkg-query', '-S', str(candidate)],
                                text=True, capture_output=True, check=False)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                owners, separator, _ = line.partition(': ')
                if not separator:
                    continue
                # dpkg-query may print diversion explanations before the
                # actual owner. They are prose, not package identifiers.
                for owner in owners.split(', '):
                    if re.fullmatch(r'[a-z0-9][a-z0-9+.-]*(?::[a-z0-9][a-z0-9-]*)?', owner):
                        return owner
    raise RuntimeError(f'No Debian package owns {path.name}.')


def collect_debian(paths, component, file, issues):
    packages = set()
    for path in paths:
        try:
            packages.add(debian_package(path))
        except (OSError, RuntimeError) as exc:
            issues.append(str(exc))
    for package in sorted(packages):
        try:
            details = subprocess.check_output([
                'dpkg-query', '-W', '-f=${Package}\t${Version}\t${source:Package}\t${source:Version}',
                package], text=True).strip().split('\t')
            name, version, source, source_version = details
        except (OSError, subprocess.CalledProcessError, ValueError):
            issues.append(f'{package}: cannot read installed Debian version metadata.')
            continue
        entry = component(name, version, 'native-debian',
                          f'https://launchpad.net/ubuntu/+source/{source}/{quote(source_version, safe="")}')
        entry['binary_package'] = package
        entry['source_package'] = source
        entry['source_version'] = source_version
        directory = Path('/usr/share/doc') / name
        file(entry, directory / 'copyright', f'native/{name}/copyright',
             f'debian:{package}/usr/share/doc/{name}/copyright')


def parse_vcpkg_status(text, architecture=None):
    """Return exact installed port versions, omitting feature-only stanzas."""
    packages = {}
    for stanza in re.split(r'\n\s*\n', text):
        fields = {}
        for line in stanza.splitlines():
            if ': ' in line and not line.startswith(' '):
                key, value = line.split(': ', 1)
                fields[key] = value
        if fields.get('Status') == 'install ok installed' and 'Feature' not in fields:
            if architecture is not None and fields.get('Architecture') != architecture:
                continue
            name = fields.get('Package')
            if name:
                packages[name] = fields
    return packages


def collect_vcpkg(root, component, file, issues):
    root = Path(root)
    status = root.parent / 'vcpkg' / 'status'
    if not status.is_file():
        issues.append('OCR vcpkg installed package status is missing; pass --ocr-root for its target triplet.')
        return
    ports = parse_vcpkg_status(status.read_text(encoding='utf-8'), architecture=root.name)
    commit = os.environ.get('VCPKG_COMMIT', '')
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        issues.append('VCPKG_COMMIT must identify the exact 40-character source baseline.')
    matching = {name: fields for name, fields in ports.items()
                if fields.get('Architecture') == root.name}
    if not {'tesseract', 'leptonica'} <= matching.keys():
        issues.append('The OCR vcpkg target must contain installed Tesseract and Leptonica.')
    for name, fields in sorted(matching.items()):
        version = fields.get('Version', fields.get('Version-Semver', fields.get('Version-Date', '')))
        if not version:
            issues.append(f'{name}: missing vcpkg version.')
        revision = fields.get('Port-Version', '0')
        entry = component(name, version, 'native-vcpkg',
                          f'https://github.com/microsoft/vcpkg/tree/{commit}/ports/{name}')
        entry['port_revision'] = revision
        entry['vcpkg_commit'] = commit
        entry['architecture'] = fields.get('Architecture')
        entry['source_reference'] = 'The pinned port contains source URL, SHA-512, and build patches.'
        notice = root / 'share' / name / 'copyright'
        file(entry, notice, f'native/vcpkg/{name}/copyright',
             f'vcpkg:{name}:{version}#{revision}/share/{name}/copyright')


def collect_runtime(base, runtime_notices, component, file, issues):
    """Require complete Python notices and the actual Tcl/Tk runtime notices."""
    notices = Path(runtime_notices) if runtime_notices else None
    version = '.'.join(map(str, sys.version_info[:3]))
    minor = f'python{sys.version_info.major}.{sys.version_info.minor}'
    entry = component('Python', version, 'runtime',
                      f'https://github.com/python/cpython/tree/v{version}')
    candidates = [base / 'LICENSE.txt', base / 'lib' / minor / 'LICENSE.txt']
    if notices:
        candidates.append(notices / 'Python' / 'LICENSE.txt')
    found = next((p for p in candidates if p.is_file()), candidates[0])
    file(entry, found, 'runtime/Python/LICENSE.txt', f'CPython v{version}/LICENSE')
    # CI stages this from the exact CPython source tag; the core LICENSE omits
    # some third-party notices reproduced in Doc/license.rst.
    candidates = [base / 'Doc' / 'license.rst']
    if notices:
        candidates.insert(0, notices / 'Python' / 'license.rst')
    found = next((p for p in candidates if p.is_file()), candidates[0])
    file(entry, found, 'runtime/Python/bundled-library-notices.rst',
         f'CPython v{version}/Doc/license.rst')
    import tkinter
    tcl = tkinter.Tcl()
    tcl_library = Path(tcl.eval('info library'))
    for name, ver in (('Tcl', str(tcl.call('info', 'patchlevel'))), ('Tk', str(tkinter.TkVersion))):
        entry = component(name, ver, 'runtime', 'https://www.tcl-lang.org/')
        lower = name.lower()
        candidates = list((base / 'tcl').glob(f'{lower}*/license.terms'))
        candidates += list(tcl_library.parent.glob(f'{lower}*/license.terms'))
        if notices:
            candidates.append(notices / name / 'license.terms')
        found = next((p for p in candidates if p.is_file()), None)
        if found:
            file(entry, found, f'runtime/{name}/license.terms', f'{name} {ver}/license.terms')
        elif sys.platform.startswith('linux'):
            # Ubuntu ships the upstream notices inside the runtime package's
            # Debian copyright, including notices for Tk's bundled libraries.
            copyright = Path('/usr/share/doc') / f'lib{lower}{ver[:3]}' / 'copyright'
            file(entry, copyright, f'runtime/{name}/copyright',
                 f'debian:lib{lower}{ver[:3]}/copyright')
        else:
            issues.append(f'{name} {ver}: no installed runtime license.terms.')


def collect_native(tesseract, ocr_root, component, file, issues):
    if sys.platform == 'win32':
        if ocr_root is None:
            issues.append('Windows OCR requires --ocr-root with its vcpkg package notices.')
        else:
            collect_vcpkg(ocr_root, component, file, issues)
    elif sys.platform.startswith('linux'):
        try:
            from tools.native_dependencies import linux_links
        except ModuleNotFoundError:
            from native_dependencies import linux_links
        paths, errors = linux_links(tesseract)
        issues.extend(errors)
        collect_debian([tesseract, *paths], component, file, issues)
    else:
        issues.append(f'Unsupported native notice collector: {sys.platform}.')
