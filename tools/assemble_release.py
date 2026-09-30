"""Validate native build artifacts and prepare one immutable multi-platform release."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
import zipfile

TARGETS = ('macOS-arm64', 'Windows-x86_64', 'Linux-x86_64')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_contents(path):
    with zipfile.ZipFile(path) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate source archive members')
        return {name: hashlib.sha256(source.read(name)).hexdigest()
                for name in names if not name.endswith('/')}


def assemble(root, artifacts, output, commit):
    version = tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('Release version must be MAJOR.MINOR.PATCH')
    output.mkdir(parents=True, exist_ok=False)
    sources = None
    selected = []
    reports = {}
    for target in TARGETS:
        folder = artifacts/f'KnotStudio-{target}'
        verification = json.loads((folder/'verification.json').read_text(encoding='utf-8'))
        info = json.loads((folder/'build-info.json').read_text(encoding='utf-8'))
        if not verification.get('ok') or not verification.get('gui_requested'):
            raise ValueError(f'{target}: packaged GUI checks did not pass')
        if info.get('source_commit') != commit or info.get('version') != version:
            raise ValueError(f'{target}: wrong source commit or version')
        if f"{info.get('platform')}-{info.get('architecture')}" != target:
            raise ValueError(f'{target}: wrong platform or architecture')
        if any(verification.get(key) != info.get(key)
               for key in ('source_commit', 'version', 'platform', 'architecture')):
            raise ValueError(f'{target}: verification report belongs to a different build')
        if target.startswith('Windows') and not verification.get('installer_self_test', {}).get('ok'):
            raise ValueError('Windows installer verification did not pass')
        checksums = {}
        for line in (folder/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
            hash_value, name = line.split(maxsplit=1)
            name = name.strip()
            if Path(name).name != name or not re.fullmatch(r'[0-9a-f]{64}', hash_value):
                raise ValueError('Invalid checksum entry')
            if name in checksums:
                raise ValueError('Duplicate checksum entry')
            checksums[name] = hash_value
            if digest(folder/name) != hash_value:
                raise ValueError(f'{target}: checksum mismatch: {name}')
        source_name = f'KnotStudio-{version}-source.zip'
        expected = [source_name]
        stem = f'KnotStudio-{version}-{target}'
        expected += [stem+'.tar.gz'] if target.startswith('Linux') else [stem+'.zip']
        if target.startswith('Windows'):
            expected += [stem+'-Setup.exe']
        if set(checksums) != set(expected):
            raise ValueError(f'{target}: unexpected or missing release files')
        content = source_contents(folder/source_name)
        if sources is not None and sources != content:
            raise ValueError('Platforms did not build from identical source files')
        if sources is None:
            sources = content
            selected.append(folder/source_name)
        selected.extend(folder/name for name in expected if name != source_name)
        reports[target] = {'build': info, 'verification': verification}
    # Compare the bundled source to the checkout, not only to the other jobs.
    from tools.check_release import source_files
    expected = {f'KnotStudio-{version}/{p.relative_to(root).as_posix()}': digest(p)
                for p in source_files(root) if p.is_file() and not p.is_symlink()}
    if sources != expected:
        raise ValueError('Source archives differ from the release checkout')
    for p in selected:
        shutil.copy2(p, output/p.name)
    (output/'verification.json').write_text(json.dumps(reports, indent=2)+'\n', encoding='utf-8')
    assets = sorted(output.iterdir())
    (output/'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in assets), encoding='utf-8')
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(assemble(root, args.artifacts, args.output, args.commit))


if __name__ == '__main__':
    # Permit the same import layout when executed directly as from pytest.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
