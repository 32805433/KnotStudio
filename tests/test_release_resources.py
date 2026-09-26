"""Examples and help resolve after installation or app relocation."""
import json
from pathlib import Path
import sys

import pytest

from recognizer import resources


def test_source_resources_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    source = Path(resources.__file__).resolve().parent.parent
    monkeypatch.chdir(tmp_path)
    assert resources.resource_root() == source
    entries = resources.example_entries()
    assert entries
    assert all(resources.example_path(entry).is_file() for entry in entries)


def test_frozen_app_uses_the_relocated_resource_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path / 'Moved App' / 'Contents' / 'Frameworks'), raising=False)
    assert resources.resource_root() == Path(sys._MEIPASS)


def test_legacy_wheel_resources_use_the_installation_prefix(tmp_path, monkeypatch):
    package = tmp_path / 'environment/lib/python3.13/site-packages/recognizer/resources.py'
    monkeypatch.setattr(resources, '__file__', str(package))
    monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'environment'))
    monkeypatch.delattr(sys, 'frozen', raising=False)
    assert resources.resource_root() == tmp_path / 'environment/share/knot-studio'


def test_wheel_resources_remain_with_package_after_relocation(tmp_path, monkeypatch):
    package = tmp_path/'arbitrary-target'/'recognizer'
    assets = package/'_resources'
    (assets/'examples').mkdir(parents=True)
    (assets/'examples/index.json').write_text('{"examples": []}')
    monkeypatch.setattr(resources, '__file__', str(package/'resources.py'))
    monkeypatch.setattr(sys, 'prefix', str(tmp_path/'unrelated-python'))
    monkeypatch.delattr(sys, 'frozen', raising=False)
    assert resources.resource_root() == assets


@pytest.mark.parametrize('name', ['', '../private.png', '/private.png', 'missing.png'])
def test_sample_resolution_rejects_empty_missing_and_outside_paths(tmp_path, monkeypatch, name):
    directory = tmp_path / 'examples'
    directory.mkdir()
    (tmp_path / 'private.png').write_bytes(b'private')
    monkeypatch.setattr(resources, 'resource_root', lambda: tmp_path)
    with pytest.raises(ValueError, match='unavailable'):
        resources.example_path({'path': name})


def test_sample_resolution_cannot_escape_through_a_symlink(tmp_path, monkeypatch):
    directory = tmp_path / 'examples'
    directory.mkdir()
    private = tmp_path / 'private.png'
    private.write_bytes(b'private')
    (directory / 'shortcut.png').symlink_to(private)
    monkeypatch.setattr(resources, 'resource_root', lambda: tmp_path)
    with pytest.raises(ValueError, match='unavailable'):
        resources.example_path({'path': 'shortcut.png'})


def test_generated_help_contains_examples_and_all_linked_resources(tmp_path):
    pytest.importorskip('markdown')  # Build-only dependency, also used by wheel creation.
    from tools.build_docs import build_bundle, check_links
    root = Path(__file__).resolve().parents[1]
    output = build_bundle(root, tmp_path/'resources')
    manifest = json.loads((root / 'examples/index.json').read_text())
    for entry in manifest['examples']:
        assert (output/'examples'/entry['path']).read_bytes() == (root/'examples'/entry['path']).read_bytes()
    assert (output/'docs/USER_GUIDE.html').is_file()
    assert (output/'LICENSE').read_bytes() == (root/'LICENSE').read_bytes()
    assert 'CONTRIBUTING.html' in (output/'docs/DEVELOPING.html').read_text()
    assert '../source/recognizer/relaxation.py' in (output/'docs/ENERGY_TECHNICAL.html').read_text()
    (output/'docs/twist-signs.svg').unlink()
    with pytest.raises(ValueError, match='twist-signs.svg'):
        check_links(output)


def test_rebuilding_help_discards_removed_resources_and_protects_source(tmp_path):
    pytest.importorskip('markdown')
    from tools.build_docs import ASSET_DIRECTORIES, ROOT_DOCUMENTS, build_bundle
    source = tmp_path/'source'
    source.mkdir()
    for name in ROOT_DOCUMENTS:
        (source/name).write_text('Example')
    for name in (*ASSET_DIRECTORIES, 'recognizer'):
        (source/name).mkdir()
    for name in ('USER_GUIDE.md', 'retired.md'):
        (source/'docs'/name).write_text('# Test')
    (source/'examples/gallery.html').write_text('<h1>Examples</h1>')
    (source/'examples/old.png').write_bytes(b'old asset')
    (source/'recognizer/retired.py').write_text('# Old module')
    output = build_bundle(source, tmp_path/'generated')
    assert (output/'docs/retired.html').is_file()
    for name in ('docs/retired.md', 'examples/old.png', 'recognizer/retired.py'):
        (source/name).unlink()
    build_bundle(source, output)
    for name in ('docs/retired.md', 'docs/retired.html', 'examples/old.png', 'source/recognizer/retired.py'):
        assert not (output/name).exists()
    for forbidden in (source, source.parent, source/'docs/generated', source/'recognizer/_resources'):
        with pytest.raises(ValueError, match='source assets'):
            build_bundle(source, forbidden)
    for forbidden in (source/'tools', source/'tests', source/'packaging', source/'.git', tmp_path/'unrelated'):
        forbidden.mkdir()
        sentinel = forbidden/'keep-me'
        sentinel.write_text('must not be deleted')
        with pytest.raises(ValueError, match='not a generated resource bundle'):
            build_bundle(source, forbidden)
        assert sentinel.read_text() == 'must not be deleted'
    assert (source/'docs/USER_GUIDE.md').read_text() == '# Test'


@pytest.mark.parametrize('failure', ['signature', 'timeout', 'invalid-report'])
def test_bundle_verifier_writes_failure_report_for_failed_external_checks(tmp_path, monkeypatch, failure):
    import plistlib
    import subprocess
    from tools import verify_macos
    app = tmp_path/'Test.app'
    (app/'Contents/MacOS').mkdir(parents=True)
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps({'LSMinimumSystemVersion': '15.7.5'}))
    (app/'Contents/MacOS/KnotStudio').write_bytes(b'\xcf\xfa\xed\xfe')
    def run(command, **kwargs):
        if command[0] == 'otool':
            return subprocess.CompletedProcess(command, 0, stdout='binary:\n', stderr='')
        if command[0] == 'codesign':
            if failure == 'signature':
                raise subprocess.CalledProcessError(1, command, stderr='invalid signature')
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, 180)
        Path(command[2]).write_text('not JSON' if failure == 'invalid-report' else '{"ok": true}')
        return subprocess.CompletedProcess(command, 0, stdout=b'', stderr=b'')
    monkeypatch.setattr(verify_macos.subprocess, 'run', run)
    report = tmp_path/'verification.json'
    with pytest.raises(RuntimeError, match='Bundle verification failed'):
        verify_macos.verify(app, report)
    result = json.loads(report.read_text())
    assert not result['ok']
    assert result['native_libraries_checked'] == 1
    assert result['external_dependencies']
