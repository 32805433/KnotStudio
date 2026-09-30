"""Release publication gates use local artifacts and mocked GitHub responses."""
import hashlib
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from tools import assemble_release, prepare_release


VERSION = '0.2.0'
COMMIT = 'a' * 40


def write_json(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


def update_json(path, **changes):
    data = json.loads(path.read_text(encoding='utf-8'))
    data.update(changes)
    write_json(path, data)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checksums(folder):
    assets = sorted(path for path in folder.iterdir()
                    if path.name not in ('SHA256SUMS.txt', 'build-info.json', 'verification.json'))
    (folder/'SHA256SUMS.txt').write_text(''.join(f'{checksum(path)}  {path.name}\n' for path in assets),
                                      encoding='utf-8')


def write_source(path, files, index=0):
    with zipfile.ZipFile(path, 'w') as archive:
        for name, data in reversed(list(files.items())) if index else files.items():
            member = zipfile.ZipInfo(f'KnotStudio-{VERSION}/{name}', date_time=(2026, 1, index + 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED if index else zipfile.ZIP_STORED
            archive.writestr(member, data)


@pytest.fixture
def bundle(tmp_path):
    root, artifacts, output = (tmp_path/name for name in ('checkout', 'native artifacts Ω', 'release'))
    root.mkdir()
    files = {'pyproject.toml': f'[project]\nversion = "{VERSION}"\n'.encode(),
             'README.md': 'Portable source Ω\n'.encode(),
             'recognizer/example.py': b'VALUE = 42\n',
             'docs/RELEASE_NOTES.md': b'Tested desktop release.\n'}
    for name, data in files.items():
        path = root/name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
    # Native jobs have different local output/caches, which are not source.
    (root/'build').mkdir()
    (root/'build'/'local-cache').write_bytes(b'excluded')
    folders = {}
    for index, target in enumerate(assemble_release.TARGETS):
        folder = artifacts/f'KnotStudio-{target}'
        folder.mkdir(parents=True)
        folders[target] = folder
        platform, architecture = target.split('-', 1)
        info = dict(version=VERSION, source_commit=COMMIT, source_dirty=False,
                    platform=platform, architecture=architecture)
        write_json(folder/'build-info.json', info)
        verification = dict(info, ok=True, gui_requested=True)
        if platform == 'Windows':
            verification['installer_self_test'] = {'ok': True, 'checks': [{'name': 'tk_gui', 'success': True}]}
        write_json(folder/'verification.json', verification)
        write_source(folder/f'KnotStudio-{VERSION}-source.zip', files, index)
        suffix = '.tar.gz' if platform == 'Linux' else '.zip'
        (folder/f'KnotStudio-{VERSION}-{target}{suffix}').write_bytes(f'tested package: {target}'.encode())
        if platform == 'Windows':
            (folder/f'KnotStudio-{VERSION}-{target}-Setup.exe').write_bytes(b'tested installer')
        write_checksums(folder)
    return SimpleNamespace(root=root, artifacts=artifacts, output=output, folders=folders, files=files)


def assemble(bundle):
    return assemble_release.assemble(bundle.root, bundle.artifacts, bundle.output, COMMIT)


def test_consolidates_matching_source_and_all_verified_native_assets(bundle):
    assert assemble(bundle) == VERSION
    expected = {f'KnotStudio-{VERSION}-source.zip', f'KnotStudio-{VERSION}-macOS-arm64.zip',
                f'KnotStudio-{VERSION}-Windows-x86_64.zip', f'KnotStudio-{VERSION}-Windows-x86_64-Setup.exe',
                f'KnotStudio-{VERSION}-Linux-x86_64.tar.gz', 'verification.json', 'SHA256SUMS.txt'}
    assert {path.name for path in bundle.output.iterdir()} == expected
    with zipfile.ZipFile(bundle.output/f'KnotStudio-{VERSION}-source.zip') as archive:
        assert {name.removeprefix(f'KnotStudio-{VERSION}/'): archive.read(name)
                for name in archive.namelist()} == bundle.files
    reports = json.loads((bundle.output/'verification.json').read_text(encoding='utf-8'))
    assert set(reports) == set(assemble_release.TARGETS)
    assert all(report['build']['source_commit'] == COMMIT and report['verification']['ok']
               for report in reports.values())
    sums = dict(line.split(maxsplit=1) for line in (bundle.output/'SHA256SUMS.txt').read_text().splitlines())
    assert set(sums.values()) == expected - {'SHA256SUMS.txt'}
    assert all(checksum(bundle.output/name) == digest for digest, name in sums.items())


@pytest.mark.parametrize('target', assemble_release.TARGETS)
def test_rejects_missing_native_platform_without_copying_partial_release(bundle, target):
    shutil.rmtree(bundle.folders[target])
    with pytest.raises(FileNotFoundError):
        assemble(bundle)
    assert not list(bundle.output.iterdir())


def test_refuses_to_overwrite_an_existing_release_directory(bundle):
    bundle.output.mkdir()
    sentinel = bundle.output/'existing.zip'
    sentinel.write_bytes(b'existing published bytes')
    with pytest.raises(FileExistsError):
        assemble(bundle)
    assert sentinel.read_bytes() == b'existing published bytes'
    assert list(bundle.output.iterdir()) == [sentinel]


def test_rejects_native_package_changed_after_checksum_generation(bundle):
    package = bundle.folders['Linux-x86_64']/f'KnotStudio-{VERSION}-Linux-x86_64.tar.gz'
    package.write_bytes(b'changed after testing')
    with pytest.raises(ValueError, match='checksum mismatch'):
        assemble(bundle)


@pytest.mark.parametrize('field,value', [('source_commit', 'b' * 40), ('version', '9.9.9'),
                                        ('platform', 'Windows'), ('architecture', 'x86_64')])
def test_rejects_build_identity_that_does_not_match_release_target(bundle, field, value):
    update_json(bundle.folders['macOS-arm64']/'build-info.json', **{field: value})
    with pytest.raises(ValueError):
        assemble(bundle)


@pytest.mark.parametrize('field,value', [('source_commit', 'b' * 40), ('version', '9.9.9'),
                                        ('platform', 'Windows'), ('architecture', 'x86_64')])
def test_rejects_verification_from_a_different_build(bundle, field, value):
    update_json(bundle.folders['macOS-arm64']/'verification.json', **{field: value})
    with pytest.raises(ValueError):
        assemble(bundle)


@pytest.mark.parametrize('report_name', ['build-info.json', 'verification.json'])
@pytest.mark.parametrize('flag', [{}, {'source_dirty': None}, {'source_dirty': True},
                                {'source_dirty': 0}, {'source_dirty': ''},
                                {'source_dirty': 'false'}, {'source_dirty': []}],
                         ids=['missing', 'null', 'dirty', 'zero', 'empty-string', 'false-string', 'empty-list'])
def test_release_requires_explicit_clean_source_in_both_build_and_verification(bundle, report_name, flag):
    report = bundle.folders['Linux-x86_64']/report_name
    data = json.loads(report.read_text(encoding='utf-8'))
    del data['source_dirty']
    data.update(flag)
    write_json(report, data)
    with pytest.raises(ValueError, match='source_dirty=false'):
        assemble(bundle)
    assert not list(bundle.output.iterdir())


@pytest.mark.parametrize('field', ['ok', 'gui_requested'])
def test_rejects_failed_or_headless_only_verification(bundle, field):
    update_json(bundle.folders['Linux-x86_64']/'verification.json', **{field: False})
    with pytest.raises(ValueError, match='GUI checks'):
        assemble(bundle)


@pytest.mark.parametrize('installer', [None, {}, {'ok': False}])
def test_windows_requires_a_successful_installed_application_check(bundle, installer):
    path = bundle.folders['Windows-x86_64']/'verification.json'
    data = json.loads(path.read_text())
    if installer is None:
        del data['installer_self_test']
    else:
        data['installer_self_test'] = installer
    write_json(path, data)
    with pytest.raises(ValueError, match='[Ii]nstaller'):
        assemble(bundle)


@pytest.mark.parametrize('all_platforms', [False, True])
def test_source_must_match_other_platforms_and_the_release_checkout(bundle, all_platforms):
    changed = dict(bundle.files, **{'README.md': b'different source revision\n'})
    folders = bundle.folders.values() if all_platforms else [bundle.folders['Linux-x86_64']]
    for folder in folders:
        write_source(folder/f'KnotStudio-{VERSION}-source.zip', changed)
        write_checksums(folder)
    with pytest.raises(ValueError, match='[Ss]ource'):
        assemble(bundle)


@pytest.mark.parametrize('kind', ['extra', 'missing', 'duplicate', 'traversal'])
def test_rejects_unexpected_or_ambiguous_checksum_manifests(bundle, kind):
    folder = bundle.folders['Windows-x86_64']
    sums = folder/'SHA256SUMS.txt'
    if kind == 'extra':
        (folder/'extra.zip').write_bytes(b'unexpected payload')
        write_checksums(folder)
    elif kind == 'missing':
        lines = sums.read_text().splitlines()
        sums.write_text('\n'.join(line for line in lines if not line.endswith('-Setup.exe'))+'\n')
    elif kind == 'duplicate':
        sums.write_text(sums.read_text()+sums.read_text().splitlines()[0]+'\n')
    else:
        sums.write_text('0'*64+'  ../outside.zip\n')
    with pytest.raises(ValueError):
        assemble(bundle)


@pytest.fixture(autouse=True)
def no_external_release_commands(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail(f'Unexpected external release command: {args}')
    monkeypatch.setattr(prepare_release, 'subprocess', SimpleNamespace(check_output=unexpected, run=unexpected))


def mock_api(monkeypatch, releases=None, refs=None, failure=None):
    calls = []
    def output(command, **kwargs):
        assert command == ['git', 'rev-parse', 'HEAD']
        return COMMIT + '\n'
    def api(*args):
        calls.append(args)
        kind = 'tag' if '--method' in args else 'releases' if '--paginate' in args else 'refs'
        if failure and failure[0] == kind:
            if isinstance(failure[1], Exception):
                raise failure[1]
            return failure[1]
        if kind == 'tag':
            assert args == ('api', '--method', 'POST', 'repos/{owner}/{repo}/git/refs',
                            '-f', 'ref=refs/tags/v'+VERSION, '-f', 'sha='+COMMIT)
            return json.dumps({'ref': 'refs/tags/v'+VERSION, 'object': {'sha': COMMIT}})
        return json.dumps(([[]] if releases is None else releases) if kind == 'releases'
                          else ([] if refs is None else refs))
    monkeypatch.setattr(prepare_release.subprocess, 'check_output', output)
    monkeypatch.setattr(prepare_release, 'gh', api)
    return calls


def test_preflight_accepts_new_tag_and_ignores_longer_tag_prefixes(bundle, monkeypatch):
    calls = mock_api(monkeypatch, releases=[[{'tag_name': 'v0.1.0'}]],
                     refs=[{'ref': 'refs/tags/v0.2.0-preview'}])
    assert prepare_release.preflight(bundle.root, 'v'+VERSION) == COMMIT
    assert len(calls) == 2 and '--paginate' in calls[0]


@pytest.mark.parametrize('tag', ['v0.1.0', '0.2.0', 'v0.2.0-preview', 'v0.2.0;command'])
def test_preflight_rejects_wrong_or_nonrelease_versions_before_api_access(bundle, tag):
    with pytest.raises(ValueError, match='version'):
        prepare_release.preflight(bundle.root, tag)


@pytest.mark.parametrize('existing', ['release', 'draft', 'tag'])
def test_existing_tags_and_releases_are_never_replaced(bundle, monkeypatch, existing):
    releases = [[{'tag_name': 'v0.1.0'}], [{'tag_name': 'v'+VERSION, 'draft': existing == 'draft'}]]
    mock_api(monkeypatch, releases=releases if existing != 'tag' else [[]],
             refs=[{'ref': 'refs/tags/v'+VERSION}] if existing == 'tag' else [])
    monkeypatch.setattr(prepare_release, '__file__', str(bundle.root/'tools/prepare_release.py'))
    monkeypatch.setattr(sys, 'argv', ['prepare_release.py', '--tag', 'v'+VERSION,
                                   '--assets', str(bundle.artifacts)])
    with pytest.raises(ValueError, match='already exists'):
        prepare_release.main()


@pytest.mark.parametrize('stage', ['releases', 'refs'])
@pytest.mark.parametrize('response', ['failed-command', 'bad-json', 'object', 'null', 'bad-item'])
def test_api_failures_and_malformed_responses_fail_closed(bundle, monkeypatch, stage, response):
    failures = {'failed-command': subprocess.CalledProcessError(1, ['gh', 'api'], stderr='unavailable'),
                'bad-json': '<html>gateway error</html>', 'object': '{}', 'null': 'null',
                'bad-item': '[{}]' if stage == 'refs' else '[[{}]]'}
    calls = mock_api(monkeypatch, failure=(stage, failures[response]))
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        prepare_release.preflight(bundle.root, 'v'+VERSION)
    assert len(calls) == (1 if stage == 'releases' else 2)


def test_draft_creation_targets_verified_commit_and_keeps_assets_as_separate_arguments(bundle, monkeypatch):
    assemble(bundle)
    calls = mock_api(monkeypatch)
    commands = []
    monkeypatch.setattr(prepare_release.subprocess, 'run', lambda command, **kwargs: commands.append((command, kwargs)))
    monkeypatch.setattr(prepare_release, '__file__', str(bundle.root/'tools/prepare_release.py'))
    monkeypatch.setattr(sys, 'argv', ['prepare_release.py', '--tag', 'v'+VERSION,
                                   '--assets', str(bundle.output)])
    prepare_release.main()
    assert len(commands) == 1
    command, options = commands[0]
    assert command[:4] == ['gh', 'release', 'create', 'v'+VERSION]
    assert '--draft' in command and command[command.index('--target')+1] == COMMIT
    assert '--verify-tag' in command and len(calls) == 3
    assert calls[-1][1:3] == ('--method', 'POST')
    assert command[command.index('--notes-file')+1] == str(bundle.root/'docs/RELEASE_NOTES.md')
    assert set(command[command.index('--notes-file')+2:]) == {str(path) for path in bundle.output.iterdir()}
    assert '--clobber' not in command and options == {'check': True}


def test_draft_creation_rejects_unvalidated_asset_directory(bundle, monkeypatch):
    calls = mock_api(monkeypatch)
    monkeypatch.setattr(prepare_release, '__file__', str(bundle.root/'tools/prepare_release.py'))
    monkeypatch.setattr(sys, 'argv', ['prepare_release.py', '--tag', 'v'+VERSION,
                                   '--assets', str(bundle.artifacts)])
    with pytest.raises(ValueError, match='Validated release assets'):
        prepare_release.main()
    assert len(calls) == 2  # No tag is created for unvalidated assets.


@pytest.mark.parametrize('response', ['race', 'network', '{}', 'bad-json', 'wrong-sha'])
def test_atomic_tag_creation_failure_prevents_release_creation(bundle, monkeypatch, response):
    assemble(bundle)
    failure = {'race': subprocess.CalledProcessError(1, ['gh', 'api'], stderr='Reference already exists'),
               'network': subprocess.CalledProcessError(1, ['gh', 'api'], stderr='Network failure'),
               '{}': '{}', 'bad-json': '<html>gateway error</html>',
               'wrong-sha': json.dumps({'ref': 'refs/tags/v'+VERSION, 'object': {'sha': 'b'*40}})}[response]
    calls = mock_api(monkeypatch, failure=('tag', failure))
    monkeypatch.setattr(prepare_release, '__file__', str(bundle.root/'tools/prepare_release.py'))
    monkeypatch.setattr(sys, 'argv', ['prepare_release.py', '--tag', 'v'+VERSION,
                                   '--assets', str(bundle.output)])
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        prepare_release.main()
    assert len(calls) == 3 and calls[-1][1:3] == ('--method', 'POST')


def test_failed_draft_upload_leaves_created_tag_for_explicit_recovery(bundle, monkeypatch):
    assemble(bundle)
    calls = mock_api(monkeypatch)
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr='Upload failed')
    monkeypatch.setattr(prepare_release.subprocess, 'run', fail)
    monkeypatch.setattr(prepare_release, '__file__', str(bundle.root/'tools/prepare_release.py'))
    monkeypatch.setattr(sys, 'argv', ['prepare_release.py', '--tag', 'v'+VERSION,
                                   '--assets', str(bundle.output)])
    with pytest.raises(subprocess.CalledProcessError):
        prepare_release.main()
    assert len(calls) == 3  # There is no automatic DELETE or replacement of the tag.
