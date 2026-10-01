"""Native build dependency and relocation checks without native build tools."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import uuid
import zipfile

import pytest

from tools import build_desktop, build_opencv, native_dependencies, portable_licenses, verify_desktop


@pytest.mark.parametrize('system,machine,expected', [
    ('darwin', 'arm64', ('macOS', 'arm64')), ('darwin', 'aarch64', ('macOS', 'arm64')),
    ('darwin', 'x86_64', ('macOS', 'x86_64')), ('win32', 'AMD64', ('Windows', 'x86_64')),
    ('linux', 'x86_64', ('Linux', 'x86_64')),
])
def test_target_platform_normalizes_native_architecture_names(system, machine, expected):
    assert build_desktop.target_platform(system, machine) == expected


@pytest.mark.parametrize('system,machine', [('win32', 'arm64'), ('win32', 'x86'),
                                           ('linux', 'aarch64'), ('freebsd', 'x86_64')])
def test_builder_rejects_unsupported_desktop_targets(system, machine):
    with pytest.raises(ValueError, match='Unsupported native desktop target'):
        build_desktop.target_platform(system, machine)


def test_builder_uses_the_project_version_and_archives_only_matching_source(tmp_path, monkeypatch):
    root = tmp_path/'checkout Ω'
    root.mkdir()
    (root/'pyproject.toml').write_text('[project]\nversion="0.8.3"\n', encoding='utf-8')
    (root/'README.md').write_text('Matching source Ω\n', encoding='utf-8')
    for folder in ('build', 'dist', '.git', '.venv-build'):
        (root/folder).mkdir()
        (root/folder/'excluded-output').write_bytes(b'not application source')
    monkeypatch.setattr(build_desktop, 'ROOT', root)
    version = build_desktop.project_version(root)
    assert version == '0.8.3'
    archive = build_desktop.source_archive(version)
    assert archive == root/'dist'/'KnotStudio-0.8.3-source.zip'
    with zipfile.ZipFile(archive) as source:
        assert set(source.namelist()) == {'KnotStudio-0.8.3/pyproject.toml', 'KnotStudio-0.8.3/README.md'}
        assert source.read('KnotStudio-0.8.3/README.md') == (root/'README.md').read_bytes()


@pytest.mark.parametrize('tracked,untracked,expected', [
    (b'', b'', False),
    (b'recognizer/ui.py\0', b'', True),
    (b'', 'docs/new guide Ω.md\0'.encode(), True),
    (b'docs/deleted.md\0', b'build/output\0', True),
    (b'build/output\0', b'dist/archive.zip\0.venv-test/cache\0package.egg-info/metadata\0.DS_Store\0', False),
    (b'docs/name\nwith-newline.md\0', b'', True),
    (b'new-\xff-file.py\0', b'', True),
])
def test_source_dirty_distinguishes_application_edits_from_build_outputs(monkeypatch, tracked, untracked, expected):
    calls = []
    def output(command, **kwargs):
        calls.append(command)
        assert kwargs['cwd'] == build_desktop.ROOT
        return tracked if command[1] == 'diff' else untracked
    monkeypatch.setattr(build_desktop, 'subprocess', SimpleNamespace(check_output=output))
    assert build_desktop.source_dirty() is expected
    assert calls == [['git', 'diff', '--name-only', '-z', 'HEAD'],
                     ['git', 'ls-files', '--others', '--exclude-standard', '-z']]


@pytest.mark.parametrize('failure', [OSError('git unavailable'), subprocess.CalledProcessError(128, ['git'])])
def test_source_without_git_evidence_is_not_reported_clean(monkeypatch, failure):
    def output(*args, **kwargs):
        raise failure
    monkeypatch.setattr(build_desktop, 'subprocess', SimpleNamespace(
        check_output=output, CalledProcessError=subprocess.CalledProcessError))
    assert build_desktop.source_dirty() is True


@pytest.mark.parametrize('system,architecture', [('darwin', 'arm64'), ('win32', 'AMD64'), ('linux', 'x86_64')])
def test_opencv_build_flags_keep_only_needed_modules_and_use_native_toolchain(system, architecture):
    flags = build_opencv.cmake_flags(system, architecture)
    options = dict(flag.removeprefix('-D').split('=', 1) for flag in flags)
    assert options['BUILD_LIST'] == 'core,imgproc,python3'
    assert options['BUILD_SHARED_LIBS'] == 'OFF'
    for key in ('WITH_FFMPEG', 'WITH_GSTREAMER', 'WITH_MSMF', 'WITH_DSHOW', 'WITH_GTK', 'WITH_QT',
                'BUILD_opencv_imgcodecs', 'BUILD_opencv_videoio', 'BUILD_opencv_highgui'):
        assert options[key] == 'OFF'
    if system == 'darwin':
        assert options['CMAKE_OSX_ARCHITECTURES'] == 'arm64'
        assert options['CMAKE_OSX_DEPLOYMENT_TARGET'] == '15.0'
    else:
        assert not any(key.startswith('CMAKE_OSX') for key in options)
    if system == 'win32':
        assert options['CMAKE_MSVC_RUNTIME_LIBRARY'] == 'MultiThreadedDLL'
    else:
        assert 'CMAKE_MSVC_RUNTIME_LIBRARY' not in options
    flags.append('modified by caller')
    assert 'modified by caller' not in build_opencv.cmake_flags(system, architecture)


def test_opencv_rejects_unsupported_build_platform():
    with pytest.raises(ValueError, match='Unsupported'):
        build_opencv.cmake_flags('freebsd', 'x86_64')


def test_windows_opencv_manifest_patch_removes_only_disabled_ffmpeg_requirement(tmp_path):
    setup = tmp_path/'setup.py'
    source = r'''is64 = True
required = {
    "cv2": ["cv2.pyd", "data/notice.txt"] + (
            [r"bin/opencv_videoio_ffmpeg\d{3}%s\.dll" % ("_64" if is64 else "")]
            if os.name == "nt"
            else []
    ),
    "other": ["other/data.bin"],
}
def validate_members(members):
    if any(not isinstance(member, str) or not member for member in members):
        raise ValueError("Invalid required wheel member")
    return members
'''
    setup.write_text(source, encoding='utf-8')
    ast.parse(source)
    build_opencv.patch_windows_wheel_manifest(setup)
    patched = setup.read_text(encoding='utf-8')
    namespace = {'os': SimpleNamespace(name='nt')}
    exec(compile(ast.parse(patched), str(setup), 'exec'), namespace)
    assert namespace['required'] == {'cv2': ['cv2.pyd', 'data/notice.txt'], 'other': ['other/data.bin']}
    assert namespace['validate_members'](['cv2.pyd']) == ['cv2.pyd']
    with pytest.raises(ValueError, match='Invalid required wheel member'):
        namespace['validate_members']([''])
    build_opencv.patch_windows_wheel_manifest(setup)
    assert setup.read_text(encoding='utf-8') == patched


@pytest.mark.parametrize('source', [
    'required = {"cv2": ["new-videoio-layout.dll"]}\n',
    ('            [r"bin/opencv_videoio_ffmpeg\\d{3}%s\\.dll" % ("_64" if is64 else "")]\n'
     '            if os.name == "nt"\n'
     '            else []\n') * 2,
])
def test_windows_opencv_manifest_patch_rejects_changed_or_ambiguous_upstream_source(tmp_path, source):
    setup = tmp_path/'setup.py'
    setup.write_text(source, encoding='utf-8')
    with pytest.raises(RuntimeError, match='patch no longer applies'):
        build_opencv.patch_windows_wheel_manifest(setup)
    assert setup.read_text(encoding='utf-8') == source


@pytest.mark.parametrize('name', ['ld-linux-x86-64.so.2', 'libc.so.6', 'libm.so.6', 'libpthread.so.0',
                                'libX11.so.6', 'libfontconfig.so.1', 'libfreetype.so.6'])
def test_documented_ubuntu_desktop_libraries_may_remain_system_provided(name):
    assert native_dependencies.LINUX_SYSTEM_LIBRARIES.fullmatch(name)


@pytest.mark.parametrize('name', ['libstdc++.so.6', 'libgcc_s.so.1', 'libtesseract.so.5', 'liblept.so.5',
                                'libtcl8.6.so', 'libtk8.6.so', 'libpython3.13.so.1.0', 'libcustom.so'])
def test_nonbaseline_linux_dependencies_must_be_bundled(name):
    assert not native_dependencies.LINUX_SYSTEM_LIBRARIES.fullmatch(name)


@pytest.mark.parametrize('name', ['VCRUNTIME140.dll', 'MSVCP140.dll', 'concrt140.dll', 'vcomp140.dll', 'python313.dll'])
def test_windows_language_runtimes_are_bundled_even_if_installed_system_wide(tmp_path, monkeypatch, name):
    system = tmp_path/'Windows'/'System32'
    system.mkdir(parents=True)
    runtime = system/name
    runtime.write_bytes(b'installed runtime')
    monkeypatch.setenv('SystemRoot', str(system.parent))
    assert not native_dependencies.windows_system_library(name, runtime)
    assert not native_dependencies.windows_system_library(name)


def test_windows_os_dlls_require_system_location_except_api_sets(tmp_path, monkeypatch):
    system = tmp_path/'Windows'/'System32'
    system.mkdir(parents=True)
    kernel = system/'kernel32.dll'
    kernel.write_bytes(b'OS library')
    monkeypatch.setenv('SystemRoot', str(system.parent))
    assert native_dependencies.windows_system_library('kernel32.dll', kernel)
    assert native_dependencies.windows_system_library('kernel32.dll')
    assert native_dependencies.windows_system_library('api-ms-win-core-file-l1-1-0.dll')
    assert native_dependencies.windows_system_library('ext-ms-win-ntuser-window-l1-1-0.dll')
    assert not native_dependencies.windows_system_library('kernel32.dll', tmp_path/'kernel32.dll')
    assert not native_dependencies.windows_system_library('not-installed.dll')


def test_linux_dependency_probe_preserves_unresolved_dependencies(monkeypatch, tmp_path):
    def run(command, **kwargs):
        assert command == ['ldd', str(tmp_path/'KnotStudio')]
        assert kwargs['env'] == {'LD_LIBRARY_PATH': 'bundle'}
        return SimpleNamespace(returncode=0, stderr='', stdout='''
linux-vdso.so.1 (0x123)
libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x234)
libtesseract.so.5 => not found
/lib64/ld-linux-x86-64.so.2 (0x345)
''')
    monkeypatch.setattr(native_dependencies, 'subprocess', SimpleNamespace(run=run))
    paths, missing = native_dependencies.linux_links(tmp_path/'KnotStudio', env={'LD_LIBRARY_PATH': 'bundle'})
    assert paths == [Path('/lib/x86_64-linux-gnu/libc.so.6'), Path('/lib64/ld-linux-x86-64.so.2')]
    assert missing == ['libtesseract.so.5 => not found']


def test_failed_linux_dependency_probe_is_not_treated_as_an_empty_dependency_set(monkeypatch, tmp_path):
    monkeypatch.setattr(native_dependencies, 'subprocess', SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout='', stderr='inspection failed')))
    paths, errors = native_dependencies.linux_links(tmp_path/'KnotStudio')
    assert paths == [] and len(errors) == 1 and 'inspection failed' in errors[0]


def test_linux_dependency_probe_keeps_spaces_and_unicode_in_relocated_paths(monkeypatch, tmp_path):
    library = Path('/tmp/Knot Studio relocation Ω/Moved Knot Studio/_internal/libtesseract.so.5')
    output = f'libtesseract.so.5 => {library.as_posix()} (0x123abc)\n'
    monkeypatch.setattr(native_dependencies, 'subprocess', SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output, stderr='')))
    paths, errors = native_dependencies.linux_links(tmp_path/'KnotStudio')
    assert paths == [library] and not errors


@pytest.mark.parametrize('owners', ['libc6:amd64', 'libc6:amd64, libc6:i386'])
def test_debian_package_ignores_diversion_explanations_before_actual_owners(monkeypatch, owners):
    output = ('diversion by libc6 from: /lib64/ld-linux-x86-64.so.2\n'
              'diversion by libc6 to: /lib64/ld-linux-x86-64.so.2.usr-is-merged\n'
              f'{owners}: /lib64/ld-linux-x86-64.so.2\n')
    monkeypatch.setattr(portable_licenses, 'subprocess', SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output)))
    assert portable_licenses.debian_package(Path('/lib64/ld-linux-x86-64.so.2')) == 'libc6:amd64'


def test_debian_package_checks_resolved_candidate_after_diversion_only_output(monkeypatch):
    candidate = Path('relative-library/libtesseract.so.5')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        output = ('diversion by package from: original library\n' if len(calls) == 1
                  else f'libtesseract5:amd64: {candidate.resolve()}\n')
        return SimpleNamespace(returncode=0, stdout=output)
    monkeypatch.setattr(portable_licenses, 'subprocess', SimpleNamespace(run=run))
    assert portable_licenses.debian_package(candidate) == 'libtesseract5:amd64'
    assert calls == [['dpkg-query', '-S', str(candidate)], ['dpkg-query', '-S', str(candidate.resolve())]]


def test_linux_rpath_relocation_preserves_wheel_paths_and_removes_build_machine_paths(tmp_path, monkeypatch):
    app = tmp_path/'Knot Studio Ω'
    internal = app/'_internal'
    wheels = internal/'package'/'lib'
    wheels.mkdir(parents=True)
    executable, runtime, extension = app/'KnotStudio', internal/'libpython3.13.so', wheels/'extension.so'
    for path in (executable, runtime, extension):
        path.write_bytes(b'\x7fELFfixture')
    (internal/'notice.txt').write_text('Non-binary resources are not patched.', encoding='utf-8')
    initial = {executable: '$ORIGIN/_internal:/opt/hostedtoolcache/Python/lib',
               runtime: '/opt/hostedtoolcache/Python/lib:$ORIGIN',
               extension: '${ORIGIN}/../../package.libs:$ORIGIN/../libs:/opt/build/lib'}
    inspected, patched = [], {}
    def inspect(command, **kwargs):
        assert command[:2] == ['patchelf', '--print-rpath']
        path = Path(command[2])
        inspected.append(path)
        return initial[path]+'\n'
    def patch(command, **kwargs):
        assert command[:2] == ['patchelf', '--set-rpath'] and kwargs == {'check': True}
        patched[Path(command[3])] = command[2].split(':')
    monkeypatch.setattr(build_desktop, 'shutil', SimpleNamespace(which=lambda name: 'patchelf'))
    monkeypatch.setattr(build_desktop, 'subprocess', SimpleNamespace(check_output=inspect, run=patch))
    build_desktop.relocate_linux_libraries(app)
    assert set(inspected) == set(initial)
    assert patched[executable] == ['$ORIGIN/_internal', '$ORIGIN']
    assert patched[runtime] == ['$ORIGIN', '$ORIGIN/.']
    assert patched[extension] == ['${ORIGIN}/../../package.libs', '$ORIGIN/../libs', '$ORIGIN', '$ORIGIN/../..']
    assert all(not entry.startswith('/') for entries in patched.values() for entry in entries)


@pytest.mark.parametrize('platform', ['darwin', 'win32', 'linux'])
def test_relocated_selftest_cannot_use_build_machine_python_ocr_or_library_paths(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(verify_desktop, 'sys', SimpleNamespace(platform=platform))
    for key in ('PYTHONPATH', 'PYTHONHOME', 'DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH',
                '_PYI_APPLICATION_HOME_DIR', 'TESSDATA_PREFIX', 'TESSERACT_CMD', 'KNOTSTUDIO_TESSERACT'):
        monkeypatch.setenv(key, 'build machine dependency')
    monkeypatch.setenv('PATH', 'build machine tools')
    monkeypatch.setenv('SystemRoot', str(tmp_path/'Windows'))
    monkeypatch.setenv('DISPLAY', ':99')
    env = verify_desktop.isolated_environment(tmp_path/'empty-path')
    assert env['PYTHONNOUSERSITE'] == '1' and env['DISPLAY'] == ':99'
    assert not any(value == 'build machine dependency' for value in env.values())
    expected = tmp_path/'Windows'/'System32' if platform == 'win32' else tmp_path/'empty-path'
    assert env['PATH'] == str(expected)


@pytest.mark.parametrize('key', ['SYSTEMROOT', 'systemroot', 'SyStEmRoOt'])
def test_windows_isolation_honors_case_insensitive_systemroot_in_copied_environment(tmp_path, monkeypatch, key):
    system_root = tmp_path/'Custom Windows Ω'
    original = {key: str(system_root), 'PATH': 'build-machine tools'}
    monkeypatch.setattr(verify_desktop, 'os', SimpleNamespace(environ=original))
    monkeypatch.setattr(verify_desktop, 'sys', SimpleNamespace(platform='win32'))
    isolated = verify_desktop.isolated_environment(tmp_path/'empty-path')
    assert isolated['PATH'] == str(system_root/'System32')
    assert isolated[key] == str(system_root)
    assert original['PATH'] == 'build-machine tools'


@pytest.mark.parametrize('exit_code,reported_ok,expected', [(0, True, True), (23, True, False), (0, False, False), (0, None, False)])
def test_relocated_process_requires_both_successful_exit_and_successful_report(tmp_path, monkeypatch, exit_code, reported_ok, expected):
    def run(command, **kwargs):
        assert command[1] == '--self-test' and command[-1] == '--gui'
        assert kwargs['cwd'] == tmp_path
        if reported_ok is not None:
            Path(command[2]).write_text(json.dumps({'ok': reported_ok}), encoding='utf-8')
        return SimpleNamespace(returncode=exit_code, stderr=b'process diagnostic')
    monkeypatch.setattr(verify_desktop, 'subprocess', SimpleNamespace(run=run))
    result = verify_desktop.self_test(tmp_path/'Moved app Ω'/'KnotStudio', tmp_path, gui=True)
    assert result['ok'] is expected and result['exit_code'] == exit_code
    if reported_ok is None:
        assert result['detail'] == 'process diagnostic'


@pytest.fixture
def installer_probe(tmp_path, monkeypatch):
    installer = tmp_path/'Knot Studio Ω Setup.exe'
    installer.write_bytes(b'inert installer fixture')
    calls, installed_paths = [], []
    state = SimpleNamespace(failure=None)
    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs == {'check': True, 'timeout': 180}
        if command[0] == str(installer.resolve()):
            directory = Path(next(arg.removeprefix('/DIR=') for arg in command if arg.startswith('/DIR=')))
            assert directory.is_relative_to(tmp_path)
            directory.mkdir()
            (directory/'unins000.exe').write_bytes(b'inert uninstaller fixture')
            installed_paths.append(directory)
            if state.failure == 'install':
                raise subprocess.CalledProcessError(1, command)
        else:
            assert command == [str(installed_paths[-1]/'unins000.exe'), '/VERYSILENT',
                               '/SUPPRESSMSGBOXES', '/NORESTART']
    def self_test(executable, temporary, gui=False):
        assert executable == installed_paths[-1]/'KnotStudio.exe'
        assert temporary == installed_paths[-1].parent and gui is True
        if state.failure == 'timeout':
            raise subprocess.TimeoutExpired([str(executable)], 240)
        return {'ok': state.failure != 'report', 'exit_code': 0}
    monkeypatch.setattr(verify_desktop, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr(verify_desktop, 'subprocess', SimpleNamespace(run=run, SubprocessError=subprocess.SubprocessError))
    monkeypatch.setattr(verify_desktop, 'self_test', self_test)
    monkeypatch.setattr(verify_desktop, 'tempfile', SimpleNamespace(
        TemporaryDirectory=lambda **kwargs: tempfile.TemporaryDirectory(dir=tmp_path, **kwargs)))
    return SimpleNamespace(installer=installer, calls=calls, installed_paths=installed_paths, state=state)


def test_installer_verification_uses_unique_registration_and_cleans_up_each_install(tmp_path, installer_probe):
    probe = installer_probe
    identifiers = []
    for index in range(2):
        report = tmp_path/f'verification-{index}.json'
        report.write_text(json.dumps({'ok': True, 'source_commit': 'a'*40}), encoding='utf-8')
        result = verify_desktop.verify_installer(probe.installer, report, gui=True)
        assert result['ok'] is True and result['installer_self_test']['ok'] is True
        assert result['source_commit'] == 'a'*40
        command = probe.calls[index*2][0]
        identifier = next(arg.removeprefix('/KnotStudioVerify=') for arg in command
                          if arg.startswith('/KnotStudioVerify='))
        assert uuid.UUID(hex=identifier).hex == identifier
        assert '/NOICONS' in command and f'/GROUP=Knot Studio Verification {identifier}' in command
        assert '/VERYSILENT' in command and '/NORESTART' in command
        identifiers.append(identifier)
        assert not probe.installed_paths[-1].exists()
        assert json.loads(report.read_text(encoding='utf-8')) == result
    assert len(probe.calls) == 4
    assert identifiers[0] != identifiers[1]
    assert probe.installed_paths[0] != probe.installed_paths[1]


@pytest.mark.parametrize('failure', ['report', 'timeout', 'install'])
def test_failed_installer_verification_still_uninstalls_and_records_failure(tmp_path, installer_probe, failure):
    probe = installer_probe
    probe.state.failure = failure
    report = tmp_path/'verification.json'
    report.write_text(json.dumps({'ok': True}), encoding='utf-8')
    with pytest.raises(RuntimeError, match='Installer verification failed'):
        verify_desktop.verify_installer(probe.installer, report, gui=True)
    assert len(probe.calls) == 2
    assert Path(probe.calls[1][0][0]).name == 'unins000.exe'
    assert not probe.installed_paths[-1].exists()
    recorded = json.loads(report.read_text(encoding='utf-8'))
    assert recorded['ok'] is False and recorded['installer_self_test']['ok'] is False


def status_stanza(name, architecture='x64-windows-static-md', **fields):
    record = dict(Package=name, Architecture=architecture, Version='5.5.1', Status='install ok installed', **fields)
    return '\n'.join(f'{key}: {value}' for key, value in record.items())+'\n\n'


def test_vcpkg_notices_use_installed_core_port_metadata_not_feature_records():
    text = status_stanza('tesseract', **{'Port-Version': '2'})
    text += status_stanza('tesseract', Feature='tools')
    text += 'Package: incomplete\nStatus: install ok unpacked\n\n'
    parsed = portable_licenses.parse_vcpkg_status(text.replace('\n', '\r\n'))
    assert set(parsed) == {'tesseract'}
    assert parsed['tesseract']['Version'] == '5.5.1'
    assert parsed['tesseract']['Port-Version'] == '2'
    assert 'Feature' not in parsed['tesseract']


def test_vcpkg_collector_keeps_target_ports_when_host_ports_follow_them(tmp_path, monkeypatch):
    target = tmp_path/'installed'/'x64-windows-static-md'
    status = target.parent/'vcpkg'/'status'
    status.parent.mkdir(parents=True)
    status.write_text(status_stanza('tesseract')+status_stanza('leptonica')+
                      status_stanza('tesseract', architecture='x64-windows'), encoding='utf-8')
    monkeypatch.setenv('VCPKG_COMMIT', 'a'*40)
    records, notices, issues = [], [], []
    def component(name, version, kind, source):
        record = dict(name=name, version=version, kind=kind, source=source)
        records.append(record)
        return record
    portable_licenses.collect_vcpkg(target, component, lambda *args: notices.append(args), issues)
    assert issues == []
    assert {record['name'] for record in records} == {'tesseract', 'leptonica'}
    assert all(record['architecture'] == target.name and record['vcpkg_commit'] == 'a'*40 for record in records)
    assert len(notices) == 2


@pytest.mark.parametrize('unknown_dependency', [False, True])
def test_linux_notices_cover_ocr_and_tk_dependencies_without_hiding_unknown_owners(tmp_path, monkeypatch, unknown_dependency):
    base = tmp_path/'python'
    tesseract = tmp_path/'usr/bin/tesseract'
    tkinter_extension = base/'lib/_tkinter.so'
    bundled_python = base/'lib/libpython3.13.so.1.0'
    libraries = {name: tmp_path/'usr/lib'/name for name in
                 ('libtesseract.so.5', 'libc.so.6', 'libtcl8.6.so', 'libtk8.6.so', 'libX11.so.6',
                  'libpython3.13.so.1.0', 'libunowned.so')}
    owners = {tesseract: 'tesseract-ocr', libraries['libtesseract.so.5']: 'libtesseract5',
              libraries['libc.so.6']: 'libc6', libraries['libtcl8.6.so']: 'libtcl8.6',
              libraries['libtk8.6.so']: 'libtk8.6', libraries['libX11.so.6']: 'libx11-6',
              libraries['libpython3.13.so.1.0']: 'libpython3.13'}
    probes, owner_checks, notices, components, issues = [], [], [], [], []
    def links(binary):
        probes.append(binary)
        if binary == tesseract:
            return [libraries['libtesseract.so.5'], libraries['libc.so.6']], []
        assert binary == tkinter_extension
        result = [libraries[name] for name in ('libtcl8.6.so', 'libtk8.6.so', 'libX11.so.6',
                                               'libc.so.6', 'libpython3.13.so.1.0')]
        return [*result, bundled_python, *([libraries['libunowned.so']] if unknown_dependency else [])], []
    def owner(path):
        owner_checks.append(path)
        if path not in owners:
            raise RuntimeError(f'No Debian package owns {path.name}.')
        return owners[path]
    def metadata(command, **kwargs):
        name = command[-1]
        return f'{name}\t1.2.3\t{name}-source\t1.2.3\n'
    def component(name, version, kind, source):
        entry = dict(name=name, version=version, kind=kind, source=source)
        components.append(entry)
        return entry
    monkeypatch.setattr(portable_licenses, 'sys', SimpleNamespace(platform='linux', base_prefix=str(base)))
    monkeypatch.setitem(sys.modules, '_tkinter', SimpleNamespace(__file__=str(tkinter_extension)))
    monkeypatch.setattr(native_dependencies, 'linux_links', links)
    monkeypatch.setattr(portable_licenses, 'debian_package', owner)
    monkeypatch.setattr(portable_licenses, 'subprocess', SimpleNamespace(
        check_output=metadata, CalledProcessError=subprocess.CalledProcessError))
    portable_licenses.collect_native(tesseract, None, component, lambda *args: notices.append(args), issues)
    assert probes == [tesseract, tkinter_extension]
    assert bundled_python not in owner_checks
    assert libraries['libpython3.13.so.1.0'] in owner_checks
    assert {entry['name'] for entry in components} == set(owners.values())
    assert len(components) == len(owners) == len(notices)
    assert issues == (['No Debian package owns libunowned.so.'] if unknown_dependency else [])


@pytest.mark.parametrize('provenance', ['matching', 'wrong-version', 'wrong-checksum', 'missing', 'malformed'])
def test_windows_tcl_notice_fallback_requires_exact_runtime_version_and_verified_bytes(tmp_path, monkeypatch, provenance):
    base, staged = tmp_path/'Python', tmp_path/'staged notices'
    tcl_library = base/'tcl'/'tcl8.6'
    tk_notice = base/'tcl'/'tk8.6'/'license.terms'
    tcl_notice = staged/'Tcl'/'license.terms'
    tcl_library.mkdir(parents=True)
    for path, text in ((base/'LICENSE.txt', 'Python license'),
                       (staged/'Python'/'license.rst', 'Python bundled notices'),
                       (tk_notice, 'Installed Tk notice'), (tcl_notice, 'Pinned Tcl notice fixture')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    record = {'version': '8.6.15', 'sha256': hashlib.sha256(tcl_notice.read_bytes()).hexdigest()}
    if provenance == 'wrong-version':
        record['version'] = '8.6.14'
    if provenance == 'wrong-checksum':
        record['sha256'] = '0'*64
    if provenance != 'missing':
        tcl_notice.with_name('provenance.json').write_text(
            'not JSON' if provenance == 'malformed' else json.dumps(record), encoding='utf-8')
    fake_tcl = SimpleNamespace(eval=lambda expression: str(tcl_library), call=lambda *args: '8.6.15')
    monkeypatch.setitem(sys.modules, 'tkinter', SimpleNamespace(Tcl=lambda: fake_tcl, TkVersion=8.6))
    monkeypatch.setattr(portable_licenses, 'sys', SimpleNamespace(platform='win32', version_info=sys.version_info))
    components, copied, issues = [], {}, []
    def component(name, version, kind, source):
        entry = dict(name=name, version=version)
        components.append(entry)
        return entry
    def copy_notice(entry, path, destination, source):
        assert path.is_file()
        copied[destination] = path
    portable_licenses.collect_runtime(base, staged, component, copy_notice, issues)
    assert copied['runtime/Tcl/license.terms'] == tcl_notice
    assert copied['runtime/Tk/license.terms'] == tk_notice
    tcl = next(entry for entry in components if entry['name'] == 'Tcl')
    assert tcl['version'] == '8.6.15'
    if provenance == 'matching':
        assert issues == [] and tcl['notice_provenance'] == record
    else:
        assert len(issues) == 1
        expected = {'wrong-version': 'describes Tcl 8.6.14', 'wrong-checksum': 'checksum differs',
                    'missing': 'valid provenance.json', 'malformed': 'valid provenance.json'}[provenance]
        assert expected in issues[0]
