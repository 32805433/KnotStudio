#!/usr/bin/env python3
"""Build and install the release's OpenCV core/imgproc-only Python wheel.

Uses the upstream source distribution, pinned by SHA-256, and the active Python.
Builds natively on macOS, Windows x64, and Linux x86_64 without video or codecs.
The resulting wheel stays in ignored build/opencv/wheels and is cached by a
manifest of Python, architecture, source, and build options.
"""
from __future__ import annotations
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = '5.0.0.93'
SOURCE_URL = ('https://files.pythonhosted.org/packages/1d/99/'
              '76b7c80252aa83c1af16393454aafd125a0287101afe8deb0a6821af0e30/'
              'opencv_python_headless-5.0.0.93.tar.gz')
SOURCE_SHA256 = 'b82f9831daab90b725c7c1ee1b36cb5732c367096ac76d119e64e14eb70d5f3c'
CMAKE_ARGS = [
    '-DBUILD_LIST=core,imgproc,python3', '-DBUILD_SHARED_LIBS=OFF',
    '-DBUILD_opencv_imgcodecs=OFF', '-DBUILD_opencv_videoio=OFF',
    '-DBUILD_opencv_highgui=OFF', '-DBUILD_opencv_dnn=OFF',
    '-DBUILD_opencv_apps=OFF', '-DBUILD_TESTS=OFF', '-DBUILD_PERF_TESTS=OFF',
    '-DBUILD_EXAMPLES=OFF', '-DBUILD_DOCS=OFF', '-DBUILD_JAVA=OFF',
    '-DWITH_FFMPEG=OFF', '-DWITH_GSTREAMER=OFF', '-DWITH_AVFOUNDATION=OFF',
    '-DWITH_OPENCL=OFF', '-DWITH_OPENCL_SVM=OFF', '-DWITH_OPENGL=OFF',
    '-DWITH_IPP=OFF', '-DWITH_ITT=OFF', '-DWITH_TBB=OFF',
    '-DWITH_LAPACK=OFF', '-DWITH_EIGEN=OFF', '-DWITH_PROTOBUF=OFF',
    '-DWITH_QUIRC=OFF', '-DWITH_JPEG=OFF', '-DWITH_PNG=OFF',
    '-DWITH_TIFF=OFF', '-DWITH_WEBP=OFF', '-DWITH_OPENEXR=OFF',
    '-DWITH_JASPER=OFF', '-DWITH_OPENJPEG=OFF', '-DWITH_AVIF=OFF',
    '-DBUILD_ZLIB=ON', '-DWITH_VTK=OFF', '-DWITH_QT=OFF', '-DWITH_GTK=OFF',
    '-DWITH_MSMF=OFF', '-DWITH_DSHOW=OFF', '-DWITH_V4L=OFF',
]


def cmake_flags(system=None, architecture=None):
    system = system or sys.platform
    architecture = architecture or platform.machine()
    flags = list(CMAKE_ARGS)
    if system == 'darwin':
        flags += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0',
                  f'-DCMAKE_OSX_ARCHITECTURES={architecture}']
    elif system == 'win32':
        # The official Python interpreter uses the dynamic MSVC runtime.
        flags += ['-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL', '-DBUILD_WITH_STATIC_CRT=OFF']
    elif not system.startswith('linux'):
        raise ValueError(f'Unsupported build platform: {system}')
    return flags


def patch_windows_wheel_manifest(setup: Path):
    """Remove the one required wheel member belonging to disabled videoio."""
    before = (
        '            [r"bin/opencv_videoio_ffmpeg\\d{3}%s\\.dll" % ("_64" if is64 else "")]\n'
        '            if os.name == "nt"\n'
        '            else []'
    )
    after = '            []  # Knot Studio minimal build: no videoio/FFmpeg DLL.'
    content = setup.read_text(encoding='utf-8')
    if content.count(before) == 1:
        setup.write_text(content.replace(before, after, 1), encoding='utf-8')
    elif after not in content:
        raise RuntimeError('OpenCV Windows wheel manifest patch no longer applies')


def run(args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(destination):
    if destination.is_file() and sha256(destination) == SOURCE_SHA256:
        return
    from pip._vendor import certifi
    context = ssl.create_default_context(cafile=certifi.where())
    temporary = destination.with_suffix('.download')
    with urllib.request.urlopen(SOURCE_URL, context=context, timeout=90) as source:
        with temporary.open('wb') as output:
            shutil.copyfileobj(source, output)
    if sha256(temporary) != SOURCE_SHA256:
        temporary.unlink()
        raise RuntimeError('OpenCV source checksum mismatch')
    temporary.replace(destination)


def verify_installed():
    # A separate interpreter prevents reusing a previous cv2 from sys.modules.
    code = '''import cv2, json, pathlib, subprocess, sys
p = pathlib.Path(cv2.__file__).parent
marker = json.loads((p/'knotstudio_build.json').read_text())
assert marker['source_sha256'] == %r
info = cv2.getBuildInformation()
assert 'FFMPEG:                      YES' not in info
assert not hasattr(cv2, 'VideoCapture'), 'Video module must not be bundled'
assert not hasattr(cv2, 'imread'), 'Codec module must not be bundled'
extensions = list(p.glob('*.so')) + list(p.glob('*.pyd'))
assert extensions, 'OpenCV native extension missing'
for extension in extensions if sys.platform == 'darwin' else []:
    links = subprocess.check_output(['/usr/bin/otool', '-L', str(extension)], text=True)
    for line in links.splitlines()[1:]:
        target = line.strip().split(' (')[0]
        assert target.startswith(('/usr/lib/', '/System/Library/')), target
print(json.dumps({'version':cv2.__version__, 'modules':marker['modules'],
                  'native_link_check': 'system libraries only' if sys.platform == 'darwin'
                  else 'checked by final packaged application verification'}))
''' % SOURCE_SHA256
    run([sys.executable, '-c', code], cwd=ROOT)


def build(install=True, jobs=None):
    work = ROOT/'build'/'opencv'
    wheels = work/'wheels'
    wheels.mkdir(parents=True, exist_ok=True)
    flags = cmake_flags()
    manifest = {'version': VERSION, 'source_url': SOURCE_URL,
                'source_sha256': SOURCE_SHA256,
                'python': platform.python_version(), 'architecture':platform.machine(),
                'platform': sys.platform, 'platform_release': platform.release(),
                'requested_modules':['core','imgproc','python3'],
                'modules':['core','flann','geometry','imgproc','python3'], 'cmake_args':flags,
                'build_tools': {name: metadata.version(name) for name in
                                ('numpy','cmake','ninja','scikit-build','setuptools','wheel')},
                'minimum_macos': '15.0' if sys.platform == 'darwin' else None,
                'typing_patch_revision':2, 'notices_revision':2,
                'windows_wheel_patch_revision': 1 if sys.platform == 'win32' else 0}
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache = wheels/key
    cached = list(cache.glob('*.whl')) if cache.is_dir() else []
    if not cached:
        archive = work/f'opencv_python_headless-{VERSION}.tar.gz'
        # Reuse a manually downloaded verified archive, if present.
        first = ROOT/'build'/'opencv-source'/archive.name
        if not archive.exists() and first.exists() and sha256(first) == SOURCE_SHA256:
            shutil.copy2(first, archive)
        download(archive)
        source = work/f'opencv_python_headless-{VERSION}'
        if not source.is_dir():
            with tarfile.open(archive) as package:
                package.extractall(work, filter='data')
        if sys.platform == 'win32':
            # Upstream always requires its FFmpeg DLL in Windows wheels even
            # when videoio is disabled. Keep all other wheel members required.
            patch_windows_wheel_manifest(source / 'setup.py')
        # OpenCV 5's typing generator assumes every optional module is present.
        # Restrict its refinements to symbols compiled in this minimal build.
        refinement = source/'opencv/modules/python/src2/typing_stubs_generation/api_refinement.py'
        text = refinement.read_text(encoding='utf-8')
        before = '    for symbol_name, refine_symbol in NODES_TO_REFINE.items():\n        refine_symbol(root, symbol_name)'
        after = ('    # Knot Studio minimal-build patch: optional modules may be absent.\n'
                 '    from .ast_utils import ScopeNotFoundError, SymbolNotFoundError\n'
                 '    for symbol_name, refine_symbol in NODES_TO_REFINE.items():\n'
                 '        try:\n'
                 '            find_function_node(root, symbol_name)\n'
                 '        except (ScopeNotFoundError, SymbolNotFoundError):\n'
                 '            continue\n'
                 '        refine_symbol(root, symbol_name)')
        if before in text:
            refinement.write_text(text.replace(before, after, 1), encoding='utf-8')
        elif after not in text:
            raise RuntimeError('OpenCV typing patch no longer applies')
        generation = source/'opencv/modules/python/src2/typing_stubs_generation/generation.py'
        text = generation.read_text(encoding='utf-8')
        before = '        node.resolve(root)\n        if isinstance(node, AliasTypeNode):'
        after = ('        # Knot Studio minimal-build patch: no aliases for absent classes.\n'
                 '        from .nodes.type_node import TypeResolutionError\n'
                 '        try:\n'
                 '            node.resolve(root)\n'
                 '        except TypeResolutionError:\n'
                 '            continue\n'
                 '        if isinstance(node, AliasTypeNode):')
        if before in text:
            generation.write_text(text.replace(before, after, 1), encoding='utf-8')
        elif after not in text:
            raise RuntimeError('OpenCV alias patch no longer applies')
        env = dict(os.environ)
        env.update(ENABLE_HEADLESS='1', ENABLE_CONTRIB='0', CMAKE_ARGS=' '.join(flags),
                   CMAKE_BUILD_PARALLEL_LEVEL=str(jobs or min(os.cpu_count() or 2, 8)),
                   # Upstream setup.py sets CMAKE_GENERATOR_PLATFORM=x64 on
                   # Windows, which is incompatible with the Ninja generator.
                   CMAKE_GENERATOR='Visual Studio 17 2022' if sys.platform == 'win32' else 'Ninja',
                   PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''))
        if sys.platform == 'darwin':
            env['MACOSX_DEPLOYMENT_TARGET'] = '15.0'
        raw = work/'raw-wheel'/key
        raw.mkdir(parents=True, exist_ok=True)
        run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
             '--wheel-dir', raw, source], cwd=ROOT, env=env)
        wheel = next(raw.glob('*.whl'))
        unpacked = work/'unpacked'/key
        unpacked.mkdir(parents=True, exist_ok=True)
        run([sys.executable, '-m', 'wheel', 'unpack', wheel, '-d', unpacked])
        package = next(unpacked.glob('opencv_python_headless-*'))
        # Preserve notices from the actual built modules, including legacy BSD
        # source headers that the generic wheel notice may not reproduce.
        notices = next(package.glob('*.dist-info'))/'licenses'
        notices.mkdir(exist_ok=True)
        sections = []
        seen = set()
        for relative in ('modules/core','modules/flann','modules/geometry','modules/imgproc',
                         'hal/carotene'):
            for file in sorted((source/'opencv'/relative).rglob('*')):
                if file.suffix not in {'.cpp','.c','.hpp','.h'}:
                    continue
                content = file.read_text(encoding='utf-8', errors='replace')
                for match in re.finditer(r'/\*.*?\*/', content[:16000], re.S):
                    block = match.group()
                    if ('copyright' in block.lower() and
                        any(term in block.lower() for term in ('license','redistribution','permission'))
                        and block not in seen):
                        seen.add(block)
                        sections.append(file.relative_to(source/'opencv').as_posix()+'\n'+block)
        (notices/'MINIMAL-OPENCV-NOTICES.txt').write_text('\n\n'.join(sections)+'\n', encoding='utf-8')
        for dependency in ('zlib','flatbuffers','dlpack'):
            for license_file in (source/'opencv'/'3rdparty'/dependency).glob('LICENSE*'):
                shutil.copy2(license_file, notices/(dependency+'-'+license_file.name))
        (notices/'KNOTSTUDIO-BUILD-NOTICE.txt').write_text(
            'Minimal OpenCV built from the source and options in cv2/knotstudio_build.json.\n'
            'Runtime modules: core, flann, geometry, imgproc, python3. No FFmpeg or video codecs.\n'
            'Two build-only typing generation guards skip omitted optional symbols and aliases.\n'
            'On Windows, the wheel file manifest omits the disabled videoio/FFmpeg DLL.\n'
            'The exact patches are in tools/build_opencv.py in the Knot Studio source.\n'
            'The upstream generic LICENSE-3RD-PARTY also describes optional components not built here.\n', encoding='utf-8')
        (package/'cv2'/'knotstudio_build.json').write_text(json.dumps(manifest,indent=2)+'\n', encoding='utf-8')
        cache.mkdir(parents=True,exist_ok=True)
        run([sys.executable, '-m', 'wheel', 'pack', package, '-d', cache])
        cached = list(cache.glob('*.whl'))
    if len(cached) != 1:
        raise RuntimeError('Expected exactly one cached OpenCV wheel')
    if install:
        run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', cached[0]])
        verify_installed()
    print(cached[0], flush=True)
    return cached[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-install', action='store_true')
    parser.add_argument('--jobs', type=int, default=None)
    args = parser.parse_args()
    build(install=not args.no_install, jobs=args.jobs)
