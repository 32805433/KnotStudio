# Shared native build. Configuration is supplied by tools/build_desktop.py.
import os
from pathlib import Path
import sys
import tomllib
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH).parent
sys.path.insert(0, str(root))
from tools.native_dependencies import ocr_binaries
assets = root / '.build-assets'
with (root / 'pyproject.toml').open('rb') as stream:
    version = tomllib.load(stream)['project']['version']
ocr_root = os.environ.get('KNOTSTUDIO_OCR_ROOT')
binaries = ocr_binaries(Path(os.environ['KNOTSTUDIO_TESSERACT']), Path(ocr_root) if ocr_root else None)
datas = [(str(root / 'recognizer' / 'models'), 'recognizer/models'),
         (str(assets / 'resources'), '.'),
         (str(assets / 'licenses'), 'licenses'),
         (str(assets / 'build-info.json'), '.'),
         (str(Path(os.environ['KNOTSTUDIO_TESSDATA']) / 'eng.traineddata'), 'ocr/tessdata')]
# scikit-image uses lazy module imports and .pyi declarations at runtime.
datas += collect_data_files('skimage', include_py_files=False)
hiddenimports = collect_submodules('skimage', filter=lambda name: '.tests' not in name)
hiddenimports += ['PIL._tkinter_finder']
a = Analysis([str(root / 'packaging' / 'entrypoint.py')], pathex=[str(root)],
             binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             excludes=['pytest', 'IPython', 'matplotlib', 'pandas', 'torch', 'tensorflow'],
             noarchive=False)
pyz = PYZ(a.pure)
options = {}
if sys.platform == 'win32':
    from PyInstaller.utils.win32.versioninfo import VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable, StringStruct, VarFileInfo, VarStruct
    numbers = tuple(int(part) for part in version.split('.'))
    numbers = (numbers + (0, 0, 0, 0))[:4]
    options.update(icon=str(assets / 'KnotStudio.ico'),
                   manifest=str(root / 'packaging' / 'windows' / 'KnotStudio.manifest'),
                   version=VSVersionInfo(
                       ffi=FixedFileInfo(filevers=numbers, prodvers=numbers, mask=0x3f,
                                         flags=0, OS=0x40004, fileType=1, subtype=0, date=(0, 0)),
                       kids=[StringFileInfo([StringTable('040904B0', [
                           StringStruct('CompanyName', 'Knot Studio contributors'),
                           StringStruct('FileDescription', 'Knot Studio'),
                           StringStruct('FileVersion', version),
                           StringStruct('ProductName', 'Knot Studio'),
                           StringStruct('ProductVersion', version),
                           StringStruct('OriginalFilename', 'KnotStudio.exe')])]),
                             VarFileInfo([VarStruct('Translation', [1033, 1200])])]))
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='KnotStudio',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, argv_emulation=False,
          target_arch=os.environ.get('KNOTSTUDIO_ARCH') if sys.platform == 'darwin' else None,
          codesign_identity=os.environ.get('KNOTSTUDIO_CODESIGN') or None,
          entitlements_file=None, **options)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='KnotStudio')
if sys.platform == 'darwin':
    app = BUNDLE(coll, name='Knot Studio.app', icon=str(assets / 'KnotStudio.icns'),
                 bundle_identifier='org.knotstudio.desktop', version=version,
                 info_plist={
                     'CFBundleDisplayName': 'Knot Studio',
                     'CFBundleShortVersionString': version,
                     'CFBundleVersion': version,
                     'LSMinimumSystemVersion': '15.7.5',
                     'NSHighResolutionCapable': True,
                     'NSHumanReadableCopyright': 'Knot Studio contributors. GPL-3.0-or-later; see bundled notices.',
                     'CFBundleDocumentTypes': [
                         {'CFBundleTypeName': 'Knot Studio diagram', 'CFBundleTypeRole': 'Editor',
                          'LSHandlerRank': 'Alternate', 'CFBundleTypeExtensions': ['json']},
                         {'CFBundleTypeName': 'Diagram image', 'CFBundleTypeRole': 'Viewer',
                          'LSHandlerRank': 'Alternate',
                          'CFBundleTypeExtensions': ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'webp']}
                     ]})
