# Native macOS build. Configuration is supplied by tools/build_macos.py.
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH).parent
assets = root/'.build-assets'
datas = [(str(root/'recognizer'/'models'), 'recognizer/models'),
         (str(assets/'resources'), '.'),
         (str(assets/'licenses'), 'licenses'),
         (str(assets/'build-info.json'), '.'),
         (str(Path(os.environ['KNOTSTUDIO_TESSDATA'])/'eng.traineddata'), 'ocr/tessdata')]
# scikit-image uses lazy module imports and .pyi declarations at runtime.
datas += collect_data_files('skimage', include_py_files=False)
hiddenimports = collect_submodules('skimage', filter=lambda name: '.tests' not in name)
hiddenimports += ['PIL._tkinter_finder']
a = Analysis([str(root/'packaging'/'entrypoint.py')], pathex=[str(root)],
             binaries=[(os.environ['KNOTSTUDIO_TESSERACT'], 'ocr/bin')],
             datas=datas, hiddenimports=hiddenimports,
             excludes=['pytest', 'IPython', 'matplotlib', 'pandas', 'torch', 'tensorflow'],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='KnotStudio',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, argv_emulation=False,
          target_arch=os.environ.get('KNOTSTUDIO_ARCH'),
          codesign_identity=os.environ.get('KNOTSTUDIO_CODESIGN') or None,
          entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='KnotStudio')
app = BUNDLE(coll, name='Knot Studio.app', icon=str(assets/'KnotStudio.icns'),
             bundle_identifier='org.knotstudio.desktop', version='0.1.0',
             info_plist={
                 'CFBundleDisplayName': 'Knot Studio',
                 'CFBundleShortVersionString': '0.1.0',
                 'CFBundleVersion': '1',
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
