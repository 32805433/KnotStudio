"""Fetch checksum-locked OCR language data and runtime notices for release builds."""
from pathlib import Path
import hashlib
import json
import shutil
import ssl
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    'tessdata/eng.traineddata': (
        'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/eng.traineddata',
        '7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2'),
    'runtime/Python/license.rst': (
        'https://raw.githubusercontent.com/python/cpython/v3.13.2/Doc/license.rst',
        '62f2c9c2c75d511170eb464ad5f83b78cc1f37eb2eb49c2846c9aa6c4557ee99'),
    'runtime/Python/LICENSE.txt': (
        'https://raw.githubusercontent.com/python/cpython/v3.13.2/LICENSE',
        '78b12c3a81360b357002334f0e70ea0e92eebf7a9b358805c03c48484945f3bb'),
}
WINDOWS_TCL = {
    'name': 'Tcl', 'version': '8.6.15', 'python': '3.13.2',
    'source_url': 'https://raw.githubusercontent.com/python/cpython-bin-deps/tcltk-8.6.15.0/amd64/tcllicense.terms',
    'sha256': '41613eabfc08921a7da9c4bfd7f3ce5d5406f55214b253d89f392ed86eebeb8f',
}


def prepare(destination):
    from pip._vendor import certifi
    context = ssl.create_default_context(cafile=certifi.where())
    inputs = dict(INPUTS)
    if sys.platform == 'win32':
        # CPython's Windows installer copies the Tcl library but omits the
        # license file kept at the binary dependency archive's top level.
        inputs['runtime/Tcl/license.terms'] = (WINDOWS_TCL['source_url'], WINDOWS_TCL['sha256'])
    for relative, (url, expected) in inputs.items():
        target = destination / relative
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.download')
        try:
            with urllib.request.urlopen(url, context=context, timeout=90) as source:
                with temporary.open('wb') as output:
                    shutil.copyfileobj(source, output)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Checksum mismatch: {relative}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    if sys.platform == 'win32':
        (destination / 'runtime' / 'Tcl' / 'provenance.json').write_text(
            json.dumps(WINDOWS_TCL, indent=2) + '\n', encoding='utf-8')
    return destination


if __name__ == '__main__':
    print(prepare(ROOT / 'build' / 'inputs'))
