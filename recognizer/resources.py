"""Resolve read-only release assets independently of the working directory."""
import json
import os
from pathlib import Path
import sys

from .desktop_platform import executable_name


def resource_root():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    source = Path(__file__).resolve().parent.parent
    if (source/'examples'/'index.json').is_file():
        return source
    packaged = Path(__file__).resolve().parent/'_resources'
    if (packaged/'examples'/'index.json').is_file():
        return packaged
    return Path(sys.prefix)/'share'/'knot-studio'


def bundled_tesseract():
    executable = resource_root()/'ocr'/'bin'/executable_name('tesseract')
    if executable.is_file():
        os.environ['TESSDATA_PREFIX'] = str(resource_root()/'ocr'/'tessdata')
        return str(executable)
    return None


def example_entries():
    path = resource_root()/'examples'/'index.json'
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding='utf-8'))
    return data.get('examples', []) if isinstance(data, dict) else data


def example_path(entry):
    directory = (resource_root()/'examples').resolve()
    relative = entry.get('file', entry.get('path', ''))
    path = (directory/relative).resolve()
    if not relative or not path.is_relative_to(directory) or not path.is_file():
        raise ValueError('This bundled example is unavailable.')
    return path
