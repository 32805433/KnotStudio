#!/usr/bin/env python3
"""Check the source release without consulting an original checkout or corpus.

The default checks use only Python's standard library. Use --import-modules
after installing the application dependencies to check runtime imports too.
Build output and virtual environments are deliberately excluded; this audits
the repository contents intended for publication.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit


IGNORED = {'.git', '.venv', '.venv-build', '.pytest_cache', '__pycache__',
           'build', 'dist', '.build-assets', '.ruff_cache', '.mypy_cache'}
FORBIDDEN = {'dataset', 'database', 'archive', 'evaluation', 'datareview',
             'data_review', 'review_app'}
REQUIRED = ('README.md', 'LICENSE', 'pyproject.toml',
            'recognizer/__init__.py', 'recognizer/pipeline.py',
            'recognizer/ui.py', 'recognizer/models/label_components_v2.json',
            'recognizer/models/ink_forest_v1.json', 'examples/index.json')
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.toml', '.json', '.yml', '.yaml',
                 '.sh', '.command', '.spec', '.html', '.plist', '.ps1', '.cmd', '.bat', '.iss'}


def source_files(root):
    for directory, children, filenames in os.walk(root, followlinks=False):
        children[:] = sorted(name for name in children
                             if name not in IGNORED and not name.endswith('.egg-info')
                             and not name.startswith('.venv-'))
        for name in children:
            path = Path(directory) / name
            if path.is_symlink():
                yield path
        for filename in sorted(filenames):
            if filename != '.DS_Store':
                yield Path(directory) / filename


def module_exists(root, name):
    path = root.joinpath(*name.split('.'))
    return path.with_suffix('.py').is_file() or (path / '__init__.py').is_file()


def audit_imports(root, source, errors):
    relative = source.relative_to(root)
    try:
        tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(relative))
    except (SyntaxError, UnicodeError) as error:
        errors.append(f'{relative}: invalid Python: {error}')
        return
    package = list(relative.with_suffix('').parts[:-1])
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > len(package):
                    errors.append(f'{relative}:{node.lineno}: relative import escapes the package')
                    continue
                base = package[:len(package)-node.level+1]
                if node.module:
                    modules = ['.'.join(base + node.module.split('.'))]
                else:
                    modules = ['.'.join(base + [alias.name]) for alias in node.names
                               if alias.name != '*']
            elif node.module:
                modules = [node.module]
        for module in modules:
            parts = module.split('.')
            if any(part in FORBIDDEN or part.startswith('dataset_') for part in parts):
                errors.append(f'{relative}:{node.lineno}: excluded research import {module}')
            if parts[0] == 'recognizer' and not module_exists(root, module):
                errors.append(f'{relative}:{node.lineno}: missing internal module {module}')


def audit_markdown_links(root, source, text, errors):
    # Markdown inline links and image links; URLs and section-only links need no
    # filesystem lookup. Targets containing spaces may use angle brackets.
    for match in re.finditer(r'!?\[[^\]\n]*\]\((<[^>]+>|[^\n)]+)\)', text):
        target = match.group(1).strip()
        if target.startswith('<'):
            target = target[1:target.index('>')]
        else:
            target = re.split(r'\s+["\']', target, maxsplit=1)[0]
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        path = (source.parent / unquote(parsed.path)).resolve()
        if not path.is_relative_to(root):
            errors.append(f'{source.relative_to(root)}: local link escapes release: {target}')
        elif not path.exists():
            errors.append(f'{source.relative_to(root)}: missing local link: {target}')


def audit_examples(root, errors):
    path = root / 'examples/index.json'
    if not path.is_file():
        return
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        entries = manifest['examples']
        assert isinstance(entries, list) and entries, 'examples must be a nonempty list'
        media, features, identifiers = set(), set(), set()
        for entry in entries:
            name = entry['path']
            image = (path.parent / name).resolve()
            assert image.is_relative_to(path.parent.resolve()), f'image escapes examples/: {name}'
            assert image.is_file(), f'missing sample: {name}'
            assert hashlib.sha256(image.read_bytes()).hexdigest() == entry['sha256'], f'sample hash differs: {name}'
            assert entry['id'] not in identifiers, f'duplicate sample id: {entry["id"]}'
            identifiers.add(entry['id'])
            media.add(entry['medium'])
            features.update(entry['features'])
            assert entry.get('license'), f'missing sample license/provenance notice: {name}'
            assert entry.get('source'), f'missing sample source record: {name}'
        assert {'program', 'blackboard', 'ipad'} <= media, 'sample set is missing a requested medium'
        assert {'labels', 'twist-box', 'parallel-strands'} <= features, 'sample set is missing requested features'
    except (KeyError, ValueError, TypeError, AssertionError) as error:
        errors.append(f'examples/index.json: {error}')


def check_source(root):
    root = root.resolve()
    errors = []
    for name in REQUIRED:
        if not (root / name).is_file():
            errors.append(f'Missing required release file: {name}')
    files = list(source_files(root))
    for path in files:
        relative = path.relative_to(root)
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            errors.append(f'{relative}: symlink points outside release')
            continue
        if not path.is_file():
            continue
        if any(part.lower() in FORBIDDEN for part in relative.parts):
            errors.append(f'{relative}: excluded corpus/review/research content')
        if path.suffix.lower() in {'.db', '.sqlite', '.sqlite3'}:
            errors.append(f'{relative}: database must not be published with the app')
        if path.name.startswith('dataset_') and path.suffix == '.py':
            errors.append(f'{relative}: dataset implementation must not be included')
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeError:
            errors.append(f'{relative}: expected UTF-8 text')
            continue
        if (re.search(r'(?:/Users/|/home/)[A-Za-z0-9_.-]+/', text)
                or re.search(r'[A-Za-z]:[\\/]Users[\\/][^\\/\s]+[\\/]', text)):
            errors.append(f'{relative}: contains an absolute personal filesystem path')
        if path.suffix == '.py':
            audit_imports(root, path, errors)
        if path.suffix == '.md':
            audit_markdown_links(root, path, text, errors)
        if path.suffix == '.json':
            try:
                json.loads(text)
            except json.JSONDecodeError as error:
                errors.append(f'{relative}: invalid JSON: {error}')
    audit_examples(root, errors)
    return errors, files


def check_runtime_imports(root):
    # -I ignores PYTHONPATH and user site packages; a foreign working directory
    # makes accidental checkout-relative asset access fail. Only this release
    # is added, never its parent or a neighboring original development tree.
    probe = r'''
import importlib, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
package = root / 'recognizer'
names = sorted('recognizer.' + path.stem for path in package.glob('*.py')
               if path.stem not in {'__init__', '__main__'})
for name in names:
    module = importlib.import_module(name)
    if not pathlib.Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Runtime module loaded from outside release: ' + name)
for name in sys.modules:
    if name.startswith('recognizer.') and any(
            part.startswith('dataset_') or part in {'dataset', 'evaluation', 'tests'}
            for part in name.split('.')):
        raise RuntimeError('Excluded runtime dependency: ' + name)
from recognizer.label_model import load_model as labels
from recognizer.learned_ink import load_model as ink
assert labels() is not None, 'Packaged label model is missing'
assert ink() is not None, 'Packaged ink model is missing'
print('Imported %d runtime modules and both packaged models from an unrelated directory.' % len(names))
'''
    with tempfile.TemporaryDirectory(prefix='knotstudio-import-') as directory:
        result = subprocess.run([sys.executable, '-I', '-c', probe, str(root.resolve())],
                                cwd=directory, text=True, capture_output=True, timeout=120)
    if result.returncode:
        return [f'Isolated runtime import check failed:\n{result.stderr.strip()}']
    print(result.stdout.strip())
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='source release folder (defaults to this script\'s parent project)')
    parser.add_argument('--import-modules', action='store_true',
                        help='also import runtime modules and models with installed dependencies')
    args = parser.parse_args()
    errors, files = check_source(args.root)
    if args.import_modules:
        errors.extend(check_runtime_imports(args.root))
    if errors:
        for error in errors:
            print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(f'Release checks passed ({len(files)} source files).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
