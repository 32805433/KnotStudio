"""Build a self-contained, browsable copy of the release documentation."""
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit

import markdown


ROOT_DOCUMENTS = ('LICENSE', 'README.md', 'CHANGELOG.md', 'CONTRIBUTING.md',
                  'SECURITY.md', 'pyproject.toml')
ASSET_DIRECTORIES = ('docs', 'examples', 'licenses', 'demo')
RESOURCE_MARKER = '.knotstudio-generated-resources'
RESOURCE_MARKER_CONTENT = 'Knot Studio generated resource bundle, version 1\n'


def _rewrite_links(text, source, root, output_root):
    def rewrite(match):
        target = match.group(2)
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)
        resolved = (source.parent/unquote(parsed.path)).resolve()
        if not resolved.is_relative_to(root):
            return match.group(0)
        relative = resolved.relative_to(root)
        if relative.parts[0] == 'recognizer':
            relative = Path('source')/relative
        elif relative.suffix == '.md' and (output_root/relative).is_file():
            relative = relative.with_suffix('.html')
        destination = output_root/relative
        page_parent = output_root/source.relative_to(root).parent
        target = Path(os.path.relpath(destination, page_parent)).as_posix()
        if parsed.query:
            target += '?'+parsed.query
        if parsed.fragment:
            target += '#'+parsed.fragment
        return match.group(1)+'="'+target+'"'
    return re.sub(r'(href|src)="([^"\n]+)"', rewrite, text)


def build_bundle(root: Path, output: Path):
    """Write all help pages, linked files, and example assets beneath output.

    Assets are kept together for relocatable wheels and frozen applications.
    Markdown is a build dependency only; readers need no rendering package.
    """
    root, output = root.resolve(), output.resolve()
    protected = [root/name for name in (*ASSET_DIRECTORIES, 'recognizer')]
    if (root.is_relative_to(output) or
            any(output.is_relative_to(path) for path in protected)):
        raise ValueError('Generated resources must not replace or sit inside source assets.')
    if output.exists():
        if not output.is_dir():
            raise ValueError('Generated resource output must be a directory.')
        try:
            generated = (output/RESOURCE_MARKER).read_text(encoding='utf-8') == RESOURCE_MARKER_CONTENT
        except (OSError, UnicodeError):
            generated = False
        if any(output.iterdir()) and not generated:
            raise ValueError('Refusing to replace an existing directory that is not a generated resource bundle.')
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/RESOURCE_MARKER).write_text(RESOURCE_MARKER_CONTENT, encoding='utf-8')
    for name in ROOT_DOCUMENTS:
        shutil.copy2(root/name, output/name)
    for name in ASSET_DIRECTORIES:
        shutil.copytree(root/name, output/name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('.DS_Store', '__pycache__', '*.pyc'))
    source_copy = output/'source'/'recognizer'
    source_copy.mkdir(parents=True, exist_ok=True)
    for source in (root/'recognizer').glob('*.py'):
        shutil.copy2(source, source_copy/source.name)
    pages = [root/name for name in ROOT_DOCUMENTS if name.endswith('.md')]
    for name in ASSET_DIRECTORIES:
        pages.extend((root/name).rglob('*.md'))
    for source in pages:
        relative = source.relative_to(root).with_suffix('.html')
        body = markdown.markdown(source.read_text(encoding='utf-8'),
                                 extensions=['tables', 'fenced_code', 'toc'])
        body = _rewrite_links(body, source, root, output)
        guide = Path(os.path.relpath(output/'docs/USER_GUIDE.html', (output/relative).parent)).as_posix()
        examples = Path(os.path.relpath(output/'examples/gallery.html', (output/relative).parent)).as_posix()
        page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Knot Studio help</title>
<style>body{max-width:850px;margin:3rem auto;padding:0 1.4rem;color:#172234;background:#fff;
font:17px/1.65 system-ui,sans-serif}a{color:#2556af}pre{overflow:auto;background:#f3f5f8;padding:1rem}
table{border-collapse:collapse}td,th{border:1px solid #ccd3dd;padding:.4rem .7rem}img{max-width:100%}</style>
'''+f'<nav><a href="{guide}">User guide</a> · <a href="{examples}">Examples</a></nav>'+body+'</html>'
        (output/relative).write_text(page, encoding='utf-8')
    for name in ASSET_DIRECTORIES:
        for source in (root/name).rglob('*.html'):
            text = _rewrite_links(source.read_text(encoding='utf-8'), source, root, output)
            (output/source.relative_to(root)).write_text(text, encoding='utf-8')
    check_links(output)
    return output


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.targets.extend(attrs[key] for key in ('href', 'src') if attrs.get(key))
        if attrs.get('id'):
            self.ids.add(attrs['id'])


def check_links(root: Path):
    """Reject missing local resources and broken generated HTML anchors."""
    root = root.resolve()
    pages = {}
    for page in root.rglob('*.html'):
        parser = _Links()
        parser.feed(page.read_text(encoding='utf-8'))
        pages[page.resolve()] = parser
    errors = []
    for page, parser in pages.items():
        for target in parser.targets:
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            destination = ((page.parent/unquote(parsed.path)).resolve()
                           if parsed.path else page)
            if not destination.is_relative_to(root) or not destination.is_file():
                errors.append(f'{page.relative_to(root)}: missing local resource {target}')
            elif (parsed.fragment and destination in pages and
                  unquote(parsed.fragment) not in pages[destination].ids):
                errors.append(f'{page.relative_to(root)}: missing anchor {target}')
    if errors:
        raise ValueError('Broken offline help links:\n'+'\n'.join(errors))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    build_bundle(root, root/'.build-assets'/'resources')
