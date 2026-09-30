"""Preflight a new release tag or create its draft; never replace a published release."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tomllib


def gh(*args):
    return subprocess.check_output(['gh', *args], text=True).strip()


def preflight(root, tag):
    version = tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    if tag != 'v'+version or not re.fullmatch(r'v\d+\.\d+\.\d+', tag):
        raise ValueError('Tag must match the version in pyproject.toml')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    # API errors fail closed; a network failure is never treated as release absence.
    releases = json.loads(gh('api', '--paginate', '--slurp', 'repos/{owner}/{repo}/releases?per_page=100'))
    if (not isinstance(releases, list) or not releases
            or any(not isinstance(page, list) for page in releases)
            or any(not isinstance(r, dict) or not isinstance(r.get('tag_name'), str)
                   for page in releases for r in page)):
        raise ValueError('Malformed release API response')
    if any(r['tag_name'] == tag for page in releases for r in page):
        raise ValueError('A release with this tag already exists; use a new version')
    refs = json.loads(gh('api', 'repos/{owner}/{repo}/git/matching-refs/tags/'+tag))
    if (not isinstance(refs, list)
            or any(not isinstance(r, dict) or not isinstance(r.get('ref'), str) for r in refs)):
        raise ValueError('Malformed tag API response')
    if any(r['ref'] == 'refs/tags/'+tag for r in refs):
        raise ValueError('Tag already exists; release tags must be new and immutable')
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--assets', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = preflight(root, args.tag)
    if args.assets:
        files = sorted(p for p in args.assets.iterdir() if p.is_file())
        if not files or not (args.assets/'SHA256SUMS.txt').is_file():
            raise ValueError('Validated release assets are required')
        # Create the ref atomically. If another actor created this tag after
        # preflight, GitHub rejects the POST instead of silently reusing it.
        # Keep a successfully created tag even if the later draft upload fails;
        # recovery is a maintainer decision, never an automatic tag deletion.
        created = json.loads(gh('api', '--method', 'POST', 'repos/{owner}/{repo}/git/refs',
                                '-f', 'ref=refs/tags/'+args.tag, '-f', 'sha='+commit))
        if (not isinstance(created, dict) or created.get('ref') != 'refs/tags/'+args.tag
                or not isinstance(created.get('object'), dict)
                or created['object'].get('sha') != commit):
            raise ValueError('Tag creation did not confirm the verified source commit')
        # Uploaded draft assets remain private until the maintainer publishes.
        subprocess.run(['gh', 'release', 'create', args.tag, '--draft', '--verify-tag', '--target', commit,
                        '--title', 'Knot Studio '+args.tag, '--notes-file', str(root/'docs/RELEASE_NOTES.md'),
                        *map(str, files)], check=True)
    else:
        print(commit)


if __name__ == '__main__':
    main()
