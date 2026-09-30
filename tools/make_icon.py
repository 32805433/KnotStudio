"""Package the approved figure-eight artwork for each desktop platform."""
from pathlib import Path

from PIL import Image


def create(output: Path):
    artwork = Path(__file__).resolve().parents[1]/'packaging'/'icons'/'figure-eight.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(artwork) as image:
        if image.size != (1024, 1024):
            raise ValueError('The approved icon must be a 1024 × 1024 image.')
        formats = {'.icns': 'ICNS', '.ico': 'ICO', '.png': 'PNG'}
        try:
            format_name = formats[output.suffix.lower()]
        except KeyError:
            raise ValueError('Application icons must use .icns, .ico, or .png.') from None
        options = {'sizes': [(n, n) for n in (16, 24, 32, 48, 64, 128, 256)]} if format_name == 'ICO' else {}
        image.convert('RGBA').save(output, format=format_name, **options)
