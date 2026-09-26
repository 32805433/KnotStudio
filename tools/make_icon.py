"""Package the approved figure-eight artwork as a macOS application icon."""
from pathlib import Path

from PIL import Image


def create(output: Path):
    artwork = Path(__file__).resolve().parents[1]/'packaging'/'icons'/'figure-eight.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(artwork) as image:
        if image.size != (1024, 1024):
            raise ValueError('The approved icon must be a 1024 × 1024 image.')
        image.convert('RGBA').save(output, format='ICNS')
