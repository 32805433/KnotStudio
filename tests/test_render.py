"""Regression checks for preserving crossing information in diagram exports."""
from recognizer.diagram import reverse_component, pd_code, validate
from recognizer.render import arrows, render_image
from recognizer.pipeline import recognize_array
from recognizer.projection_equivalence import same_pd_projection
import numpy as np
import pytest


def one_crossing():
    return {"width": 220, "height": 220, "crossings": [{"id": 0, "point": [110, 110], "ports": [{"edge": 0, "end": 0}, {"edge": 0, "end": 1}, {"edge": 1, "end": 0}, {"edge": 1, "end": 1}], "over": 1}], "edges": [{"id": 0, "points": [[110, 110], [145, 110], [175, 95], [185, 65], [170, 40], [135, 35], [110, 55], [110, 110]], "start": [0, 0], "end": [0, 1], "component": 0, "closed": False}, {"id": 1, "points": [[110, 110], [75, 110], [45, 125], [35, 155], [50, 180], [85, 185], [110, 165], [110, 110]], "start": [0, 2], "end": [0, 3], "component": 0, "closed": False}], "component_orientations": {}}


def test_overpass_remains_continuous_and_roundtrips():
    diagram = one_crossing()
    assert validate(diagram)["valid"]
    image = render_image(diagram, scale=3)
    # The overpass stays continuous; there is visible white clearance between
    # its right edge and the rounded cap of the trimmed underpassing arc.
    assert all(image.getpixel((330, y))[0] < 100 for y in range(270, 390))
    assert all(min(image.getpixel((x, 330))) > 220 for x in range(334, 337))
    prediction = recognize_array(np.asarray(image))
    assert prediction["status"] == "ok"
    assert same_pd_projection(pd_code(diagram), prediction["pd_code"])


def test_orientation_arrow_reverses_without_moving_the_curve():
    diagram = one_crossing()
    changed = reverse_component(diagram, 0)
    before, after = arrows(diagram)[0], arrows(changed)[0]
    assert before[0] == after[0]
    assert before[1] == tuple(-value for value in after[1])
    assert diagram["edges"] == changed["edges"]


@pytest.mark.parametrize('translation', [(100, 0), (-200, 0), (0, 100), (0, -200)])
def test_png_export_includes_strands_beyond_the_original_canvas(translation):
    from copy import deepcopy
    from PIL import Image, ImageChops
    dx, dy = translation
    points = [[x+dx, y+dy] for x, y in [[20, 50], [50, 20], [80, 50], [50, 80], [20, 50]]]
    diagram = dict(width=100, height=100, crossings=[], edges=[
        dict(id=0, start=None, end=None, closed=True, points=points)])
    original = deepcopy(diagram)
    # No editable-raster/minimum-width flag: this is ordinary PNG export.
    image = render_image(diagram, scale=1)
    ink = ImageChops.difference(image, Image.new('RGB', image.size, 'white')).getbbox()
    assert ink is not None
    left, top, right, bottom = ink
    assert right-left >= 60 and bottom-top >= 60
    assert 0 < left < right < image.width and 0 < top < bottom < image.height
    assert diagram == original


@pytest.mark.parametrize('dx', [-500, 500])
def test_png_export_keeps_outside_twist_boxes_and_their_labels(dx):
    from copy import deepcopy
    from test_twist_boxes import closed_braid
    diagram = closed_braid('3/2')
    diagram['crossings'][0]['point'][0] += dx
    for edge in diagram['edges']:
        edge['points'] = [[x+dx, y] for x, y in edge['points']]
    original = deepcopy(diagram)
    image = render_image(diagram, scale=1, colored=True)
    pixels = np.asarray(image)
    # The black box border and number are separate from colored strands.
    dark = np.all(pixels < 80, axis=2)
    rows, columns = np.where(dark)
    assert len(rows) > 200
    assert columns.max()-columns.min() >= 118
    assert rows.max()-rows.min() >= 158
    assert diagram == original
