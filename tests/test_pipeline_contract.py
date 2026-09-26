"""Public recognition contracts for real pixels, invalid input, and deadlines."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image, ImageDraw
import pytest

from recognizer import pipeline
from recognizer.diagram import pd_code, validate
from recognizer.recognition_runtime import current


def circle():
    image = Image.new('1', (200, 180), 1)
    ImageDraw.Draw(image).ellipse((25, 20, 175, 160), outline=0, width=5)
    return image


def unresolved_drawing():
    image = Image.new('RGB', (320, 240), 'white')
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 35, 270, 205), outline='black', width=5)
    draw.rectangle((142, 27, 152, 48), fill='white')
    draw.line((150, 20, 150, 65), fill='black', width=4)
    return np.asarray(image)


def assert_circle(result):
    assert result['status'] == 'ok'
    assert validate(result['diagram'])['valid']
    assert result['pd_code'] == pd_code(result['diagram']) == []
    assert result['unlinked_unknot_components'] == 1


def test_bitmap_array_matches_bitmap_file_and_preserves_source(tmp_path):
    image = circle()
    path = tmp_path/'circle.png'
    image.save(path)
    array = np.asarray(image)
    before = array.copy()
    results = [pipeline.recognize(path),
               pipeline.recognize_array(array),
               pipeline.recognize_array(array.astype(float)),
               pipeline.recognize_array(np.asarray(image.convert('RGBA')))]
    for result in results:
        assert_circle(result)
        assert result['diagram'] == results[0]['diagram']
    np.testing.assert_array_equal(array, before)


def test_rgba_array_and_file_use_identical_pixels_for_review_hashes(tmp_path):
    from recognizer.box_detection import detect_twist_boxes
    rgba = np.full((80, 80, 4), 255, np.uint8)
    rgba[10:40, 10:40] = [73, 109, 181, 80]
    rgba[40:60, 40:60] = [170, 20, 30, 0]
    source = Image.fromarray(rgba)
    path = tmp_path/'translucent.png'
    source.save(path)
    white = Image.new('RGBA', source.size, 'white')
    white.alpha_composite(source)
    review = detect_twist_boxes(np.asarray(white.convert('RGB')), max_ocr_seconds=0)
    review['boxes'] = [{'id': 'unknown', 'half_twists': None}]
    # The same reviewed source must not be rejected as stale merely because
    # it was submitted as RGBA pixels rather than a decoded file.
    for result in (pipeline.recognize(path, options={'twist_boxes': review}),
                   pipeline.recognize_array(rgba, options={'twist_boxes': review})):
        assert result['status'] == 'needs_review'
        assert result['diagram'] is None
        assert 'coefficient is unreadable' in result['warnings'][0]


@pytest.mark.parametrize('array', [
    np.zeros((2, 20, 3)), np.zeros((20, 20, 2)), np.zeros((0, 0)),
    np.full((5, 5, 3), np.nan), np.full((5, 5, 3), np.inf),
    np.full((5, 5, 3), 255+1j), np.full((5, 5, 3), '255'),
])
def test_invalid_pixels_are_reported_before_attempting_recognition(array):
    with pytest.raises(ValueError):
        pipeline.recognize_array(array)
    assert current() is None


def test_invalid_style_is_not_hidden_by_an_expired_budget(monkeypatch):
    from recognizer import box_detection
    def should_not_run(*args, **kwargs):
        pytest.fail('Invalid styles must be rejected before box detection.')
    monkeypatch.setattr(box_detection, 'detect_twist_boxes', should_not_run)
    with pytest.raises(ValueError, match='board_photo'):
        pipeline.recognize_array(np.asarray(circle().convert('RGB')),
                                 options={'board_photo': 'bad style', 'time_budget': 1e-12})
    assert current() is None


def test_failed_recognition_has_unknown_component_count():
    result = pipeline.recognize_array(np.full((80, 80, 3), 255, np.uint8))
    assert result['status'] == 'failed'
    assert result['diagram'] is result['pd_code'] is None
    assert result['unlinked_unknot_components'] is None


def test_numpy_time_budget_keeps_result_json_serializable():
    result = pipeline.recognize_array(np.asarray(circle()),
                                     options={'time_budget': np.float32(5.)})
    assert_circle(result)
    assert result['diagnostics']['runtime']['budget_seconds'] == 5.
    json.dumps(result, allow_nan=False)


def expire_after_extract(monkeypatch, module):
    original = module.extract
    extracted = []
    def extract_then_expire(pixels, config):
        result = original(pixels, config)
        assert not result['complete']
        extracted.append(deepcopy(result))
        current().deadline = current().started-1
        return result
    monkeypatch.setattr(module, 'extract', extract_then_expire)
    return extracted


def assert_partial_evidence(result, trace, scale):
    assert result['status'] == 'needs_review'
    assert result['diagram'] is result['pd_code'] is None
    assert result['unlinked_unknot_components'] is None
    diagnostic = result['diagnostics']
    assert diagnostic['budget_exhausted']
    assert diagnostic['branch_points'] or diagnostic['unmatched_endpoints']
    for expected, actual in zip(trace['paths'], result['preview_paths'], strict=True):
        np.testing.assert_allclose(actual['points'], np.asarray(expected['points'])/scale)
    np.testing.assert_allclose(diagnostic['branch_points'],
                               np.asarray(trace['diagnostics']['branch_points'])/scale)
    for expected, actual in zip(trace['diagnostics']['unmatched_endpoints'],
                                diagnostic['unmatched_endpoints'], strict=True):
        np.testing.assert_allclose(actual['point'], np.asarray(expected['point'])/scale)
    assert current() is None


def test_deadline_keeps_the_first_incomplete_trace_in_original_coordinates(monkeypatch):
    extracted = expire_after_extract(monkeypatch, pipeline)
    pixels = unresolved_drawing()
    original = pixels.copy()
    result = pipeline.recognize_array(pixels, options={'twist_boxes': False})
    # The first standard candidate upsamples small drawings by a factor of 3.
    assert len(extracted) == 1
    assert_partial_evidence(result, extracted[0], scale=3.)
    np.testing.assert_array_equal(pixels, original)


def test_board_deadline_keeps_partial_trace_in_original_photo_coordinates(monkeypatch):
    from recognizer import board_photo
    pixels = unresolved_drawing()
    class Contrast:
        scale = .5
        polarity = 'dark'
        def candidates(self):
            for threshold in (80, 100):
                yield pixels, dict(method='test contrast', foreground_fraction=.03,
                                   strong=threshold, weak=threshold/2)
    monkeypatch.setattr(board_photo, 'prepared_board_contrast', lambda *a, **k: Contrast())
    extracted = expire_after_extract(monkeypatch, board_photo)
    photo = np.full((480, 640, 3), 60, np.uint8)
    result = pipeline.recognize_array(photo, options={'board_photo': 'dark', 'twist_boxes': False})
    assert len(extracted) == 1
    assert_partial_evidence(result, extracted[0], scale=.5)
    assert result['diagnostics']['preprocessing']['transform'] == 'board_photo'


def test_unread_box_does_not_invent_pd_or_component_count():
    from recognizer.box_detection import detect_twist_boxes
    pixels = np.asarray(circle().convert('RGB'))
    review = detect_twist_boxes(pixels, max_ocr_seconds=0)
    review['boxes'] = [{'id': 'unknown', 'half_twists': None}]
    result = pipeline.recognize_array(pixels, options={'twist_boxes': review})
    assert result['status'] == 'needs_review'
    assert result['diagram'] is result['pd_code'] is None
    assert result['unlinked_unknot_components'] is None
    assert 'coefficient is unreadable' in result['warnings'][0]
