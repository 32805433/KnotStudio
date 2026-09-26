import numpy as np
from PIL import Image, ImageDraw
import pytest

from recognizer.drawing import PencilGesture
from recognizer.pipeline import recognize_array


def dark(image,x,y):
    return min(image.getpixel((x,y)))<100


@pytest.mark.parametrize('under',[False,True])
def test_crossing_clearance_survives_many_mouse_events(under):
    base=Image.new('RGB',(200,200),'white')
    ImageDraw.Draw(base).line([(100,20),(100,180)],fill='#dc3545',width=7)
    saved=np.asarray(base).copy()
    pen=PencilGesture(base,5)
    for x in range(20,180,2):
        result=pen.add((x,100),(x+2,100),under=under)
    assert dark(result,100,100)
    assert dark(result,100,105)==under
    assert dark(result,106,100)==(not under)
    if under:
        assert result.getpixel((100,100))==base.getpixel((100,100))
    assert np.array_equal(np.asarray(base),saved)


@pytest.mark.parametrize('under',[False,True])
def test_two_closed_components_reconstruct_with_two_crossings(under):
    base=Image.new('RGB',(320,320),'white')
    ImageDraw.Draw(base).ellipse((80,60,220,220),outline='black',width=5)
    pen=PencilGesture(base,5)
    points=[(40,140),(270,140),(270,270),(40,270),(40,140)]
    for a,b in zip(points,points[1:]):image=pen.add(a,b,under=under)
    result=recognize_array(np.asarray(image))
    assert result['diagram'] is not None
    assert len(result['pd_code'])==2
    assert result['diagnostics']['validation']['components']==2


def test_shift_can_change_between_crossings_in_one_gesture():
    base=Image.new('RGB',(240,200),'white')
    draw=ImageDraw.Draw(base)
    for x in [70,170]:draw.line([(x,20),(x,180)],fill='black',width=5)
    pen=PencilGesture(base,5)
    pen.add((20,100),(120,100),under=False)
    result=pen.add((120,100),(220,100),under=True)
    assert dark(result,75,100) and not dark(result,70,105)
    assert dark(result,170,105) and not dark(result,175,100)
    assert dark(result,120,100)  # No gap at the modifier transition.


@pytest.mark.parametrize('under',[False,True])
def test_self_crossing_in_one_gesture_has_one_crossing(under):
    pen=PencilGesture(Image.new('RGB',(220,220),'white'),5)
    points=[(40,40),(180,180),(40,180),(180,40),(40,40)]
    for a,b in zip(points,points[1:]):image=pen.add(a,b,under=under)
    result=recognize_array(np.asarray(image))
    assert len(pen.self_crossings)==1
    assert result['diagram'] is not None
    assert len(result['pd_code'])==1
    assert result['diagnostics']['validation']['components']==1
    assert dark(image,113,113)==under
    assert dark(image,113,107)==(not under)


@pytest.mark.parametrize('under',[False,True])
def test_strokes_can_join_an_existing_endpoint(under):
    base=Image.new('RGB',(180,120),'white')
    ImageDraw.Draw(base).line([(20,60),(80,60)],fill='black',width=5)
    image=PencilGesture(base,5).add((80,60),(150,60),under=under)
    assert all(dark(image,x,60) for x in range(22,149))


@pytest.mark.parametrize('angle',[20,35,50])
@pytest.mark.parametrize('under',[False,True])
@pytest.mark.parametrize('segments',[1,12])
def test_acute_self_crossing_recognizes_and_preserves_nearby_arc(angle,under,segments):
    half_angle=np.radians(angle/2)
    dy=150*np.tan(half_angle)
    points=[(50,220-dy),(350,220+dy),(50,220+dy),(350,220-dy),(50,220-dy)]
    pen=PencilGesture(Image.new('RGB',(400,440),'white'),7)
    for a,b in zip(points,points[1:]):
        sampled=np.linspace(a,b,segments+1)
        for c,d in zip(sampled,sampled[1:]):
            image=pen.add(c,d,under=under)
    result=recognize_array(np.asarray(image),options={'remove_labels':False})
    assert result['status']=='ok',result['diagnostics']
    assert len(result['pd_code'])==1
    assert result['diagnostics']['validation']['components']==1
    # Near the crossing, the intended overpass is continuous and the other
    # branch has visible white clearance even at a shallow angle.
    length=7*.9/np.sin(2*half_angle)
    positive=np.array([np.cos(half_angle),np.sin(half_angle)])
    negative=positive*np.array([1,-1])
    over,below=(positive,negative) if under else (negative,positive)
    assert dark(image,*map(round,np.array([200,220])+length*over))
    assert not dark(image,*map(round,np.array([200,220])+length*below))
    # At 20 degrees this unrelated connector is inside the old circular erase
    # neighborhood; widening a disk alone would cut it in two.
    assert dark(image,200,round(220-dy))


@pytest.mark.parametrize('under', [False, True])
def test_blackboard_crossing_keeps_white_chalk_and_black_clearance(under):
    base = Image.new('RGB', (200, 200), (30, 40, 35))
    ImageDraw.Draw(base).line([(100, 20), (100, 180)], fill=(245, 240, 220), width=7)
    before = np.asarray(base).copy()
    pen = PencilGesture(base, 5)
    assert pen.dark
    for x in range(20, 180, 2):
        result = pen.add((x, 100), (x+2, 100), under=under)
    assert (max(result.getpixel((100, 105))) > 200) == under
    assert (max(result.getpixel((106, 100))) > 200) == (not under)
    assert result.getpixel((100, 100)) == (base.getpixel((100, 100)) if under else (255, 255, 255))
    # Under-drawing leaves existing background pixels alone; over-drawing
    # actively erases the old strand to black before laying down white chalk.
    assert result.getpixel((106, 100) if under else (100, 105)) == ((30, 40, 35) if under else (0, 0, 0))
    np.testing.assert_array_equal(base, before)


@pytest.mark.parametrize('under', [False, True])
def test_blackboard_self_crossing_has_same_geometry_as_light_drawing(under):
    pens = [PencilGesture(Image.new('RGB', (220, 220), color), 5) for color in ('white', 'black')]
    points = [(40, 40), (180, 180), (40, 180), (180, 40), (40, 40)]
    results = []
    for pen in pens:
        for a, b in zip(points, points[1:]):
            result = pen.add(a, b, under=under)
        assert len(pen.self_crossings) == 1
        results.append(np.asarray(result))
    np.testing.assert_array_equal(results[0].min(2) < 100, results[1].max(2) > 200)


@pytest.mark.parametrize('under', [False, True])
def test_blackboard_endpoint_join_remains_continuous(under):
    base = Image.new('RGB', (180, 120), 'black')
    ImageDraw.Draw(base).line([(20, 60), (80, 60)], fill='white', width=5)
    result = PencilGesture(base, 5).add((80, 60), (150, 60), under=under)
    assert all(result.getpixel((x, 60)) == (255, 255, 255) for x in range(22, 149))
