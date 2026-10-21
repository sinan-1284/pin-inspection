"""Pipeline tests on synthetic pins with known geometry."""
import cv2
import numpy as np
import pytest

from pin_inspection.align import align
from pin_inspection.config import load_config
from pin_inspection.features import measure
from pin_inspection.segment import segment

CFG = load_config()


def draw_pin(angle_deg=0, length=300, shaft_w=10, head_w=24, head_t=6, bend=0.0, size=(1000, 1400)):
    """Dark pin on a light background; head at the origin end, tip at the far end.

    `bend` is the shaft's peak sideways deviation in pixels (circular-ish bow).
    """
    img = np.full(size, 220, np.uint8)
    cx, cy = size[1] / 2, size[0] / 2
    a = np.radians(angle_deg)
    u = np.array([np.cos(a), np.sin(a)])
    v = np.array([-u[1], u[0]])
    start = np.array([cx, cy]) - u * length / 2
    tip_len = shaft_w * 2
    pts_top, pts_bot = [], []
    for t in np.linspace(0, length, 400):
        w = shaft_w if t < length - tip_len else shaft_w * (length - t) / tip_len
        off = bend * np.sin(np.pi * t / length)
        c = start + u * t + v * off
        pts_top.append(c + v * w / 2)
        pts_bot.append(c - v * w / 2)
    poly = np.array(pts_top + pts_bot[::-1], np.int32)
    cv2.fillPoly(img, [poly], 40)
    head = np.array([start + v * head_w / 2, start + v * head_w / 2 + u * head_t,
                     start - v * head_w / 2 + u * head_t, start - v * head_w / 2], np.int32)
    cv2.fillPoly(img, [head], 40)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def run(img):
    seg = segment(img, CFG["segment"])
    assert seg is not None
    al = align(seg.mask)
    return al, measure(al.mask, seg.scale, CFG["measure"])


@pytest.mark.parametrize("angle", [0, 37, 90, 180, 268])
def test_length_and_widths_any_orientation(angle):
    al, res = run(draw_pin(angle_deg=angle))
    px = res["px"]
    assert px["length"] == pytest.approx(300, abs=4)
    assert px["shaft_width"] == pytest.approx(10, abs=1.5)
    assert px["head_width"] == pytest.approx(24, abs=2.5)
    assert res["ratios"]["head_symmetry"] > 0.85


@pytest.mark.parametrize("angle", [0, 90, 200])
def test_head_direction(angle):
    al, _ = run(draw_pin(angle_deg=angle))
    assert (al.angle_deg - angle) % 360 == pytest.approx(0, abs=3) or \
        (al.angle_deg - angle) % 360 == pytest.approx(360, abs=3)


def test_straight_vs_bent():
    _, straight = run(draw_pin(angle_deg=20))
    _, bent = run(draw_pin(angle_deg=20, bend=8))
    assert straight["px"]["straightness_max_dev"] < 1.5
    assert bent["px"]["straightness_max_dev"] > 2 * straight["px"]["straightness_max_dev"] + 1


def test_mm_scale():
    cfg = dict(CFG["measure"], mm_per_px=0.1)
    seg = segment(draw_pin(), CFG["segment"])
    res = measure(align(seg.mask).mask, seg.scale, cfg)
    assert res["mm"]["length"] == pytest.approx(30, abs=0.5)


def test_no_pin():
    assert segment(np.full((600, 800, 3), 220, np.uint8), CFG["segment"]) is None
