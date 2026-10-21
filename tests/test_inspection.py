"""PASS/FAIL logic: tolerance checks and the good-pin model."""
import copy

import pytest

from pin_inspection.config import load_config
from pin_inspection.inspection import check
from pin_inspection.model import PinModel

from .test_measure import draw_pin

CFG = load_config()
SCALE = 22 / 300  # synthetic 300 px pin = 22 mm, like the real station


def measured(**kw):
    cfg = copy.deepcopy(CFG)
    cfg["measure"]["mm_per_px"] = SCALE
    from pin_inspection.pipeline import analyse
    kw = {"shaft_w": 14, "head_w": 26, **kw}
    return analyse(draw_pin(**kw), cfg).res


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(CFG)
    c["measure"]["mm_per_px"] = SCALE
    return c


@pytest.fixture(scope="module")
def good():
    return [measured(angle_deg=a) for a in (0, 15, 30, 60, 90, 120, 200, 268)]


def test_good_pins_pass(cfg, good):
    for res in good:
        verdict, checks = check(res, cfg)
        assert verdict == "PASS", [c for c in checks if not c[3]]


@pytest.mark.parametrize("kw, expect", [
    ({"length": 250}, "length"),
    ({"bend": 8}, "straightness_max_dev"),
    ({"head_w": 14}, "head_shaft_ratio"),
])
def test_bad_pins_fail_for_the_right_reason(cfg, kw, expect):
    verdict, checks = check(measured(angle_deg=40, **kw), cfg)
    assert verdict == "FAIL"
    assert expect in [c[0] for c in checks if not c[3]]


def test_model_accepts_good_and_rejects_bad(cfg, good, tmp_path):
    model = PinModel.fit(good, cfg["model"], SCALE)
    model.save(tmp_path / "m.json")
    model = PinModel.load(tmp_path / "m.json")
    assert all(model.score(r) <= model.threshold for r in good)
    assert model.score(measured(angle_deg=45)) <= model.threshold
    assert model.score(measured(angle_deg=45, length=270)) > model.threshold
    assert model.score(measured(angle_deg=45, bend=6)) > model.threshold
