"""Accuracy panel: precision vs. tolerance and sample evaluation."""
import pytest

from pin_inspection.accuracy import Precision, evaluate_samples, rate
from pin_inspection.config import load_config


def frames(values):
    return [[("length", v, "", True), ("straightness_max_dev", 0.05, "", True)] for v in values]


def test_precision_share_of_tolerance():
    p = Precision(min_frames=5)
    for c in frames([22.0, 22.1, 21.9, 22.0, 22.1, 21.9]):
        p.feed("PIN", c)
    p.feed("NO_PIN", [])
    s = p.summary({"length": [21.0, 23.3], "straightness_max_dev": [None, 0.25]})
    sd = s["length"]["sd"]
    assert sd == pytest.approx(0.0894, abs=0.001)
    assert s["length"]["pct_of_tol"] == pytest.approx(100 * 6 * sd / 2.3, abs=0.2)
    assert s["straightness_max_dev"]["sd"] == 0  # constant -> no spread
    assert s["length"]["rating"] == "acceptable"


def test_settling_frames_are_ignored():
    p = Precision(min_frames=3)
    for c in frames([20, 25, 30]):
        p.feed("SETTLING", c)  # pin still moving
    assert p.summary({"length": [21.0, 23.3]}) == {}


def test_rating_bands():
    assert rate(5) == "excellent" and rate(20) == "acceptable" and rate(45) == "poor" and rate(None) is None


def test_evaluate_empty_dir(tmp_path):
    r = evaluate_samples(tmp_path, load_config(), None)
    assert r["n"] == 0 and r["accuracy_pct"] is None
