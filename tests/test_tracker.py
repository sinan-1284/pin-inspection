"""Per-frame verdicts -> one result per pin."""
from pin_inspection.tracker import PinTracker

OK = [("length", 22.0, "21..23", True)]
BAD = [("length", 19.0, "21..23", False)]


def feed(tr, frames):
    out = []
    for t, v in enumerate(frames):
        checks = {"PASS": OK, "FAIL": BAD}.get(v, [])
        failed = [c[0] for c in checks if not c[3]]
        rec = tr.update(v, failed, checks, card_png=b"png" if v == "FAIL" else None, now=float(t))
        if rec:
            out.append(rec)
    return out


def test_one_record_per_pin():
    tr = PinTracker()
    recs = feed(tr, ["NO_PIN", "PASS", "PASS", "PASS", "NO_PIN", "FAIL", "FAIL", "NO_PIN"])
    assert [r["verdict"] for r in recs] == ["PASS", "FAIL"]
    assert recs[1]["failed"] == ["length"]
    assert tr.stats() == {**tr.stats(), "inspected": 2, "pass": 1, "fail": 1, "pass_rate": 50.0}


def test_majority_and_tie_is_fail():
    tr = PinTracker()
    recs = feed(tr, ["PASS", "PASS", "FAIL", "NO_PIN", "PASS", "FAIL", "NO_PIN"])
    assert [r["verdict"] for r in recs] == ["PASS", "FAIL"]


def test_single_frame_flicker_ignored():
    tr = PinTracker(min_frames=2)
    assert feed(tr, ["PASS", "NO_PIN"]) == []
    assert tr.stats()["inspected"] == 0


def test_fail_cards_kept_and_reset():
    tr = PinTracker()
    recs = feed(tr, ["FAIL", "FAIL", "NO_PIN"])
    assert tr.fail_card(recs[0]["id"]) == b"png"
    tr.reset()
    assert tr.stats()["inspected"] == 0 and tr.fail_card(recs[0]["id"]) is None
