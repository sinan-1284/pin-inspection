"""Frame stream -> one inspection per pin."""
from pin_inspection.config import load_config
from pin_inspection.tracker import PinTracker

CFG = load_config()


def pin(length=22.0, shaft=1.0):
    return {"px": {"length": length / 0.1554}, "mm": {"length": length, "shaft_width": shaft}}


def run(frames):
    t = PinTracker(CFG)
    return [t.update(f) for f in frames]


def test_counts_once_after_settling():
    out = run([pin(), pin(22.05), pin(21.98), pin(), pin(), pin()])
    assert [c for _, c in out] == [False, False, True, False, False, False]
    assert out[0][0] == "SETTLING" and out[3][0] == "PIN"


def test_next_pin_needs_clear_view():
    out = run([pin()] * 3 + [None, None] + [pin(21.5)] * 3)
    assert sum(c for _, c in out) == 2


def test_brief_gap_does_not_double_count():
    out = run([pin()] * 3 + [None] + [pin()] * 3)  # one empty frame < clear_frames
    assert sum(c for _, c in out) == 1


def test_moving_pin_is_not_counted():
    out = run([pin(20), pin(21), pin(22), pin(23), pin(20.5)])
    assert not any(c for _, c in out)


def test_hand_or_paper_edge_is_obstructed():
    state, counted = run([pin(length=90, shaft=4)])[0]
    assert state == "OBSTRUCTED" and not counted
