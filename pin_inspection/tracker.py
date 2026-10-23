"""Turn the per-frame verdicts of the live camera into one result per pin.

The station checks every frame, so a single pin under the camera produces
many verdicts. A pin starts when something appears after NO_PIN and ends when
the view is empty again; its result is the majority verdict over its frames
(a tie counts as FAIL, the safe side).
"""
import time
from collections import Counter, deque


class PinTracker:
    def __init__(self, min_frames=2, keep_recent=50, keep_fails=12):
        self.min_frames = min_frames
        self.recent = deque(maxlen=keep_recent)   # committed pins, newest last
        self.fails = deque(maxlen=keep_fails)     # (id, card_png) of recent FAILs
        self.counts = Counter()                   # PASS / FAIL totals
        self.started = time.time()
        self.next_id = 1
        self.cur = None

    def update(self, verdict, failed, checks, card_png=None, now=None):
        """Feed one frame's result; returns the committed pin record when a pin ends."""
        now = time.time() if now is None else now
        if verdict == "NO_PIN":
            rec = self._commit(now) if self.cur else None
            self.cur = None
            return rec
        if self.cur is None:
            self.cur = {"start": now, "votes": Counter(), "reasons": Counter(), "last": {}, "fail_card": None}
        c = self.cur
        c["votes"][verdict] += 1
        c["reasons"].update(failed)
        c["last"][verdict] = {n: v for n, v, _, _ in checks}
        if verdict == "FAIL" and card_png is not None:
            c["fail_card"] = card_png
        return None

    def _commit(self, now):
        c = self.cur
        frames = sum(c["votes"].values())
        if frames < self.min_frames:
            return None  # a flicker, not a pin
        verdict = "PASS" if c["votes"]["PASS"] > c["votes"]["FAIL"] else "FAIL"
        vals = c["last"].get(verdict, {})
        rec = {
            "id": self.next_id,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c["start"])),
            "verdict": verdict,
            "failed": [r for r, _ in c["reasons"].most_common()] if verdict == "FAIL" else [],
            "frames": frames,
            "seconds": round(now - c["start"], 1),
            "model_score": vals.get("model_score"),
            "length": vals.get("length"),
        }
        self.next_id += 1
        self.counts[verdict] += 1
        self.recent.append(rec)
        if verdict == "FAIL" and c["fail_card"] is not None:
            self.fails.append((rec["id"], c["fail_card"]))
        return rec

    def fail_card(self, pin_id):
        for i, png in self.fails:
            if i == pin_id:
                return png
        return None

    def reset(self):
        self.recent.clear()
        self.fails.clear()
        self.counts.clear()
        self.started = time.time()
        self.cur = None

    def stats(self):
        total = self.counts["PASS"] + self.counts["FAIL"]
        return {
            "inspected": total,
            "pass": self.counts["PASS"],
            "fail": self.counts["FAIL"],
            "pass_rate": round(100 * self.counts["PASS"] / total, 1) if total else None,
            "since": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)),
        }
