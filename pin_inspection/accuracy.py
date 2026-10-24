"""How good is the station? Decision accuracy on labelled samples and
measurement precision (repeatability) on pins lying still.

Precision is reported the usual QC way: the measurement spread compared to
the tolerance. For a two-sided limit that is 6 sigma / (max - min); for a
one-sided limit, 3 sigma / distance from the typical value to the limit.
Under 10 % is excellent, under 30 % acceptable, above that the gauge is too
noisy to trust near the limit.
"""
import statistics
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from .inspection import check
from .pipeline import analyse


def rate(pct):
    if pct is None:
        return None
    return "excellent" if pct < 10 else "acceptable" if pct < 30 else "poor"


def evaluate_samples(data_dir, cfg, model):
    """Inspect every labelled sample image; return a summary dict."""
    data_dir = Path(data_dir)
    items = []
    for truth in ("good", "bad"):
        root = data_dir / truth
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in (".png", ".jpg"):
                continue
            img = cv2.imread(str(p))
            a = None if img is None else analyse(img, cfg)
            verdict, checks = ("FAIL", []) if a is None else check(a.res, cfg, model)
            failed = [c[0] for c in checks if not c[3]] or (["no pin found"] if a is None else [])
            defect = "good" if truth == "good" else (str(p.parent.relative_to(root)) if p.parent != root else "other")
            items.append({"file": str(p.relative_to(data_dir)), "truth": truth, "defect": defect,
                          "verdict": verdict, "failed": failed,
                          "correct": (truth == "good") == (verdict == "PASS")})
    good = [i for i in items if i["truth"] == "good"]
    bad = [i for i in items if i["truth"] == "bad"]
    defects = {}
    for i in bad:
        d = defects.setdefault(i["defect"], {"n": 0, "caught": 0})
        d["n"] += 1
        d["caught"] += i["verdict"] == "FAIL"
    pct = lambda a, b: round(100 * a / b, 1) if b else None  # noqa: E731
    correct = sum(i["correct"] for i in items)
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n": len(items), "correct": correct, "accuracy_pct": pct(correct, len(items)),
        "good": {"n": len(good), "passed": sum(i["verdict"] == "PASS" for i in good),
                 "pct": pct(sum(i["verdict"] == "PASS" for i in good), len(good))},
        "bad": {"n": len(bad), "caught": sum(i["verdict"] == "FAIL" for i in bad),
                "pct": pct(sum(i["verdict"] == "FAIL" for i in bad), len(bad))},
        "defects": defects,
        "wrong": [i for i in items if not i["correct"]],
        "items": items,
        "model": bool(model),
    }


class Precision:
    """Spread of each measurement while a pin lies still (tracker state PIN)."""

    def __init__(self, min_frames=5, keep=20):
        self.min_frames = min_frames
        self.pins = deque(maxlen=keep)  # per finished pin: {check: (mean, sd, n)}
        self.cur = {}
        self.lock = threading.Lock()  # fed by the camera thread, read by the web server

    def feed(self, state, checks):
        with self.lock:
            self._feed(state, checks)

    def _feed(self, state, checks):
        if state == "PIN":
            for name, v, _, _ in checks:
                if isinstance(v, (int, float)):
                    self.cur.setdefault(name, []).append(float(v))
        elif state in ("NO_PIN", "OBSTRUCTED") and self.cur:
            stats = self._stats(self.cur)
            if stats:
                self.pins.append(stats)
            self.cur = {}

    def _stats(self, series):
        out = {}
        for name, v in series.items():
            if len(v) >= self.min_frames:
                out[name] = (statistics.fmean(v), statistics.stdev(v), len(v))
        return out

    def summary(self, limits):
        """Typical spread per check over recent still pins, plus the current pin."""
        with self.lock:
            current = self._stats(self.cur)
            pins = list(self.pins) + ([current] if current else [])
            cur_n = {k: len(v) for k, v in self.cur.items()}
        out = {}
        for name in limits:
            rows = [p[name] for p in pins if name in p]
            if not rows:
                continue
            sd = statistics.median(r[1] for r in rows)
            mean = statistics.fmean(r[0] for r in rows)
            lo, hi = limits[name]
            if lo is not None and hi is not None and hi > lo:
                pct = 100 * 6 * sd / (hi - lo)
            elif lo is not None or hi is not None:
                lim = lo if lo is not None else hi
                pct = 100 * 3 * sd / abs(mean - lim) if mean != lim else None
            else:
                pct = None
            cur = current.get(name)
            out[name] = {"sd": round(sd, 4), "pins": len(rows), "pct_of_tol": None if pct is None else round(pct, 1),
                         "rating": rate(pct), "current_sd": round(cur[1], 4) if cur else None,
                         "current_frames": cur[2] if cur else cur_n.get(name, 0)}
        return out
