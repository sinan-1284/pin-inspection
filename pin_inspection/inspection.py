"""PASS / FAIL decision for one measured pin."""
from .model import value


def check(res, cfg, model=None):
    """Return (verdict, checks); each check is (name, value, limits_text, ok)."""
    checks = []
    unit = "mm" if "mm" in res else "px"
    for name, (lo, hi) in cfg["inspect"]["tolerances"].items():
        try:
            v = value(res, name)
        except KeyError:
            continue  # e.g. mm limits while uncalibrated
        if unit == "px" and name in res["px"]:
            continue  # tolerances are written in mm
        ok = v is not None and (lo is None or v >= lo) and (hi is None or v <= hi)
        lim = f"{'' if lo is None else lo}..{'' if hi is None else hi}"
        checks.append((name, v, lim, ok))
    if model is not None:
        s = model.score(res)
        checks.append(("model_score", round(s, 2), f"..{model.threshold:.1f}", s <= model.threshold))
    verdict = "PASS" if checks and all(c[3] for c in checks) else "FAIL"
    return verdict, checks
