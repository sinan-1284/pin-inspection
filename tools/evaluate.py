"""Run inspection over labelled samples and print a confusion matrix.

    venv/bin/python tools/evaluate.py     # uses data/good/ and data/bad/<defect>/
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pin_inspection.config import load_config  # noqa: E402
from pin_inspection.inspection import check  # noqa: E402
from pin_inspection.model import PinModel  # noqa: E402
from pin_inspection.pipeline import analyse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg = load_config()
    mp = ROOT / cfg["model"]["path"]
    model = PinModel.load(mp) if mp.exists() else None
    print(f"Model: {'loaded' if model else 'none (tolerances only)'}\n")
    counts = {("good", "PASS"): 0, ("good", "FAIL"): 0, ("bad", "PASS"): 0, ("bad", "FAIL"): 0}
    for truth in ("good", "bad"):
        for p in sorted((ROOT / "data" / truth).rglob("*")):
            if p.suffix.lower() not in (".png", ".jpg"):
                continue
            a = analyse(cv2.imread(str(p)), cfg)
            verdict, checks = ("FAIL", []) if a is None else check(a.res, cfg, model)
            failed = [c[0] for c in checks if not c[3]] or (["no pin"] if a is None else [])
            counts[(truth, verdict)] += 1
            wrong = (truth == "good") != (verdict == "PASS")
            print(f"  {'WRONG' if wrong else 'ok   '} {p.relative_to(ROOT / 'data')}: {verdict} {' '.join(failed)}")
    g, b = counts[("good", "PASS")] + counts[("good", "FAIL")], counts[("bad", "PASS")] + counts[("bad", "FAIL")]
    print(f"\n              predicted PASS  predicted FAIL")
    print(f"  good ({g:3d})   {counts[('good', 'PASS')]:14d}  {counts[('good', 'FAIL')]:14d}   <- FAIL here = false reject")
    print(f"  bad  ({b:3d})   {counts[('bad', 'PASS')]:14d}  {counts[('bad', 'FAIL')]:14d}   <- PASS here = missed defect")


if __name__ == "__main__":
    main()
