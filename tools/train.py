"""Learn what a good pin looks like from the images in data/good/.

    venv/bin/python tools/check_pin.py --save good   # repeat for 10-20 good pins
    venv/bin/python tools/train.py

Writes models/model.json and prints each image's score plus tolerance
limits suggested by the spread of the good pins.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pin_inspection.config import load_config  # noqa: E402
from pin_inspection.model import PinModel, value  # noqa: E402
from pin_inspection.pipeline import analyse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--good", default=ROOT / "data" / "good", type=Path)
    args = ap.parse_args()

    cfg = load_config()
    paths = sorted(args.good.glob("*.png")) + sorted(args.good.glob("*.jpg"))
    results, used = [], []
    for p in paths:
        img = cv2.imread(str(p))
        a = None if img is None else analyse(img, cfg)
        if a is None:
            print(f"  skip {p.name}: no pin found")
            continue
        results.append(a.res)
        used.append(p)
    if len(results) < 3:
        sys.exit(f"Need at least 3 good pins in {args.good} (found {len(results)}); "
                 "10-20 is better. Capture them with: tools/check_pin.py --save good")
    if len(results) < 10:
        print(f"Only {len(results)} good pins — the model will work but 10-20 is recommended.\n")

    model = PinModel.fit(results, cfg["model"], cfg["measure"].get("mm_per_px"))
    model.save(ROOT / cfg["model"]["path"])

    print(f"{'image':28s} score")
    for p, r in zip(used, results):
        print(f"  {p.name:26s} {model.score(r):5.2f}")
    print(f"\nThreshold: {model.threshold:.2f}   ({len(results)} good pins, units {model.units})")
    print(f"Saved {cfg['model']['path']}\n")

    print("Good-pin spread and suggested tolerances (mean +/- max(4 sd, 3 x noise floor)):")
    print(f"  {'feature':22s} {'mean':>8s} {'sd':>7s}   suggested       current")
    floors = cfg["model"]["features"]
    for name, (lo, hi) in cfg["inspect"]["tolerances"].items():
        try:
            vals = np.array([value(r, name) for r in results], float)
        except KeyError:
            continue
        sd = vals.std(ddof=1)
        half = max(4 * sd, 3 * floors.get(name, 0))
        print(f"  {name:22s} {vals.mean():8.3f} {sd:7.3f}   "
              f"{vals.mean() - half:6.2f}..{vals.mean() + half:<6.2f}  {lo}..{hi}")


if __name__ == "__main__":
    main()
