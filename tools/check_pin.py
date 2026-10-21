"""PASS / FAIL inspection of the pin under the camera (or saved images).

    venv/bin/python tools/check_pin.py                    # one pin from the camera
    venv/bin/python tools/check_pin.py --loop             # keep inspecting, Ctrl-C to stop
    venv/bin/python tools/check_pin.py --save good        # inspect and keep the frame as a good sample
    venv/bin/python tools/check_pin.py --save bad/bent    # ... or as a labelled bad sample
    venv/bin/python tools/check_pin.py --image a.png b.png

Every result is appended to inspection_log.csv; the latest annotated card is
written to captures/last_inspection.png.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pin_inspection.capture import Camera  # noqa: E402
from pin_inspection.config import load_config  # noqa: E402
from pin_inspection.inspection import check  # noqa: E402
from pin_inspection.model import PinModel  # noqa: E402
from pin_inspection.overlay import annotate, verdict_card  # noqa: E402
from pin_inspection.pipeline import analyse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "inspection_log.csv"
LOG_FIELDS = ["time", "source", "verdict", "failed", "model_score", "length", "shaft_width",
              "head_width", "straightness_max_dev", "head_symmetry", "head_shaft_ratio", "tip_angle_deg"]


def load_model(cfg):
    path = ROOT / cfg["model"]["path"]
    if not path.exists():
        return None
    model = PinModel.load(path)
    if model.mm_per_px != cfg["measure"].get("mm_per_px"):
        print(f"WARNING: model was trained at mm_per_px={model.mm_per_px}, config is "
              f"{cfg['measure'].get('mm_per_px')} — retrain with tools/train.py")
    return model


def inspect_frame(frame, source, cfg, model):
    a = analyse(frame, cfg)
    if a is None:
        print(f"{source}: NO PIN")
        return "NO_PIN", None
    verdict, checks = check(a.res, cfg, model)
    failed = [c[0] for c in checks if not c[3]]

    print(f"{source}: {verdict}" + (f"   failed: {', '.join(failed)}" if failed else ""))
    for name, v, lim, ok in checks:
        print(f"    {'ok  ' if ok else 'FAIL'} {name:22s} {v:>9}   [{lim}]")
    if model is not None and failed:
        dev = sorted(model.contributions(a.res).items(), key=lambda kv: -abs(kv[1]))[:3]
        print("    most unusual vs. good pins: " + ", ".join(f"{k} {v:+.1f} sigma" for k, v in dev))

    lines = [(f"{name}: {v} [{lim}]", ok) for name, v, lim, ok in checks]
    card = verdict_card(annotate(a.frame, a.seg, a.al), verdict, lines)
    cv2.imwrite(str(ROOT / "captures" / "last_inspection.png"), card)

    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source, "verdict": verdict,
           "failed": " ".join(failed)}
    for name, v, _, _ in checks:
        if name in LOG_FIELDS:
            row[name] = v
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return verdict, card


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", nargs="+", help="inspect these images instead of the camera")
    ap.add_argument("--loop", action="store_true", help="inspect continuously from the camera")
    ap.add_argument("--save", metavar="LABEL", help="also save the frame to data/<LABEL>/ (good, bad/<defect>)")
    ap.add_argument("--no-model", action="store_true", help="tolerances only, ignore the trained model")
    args = ap.parse_args()

    cfg = load_config()
    model = None if args.no_model else load_model(cfg)
    if model is None and not args.no_model:
        print("(no trained model yet — checking tolerances only; see tools/train.py)")
    (ROOT / "captures").mkdir(exist_ok=True)

    if args.image:
        for p in args.image:
            frame = cv2.imread(p)
            if frame is None:
                print(f"{p}: cannot read")
                continue
            inspect_frame(frame, p, cfg, model)
        return

    with Camera(cfg["camera"]) as cam:
        while True:
            frame = cam.read()
            source = time.strftime("camera %H:%M:%S")
            verdict, _ = inspect_frame(frame, source, cfg, model)
            if args.save and verdict != "NO_PIN":
                out = ROOT / "data" / args.save
                out.mkdir(parents=True, exist_ok=True)
                path = out / time.strftime("%Y%m%d-%H%M%S.png")
                cv2.imwrite(str(path), frame)
                print(f"    saved {path.relative_to(ROOT)}")
            if not args.loop:
                break


if __name__ == "__main__":
    main()
