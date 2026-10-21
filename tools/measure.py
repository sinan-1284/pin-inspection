"""Measure the pin in front of the camera (or in an image file).

    venv/bin/python tools/measure.py                 # grab from camera
    venv/bin/python tools/measure.py --image x.jpg   # measure a saved frame
    venv/bin/python tools/measure.py --mm-per-px 0.1 # also report millimetres

Writes the raw frame, an annotated crop, the aligned mask and a JSON report
to captures/<timestamp>/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pin_inspection.align import align  # noqa: E402
from pin_inspection.capture import Camera  # noqa: E402
from pin_inspection.config import load_config  # noqa: E402
from pin_inspection.features import measure  # noqa: E402
from pin_inspection.segment import segment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def annotate(frame, seg, al, zoom=6):
    x, y, w, h = seg.roi
    crop = cv2.resize(frame[y:y + h, x:x + w], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(seg.mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(crop, contours, -1, (0, 200, 0), 1)
    f = zoom / seg.scale
    head = tuple(int(v * f) for v in al.head_xy)
    tip = tuple(int(v * f) for v in al.tip_xy)
    cv2.line(crop, head, tip, (255, 120, 0), 1)
    cv2.circle(crop, head, 6, (0, 0, 255), 2)
    cv2.putText(crop, "head", (head[0] + 10, head[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.circle(crop, tip, 6, (255, 0, 255), 2)
    cv2.putText(crop, "tip", (tip[0] + 10, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return crop


def profile_plot(profile, scale_label="px"):
    w = np.array(profile["width"])
    n, H = len(w), 200
    img = np.full((H, n, 3), 255, np.uint8)
    top = w.max() * 1.1
    pts = np.column_stack([np.arange(n), H - 1 - (w / top * (H - 20)).astype(int)])
    cv2.polylines(img, [pts.astype(np.int32)], False, (200, 80, 0), 1)
    cv2.putText(img, f"width ({scale_label}) head -> tip, max {w.max():.1f}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="measure this image instead of grabbing from the camera")
    ap.add_argument("--mm-per-px", type=float, help="scale for millimetre output (overrides config)")
    ap.add_argument("--config", default=ROOT / "config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mm_per_px:
        cfg["measure"]["mm_per_px"] = args.mm_per_px

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read {args.image}")
    else:
        with Camera(cfg["camera"]) as cam:
            frame = cam.read()

    out_dir = ROOT / "captures" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "frame.png"), frame)

    seg = segment(frame, cfg["segment"])
    if seg is None:
        sys.exit("No pin found in view")
    al = align(seg.mask)
    res = measure(al.mask, seg.scale, cfg["measure"])
    x, y, w, h = seg.roi
    res["image"] = {
        "frame_size": [frame.shape[1], frame.shape[0]],
        "pin_angle_deg": round(al.angle_deg, 2),
        "head_xy": [round(float(x + al.head_xy[0] / seg.scale), 1), round(float(y + al.head_xy[1] / seg.scale), 1)],
        "tip_xy": [round(float(x + al.tip_xy[0] / seg.scale), 1), round(float(y + al.tip_xy[1] / seg.scale), 1)],
    }

    cv2.imwrite(str(out_dir / "annotated.png"), annotate(frame, seg, al))
    cv2.imwrite(str(out_dir / "aligned_mask.png"), al.mask)
    cv2.imwrite(str(out_dir / "width_profile.png"), profile_plot(res["profile"]))
    (out_dir / "measurements.json").write_text(json.dumps(res, indent=2))

    print(f"Saved to {out_dir}")
    print(f"Pin angle: {res['image']['pin_angle_deg']} deg   head at {res['image']['head_xy']}   tip at {res['image']['tip_xy']}")
    mm = res.get("mm", {})
    for k, v in res["px"].items():
        extra = f"   {mm[k]:.3f} mm" if k in mm else ""
        print(f"  {k:24s} {v:10.2f} px{extra}")
    for k, v in res["ratios"].items():
        print(f"  {k:24s} {v:10}")
    print(f"  {'tip_angle_deg':24s} {res['tip_angle_deg']:>10}")


if __name__ == "__main__":
    main()
