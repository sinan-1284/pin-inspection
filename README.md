# Pin Inspection

Machine vision quality control for **flat-head paper pins** on a production line. A backlit camera captures each pin's silhouette, the system measures its shape, and flags defective pins as **PASS / FAIL** in real time.

The approach follows DDLC's existing [`machine_vision_pin`](https://github.com/AI-DDLC/machine_vision_pin) prototype: classical computer vision with a one-class model trained only on good pins — no deep learning, no GPU, and no need to collect examples of every defect.

> **Status:** Milestone 0/1 in progress — camera capture, segmentation, alignment and shape measurement work on the real station. No PASS/FAIL model yet. See [PLAN.md](PLAN.md).

## Quick start (on the Pi)

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt

venv/bin/python tools/measure.py                       # measure the pin under the camera
venv/bin/python tools/measure.py --image frame.png     # measure a saved frame
venv/bin/python tools/measure.py --mm-per-px 0.18      # also report millimetres
venv/bin/python -m pytest -q tests                     # run tests (synthetic pins)
```

Each run saves to `captures/<timestamp>/`: the raw `frame.png`, an `annotated.png` close-up (outline, axis, head and tip), the `aligned_mask.png` (head left), a `width_profile.png` and the full `measurements.json`.

## Measurements

| Measurement | Meaning |
|---|---|
| `length` | Head end to tip, along the pin axis |
| `shaft_width` / `shaft_width_std` | Median shaft width (middle 20–70 % of the pin) and its variation |
| `head_width` | Widest point at the head end |
| `head_thickness` | Axial extent of the head (where it is wider than the shaft) |
| `head_offset` / `head_symmetry` | How far the head sits off the shaft axis; 1.0 = perfectly centred |
| `tip_length` / `tip_angle_deg` | Length of the tapered point and its included angle |
| `shaft_length` | Length between head and taper |
| `straightness_max_dev` / `_rms_dev` / `straightness_pct` | Deviation of the shaft centreline from a straight line (bent shaft) |
| `head_shaft_ratio`, `length_shaft_ratio` | Scale-free proportions |
| `area`, `perimeter` | Silhouette area and outline length |
| `pin_angle_deg`, `head_xy`, `tip_xy` | Where the pin lies in the frame |

Values are in camera pixels; set `measure.mm_per_px` in `config.yaml` (or pass `--mm-per-px`) to also get millimetres.

### Calibration

To get millimetres, measure something of known length in the same plane as the pin — e.g. a pin whose length you measured with calipers, or a ruler laid on the paper — and set `mm_per_px = real_length_mm / measured_length_px`. Recalibrate whenever the camera height or zoom changes.

## Current station

- Logitech C920 on `/dev/video0`, looking straight down at the pin on white paper, front-lit (no backlight yet)
- Captured at 2560×1472 uncompressed (YUYV, ~2 fps). A pin is about 140 px long and the shaft about 7 px wide at the current camera height
- Measured on a 4× upscaled crop for sub-pixel edges. Repeat measurements of the same pin agree to about ±0.25 px for length and shaft width, and ±0.75 px for head width
- To make the head features more reliable, move the camera closer (or zoom) so the pin fills more of the frame, and add a backlight for a clean silhouette

## How it will work

```
Camera frame → Threshold → Find pin contour → Align (head left) → Measure shape → Score vs. good pins → PASS / FAIL
```

1. **Capture** — a side-view camera with a backlight behind the pins, so each pin is a clean dark silhouette.
2. **Segment** — threshold the frame and pick out the pin's outline.
3. **Align** — rotate the pin so it's horizontal with the head on the left, then crop.
4. **Measure** — geometric features such as head/shaft width ratio, shaft straightness and head symmetry.
5. **Score** — compare the features to the distribution learned from 10–20 known-good pins (Mahalanobis distance). Too far from "normal" = FAIL.

## Defects it targets

| Defect | Detected by |
|---|---|
| Bent shaft | Shaft straightness |
| Wrong head/shaft proportions | Head-to-shaft width ratio |
| Misshapen or off-centre head | Head symmetry |
| Wrong length / missing head or tip | Total length, head width (planned) |

## Hardware

- **Raspberry Pi 5** (4 GB or 8 GB) running Raspberry Pi OS 64-bit, with the official 27 W power supply and an active cooler
- **USB webcam** (e.g. Logitech C920) on a blue USB 3 port, mounted side-on to the pin
- Backlight behind the pins — required for a clean silhouette
- Python 3.11+ (the Raspberry Pi OS default), in the project's own `venv/`

## Health notifications

The station reports its own health so problems are noticed without someone watching the screen:

- **Discord** — alerts to a personal Discord channel via a webhook: station started/stopped, camera disconnected, Pi overheating or under-voltage, disk filling up, fail-rate spike, no pins seen for a long time; plus a daily summary (inspected / pass / fail, uptime).
- **DDLC health dashboard** — periodic health messages on `health/<machine-id>` over MQTT, the same topic [`AI-DDLC/health_ddlc`](https://github.com/AI-DDLC/health_ddlc) uses, so the station shows up alongside the other DDLC machines.

The Discord webhook URL is a secret: it lives in a local `.env` file, never in git.

## Repository layout

```
pin-inspection/
├── README.md
├── PLAN.md              # Build plan and milestones
├── config.yaml          # Camera, segmentation, model and output settings
├── requirements.txt
├── .env.example         # Template for secrets (Discord webhook URL)
├── pin_inspection/      # Core pipeline
│   ├── capture.py       # USB webcam (V4L2), optional locked focus/exposure
│   ├── segment.py       # Find the pin, sub-pixel mask on an upscaled crop
│   ├── align.py         # PCA rotation, head on the left
│   ├── features.py      # Shape measurements
│   └── health.py        # (planned) Health checks + Discord / MQTT notifications
├── tools/
│   └── measure.py       # Measure the pin under the camera / in an image
├── app/                 # Web dashboard (later milestone)
└── tests/               # Unit tests + sample images
```

## Reference

- [`AI-DDLC/machine_vision_pin`](https://github.com/AI-DDLC/machine_vision_pin) — original pin inspector prototype (CLI + MQTT)
- [`AI-DDLC/generic`](https://github.com/AI-DDLC/generic) — DDLC browser HMI pattern (live video, train/inspect modes, DDLC theme)
- [`AI-DDLC/health_ddlc`](https://github.com/AI-DDLC/health_ddlc) — DDLC machine health publisher (`health/<machine-id>` over MQTT)
