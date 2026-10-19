# Pin Inspection

Machine vision quality control for **flat-head paper pins** on a production line. A backlit camera captures each pin's silhouette, the system measures its shape, and flags defective pins as **PASS / FAIL** in real time.

The approach follows DDLC's existing [`machine_vision_pin`](https://github.com/AI-DDLC/machine_vision_pin) prototype: classical computer vision with a one-class model trained only on good pins — no deep learning, no GPU, and no need to collect examples of every defect.

> **Status:** planning. No code yet — see [PLAN.md](PLAN.md) for the build plan and milestones.

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

- USB camera (e.g. Logitech C920), mounted side-on to the pin
- Backlight behind the pins — required for a clean silhouette
- Linux PC (Ubuntu 22.04+), Python 3.12

## Repository layout (planned)

```
pin-inspection/
├── README.md
├── PLAN.md              # Build plan and milestones
├── config.yaml          # Camera, segmentation, model and output settings
├── requirements.txt
├── pin_inspection/      # Core pipeline (capture, segment, align, features, model)
├── tools/               # CLI: capture reference images, train, test on images
├── app/                 # Web dashboard (later milestone)
└── tests/               # Unit tests + sample images
```

## Reference

- [`AI-DDLC/machine_vision_pin`](https://github.com/AI-DDLC/machine_vision_pin) — original pin inspector prototype (CLI + MQTT)
- [`AI-DDLC/generic`](https://github.com/AI-DDLC/generic) — DDLC browser HMI pattern (live video, train/inspect modes, DDLC theme)
