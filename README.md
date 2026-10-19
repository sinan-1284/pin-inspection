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

- **Raspberry Pi 5** (4 GB or 8 GB) running Raspberry Pi OS 64-bit, with the official 27 W power supply and an active cooler
- **USB webcam** (e.g. Logitech C920) on a blue USB 3 port, mounted side-on to the pin
- Backlight behind the pins — required for a clean silhouette
- Python 3.11+ (the Raspberry Pi OS default), in the project's own `venv/`

## Health notifications

The station reports its own health so problems are noticed without someone watching the screen:

- **Discord** — alerts to a personal Discord channel via a webhook: station started/stopped, camera disconnected, Pi overheating or under-voltage, disk filling up, fail-rate spike, no pins seen for a long time; plus a daily summary (inspected / pass / fail, uptime).
- **DDLC health dashboard** — periodic health messages on `health/<machine-id>` over MQTT, the same topic [`AI-DDLC/health_ddlc`](https://github.com/AI-DDLC/health_ddlc) uses, so the station shows up alongside the other DDLC machines.

The Discord webhook URL is a secret: it lives in a local `.env` file, never in git.

## Repository layout (planned)

```
pin-inspection/
├── README.md
├── PLAN.md              # Build plan and milestones
├── config.yaml          # Camera, segmentation, model and output settings
├── requirements.txt
├── .env.example         # Template for secrets (Discord webhook URL)
├── pin_inspection/      # Core pipeline (capture, segment, align, features, model)
│   └── health.py        # Health checks + Discord / MQTT notifications
├── tools/               # CLI: capture reference images, train, test on images
├── app/                 # Web dashboard (later milestone)
└── tests/               # Unit tests + sample images
```

## Reference

- [`AI-DDLC/machine_vision_pin`](https://github.com/AI-DDLC/machine_vision_pin) — original pin inspector prototype (CLI + MQTT)
- [`AI-DDLC/generic`](https://github.com/AI-DDLC/generic) — DDLC browser HMI pattern (live video, train/inspect modes, DDLC theme)
- [`AI-DDLC/health_ddlc`](https://github.com/AI-DDLC/health_ddlc) — DDLC machine health publisher (`health/<machine-id>` over MQTT)
