# Pin Inspection — Build Plan

Goal: a reliable, operator-friendly inspection station for flat-head paper pins that flags bent, misshapen or wrongly-proportioned pins in real time, and reports results to the line.

The plan builds step by step, so each milestone is usable on its own. It reuses the proven ideas from [`AI-DDLC/machine_vision_pin`](https://github.com/AI-DDLC/machine_vision_pin) but rebuilds them cleanly with tests, so we understand and own every part.

---

## Milestone 0 — Raspberry Pi setup & sample data

- [ ] Flash **Raspberry Pi OS 64-bit (Lite or Desktop)**, enable SSH, set hostname (e.g. `pin-station-01`), connect to Wi-Fi/Ethernet
- [ ] Fit the **active cooler** and use the **official 27 W PSU** — under-powering or overheating makes the Pi throttle, which drops fps
- [ ] Plug the webcam into a **USB 3 (blue) port**; check it with `v4l2-ctl --list-devices` and `v4l2-ctl --list-ctrls`
- [ ] Project skeleton: `pin_inspection/` package, `tools/`, `tests/`, `config.yaml`, `requirements.txt`, own `venv/` (Python 3.11+, `opencv-python-headless` — no desktop GUI needed on the Pi)
- [ ] Mount camera + backlight; confirm a clean silhouette (pin dark, background bright)
- [ ] Lock camera focus and exposure (auto-exposure drifts and breaks thresholding)
- [ ] Check speed on the Pi: target resolution (start at 640×480, go up only if detail is needed) and frames per second for the full pipeline
- [ ] Collect a sample set:
  - 20+ **good** pins (different positions/angles in the field of view)
  - 10+ **bad** pins, labelled by defect: bent shaft, bad head, short/long, missing head
- [ ] Store samples under `tests/data/good/` and `tests/data/bad/<defect>/` (small, compressed)

**Done when:** we have labelled images of good and bad pins from the real setup.

## Milestone 1 — Core pipeline (offline, from images)

Port and clean up the reference pipeline, one module at a time, each with unit tests:

| Module | Job | Reference |
|---|---|---|
| `capture.py` | Camera wrapper with locked focus/exposure | `machine_vision_pin/capture.py` |
| `segment.py` | Otsu (or fixed) threshold → largest pin contour | `segment.py` |
| `align.py` | PCA rotation → horizontal, head on the left → crop | `align.py` |
| `features.py` | Shape measurements from the aligned mask | `features.py` |
| `model.py` | One-class Mahalanobis model, threshold from good-pin scores | `model.py` |

Improvements over the reference:

- [ ] **Decide the feature set from data.** The reference measures 6 features but only scores 3 (`head_shaft_ratio`, `shaft_straightness`, `head_symmetry`) — and its README still says 6. Evaluate all 6 on the sample set and keep the ones that separate good from bad.
- [ ] **Head-direction check** — make sure the head really ends up on the left (compare end widths), otherwise every feature is measured on the wrong end.
- [ ] **Save the model as JSON** (mean, inverse covariance, threshold, feature names) instead of a pickle — readable, diffable, and safe to load.
- [ ] Don't commit trained models or captured images to git (the reference repo has `model.pkl` committed despite ignoring it).

**Done when:** `tools/evaluate.py --good tests/data/good --bad tests/data/bad` prints a confusion matrix, and every bad sample is caught with no good samples rejected (or we know exactly why not).

## Milestone 2 — Train & live inspection (CLI)

- [ ] `tools/capture_images.py` — live preview, press `s` to save a good-pin image
- [ ] `tools/train.py` — fit the model from a folder of good images, print per-image scores and the threshold
- [ ] `tools/inspect.py` — live camera: overlay PASS/FAIL, score and features; `--image` mode for single files
- [ ] Score smoothing across a few frames so one noisy frame doesn't flip the result
- [ ] Log every inspection to CSV (timestamp, result, score, features)
- [ ] Handle "no pin in view" as its own state (not a FAIL)

**Done when:** an operator can train on good pins and run live inspection on the real station.

## Milestone 3 — Health monitoring & Discord notifications

`pin_inspection/health.py`, running inside the inspection service (so it knows camera and inspection state, not just hardware):

**Checks** (every ~30 s):

| Check | Source on the Pi | Alert when |
|---|---|---|
| Camera | frames arriving from the capture loop | no frame for 10 s / camera unplugged — and again when it recovers |
| CPU temperature | `/sys/class/thermal/thermal_zone0/temp` | above 80 °C (Pi 5 throttles at ~85 °C) |
| Power / throttling | `vcgencmd get_throttled` | under-voltage or throttling flag set |
| Disk | `psutil.disk_usage('/')` | above 90 % full (SD cards fill up with logs/images) |
| Memory / CPU load | `psutil` | sustained very high usage |
| Inspection | the inspector's own counters | fail rate above X % over the last N pins; no pin seen for N minutes during shift hours |
| Service | start-up / shutdown hooks | station started, stopped, or crashed and restarted |

**Discord** (personal channel, via webhook):

- [ ] Create a webhook in the channel (Channel settings → Integrations → Webhooks) and put its URL in `.env` as `DISCORD_WEBHOOK_URL` — never commit it
- [ ] Send **on state change only** (OK → problem, problem → recovered), with a cooldown, so a flapping sensor can't spam the channel
- [ ] Colour-coded embeds: 🔴 problem, 🟢 recovered, 🔵 info; include hostname, time and the measured value
- [ ] Optional snapshot image attached to camera / fail-rate alerts
- [ ] **Daily summary** at end of shift: pins inspected, pass / fail, top defect, uptime, max temperature
- [ ] `tools/test_discord.py` to send a test message when setting up a new station
- [ ] Never let a Discord outage break inspection: send from a background queue with timeouts and retries

**DDLC health dashboard:**

- [ ] Publish a health payload every 5 s to `health/<machine-id>` (same topic and `hardware` shape as `AI-DDLC/health_ddlc/health_pub.py`), extended with `temperature_c`, `throttled`, and a `machine_vision` section (camera ok, model loaded, fps, pass/fail counts)

**Dead-station detection:** a Pi that has crashed or lost power can't send its own alert. Cover this from outside — either a free heartbeat service (e.g. healthchecks.io, which can notify Discord when the pings stop) or the DDLC health dashboard flagging a machine that went quiet.

**Done when:** unplugging the camera, covering the cooler, or pulling the network each produces exactly one Discord alert, and one "recovered" message when fixed.

## Milestone 4 — Results out to the line

- [ ] Publish results over MQTT (same topic/payload shape as the reference: `<device_id>/pin_inspection`, see `machine_vision_pin/hmi.md`) so existing DDLC subscribers keep working
- [ ] Optional digital output for a reject mechanism (GPIO / Modbus TCP to a PLC)
- [ ] Count totals: inspected / pass / fail per shift

## Milestone 5 — Web dashboard (HMI)

Browser UI in the style of [`AI-DDLC/generic`](https://github.com/AI-DDLC/generic) (FastAPI + DDLC theme):

- [ ] Live video with PASS/FAIL overlay
- [ ] Train mode (password-protected): capture good pins and retrain from the browser
- [ ] Results view: pass/fail counters, recent fails with images, CSV export
- [ ] Settings: camera selection, exposure, sensitivity (`threshold_multiplier`)

## Milestone 6 — Hardening

- [ ] Run as a `systemd` service that starts on boot, restarts on crash, and survives camera unplug/replug
- [ ] Protect the SD card: rotate logs, cap saved images, or boot from an SSD (Pi 5 supports NVMe via a HAT)
- [ ] Measure accuracy on a larger batch (e.g. 200 pins) and tune the threshold from real false-reject / missed-defect rates
- [ ] Document station setup (camera distance, backlight, lens) so it can be rebuilt

## Later / optional

- Deep-learning fallback (e.g. anomalib PatchCore) for surface/texture defects the silhouette can't see
- Multiple pins per frame
- Real-world measurements in mm via camera calibration

---

## Open questions

1. **Line speed** — how many pins per second must be inspected? (Drives camera fps and whether frames need triggering.)
2. **Pin presentation** — do pins arrive one at a time in a fixed position, or loose/moving on a conveyor?
3. **Which defects matter most** to the customer, and what's the acceptable false-reject rate?
4. **Output** — does a reject signal need to drive a physical mechanism (PLC/GPIO), or is reporting enough?
5. **Camera** — which USB webcam exactly? It must allow manual focus and exposure (the C920 does).
6. **Notifications** — any quiet hours for Discord alerts, and should the daily summary go to the same channel?
