"""Live inspection station with a web dashboard.

    venv/bin/python tools/station.py            # then open http://<pi>:8080
    venv/bin/python tools/station.py --port 9000

The camera is checked continuously. The dashboard (app/dashboard.html) shows
the live PASS/FAIL card, pass/fail counters, recent pins and recent fail
images, and has buttons to save the current frame as a good / bad sample, to
retrain the good-pin model from data/good/ and to reset the counters. Every
finished pin is appended to inspection_log_station.csv.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from check_pin import load_model  # noqa: E402
from pin_inspection.capture import Camera, save_frame  # noqa: E402
from pin_inspection.config import load_config  # noqa: E402
from pin_inspection.inspection import check  # noqa: E402
from pin_inspection.overlay import annotate, verdict_card  # noqa: E402
from pin_inspection.pipeline import analyse  # noqa: E402
from pin_inspection.tracker import PinTracker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
APP = ROOT / "app"
LOG = ROOT / "inspection_log_station.csv"
LOG_FIELDS = ["id", "time", "verdict", "failed", "frames", "seconds", "model_score", "length"]


class Station:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = load_model(cfg)
        self.lock = threading.Lock()
        self.frame = None
        self.card_png = None
        self.status = {"verdict": "STARTING", "checks": [], "failed": []}
        self.tracker = PinTracker(min_frames=cfg.get("station", {}).get("min_frames", 2))
        self.camera_ok = False
        self.message = "Starting camera..."
        self.fps = None
        self.train_output = ""
        self.busy = False

    def counts(self):
        good = len(list((DATA / "good").glob("*.png")))
        bad = len(list((DATA / "bad").rglob("*.png"))) if (DATA / "bad").exists() else 0
        return good, bad

    def process(self, frame):
        a = analyse(frame, self.cfg)
        if a is None:
            verdict, checks = "NO_PIN", []
            img = cv2.resize(frame, None, fx=0.3, fy=0.3)
            card = verdict_card(img, "NO PIN", [("Place a pin under the camera", True)])
        else:
            verdict, checks = check(a.res, self.cfg, self.model)
            lines = [(f"{n}: {v} [{lim}]", ok) for n, v, lim, ok in checks]
            card = verdict_card(annotate(a.frame, a.seg, a.al), verdict, lines)
        ok, png = cv2.imencode(".png", card)
        png = png.tobytes()
        failed = [c[0] for c in checks if not c[3]]
        with self.lock:
            self.frame = frame
            self.card_png = png
            self.status = {"verdict": verdict, "failed": failed,
                           "checks": [[n, v, lim, ok] for n, v, lim, ok in checks],
                           "time": time.strftime("%H:%M:%S")}
            rec = self.tracker.update(verdict, failed, checks, png)
        if rec:
            self.log(rec)
        return verdict, failed, checks

    def log(self, rec):
        print(f"{rec['time'][11:]} pin #{rec['id']}: {rec['verdict']}"
              + (f" ({', '.join(rec['failed'])})" if rec["failed"] else ""), flush=True)
        new = not LOG.exists()
        with open(LOG, "a", newline="") as f:
            w = csv.DictWriter(f, LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow({**rec, "failed": " ".join(rec["failed"])})

    def save(self, label):
        with self.lock:
            frame, verdict = self.frame, self.status["verdict"]
        if frame is None or verdict == "NO_PIN":
            return "No pin in view — nothing saved"
        out = DATA / label
        out.mkdir(parents=True, exist_ok=True)
        path = out / time.strftime("%Y%m%d-%H%M%S.png")
        n = 1
        while path.exists():  # two saves in the same second
            path = out / time.strftime(f"%Y%m%d-%H%M%S-{n}.png")
            n += 1
        save_frame(path, frame)
        good, bad = self.counts()
        msg = f"Saved {path.relative_to(ROOT)} (was {verdict}) — {good} good, {bad} bad samples"
        print(f"*** {msg}", flush=True)
        return msg

    def train(self):
        if self.busy:
            return "Training already running"
        self.busy = True
        try:
            r = subprocess.run([sys.executable, str(ROOT / "tools" / "train.py")],
                               capture_output=True, text=True, cwd=ROOT)
            self.train_output = (r.stdout + r.stderr).strip()
            print("*** TRAIN\n" + self.train_output, flush=True)
            if r.returncode == 0:
                self.model = load_model(self.cfg)
                return "Model trained and loaded"
            return "Training failed — see output"
        finally:
            self.busy = False

    def reset(self):
        with self.lock:
            self.tracker.reset()
        return "Counters reset"

    def snapshot(self):
        good, bad = self.counts()
        m = self.model
        with self.lock:
            tr = self.tracker
            return {
                "live": dict(self.status),
                "stats": tr.stats(),
                "recent": list(reversed(tr.recent))[:20],
                "fails": [{"id": i, "failed": next((r["failed"] for r in tr.recent if r["id"] == i), [])}
                          for i, _ in reversed(tr.fails)],
                "camera": {"ok": self.camera_ok, "message": self.message, "fps": self.fps},
                "model": f"Model: {m.n_train} good pins, threshold {m.threshold:.1f}" if m else None,
                "samples": {"good": good, "bad": bad},
            }

    def run_camera(self):
        last = None
        while True:
            try:
                with Camera(self.cfg["camera"]) as cam:
                    self.camera_ok, self.message = True, "Camera running"
                    t_prev = time.time()
                    while True:
                        verdict, failed, checks = self.process(cam.read())
                        now = time.time()
                        dt, t_prev = now - t_prev, now
                        if dt > 0:
                            f = 1 / dt
                            self.fps = round(f if self.fps is None else 0.8 * self.fps + 0.2 * f, 1)
                        line = verdict + (f" ({', '.join(failed)})" if failed else "")
                        if line != last:  # only log changes, not every frame
                            vals = {n: v for n, v, _, _ in checks}
                            extra = (f"  len={vals.get('length')} shaft={vals.get('shaft_width')} "
                                     f"head={vals.get('head_width')} score={vals.get('model_score')}") if checks else ""
                            print(f"{time.strftime('%H:%M:%S')} {line}{extra}", flush=True)
                            last = line
            except Exception as e:  # keep the station alive and show the problem on the dashboard
                self.camera_ok, self.fps = False, None
                self.message = f"Camera error: {e} — retrying"
                print(self.message, flush=True)
                time.sleep(2)


def make_handler(st):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype, extra=None):
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser dropped a superseded image request

        def json(self, obj):
            self.send(200, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send(200, (APP / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/logo.svg":
                self.send(200, (APP / "ddlc-logo.svg").read_bytes(), "image/svg+xml")
            elif path == "/card.png":
                with st.lock:
                    png = st.card_png
                if png is None:
                    self.send(503, b"", "text/plain")
                else:
                    self.send(200, png, "image/png")
            elif m := re.fullmatch(r"/fail/(\d+)\.png", path):
                with st.lock:
                    png = st.tracker.fail_card(int(m.group(1)))
                if png is None:
                    self.send(404, b"not found", "text/plain")
                else:
                    self.send(200, png, "image/png")
            elif path == "/log.csv":
                body = LOG.read_bytes() if LOG.exists() else (",".join(LOG_FIELDS) + "\n").encode()
                self.send(200, body, "text/csv",
                          {"Content-Disposition": 'attachment; filename="inspection_log_station.csv"'})
            elif path == "/status":
                self.json(st.snapshot())
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/save":
                label = parse_qs(u.query).get("label", ["good"])[0].strip("/")
                parts = label.split("/")
                if parts[0] not in ("good", "bad") or any(not p.replace("_", "").replace("-", "").isalnum() for p in parts):
                    return self.json({"message": f"Bad label {label!r}"})
                self.json({"message": st.save(label)})
            elif u.path == "/train":
                msg = st.train()
                self.json({"message": msg, "train_output": st.train_output})
            elif u.path == "/reset":
                self.json({"message": st.reset()})
            else:
                self.send(404, b"not found", "text/plain")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    st = Station(load_config())
    threading.Thread(target=st.run_camera, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(st))
    print(f"Pin station on http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
