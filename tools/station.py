"""Live inspection station: camera loop + browser dashboard.

    venv/bin/python tools/station.py            # then open http://<pi>:8080
    venv/bin/python tools/station.py --port 9000

Each pin is inspected continuously and recorded once, when it has settled
under the camera. The dashboard (app/dashboard.html) shows the live verdict,
today's counts, measurement trends, recent inspections and Pi health, and
has buttons to save the current frame as a good / bad sample and to retrain.
"""
import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from check_pin import load_model  # noqa: E402
from pin_inspection import health  # noqa: E402
from pin_inspection.capture import Camera, save_frame  # noqa: E402
from pin_inspection.config import load_config  # noqa: E402
from pin_inspection.inspection import check  # noqa: E402
from pin_inspection.overlay import annotate  # noqa: E402
from pin_inspection.pipeline import analyse  # noqa: E402
from pin_inspection.tracker import PinTracker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PAGE = ROOT / "app" / "dashboard.html"
LOG = ROOT / "inspection_log_station.csv"
THUMBS = ROOT / "captures" / "inspections"
CHECKS = ["length", "shaft_width", "head_width", "straightness_max_dev", "head_symmetry",
          "head_shaft_ratio", "tip_angle_deg", "model_score"]
HISTORY = 500


def jpeg(img, quality=80):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


class Station:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = load_model(cfg)
        self.tracker = PinTracker(cfg)
        self.lock = threading.Lock()
        self.frame = None
        self.live_jpg = None
        self.camera_jpg = None
        self.live = {"state": "STARTING", "verdict": None, "checks": [], "failed": []}
        self.message = "Starting camera..."
        self.camera_ok = False
        self.fps = 0.0
        self.train_output = ""
        self.busy = False
        self.history = deque(maxlen=HISTORY)
        self._health, self._health_t = {}, 0
        self._load_history()

    # ---- inspection records -------------------------------------------------
    def _load_history(self):
        if not LOG.exists():
            return
        with open(LOG, newline="") as f:
            for row in csv.DictReader(f):
                self.history.append(self._row_to_record(row))

    @staticmethod
    def _row_to_record(row):
        vals = {}
        for k in CHECKS:
            try:
                vals[k] = float(row[k]) if row.get(k) not in (None, "") else None
            except ValueError:
                vals[k] = None
        return {"id": int(row["id"]), "time": row["time"], "verdict": row["verdict"],
                "failed": row["failed"].split() if row["failed"] else [], "values": vals,
                "thumb": row.get("thumb") or None}

    def record(self, verdict, checks, failed, crop):
        rid = (self.history[-1]["id"] + 1) if self.history else 1
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        THUMBS.mkdir(parents=True, exist_ok=True)
        thumb = f"{rid:06d}.jpg"
        s = 240 / max(crop.shape[:2])
        (THUMBS / thumb).write_bytes(jpeg(cv2.resize(crop, None, fx=s, fy=s), 85))
        vals = {n: v for n, v, _, _ in checks}
        row = {"id": rid, "time": t, "verdict": verdict, "failed": " ".join(failed), "thumb": thumb,
               **{k: vals.get(k, "") for k in CHECKS}}
        new = not LOG.exists()
        with open(LOG, "a", newline="") as f:
            w = csv.DictWriter(f, ["id", "time", "verdict", "failed", "thumb", *CHECKS])
            if new:
                w.writeheader()
            w.writerow(row)
        rec = self._row_to_record({k: str(v) for k, v in row.items()})
        with self.lock:
            self.history.append(rec)
        print(f"#{rid} {t} {verdict}" + (f" ({', '.join(failed)})" if failed else "") +
              f"  len={vals.get('length')} shaft={vals.get('shaft_width')} score={vals.get('model_score')}",
              flush=True)

    # ---- per frame ----------------------------------------------------------
    def process(self, frame):
        a = analyse(frame, self.cfg)
        state, count_now = self.tracker.update(a.res if a else None)
        checks, failed, verdict = [], [], None
        view = a.frame if a else frame
        small = cv2.resize(view, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        if a is not None and state != "OBSTRUCTED":
            verdict, checks = check(a.res, self.cfg, self.model)
            failed = [c[0] for c in checks if not c[3]]
            crop = annotate(a.frame, a.seg, a.al)
            x, y, w, h = a.seg.roi
            cv2.rectangle(small, (x // 4 - 6, y // 4 - 6), ((x + w) // 4 + 6, (y + h) // 4 + 6),
                          (0, 170, 0) if verdict == "PASS" else (0, 0, 220), 2)
        else:
            crop = None
            if a is not None:  # OBSTRUCTED: box what was found so the operator can clear it
                x, y, w, h = a.seg.roi
                cv2.rectangle(small, (x // 4 - 6, y // 4 - 6), ((x + w) // 4 + 6, (y + h) // 4 + 6),
                              (0, 140, 255), 2)
        if count_now:
            self.record(verdict, checks, failed, crop)
        with self.lock:
            self.frame = frame
            self.camera_jpg = jpeg(small, 70)
            self.live_jpg = jpeg(crop if crop is not None else small, 85)
            self.live = {"state": state, "verdict": verdict, "failed": failed,
                         "checks": [[n, v, lim, ok] for n, v, lim, ok in checks],
                         "angle_deg": a.res["image"]["pin_angle_deg"] if a else None,
                         "found": {"length_mm": (a.res.get("mm") or {}).get("length"),
                                   "shaft_mm": (a.res.get("mm") or {}).get("shaft_width"),
                                   "roi": [int(v) for v in a.seg.roi]} if a else None,
                         "time": time.strftime("%H:%M:%S")}

    def run_camera(self):
        while True:
            try:
                with Camera(self.cfg["camera"]) as cam:
                    self.camera_ok, self.message = True, "Camera running"
                    t0, n = time.time(), 0
                    while True:
                        self.process(cam.read())
                        n += 1
                        if time.time() - t0 >= 5:
                            self.fps, t0, n = n / (time.time() - t0), time.time(), 0
            except RuntimeError as e:
                self.camera_ok, self.message = False, f"Camera error: {e} — retrying"
                print(self.message, flush=True)
                time.sleep(2)

    # ---- buttons ------------------------------------------------------------
    def counts(self):
        good = len(list((DATA / "good").glob("*.png")))
        bad = {}
        if (DATA / "bad").exists():
            for p in (DATA / "bad").rglob("*.png"):
                k = str(p.parent.relative_to(DATA / "bad"))
                bad[k] = bad.get(k, 0) + 1
        return good, bad

    def save(self, label):
        with self.lock:
            frame, state = self.frame, self.live["state"]
        if frame is None or state in ("NO_PIN", "OBSTRUCTED", "STARTING"):
            return "No pin in view — nothing saved"
        out = DATA / label
        out.mkdir(parents=True, exist_ok=True)
        path = out / time.strftime("%Y%m%d-%H%M%S.png")
        save_frame(path, frame)
        msg = f"Saved {path.relative_to(ROOT)} (live result {self.live['verdict']})"
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

    # ---- API payloads -------------------------------------------------------
    def health(self):
        if time.time() - self._health_t > 5:
            self._health, self._health_t = health.read(str(ROOT)), time.time()
        return self._health

    def status(self):
        good, bad = self.counts()
        today = time.strftime("%Y-%m-%d")
        with self.lock:
            live = dict(self.live)
            hist = [r for r in self.history if r["time"].startswith(today)]
        m = self.model
        return {
            "live": live,
            "today": {"inspected": len(hist), "pass": sum(r["verdict"] == "PASS" for r in hist),
                      "fail": sum(r["verdict"] == "FAIL" for r in hist),
                      "last_id": self.history[-1]["id"] if self.history else 0},
            "samples": {"good": good, "bad": bad},
            "model": {"n_train": m.n_train, "threshold": round(m.threshold, 2)} if m else None,
            "camera": {"ok": self.camera_ok, "message": self.message, "fps": round(self.fps, 2)},
            "health": self.health(),
            "busy": self.busy,
        }

    def config(self):
        tol = self.cfg["inspect"]["tolerances"]
        limits = {k: v for k, v in tol.items()}
        if self.model:
            limits["model_score"] = [None, round(self.model.threshold, 2)]
        units = {k: ("mm" if k in ("length", "shaft_width", "head_width", "straightness_max_dev") else
                     "°" if k == "tip_angle_deg" else "") for k in CHECKS}
        return {"limits": limits, "units": units, "checks": CHECKS,
                "mm_per_px": self.cfg["measure"].get("mm_per_px")}


def make_handler(st):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype):
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser dropped a superseded image request

        def json(self, obj):
            self.send(200, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            path = u.path
            if path == "/":
                self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path in ("/live.jpg", "/camera.jpg"):
                with st.lock:
                    img = st.live_jpg if path == "/live.jpg" else st.camera_jpg
                self.send(200, img, "image/jpeg") if img else self.send(503, b"", "text/plain")
            elif path.startswith("/thumb/"):
                name = path.rsplit("/", 1)[1]
                p = THUMBS / name
                if name.endswith(".jpg") and name[:-4].isdigit() and p.exists():
                    self.send(200, p.read_bytes(), "image/jpeg")
                else:
                    self.send(404, b"", "text/plain")
            elif path == "/api/status":
                self.json(st.status())
            elif path == "/api/config":
                self.json(st.config())
            elif path == "/api/history":
                n = int(parse_qs(u.query).get("n", ["100"])[0])
                with st.lock:
                    self.json(list(st.history)[-n:])
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/api/save":
                label = parse_qs(u.query).get("label", ["good"])[0].strip("/")
                parts = label.split("/")
                if parts[0] not in ("good", "bad") or len(parts) > 2 or \
                        any(not p.replace("_", "").replace("-", "").isalnum() for p in parts):
                    return self.json({"message": f"Bad label {label!r}"})
                self.json({"message": st.save(label)})
            elif u.path == "/api/train":
                msg = st.train()
                self.json({"message": msg, "train_output": st.train_output})
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
