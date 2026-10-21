"""Live inspection station with a browser page for saving samples and training.

    venv/bin/python tools/station.py            # then open http://<pi>:8080
    venv/bin/python tools/station.py --port 9000

The camera is checked continuously. The page shows the latest PASS/FAIL card
and has buttons to save the current frame as a good / bad sample and to
retrain the good-pin model from data/good/.
"""
import argparse
import json
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
from pin_inspection.capture import Camera  # noqa: E402
from pin_inspection.config import load_config  # noqa: E402
from pin_inspection.inspection import check  # noqa: E402
from pin_inspection.overlay import annotate, verdict_card  # noqa: E402
from pin_inspection.pipeline import analyse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


class Station:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = load_model(cfg)
        self.lock = threading.Lock()
        self.frame = None
        self.card_png = None
        self.status = {"verdict": "STARTING", "checks": [], "failed": []}
        self.message = "Starting camera..."
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
        failed = [c[0] for c in checks if not c[3]]
        with self.lock:
            self.frame = frame
            self.card_png = png.tobytes()
            self.status = {"verdict": verdict, "failed": failed,
                           "checks": [[n, v, lim, ok] for n, v, lim, ok in checks],
                           "time": time.strftime("%H:%M:%S")}
        return verdict, failed, checks

    def save(self, label):
        with self.lock:
            frame, verdict = self.frame, self.status["verdict"]
        if frame is None or verdict == "NO_PIN":
            return "No pin in view — nothing saved"
        out = DATA / label
        out.mkdir(parents=True, exist_ok=True)
        path = out / time.strftime("%Y%m%d-%H%M%S.png")
        cv2.imwrite(str(path), frame)
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

    def run_camera(self):
        last = None
        while True:
            try:
                with Camera(self.cfg["camera"]) as cam:
                    self.message = "Camera running"
                    while True:
                        verdict, failed, checks = self.process(cam.read())
                        line = verdict + (f" ({', '.join(failed)})" if failed else "")
                        if line != last:  # only log changes, not every frame
                            vals = {n: v for n, v, _, _ in checks}
                            extra = (f"  len={vals.get('length')} shaft={vals.get('shaft_width')} "
                                     f"head={vals.get('head_width')} score={vals.get('model_score')}") if checks else ""
                            print(f"{time.strftime('%H:%M:%S')} {line}{extra}", flush=True)
                            last = line
            except RuntimeError as e:
                self.message = f"Camera error: {e} — retrying"
                print(self.message, flush=True)
                time.sleep(2)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pin Station</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f4f4f4;color:#222}
 header{padding:12px 16px;background:#222;color:#fff;font-size:18px}
 main{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
 #card{max-width:100%;max-height:80vh;border:1px solid #ccc;background:#fff}
 .side{flex:1;min-width:260px}
 button{font-size:17px;padding:12px 16px;margin:4px 4px 4px 0;border:0;border-radius:6px;cursor:pointer;color:#fff}
 .good{background:#0a0}.bad{background:#c22}.train{background:#246}
 #msg{margin:10px 0;font-weight:600}
 pre{background:#fff;border:1px solid #ccc;padding:8px;white-space:pre-wrap;font-size:12px;max-height:45vh;overflow:auto}
 input{font-size:15px;padding:8px;width:130px}
</style></head><body>
<header>Pin Station — <span id="counts"></span> — model: <span id="model"></span></header>
<main>
 <img id="card" alt="latest result">
 <div class="side">
  <button class="good" onclick="post('/save?label=good')">Save as GOOD</button><br>
  <input id="defect" value="bent" title="defect name">
  <button class="bad" onclick="post('/save?label=bad/'+encodeURIComponent(document.getElementById('defect').value))">Save as BAD</button><br>
  <button class="train" onclick="post('/train')">Train model</button>
  <div id="msg"></div>
  <pre id="out"></pre>
 </div>
</main>
<script>
async function post(u){document.getElementById('msg').textContent='...';
 const r=await fetch(u,{method:'POST'});const j=await r.json();
 document.getElementById('msg').textContent=j.message; if(j.train_output)document.getElementById('out').textContent=j.train_output;}
async function tick(){
 try{const s=await (await fetch('/status')).json();
  document.getElementById('counts').textContent=s.good+' good / '+s.bad+' bad samples';
  document.getElementById('model').textContent=s.model;
  document.getElementById('card').src='/card.png?t='+Date.now();}catch(e){}
 setTimeout(tick,700);}
tick();
</script></body></html>"""


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
            path = urlparse(self.path).path
            if path == "/":
                self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/card.png":
                with st.lock:
                    png = st.card_png
                if png is None:
                    self.send(503, b"", "text/plain")
                else:
                    self.send(200, png, "image/png")
            elif path == "/status":
                good, bad = st.counts()
                with st.lock:
                    s = dict(st.status)
                m = st.model
                s.update(good=good, bad=bad, message=st.message,
                         model=f"trained on {m.n_train}, threshold {m.threshold:.1f}" if m else "not trained")
                self.json(s)
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
