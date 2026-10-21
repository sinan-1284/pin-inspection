"""USB webcam capture (V4L2) with optional locked focus/exposure."""
import os
import time

import cv2


class Camera:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg["device"], cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cfg['device']}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg["fourcc"]))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
        if cfg.get("autofocus") is not None:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, int(cfg["autofocus"]))
        if cfg.get("focus") is not None:
            self.cap.set(cv2.CAP_PROP_FOCUS, cfg["focus"])
        if cfg.get("exposure") is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2 manual mode
            self.cap.set(cv2.CAP_PROP_EXPOSURE, cfg["exposure"])
        self._settle(cfg.get("settle_s", 0))

    def _settle(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.cap.read()

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera returned no frame")
        return frame

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def save_frame(path, frame):
    """Write a PNG and force it to disk, so a power cut right after a click
    can't leave an empty file (the SD card write cache is otherwise lost)."""
    ok, png = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(png.tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
