"""Frame -> measurements, shared by the CLI tools."""
from dataclasses import dataclass

import cv2
import numpy as np

from .align import Alignment, align
from .features import measure
from .segment import Segmentation, segment


@dataclass
class Analysis:
    frame: np.ndarray       # the frame as measured (aspect-corrected); seg/al coords refer to it
    seg: Segmentation
    al: Alignment
    res: dict


def correct_aspect(frame, aspect):
    """Undo the camera's horizontal stretch so pixels are square."""
    if not aspect or aspect == 1:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(round(w / aspect)), h), interpolation=cv2.INTER_AREA)


def expected_length_px(cfg):
    """Nominal pin length in pixels (middle of the length tolerance), if calibrated."""
    mmpp = cfg["measure"].get("mm_per_px")
    lim = cfg.get("inspect", {}).get("tolerances", {}).get("length")
    if not mmpp or not lim or None in lim:
        return None
    return (lim[0] + lim[1]) / 2 / mmpp


def analyse(frame, cfg):
    """Segment, align and measure the pin in `frame`; None if no pin is found."""
    frame = correct_aspect(frame, cfg["camera"].get("pixel_aspect", 1))
    seg = segment(frame, dict(cfg["segment"], expected_length_px=expected_length_px(cfg)))
    if seg is None:
        return None
    al = align(seg.mask)
    res = measure(al.mask, seg.scale, cfg["measure"])
    x, y, _, _ = seg.roi
    res["image"] = {
        "frame_size": [frame.shape[1], frame.shape[0]],
        "pin_angle_deg": round(al.angle_deg, 2),
        "head_xy": [round(float(x + al.head_xy[0] / seg.scale), 1), round(float(y + al.head_xy[1] / seg.scale), 1)],
        "tip_xy": [round(float(x + al.tip_xy[0] / seg.scale), 1), round(float(y + al.tip_xy[1] / seg.scale), 1)],
    }
    return Analysis(frame, seg, al, res)
