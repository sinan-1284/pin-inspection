"""Frame -> measurements, shared by the CLI tools."""
from dataclasses import dataclass

from .align import Alignment, align
from .features import measure
from .segment import Segmentation, segment


@dataclass
class Analysis:
    seg: Segmentation
    al: Alignment
    res: dict


def analyse(frame, cfg):
    """Segment, align and measure the pin in `frame`; None if no pin is found."""
    seg = segment(frame, cfg["segment"])
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
    return Analysis(seg, al, res)
