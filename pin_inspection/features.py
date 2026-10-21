"""Shape measurements from an aligned pin mask (head left, tip right).

All measurements are returned in full-resolution camera pixels, and also in
millimetres when a scale (mm_per_px) is configured.
"""
import cv2
import numpy as np


def _first_run(flags):
    """Length of the run of True values starting at the first True."""
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return 0, 0
    start = idx[0]
    end = start
    while end + 1 < len(flags) and flags[end + 1]:
        end += 1
    return start, end - start + 1


def measure(aligned, scale, cfg):
    """Return a dict of measurements. `scale` is the mask upscale factor."""
    m = aligned > 0
    cols = np.flatnonzero(m.any(axis=0))
    m = m[:, cols[0]:cols[-1] + 1]
    n = m.shape[1]
    width = m.sum(axis=0).astype(float)
    rows = np.arange(m.shape[0])[:, None]
    top = np.where(m, rows, m.shape[0]).min(axis=0).astype(float)
    bottom = np.where(m, rows, -1).max(axis=0).astype(float)
    centre = (top + bottom) / 2

    # Shaft: the middle of the pin, clear of head and tip.
    shaft_w = float(np.median(width[int(n * 0.2):int(n * 0.7)]))

    # Head: wide region at the left end.
    head_start, head_len = _first_run(width[:max(1, n // 4)] > cfg["head_width_factor"] * shaft_w)
    head_end = head_start + head_len
    head_w = float(width[:max(1, n // 6)].max())

    # Tip: narrowing region at the right end.
    rev = width[::-1] < cfg["tip_width_factor"] * shaft_w
    tip_len = 0
    while tip_len < n // 2 and rev[tip_len]:
        tip_len += 1
    tip_start = n - tip_len

    # Straightness: deviation of the shaft centreline from a straight line.
    s0, s1 = max(head_end, int(n * 0.05)) + 1, min(tip_start, int(n * 0.95)) - 1
    xs = np.arange(s0, s1)
    coef = np.polyfit(xs, centre[s0:s1], 1)
    resid = centre[s0:s1] - np.polyval(coef, xs)
    max_dev = float(np.abs(resid).max())
    rms_dev = float(np.sqrt(np.mean(resid ** 2)))

    # Head symmetry about the shaft axis (extrapolated into the head).
    hx = np.arange(0, max(head_end, 1))
    axis_y = np.polyval(coef, hx)
    above = float((axis_y - top[hx]).max())
    below = float((bottom[hx] - axis_y).max())
    head_sym = min(above, below) / max(above, below) if max(above, below) > 0 else 0.0
    head_offset = (below - above) / 2

    # Tip angle from the taper length and the shaft half-width.
    tip_angle = float(np.degrees(2 * np.arctan2(shaft_w / 2, tip_len))) if tip_len else None

    contours, _ = cv2.findContours(aligned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)

    px = {
        "length": n / scale,
        "shaft_width": shaft_w / scale,
        "shaft_width_std": float(np.std(width[int(n * 0.2):int(n * 0.7)])) / scale,
        "head_width": head_w / scale,
        "head_thickness": head_len / scale,
        "head_offset": head_offset / scale,
        "tip_length": tip_len / scale,
        "shaft_length": (tip_start - head_end) / scale,
        "straightness_max_dev": max_dev / scale,
        "straightness_rms_dev": rms_dev / scale,
        "area": float(m.sum()) / scale ** 2,
        "perimeter": perimeter / scale,
    }
    out = {"px": {k: round(v, 2) for k, v in px.items()}}
    out["ratios"] = {
        "head_shaft_ratio": round(head_w / shaft_w, 3),
        "length_shaft_ratio": round(n / shaft_w, 1),
        "head_symmetry": round(head_sym, 3),
        "straightness_pct": round(100 * max_dev / n, 3),
    }
    out["tip_angle_deg"] = round(tip_angle, 1) if tip_angle else None
    mmpp = cfg.get("mm_per_px")
    if mmpp:
        out["mm"] = {k: round(v * mmpp, 3) for k, v in px.items() if k != "area"}
        out["mm"]["area_mm2"] = round(px["area"] * mmpp ** 2, 3)
    out["profile"] = {
        "width": (width / scale).round(2).tolist(),
        "centre": ((centre - np.polyval(coef, np.arange(n))) / scale).round(2).tolist(),
    }
    return out
