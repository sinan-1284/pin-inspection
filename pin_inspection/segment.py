"""Find the pin in a frame and return a clean, upscaled binary mask of it.

The station is currently front-lit (pin on white paper), so the pin is not a
uniform dark silhouette: the shiny shaft has dark edges and a bright centre
line. We therefore threshold the absolute difference from a blurred
background estimate, then close gaps and fill holes.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Segmentation:
    mask: np.ndarray        # uint8 0/255 mask of the pin, upscaled ROI
    roi: tuple              # (x, y, w, h) of the ROI in the full frame
    scale: int              # upscale factor of mask vs. full frame


def _largest_component(mask, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None, None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        return None, None
    return (lab == i).astype(np.uint8) * 255, stats[i, :4]


def _pin_like_component(mask, min_area, expected_len):
    """Pick the blob that looks most like a pin, not simply the biggest.

    Paper edges, creases and shadows can be far larger than the pin. With an
    expected pin length (pixels), keep blobs whose long side is 0.6-1.6x that
    and that are thin, and take the one closest to the expected length. Falls
    back to the largest blob if nothing qualifies (the caller then decides it
    is not a pin).
    """
    if not expected_len:
        return _largest_component(mask, min_area)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, best_err = None, None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        pts = np.column_stack(np.nonzero(lab == i)[::-1]).astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        long_side, short_side = max(w, h), min(w, h)
        if not (0.6 * expected_len <= long_side <= 1.6 * expected_len) or short_side > 0.25 * expected_len:
            continue
        err = abs(long_side - expected_len)
        if best is None or err < best_err:
            best, best_err = i, err
    if best is None:
        return _largest_component(mask, min_area)
    return (lab == best).astype(np.uint8) * 255, stats[best, :4]


def _fill_holes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
    return out


def _foreground(gray, blur):
    bg = cv2.medianBlur(gray, blur)
    diff = cv2.absdiff(gray, bg)
    _, m = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def segment(frame, cfg):
    """Return a Segmentation for the largest pin-like blob, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape
    mx, my = int(w * cfg["search_margin"]), int(h * cfg["search_margin"])
    search = gray[my:h - my, mx:w - mx]

    # Coarse pass at full resolution to locate the pin.
    m = _foreground(search, cfg["bg_blur"])
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    blob, box = _pin_like_component(m, cfg["min_area_px"], cfg.get("expected_length_px"))
    if blob is None:
        return None
    bx, by, bw, bh = box
    pad = max(10, cfg["bg_blur"] // 2)
    x0, y0 = max(0, mx + bx - pad), max(0, my + by - pad)
    x1, y1 = min(w, mx + bx + bw + pad), min(h, my + by + bh + pad)

    # Fine pass on an upscaled crop for sub-pixel edges. The background is
    # estimated on a wider crop so the blur window never sits on the pin only.
    s = cfg["upscale"]
    bpad = cfg["bg_blur"]
    X0, Y0 = max(0, x0 - bpad), max(0, y0 - bpad)
    X1, Y1 = min(w, x1 + bpad), min(h, y1 + bpad)
    wide = cv2.resize(gray[Y0:Y1, X0:X1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    blur = cfg["bg_blur"] * s | 1
    blur = min(blur, 255)  # cv2.medianBlur limit for uint8 with large kernels
    fine = _foreground(wide, blur)
    fine = fine[(y0 - Y0) * s:(y1 - Y0) * s, (x0 - X0) * s:(x1 - X0) * s]
    # Close across the bright specular streak/spot so the outline is solid
    # before filling, then open to drop single-pixel noise on the edges.
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s + 1, 2 * s + 1))
    opn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s + 1, s + 1))
    fine = cv2.morphologyEx(fine, cv2.MORPH_CLOSE, close)
    fine = cv2.morphologyEx(fine, cv2.MORPH_OPEN, opn)
    fine, _ = _largest_component(fine, cfg["min_area_px"] * s * s // 4)
    if fine is None:
        return None
    return Segmentation(_fill_holes(fine), (x0, y0, x1 - x0, y1 - y0), s)
