"""Annotated close-up images of the pin."""
import cv2
import numpy as np

GREEN, RED, GREY = (0, 170, 0), (0, 0, 220), (90, 90, 90)


def annotate(frame, seg, al, zoom=6):
    x, y, w, h = seg.roi
    crop = cv2.resize(frame[y:y + h, x:x + w], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(seg.mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(crop, contours, -1, (0, 200, 0), 1)
    f = zoom / seg.scale
    head = tuple(int(v * f) for v in al.head_xy)
    tip = tuple(int(v * f) for v in al.tip_xy)
    cv2.line(crop, head, tip, (255, 120, 0), 1)
    cv2.circle(crop, head, 6, (0, 0, 255), 2)
    cv2.putText(crop, "head", (head[0] + 10, head[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.circle(crop, tip, 6, (255, 0, 255), 2)
    cv2.putText(crop, "tip", (tip[0] + 10, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return crop


def verdict_card(crop, verdict, lines, width=520):
    """Put the close-up next to a PASS/FAIL panel listing the reasons."""
    colour = {"PASS": GREEN, "FAIL": RED}.get(verdict, GREY)
    h = max(crop.shape[0], 120 + 24 * len(lines))
    panel = np.full((h, width, 3), 255, np.uint8)
    cv2.rectangle(panel, (0, 0), (width, 90), colour, cv2.FILLED)
    cv2.putText(panel, verdict, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5)
    for i, (text, ok) in enumerate(lines):
        cv2.putText(panel, text, (12, 125 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0) if ok else RED, 1 if ok else 2)
    img = np.full((h, crop.shape[1], 3), 255, np.uint8)
    img[:crop.shape[0]] = crop
    return np.hstack([img, panel])
