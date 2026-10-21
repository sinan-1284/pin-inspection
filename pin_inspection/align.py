"""Rotate the pin mask so its axis is horizontal with the head on the left."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Alignment:
    mask: np.ndarray        # aligned, tightly cropped mask (head left, tip right)
    angle_deg: float        # image angle of the head->tip direction (0 = right, 90 = down)
    head_xy: tuple          # head end point in mask (pre-alignment) coordinates
    tip_xy: tuple           # tip end point in mask (pre-alignment) coordinates


def _end_width(mask, frac):
    cols = np.count_nonzero(mask, axis=0)
    cols = cols[cols > 0]
    n = max(1, int(len(cols) * frac))
    return cols[:n].mean(), cols[-n:].mean()


def align(mask):
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    centre = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - centre).T))
    axis = vecs[:, -1]  # major axis (largest eigenvalue)

    # Rotate so the major axis lies along +x.
    angle = np.degrees(np.arctan2(axis[1], axis[0]))
    h, w = mask.shape
    M = cv2.getRotationMatrix2D(tuple(centre), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - centre[0]
    M[1, 2] += nh / 2 - centre[1]
    rot = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(rot)
    rot = rot[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # The head is the wider end; flip so it is on the left.
    left, right = _end_width(rot, 0.1)
    if right > left:
        rot = rot[:, ::-1]
        axis = -axis
        angle += 180
    # axis now points head -> tip
    proj = (pts - centre) @ axis
    head = centre + axis * proj.min()
    tip = centre + axis * proj.max()
    angle_deg = (np.degrees(np.arctan2(axis[1], axis[0])) + 360) % 360
    return Alignment(np.ascontiguousarray(rot), float(angle_deg), tuple(head), tuple(tip))
