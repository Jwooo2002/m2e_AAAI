from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


REGION_LABELS = {
    0: "background",
    1: "skin",
    2: "left_eye",
    3: "right_eye",
    4: "nose",
    5: "mouth",
    6: "left_eyebrow",
    7: "right_eyebrow",
    8: "jaw_or_contour",
}

PALETTE = np.array(
    [
        [0, 0, 0],
        [236, 196, 160],
        [50, 130, 220],
        [80, 190, 240],
        [80, 190, 110],
        [220, 70, 80],
        [150, 90, 190],
        [190, 110, 210],
        [245, 190, 70],
    ],
    dtype=np.uint8,
)


def build_semantic_mask(image_shape: tuple[int, int], landmarks: list[dict]) -> tuple[np.ndarray, list[dict], list[str]]:
    h, w = image_shape
    pts = np.array([[p["x"], p["y"]] for p in landmarks], dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    warnings: list[str] = []

    def fill(indices: list[int], label: int, hull: bool = False) -> None:
        poly = pts[indices].copy()
        if len(poly) < 3:
            warnings.append(f"region {REGION_LABELS[label]} skipped: fewer than 3 points")
            return
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        poly_i = np.round(poly).astype(np.int32)
        if hull:
            poly_i = cv2.convexHull(poly_i)
        cv2.fillPoly(mask, [poly_i], int(label))

    # WFLW landmarks are used as polygons. The outer face contour becomes the
    # coarse foreground/skin region first, then smaller facial parts overwrite it.
    face = np.round(pts[0:33]).astype(np.int32)
    face[:, 0] = np.clip(face[:, 0], 0, w - 1)
    face[:, 1] = np.clip(face[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [face], 1)

    # Keep the jaw/contour as its own semantic band instead of only "skin".
    # This makes contour-generated events measurable separately.
    contour_line = np.zeros_like(mask)
    cv2.polylines(contour_line, [face], isClosed=True, color=8, thickness=max(2, min(h, w) // 120))
    mask[contour_line == 8] = 8

    # Each fill call converts a WFLW landmark subset to a region id:
    # nose=4, mouth=5, eyes=2/3, eyebrows=6/7. Convex hulls make compact masks
    # from sparse keypoints.
    fill(list(range(51, 60)), 4, hull=True)
    fill(list(range(76, 96)), 5, hull=True)
    fill(list(range(60, 68)), 2, hull=True)
    fill(list(range(68, 76)), 3, hull=True)
    fill(list(range(33, 42)), 6, hull=True)
    fill(list(range(42, 51)), 7, hull=True)

    stats = []
    for region_id, label in REGION_LABELS.items():
        ys, xs = np.where(mask == region_id)
        if len(xs) == 0:
            if region_id != 0:
                warnings.append(f"empty semantic region: {label}")
            bbox = None
        else:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        stats.append({"region_id": region_id, "label": label, "area": int(len(xs)), "bbox": bbox})
    return mask, stats, warnings


def save_semantic_png(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(PALETTE[mask], cv2.COLOR_RGB2BGR))
