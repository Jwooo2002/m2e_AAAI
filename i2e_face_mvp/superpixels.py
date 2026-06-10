from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .semantics import REGION_LABELS


def slic_superpixels(
    image_rgb: np.ndarray,
    foreground_mask: np.ndarray,
    n_segments: int = 180,
    compactness: float = 12.0,
    iterations: int = 6,
) -> np.ndarray:
    h, w = foreground_mask.shape
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_y, fg_x = np.where(foreground_mask)
    if len(fg_x) == 0:
        raise ValueError("foreground mask is empty")

    step = max(6, int(np.sqrt((h * w) / max(1, n_segments))))
    centers = []
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if foreground_mask[y, x]:
                yy0, yy1 = max(0, y - step), min(h, y + step + 1)
                xx0, xx1 = max(0, x - step), min(w, x + step + 1)
                local_y, local_x = np.where(foreground_mask[yy0:yy1, xx0:xx1])
                if len(local_x):
                    x = int(xx0 + local_x[len(local_x) // 2])
                    y = int(yy0 + local_y[len(local_y) // 2])
                centers.append([lab[y, x, 0], lab[y, x, 1], lab[y, x, 2], float(x), float(y)])
    if not centers:
        centers.append([*lab[int(fg_y.mean()), int(fg_x.mean())], float(fg_x.mean()), float(fg_y.mean())])

    centers_np = np.array(centers, dtype=np.float32)
    labels = np.full((h, w), -1, dtype=np.int32)
    distances = np.full((h, w), np.inf, dtype=np.float32)
    xy_weight = compactness / float(step)

    for _ in range(iterations):
        distances.fill(np.inf)
        labels.fill(-1)
        for cid, (l, a, b, cx, cy) in enumerate(centers_np):
            x0, x1 = max(0, int(cx - step)), min(w, int(cx + step + 1))
            y0, y1 = max(0, int(cy - step)), min(h, int(cy + step + 1))
            patch = lab[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dc = np.sqrt((patch[:, :, 0] - l) ** 2 + (patch[:, :, 1] - a) ** 2 + (patch[:, :, 2] - b) ** 2)
            ds = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            d = dc + xy_weight * ds
            fg = foreground_mask[y0:y1, x0:x1]
            current = distances[y0:y1, x0:x1]
            better = (d < current) & fg
            current[better] = d[better]
            labels[y0:y1, x0:x1][better] = cid

        for cid in range(len(centers_np)):
            ys, xs = np.where(labels == cid)
            if len(xs) == 0:
                continue
            colors = lab[ys, xs]
            centers_np[cid] = [colors[:, 0].mean(), colors[:, 1].mean(), colors[:, 2].mean(), xs.mean(), ys.mean()]

    return relabel_connected(labels, foreground_mask)


def relabel_connected(labels: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    blob_map = np.zeros(labels.shape, dtype=np.int32)
    next_id = 1
    for label in sorted(int(v) for v in np.unique(labels) if v >= 0):
        component_mask = ((labels == label) & foreground_mask).astype(np.uint8)
        n, cc = cv2.connectedComponents(component_mask, connectivity=8)
        for cid in range(1, n):
            blob_map[cc == cid] = next_id
            next_id += 1
    return blob_map


def blob_metadata(image_rgb: np.ndarray, blob_map: np.ndarray, semantic_mask: np.ndarray, min_pixels: int = 12) -> tuple[np.ndarray, list[dict], list[str]]:
    warnings: list[str] = []
    cleaned = blob_map.copy()
    for bid in [int(v) for v in np.unique(blob_map) if v > 0]:
        if int((blob_map == bid).sum()) < min_pixels:
            cleaned[blob_map == bid] = 0
            warnings.append(f"tiny blob discarded: {bid}")
    cleaned = compact_blob_ids(cleaned)

    neighbors = find_neighbors(cleaned)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    metadata = []
    for bid in [int(v) for v in np.unique(cleaned) if v > 0]:
        ys, xs = np.where(cleaned == bid)
        sem_values, sem_counts = np.unique(semantic_mask[ys, xs], return_counts=True)
        sem_id = int(sem_values[int(np.argmax(sem_counts))])
        rgb = image_rgb[ys, xs]
        metadata.append(
            {
                "blob_id": bid,
                "semantic_label": REGION_LABELS.get(sem_id, "unknown"),
                "semantic_id": sem_id,
                "pixel_count": int(len(xs)),
                "centroid": [float(xs.mean()), float(ys.mean())],
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "mean_rgb": [float(v) for v in rgb.mean(axis=0)],
                "local_texture_strength": float(grad[ys, xs].mean()),
                "neighbor_blob_ids": sorted(neighbors.get(bid, set())),
            }
        )
    return cleaned, metadata, warnings


def compact_blob_ids(blob_map: np.ndarray) -> np.ndarray:
    out = np.zeros_like(blob_map)
    for new_id, old_id in enumerate([int(v) for v in np.unique(blob_map) if v > 0], start=1):
        out[blob_map == old_id] = new_id
    return out


def find_neighbors(blob_map: np.ndarray) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    right_a, right_b = blob_map[:, :-1], blob_map[:, 1:]
    down_a, down_b = blob_map[:-1, :], blob_map[1:, :]
    for a_arr, b_arr in ((right_a, right_b), (down_a, down_b)):
        diff = (a_arr != b_arr) & (a_arr > 0) & (b_arr > 0)
        for a, b in zip(a_arr[diff].flat, b_arr[diff].flat):
            neighbors[int(a)].add(int(b))
            neighbors[int(b)].add(int(a))
    return neighbors


def save_blob_map_png(path: Path, blob_map: np.ndarray) -> None:
    ids = blob_map.astype(np.uint32)
    rgb = np.zeros((*blob_map.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = ((ids * 37) % 255).astype(np.uint8)
    rgb[:, :, 1] = ((ids * 67) % 255).astype(np.uint8)
    rgb[:, :, 2] = ((ids * 97) % 255).astype(np.uint8)
    rgb[blob_map == 0] = 0
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
