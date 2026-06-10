from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .events import EVENT_DTYPE, save_events_csv
except ImportError:  # Allows `python i2e_face_mvp/bound_motion_events.py ...`.
    from events import EVENT_DTYPE, save_events_csv


SUPPORTED_MOTION_SCHEMA_VERSION = "bound_blob_motion.v1"


def run_bound_motion_event_generation(
    image_path: Path,
    semantic_mask_path: Path,
    blob_map_path: Path,
    bound_blob_motion_path: Path,
    output_dir: Path,
    duration_ms: float = 140.0,
    steps: int = 16,
    contrast_threshold: float = 0.08,
    max_events_per_step: int = 6000,
) -> dict[str, Any]:
    image_rgb = load_image_rgb(image_path)
    semantic_mask = np.load(semantic_mask_path)
    blob_map = np.load(blob_map_path)
    motion_payload = json.loads(bound_blob_motion_path.read_text(encoding="utf-8"))
    blob_motion = load_bound_blob_motion_rows(motion_payload)
    validate_inputs(image_rgb, semantic_mask, blob_map)

    flow_x, flow_y, semantic_by_blob = build_dense_flow_from_bound_motion(blob_map, blob_motion)
    events = generate_events_from_directional_change(
        image_rgb,
        semantic_mask,
        blob_map,
        flow_x,
        flow_y,
        semantic_by_blob,
        duration_ms=duration_ms,
        steps=steps,
        contrast_threshold=contrast_threshold,
        max_events_per_step=max_events_per_step,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "events.npy", events)
    save_events_csv(output_dir / "events.csv", events)
    save_event_preview(output_dir / "event_preview.png", semantic_mask.shape, events)
    stats = build_event_statistics(
        semantic_mask,
        blob_map,
        events,
        flow_x,
        flow_y,
        duration_ms=duration_ms,
        steps=steps,
        contrast_threshold=contrast_threshold,
        motion_source=str(bound_blob_motion_path),
    )
    (output_dir / "event_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_bound_blob_motion_rows(motion_payload: dict[str, Any]) -> list[dict[str, Any]]:
    schema_version = motion_payload.get("schema_version")
    if schema_version not in (None, SUPPORTED_MOTION_SCHEMA_VERSION):
        raise ValueError(
            f"unsupported bound motion schema_version {schema_version!r}; "
            f"expected {SUPPORTED_MOTION_SCHEMA_VERSION!r}"
        )
    blob_motion = motion_payload.get("blobs")
    if blob_motion is None:
        blob_motion = motion_payload.get("bound_blob_motion")
    if not isinstance(blob_motion, list):
        raise ValueError("bound_blob_motion.json must contain a canonical 'blobs' list or legacy 'bound_blob_motion' list")
    return blob_motion


def generate_events_from_directional_change(
    image_rgb: np.ndarray,
    semantic_mask: np.ndarray,
    blob_map: np.ndarray,
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    semantic_by_blob: dict[int, int],
    duration_ms: float,
    steps: int,
    contrast_threshold: float,
    max_events_per_step: int,
) -> np.ndarray:
    if duration_ms <= 0.0:
        raise ValueError("duration_ms must be positive")
    steps = max(1, int(steps))
    threshold = float(contrast_threshold)
    if threshold <= 0.0:
        raise ValueError("contrast_threshold must be positive")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    log_gray = np.log(gray + 1e-3)
    grad_x = cv2.Sobel(log_gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(log_gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0

    active = (semantic_mask > 0) & (blob_map > 0) & ((np.abs(flow_x) + np.abs(flow_y)) > 1e-6)
    if not active.any():
        return np.empty((0,), dtype=EVENT_DTYPE)

    # Brightness constancy gives dL/dt = -grad(L) dot velocity. The supplied
    # blob vectors are total displacements over the event window, so each step
    # contributes one equal slice of directional log-intensity change.
    delta_per_step = -(grad_x * flow_x + grad_y * flow_y) / float(steps)
    residual = np.zeros(blob_map.shape, dtype=np.float32)
    events: list[tuple[int, int, float, int, int, int]] = []

    for step in range(1, steps + 1):
        residual[active] += delta_per_step[active]
        crossings = active & (np.abs(residual) >= threshold)
        ys, xs = np.where(crossings)
        if len(xs) == 0:
            continue

        strengths = np.abs(residual[ys, xs])
        if len(xs) > max_events_per_step:
            keep = np.argsort(strengths)[::-1][:max_events_per_step]
            ys = ys[keep]
            xs = xs[keep]
            strengths = strengths[keep]

        polarities = np.where(residual[ys, xs] > 0.0, 1, -1).astype(np.int8)
        counts = np.maximum(1, np.floor(strengths / threshold).astype(np.int32))
        t = float(duration_ms) * (float(step) / float(steps))
        for x, y, polarity, count in zip(xs, ys, polarities, counts):
            bid = int(blob_map[y, x])
            sid = int(semantic_by_blob.get(bid, int(semantic_mask[y, x])))
            for _ in range(int(count)):
                events.append((int(x), int(y), t, int(polarity), bid, sid))
        residual[ys, xs] -= polarities.astype(np.float32) * threshold * counts.astype(np.float32)

    if not events:
        return np.empty((0,), dtype=EVENT_DTYPE)
    arr = np.array(events, dtype=EVENT_DTYPE)
    arr.sort(order=["t", "blob_id", "y", "x"])
    return arr


def build_dense_flow_from_bound_motion(
    blob_map: np.ndarray,
    blob_motion: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    flow_x = np.zeros(blob_map.shape, dtype=np.float32)
    flow_y = np.zeros(blob_map.shape, dtype=np.float32)
    semantic_by_blob: dict[int, int] = {}
    for row in blob_motion:
        blob_id = int(row["blob_id"])
        motion = row.get("motion_px")
        if not isinstance(motion, list) or len(motion) != 2:
            raise ValueError(f"blob {blob_id} must define motion_px as [dx, dy]")
        mask = blob_map == blob_id
        if not mask.any():
            continue
        flow_x[mask] = float(motion[0])
        flow_y[mask] = float(motion[1])
        semantic_by_blob[blob_id] = int(row.get("semantic_id", 0))
    return flow_x, flow_y, semantic_by_blob


def build_event_statistics(
    semantic_mask: np.ndarray,
    blob_map: np.ndarray,
    events: np.ndarray,
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    duration_ms: float,
    steps: int,
    contrast_threshold: float,
    motion_source: str,
) -> dict[str, Any]:
    total = int(len(events))
    mag = np.sqrt(flow_x * flow_x + flow_y * flow_y)
    foreground = blob_map > 0
    on = int((events["p"] > 0).sum()) if total else 0
    off = int((events["p"] < 0).sum()) if total else 0
    return {
        "mode": "direct_gradient_bound_blob_motion",
        "motion_source": motion_source,
        "image_shape": [int(semantic_mask.shape[0]), int(semantic_mask.shape[1])],
        "duration_ms": float(duration_ms),
        "steps": int(steps),
        "contrast_threshold": float(contrast_threshold),
        "total_events": total,
        "polarity_distribution": {
            "on": on,
            "off": off,
            "on_fraction": safe_div(on, total),
            "off_fraction": safe_div(off, total),
        },
        "flow": {
            "active_blob_pixels": int(foreground.sum()),
            "moving_pixels": int(((np.abs(flow_x) + np.abs(flow_y)) > 1e-6).sum()),
            "mean_magnitude_px": float(mag[foreground].mean()) if foreground.any() else 0.0,
            "max_magnitude_px": float(mag.max()) if mag.size else 0.0,
        },
        "semantic_event_counts": semantic_event_counts(semantic_mask, events),
        "blob_event_counts": blob_event_counts(blob_map, events),
        "outputs": ["events.npy", "events.csv", "event_statistics.json", "event_preview.png"],
        "notes": [
            "Events are generated directly from RGB log-intensity gradients and bound blob motion vectors.",
            "No LivePortrait motion is used.",
            "No RGB video or image-to-video-to-event stage is used.",
        ],
    }


def semantic_event_counts(semantic_mask: np.ndarray, events: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    event_semantic = events["semantic_id"].astype(np.int32) if len(events) else np.empty((0,), dtype=np.int32)
    for sid in sorted(int(v) for v in np.unique(semantic_mask)):
        count = int((event_semantic == sid).sum()) if len(events) else 0
        rows.append(
            {
                "semantic_id": sid,
                "pixel_count": int((semantic_mask == sid).sum()),
                "event_count": count,
            }
        )
    return rows


def blob_event_counts(blob_map: np.ndarray, events: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    event_blob = events["blob_id"].astype(np.int32) if len(events) else np.empty((0,), dtype=np.int32)
    for bid in sorted(int(v) for v in np.unique(blob_map) if int(v) > 0):
        count = int((event_blob == bid).sum()) if len(events) else 0
        rows.append(
            {
                "blob_id": bid,
                "pixel_count": int((blob_map == bid).sum()),
                "event_count": count,
            }
        )
    rows.sort(key=lambda row: (-row["event_count"], row["blob_id"]))
    return rows


def save_event_preview(path: Path, image_shape: tuple[int, int], events: np.ndarray) -> None:
    h, w = image_shape
    preview = np.zeros((h, w, 3), dtype=np.float32)
    if len(events):
        on = events["p"] > 0
        np.add.at(preview[:, :, 1], (events["y"][on], events["x"][on]), 30.0)
        np.add.at(preview[:, :, 0], (events["y"][~on], events["x"][~on]), 30.0)
    preview = np.clip(preview, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


def load_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def validate_inputs(image_rgb: np.ndarray, semantic_mask: np.ndarray, blob_map: np.ndarray) -> None:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")
    if image_rgb.shape[:2] != semantic_mask.shape:
        raise ValueError(f"image and semantic_mask shape mismatch: {image_rgb.shape[:2]} vs {semantic_mask.shape}")
    if semantic_mask.shape != blob_map.shape:
        raise ValueError(f"semantic_mask and blob_map shape mismatch: {semantic_mask.shape} vs {blob_map.shape}")


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate direct events from RGB gradients and bound blob motion.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--semantic-mask", type=Path, required=True)
    parser.add_argument("--blob-map", type=Path, required=True)
    parser.add_argument("--bound-blob-motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-ms", type=float, default=140.0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--contrast-threshold", type=float, default=0.08)
    parser.add_argument("--max-events-per-step", type=int, default=6000)
    args = parser.parse_args()

    stats = run_bound_motion_event_generation(
        image_path=args.image,
        semantic_mask_path=args.semantic_mask,
        blob_map_path=args.blob_map,
        bound_blob_motion_path=args.bound_blob_motion,
        output_dir=args.output_dir,
        duration_ms=args.duration_ms,
        steps=args.steps,
        contrast_threshold=args.contrast_threshold,
        max_events_per_step=args.max_events_per_step,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "total_events": stats["total_events"]}, indent=2))


if __name__ == "__main__":
    main()
