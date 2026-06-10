from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .events import EVENT_DTYPE, save_event_stream_gif, save_events_csv, warp_frame_and_labels
from .semantics import build_semantic_mask
from .superpixels import blob_metadata, slic_superpixels
from .wflw import search_wflw_annotations
from .pipeline import load_sample_at_or_after


def run_uploaded_flow(
    flow_path: Path,
    output_dir: Path,
    output_subdir: str = "uploaded_optical_flow",
    sample_index: int = 0,
    frames: int = 16,
    duration_ms: float = 140.0,
    max_flow_px: float = 24.0,
    threshold: float = 0.15,
    resize_image_to_flow: bool = False,
) -> dict:
    train, _test = search_wflw_annotations()
    sample = load_sample_at_or_after(train, sample_index)
    image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(sample.image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    semantic_mask, _stats, _warnings = build_semantic_mask((h, w), sample.landmarks)
    raw_flow = np.load(flow_path)
    if resize_image_to_flow:
        flow_h, flow_w = flow_spatial_shape(raw_flow)
        image_rgb = cv2.resize(image_rgb, (flow_w, flow_h), interpolation=cv2.INTER_AREA)
        semantic_mask = cv2.resize(semantic_mask, (flow_w, flow_h), interpolation=cv2.INTER_NEAREST)
        h, w = image_rgb.shape[:2]

    blob_map, metadata, _blob_warnings = blob_metadata(
        image_rgb,
        slic_superpixels(image_rgb, semantic_mask > 0, n_segments=180),
        semantic_mask,
    )

    flow_sequence = prepare_flow_sequence(raw_flow, (h, w), frames=frames, max_flow_px=max_flow_px)
    events = generate_events_from_flow_sequence(
        image_rgb,
        blob_map,
        metadata,
        flow_sequence,
        duration_ms=duration_ms,
        threshold=threshold,
    )

    out = output_dir / output_subdir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "events_uploaded_flow.npy", events)
    save_events_csv(out / "events_uploaded_flow.csv", events)
    save_uploaded_flow_preview_gif(out / "uploaded_flow_motion_preview.gif", image_rgb, flow_sequence, semantic_mask > 0)
    save_event_stream_gif(out / "uploaded_flow_event_stream.gif", image_rgb, events, frames=frames)
    save_event_stream_gif(out / "uploaded_flow_event_stream_black.gif", image_rgb, events, background_mode="black", frames=frames)
    save_flow_overlay(out / "uploaded_flow_overlay.png", image_rgb, semantic_mask, flow_sequence[len(flow_sequence) // 2])

    summary = {
        "flow_path": str(flow_path),
        "flow_shape": list(raw_flow.shape),
        "sample_index": int(sample_index),
        "image_path": str(sample.image_path),
        "num_events": int(len(events)),
        "on_events": int((events["p"] > 0).sum()) if len(events) else 0,
        "off_events": int((events["p"] < 0).sum()) if len(events) else 0,
        "frames": int(frames),
        "duration_ms": float(duration_ms),
        "threshold": float(threshold),
        "resize_image_to_flow": bool(resize_image_to_flow),
        "event_image_shape": [int(h), int(w)],
        "max_flow_px_after_resize": float(np.sqrt((flow_sequence * flow_sequence).sum(axis=-1)).max()),
        "outputs": [
            "uploaded_flow_motion_preview.gif",
            "uploaded_flow_event_stream.gif",
            "uploaded_flow_event_stream_black.gif",
            "uploaded_flow_overlay.png",
        ],
    }
    (out / "uploaded_flow_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def prepare_flow_sequence(raw_flow: np.ndarray, image_shape: tuple[int, int], frames: int, max_flow_px: float) -> np.ndarray:
    if raw_flow.ndim == 3 and raw_flow.shape[-1] == 2:
        raw_flow = raw_flow[None, ...]
    if raw_flow.ndim != 4 or raw_flow.shape[-1] != 2:
        raise ValueError(f"Expected flow shape (T,H,W,2) or (H,W,2), got {raw_flow.shape}")

    h, w = image_shape
    idx = np.linspace(0, raw_flow.shape[0] - 1, max(1, int(frames))).round().astype(np.int32)
    selected = raw_flow[idx].astype(np.float32)
    src_h, src_w = selected.shape[1:3]
    resized = []
    for flow in selected:
        fx = cv2.resize(flow[:, :, 0], (w, h), interpolation=cv2.INTER_LINEAR) * (w / float(src_w))
        fy = cv2.resize(flow[:, :, 1], (w, h), interpolation=cv2.INTER_LINEAR) * (h / float(src_h))
        resized.append(np.stack([fx, fy], axis=-1))
    flow_sequence = np.stack(resized, axis=0).astype(np.float32)

    mag = np.sqrt((flow_sequence * flow_sequence).sum(axis=-1))
    scale = np.ones_like(mag, dtype=np.float32)
    high = mag > max_flow_px
    scale[high] = float(max_flow_px) / np.maximum(mag[high], 1e-6)
    flow_sequence[:, :, :, 0] *= scale
    flow_sequence[:, :, :, 1] *= scale
    return flow_sequence


def flow_spatial_shape(raw_flow: np.ndarray) -> tuple[int, int]:
    if raw_flow.ndim == 4 and raw_flow.shape[-1] == 2:
        return int(raw_flow.shape[1]), int(raw_flow.shape[2])
    if raw_flow.ndim == 3 and raw_flow.shape[-1] == 2:
        return int(raw_flow.shape[0]), int(raw_flow.shape[1])
    raise ValueError(f"Expected flow shape (T,H,W,2) or (H,W,2), got {raw_flow.shape}")


def generate_events_from_flow_sequence(
    image_rgb: np.ndarray,
    blob_map: np.ndarray,
    blob_metadata_list: list[dict],
    flow_sequence: np.ndarray,
    duration_ms: float,
    threshold: float = 0.075,
    max_events_per_frame: int = 5000,
) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    previous = np.log(gray + 1e-3)
    events = []
    meta_by_id = {int(m["blob_id"]): m for m in blob_metadata_list}

    for i, flow in enumerate(flow_sequence, start=1):
        current, warped_blob_map = warp_frame_and_labels(gray, blob_map, flow[:, :, 0], flow[:, :, 1])
        current_log = np.log(current + 1e-3)
        delta = current_log - previous
        active = (warped_blob_map > 0) & (np.abs(delta) >= threshold)
        ys, xs = np.where(active)
        if len(xs) > max_events_per_frame:
            strength = np.abs(delta[ys, xs])
            keep = np.argsort(strength)[::-1][:max_events_per_frame]
            ys = ys[keep]
            xs = xs[keep]
        t = duration_ms * (i / len(flow_sequence))
        polarities = np.where(delta[ys, xs] > 0, 1, -1).astype(np.int8)
        blob_ids = warped_blob_map[ys, xs].astype(np.int32)
        for x, y, p, bid in zip(xs, ys, polarities, blob_ids):
            sem_id = int(meta_by_id.get(int(bid), {}).get("semantic_id", 0))
            events.append((int(x), int(y), float(t), int(p), int(bid), sem_id))
        previous = current_log

    if not events:
        return np.empty((0,), dtype=EVENT_DTYPE)
    arr = np.array(events, dtype=EVENT_DTYPE)
    arr.sort(order=["t", "blob_id", "y", "x"])
    return arr


def save_uploaded_flow_preview_gif(path: Path, image_rgb: np.ndarray, flow_sequence: np.ndarray, foreground_mask: np.ndarray) -> None:
    h, w = image_rgb.shape[:2]
    rgb = image_rgb.astype(np.float32) / 255.0
    rgb = rgb.copy()
    rgb[~foreground_mask] = 0.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    display_mask = foreground_mask.astype(bool)
    frames = []
    for flow in flow_sequence:
        warped = cv2.remap(
            rgb,
            xx - flow[:, :, 0],
            yy - flow[:, :, 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped[~display_mask] = 0.0
        frame = cv2.resize(np.clip(warped * 255, 0, 255).astype(np.uint8), (720, 720), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(frame))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)


def save_flow_overlay(path: Path, image_rgb: np.ndarray, semantic_mask: np.ndarray, flow: np.ndarray) -> None:
    canvas = (image_rgb.astype(np.float32) * 0.35).astype(np.uint8)
    foreground = semantic_mask > 0
    canvas[~foreground] = 0
    h, w = semantic_mask.shape
    stride = max(20, min(h, w) // 24)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if not foreground[y, x]:
                continue
            dx = float(flow[y, x, 0])
            dy = float(flow[y, x, 1])
            end = (int(np.clip(x + dx * 2.0, 0, w - 1)), int(np.clip(y + dy * 2.0, 0, h - 1)))
            cv2.arrowedLine(canvas, (x, y), end, (60, 240, 60), 2, tipLength=0.35)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply an uploaded optical-flow sequence to the current WFLW sample.")
    parser.add_argument("--flow", type=Path, default=Path("flow_px_all.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wflw_mvp_sample"))
    parser.add_argument("--output-subdir", type=str, default="uploaded_optical_flow")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--duration-ms", type=float, default=140.0)
    parser.add_argument("--max-flow-px", type=float, default=24.0)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--resize-image-to-flow", action="store_true")
    args = parser.parse_args()

    summary = run_uploaded_flow(
        args.flow,
        args.output_dir,
        output_subdir=args.output_subdir,
        sample_index=args.sample_index,
        frames=args.frames,
        duration_ms=args.duration_ms,
        max_flow_px=args.max_flow_px,
        threshold=args.threshold,
        resize_image_to_flow=args.resize_image_to_flow,
    )
    print(json.dumps({"output_dir": str(args.output_dir / args.output_subdir), "num_events": summary["num_events"]}, indent=2))


if __name__ == "__main__":
    main()
