from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


EVENT_DTYPE = np.dtype(
    [
        ("x", "i4"),
        ("y", "i4"),
        ("t", "f4"),
        ("p", "i1"),
        ("blob_id", "i4"),
        ("semantic_id", "i2"),
    ]
)


def generate_events(
    image_rgb: np.ndarray,
    blob_map: np.ndarray,
    blob_metadata: list[dict],
    blob_prompts: list[dict],
    steps: int = 16,
    max_events_per_step: int = 4500,
) -> np.ndarray:
    return generate_events_from_dense_motion(
        image_rgb,
        blob_map,
        blob_metadata,
        blob_prompts,
        steps=steps,
        max_events_per_step=max_events_per_step,
    )


def generate_events_from_dense_motion(
    image_rgb: np.ndarray,
    blob_map: np.ndarray,
    blob_metadata: list[dict],
    blob_prompts: list[dict],
    steps: int = 16,
    max_events_per_step: int = 4500,
) -> np.ndarray:
    flow_x, flow_y, threshold_map, duration_ms = build_dense_motion_field(blob_map, blob_prompts)
    return generate_events_from_flow(
        image_rgb,
        blob_map,
        blob_metadata,
        flow_x,
        flow_y,
        threshold_map=threshold_map,
        duration_ms=duration_ms,
        steps=steps,
        max_events_per_step=max_events_per_step,
    )


def generate_events_from_flow(
    image_rgb: np.ndarray,
    blob_map: np.ndarray,
    blob_metadata: list[dict],
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    threshold_map: np.ndarray | None = None,
    duration_ms: float = 120.0,
    steps: int = 16,
    max_events_per_step: int = 4500,
) -> np.ndarray:
    if duration_ms <= 0.0 or not np.any(np.abs(flow_x) + np.abs(flow_y) > 1e-4):
        return np.empty((0,), dtype=EVENT_DTYPE)
    steps = max(1, int(steps))
    if threshold_map is None:
        threshold_map = np.full(blob_map.shape, 0.08, dtype=np.float32)
    else:
        threshold_map = threshold_map.astype(np.float32)

    # Event simulation is based on brightness change, so RGB is reduced to
    # grayscale and then log-scaled like a contrast-based event camera model.
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    previous = np.log(gray + 1e-3)
    events = []
    meta_by_id = {int(m["blob_id"]): m for m in blob_metadata}

    for step in range(1, steps + 1):
        # The flow is applied as a short forward-and-return cycle. This avoids a
        # one-way slide and gives both ON and OFF events from the same moving
        # facial edges.
        alpha = motion_cycle_alpha(step / steps)

        # Warp the whole image and the blob labels with the same dense flow.
        # This is the key change from independent per-blob event generation:
        # all pixels now share one coherent motion timeline.
        current, warped_blob_map = warp_frame_and_labels(gray, blob_map, flow_x * alpha, flow_y * alpha)
        current_log = np.log(current + 1e-3)
        delta = current_log - previous

        # ON event: brightness increased enough. OFF event: brightness decreased
        # enough. Background is excluded by requiring warped_blob_map > 0.
        active = (warped_blob_map > 0) & (np.abs(delta) >= threshold_map)
        ys, xs = np.where(active)
        if len(xs) == 0:
            previous = current_log
            continue
        strengths = np.abs(delta[ys, xs])
        if len(xs) > max_events_per_step:
            keep = np.argsort(strengths)[::-1][:max_events_per_step]
            ys = ys[keep]
            xs = xs[keep]
            strengths = strengths[keep]
        t = duration_ms * (step / steps)
        blob_ids = warped_blob_map[ys, xs].astype(np.int32)
        polarities = np.where(delta[ys, xs] > 0, 1, -1).astype(np.int8)
        for x, y, polarity, bid in zip(xs, ys, polarities, blob_ids):
            sem_id = int(meta_by_id.get(int(bid), {}).get("semantic_id", 0))
            events.append((int(x), int(y), float(t), int(polarity), int(bid), sem_id))
        previous = current_log

    if not events:
        return np.empty((0,), dtype=EVENT_DTYPE)
    arr = np.array(events, dtype=EVENT_DTYPE)
    arr.sort(order=["t", "blob_id", "y", "x"])
    return arr


def build_dense_motion_field(
    blob_map: np.ndarray,
    blob_prompts: list[dict],
    default_threshold: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    h, w = blob_map.shape
    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)
    threshold_map = np.full((h, w), float(default_threshold), dtype=np.float32)
    duration_ms = 0.0

    for prompt in blob_prompts:
        # Blob prompts are converted to a dense optical-flow-like field. Each
        # blob receives a displacement vector, then the vectors are smoothed
        # across the foreground so neighboring blobs move more coherently.
        bid = int(prompt["blob_id"])
        mask = blob_map == bid
        if not mask.any():
            continue
        motion = prompt["motion_context"]
        direction = np.array(motion["direction"], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction /= norm
        displacement = direction * float(motion["magnitude"])
        flow_x[mask] = float(displacement[0])
        flow_y[mask] = float(displacement[1])
        threshold_map[mask] = float(prompt["event_generation_context"].get("contrast_threshold", default_threshold))
        duration_ms = max(duration_ms, float(motion.get("duration_ms", 0.0)))

    foreground = (blob_map > 0).astype(np.float32)
    flow_x = smooth_foreground_field(flow_x, foreground)
    flow_y = smooth_foreground_field(flow_y, foreground)
    flow_x[blob_map == 0] = 0.0
    flow_y[blob_map == 0] = 0.0
    return flow_x, flow_y, threshold_map, duration_ms


def smooth_foreground_field(field: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    weighted = cv2.GaussianBlur(field * foreground, (0, 0), sigmaX=5.0, sigmaY=5.0)
    weights = cv2.GaussianBlur(foreground, (0, 0), sigmaX=5.0, sigmaY=5.0)
    smoothed = np.zeros_like(field, dtype=np.float32)
    valid = weights > 1e-4
    smoothed[valid] = weighted[valid] / weights[valid]
    return smoothed


def warp_frame_and_labels(
    gray: np.ndarray,
    blob_map: np.ndarray,
    flow_x: np.ndarray,
    flow_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # cv2.remap is backward warping: output(x, y) samples input at
    # (x - flow_x, y - flow_y). Blob ids are warped too so each event keeps
    # blob_id/semantic_id metadata after motion.
    map_x = xx - flow_x.astype(np.float32)
    map_y = yy - flow_y.astype(np.float32)
    warped_gray = cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    warped_blob = cv2.remap(
        blob_map.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.int32)
    return warped_gray, warped_blob


def save_motion_preview_gif(
    path: Path,
    image_rgb: np.ndarray,
    blob_map: np.ndarray,
    blob_prompts: list[dict],
    foreground_mask: np.ndarray | None = None,
    background_mode: str = "rgb",
    frames: int = 16,
    frame_duration_ms: int = 90,
    max_side: int = 720,
) -> None:
    flow_x, flow_y, _threshold_map, duration_ms = build_dense_motion_field(blob_map, blob_prompts)
    save_motion_preview_from_flow_gif(
        path,
        image_rgb,
        flow_x,
        flow_y,
        foreground_mask=foreground_mask,
        background_mode=background_mode,
        frames=frames,
        frame_duration_ms=frame_duration_ms,
        max_side=max_side,
    )


def save_motion_preview_from_flow_gif(
    path: Path,
    image_rgb: np.ndarray,
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    foreground_mask: np.ndarray | None = None,
    background_mode: str = "rgb",
    frames: int = 16,
    frame_duration_ms: int = 90,
    max_side: int = 720,
) -> None:
    frames = max(1, int(frames))
    h, w = image_rgb.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))

    rgb = image_rgb.astype(np.float32) / 255.0
    display_mask = None
    if background_mode == "foreground_rgb":
        if foreground_mask is None:
            raise ValueError("foreground_mask is required for foreground_rgb mode")
        rgb = rgb.copy()
        rgb[~foreground_mask.astype(bool)] = 0.0
        display_mask = cv2.resize(foreground_mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    images = []
    for step in range(frames):
        alpha = motion_cycle_alpha(step / max(1, frames - 1))
        map_x = xx - flow_x * alpha
        map_y = yy - flow_y * alpha
        border_mode = cv2.BORDER_CONSTANT if background_mode == "foreground_rgb" else cv2.BORDER_REFLECT101
        warped = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=0)
        if background_mode == "foreground_rgb":
            warped_mask = cv2.remap(
                foreground_mask.astype(np.uint8),
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ) > 0
            warped[~warped_mask] = 0.0
        elif background_mode != "rgb":
            raise ValueError(f"unknown background_mode: {background_mode}")
        frame = np.clip(warped * 255.0, 0, 255).astype(np.uint8)
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if display_mask is not None:
            frame[~display_mask] = 0
        images.append(Image.fromarray(frame))

    images[0].save(path, save_all=True, append_images=images[1:], duration=frame_duration_ms, loop=0, optimize=False)


def temporal_alpha(value: float, profile: str) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    if profile == "sinusoidal":
        return float(np.sin(value * np.pi / 2.0))
    if profile == "ease_in_out":
        return float(value * value * (3.0 - 2.0 * value))
    return value


def motion_cycle_alpha(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    if value <= 0.5:
        return temporal_alpha(value * 2.0, "ease_in_out")
    return temporal_alpha((1.0 - value) * 2.0, "ease_in_out")


def save_events_csv(path: Path, events: np.ndarray) -> None:
    header = "x,y,t,p,blob_id,semantic_id"
    if len(events) == 0:
        path.write_text(header + "\n", encoding="utf-8")
        return
    matrix = np.column_stack([events[name] for name in ["x", "y", "t", "p", "blob_id", "semantic_id"]])
    np.savetxt(path, matrix, fmt=["%d", "%d", "%.4f", "%d", "%d", "%d"], delimiter=",", header=header, comments="")


def save_event_preview(path: Path, image_shape: tuple[int, int], events: np.ndarray) -> None:
    h, w = image_shape
    preview = np.zeros((h, w, 3), dtype=np.float32)
    if len(events):
        on = events["p"] > 0
        np.add.at(preview[:, :, 1], (events["y"][on], events["x"][on]), 28.0)
        np.add.at(preview[:, :, 0], (events["y"][~on], events["x"][~on]), 28.0)
    preview = np.clip(preview, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


def save_event_stream_gif(
    path: Path,
    image_rgb: np.ndarray,
    events: np.ndarray,
    foreground_mask: np.ndarray | None = None,
    background_mode: str = "dim",
    frames: int = 16,
    frame_duration_ms: int = 90,
    max_side: int = 720,
) -> None:
    if image_rgb.ndim != 3:
        raise ValueError("image_rgb must be an HxWx3 RGB image")
    frames = max(1, int(frames))
    h, w = image_rgb.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))

    base = cv2.resize(image_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    display_mask = None
    if background_mode == "foreground_rgb":
        if foreground_mask is None:
            raise ValueError("foreground_mask is required for foreground_rgb mode")
        display_mask = cv2.resize(foreground_mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
        muted = (base.astype(np.float32) * 0.12).astype(np.uint8)
        muted[~display_mask] = 0
        muted[display_mask] = (base[display_mask].astype(np.float32) * 0.72).astype(np.uint8)
        base = muted
    elif background_mode == "black":
        base = np.zeros_like(base, dtype=np.uint8)
    elif background_mode == "dim":
        base = (base.astype(np.float32) * 0.34).astype(np.uint8)
    else:
        raise ValueError(f"unknown background_mode: {background_mode}")

    if len(events) == 0:
        images = [Image.fromarray(base)]
        images[0].save(path, save_all=True, append_images=[], duration=frame_duration_ms, loop=0)
        return

    t = events["t"].astype(np.float32)
    t_min = float(t.min())
    t_max = float(t.max())
    if t_min == t_max:
        edges = np.linspace(t_min, t_min + 1.0, frames + 1, dtype=np.float32)
    else:
        edges = np.linspace(t_min, t_max, frames + 1, dtype=np.float32)

    gif_frames = []
    cumulative = np.zeros((out_h, out_w, 3), dtype=np.float32)
    xs = np.clip(np.round(events["x"].astype(np.float32) * scale).astype(np.int32), 0, out_w - 1)
    ys = np.clip(np.round(events["y"].astype(np.float32) * scale).astype(np.int32), 0, out_h - 1)

    for i in range(frames):
        if i == frames - 1:
            mask = (t >= edges[i]) & (t <= edges[i + 1])
        else:
            mask = (t >= edges[i]) & (t < edges[i + 1])
        frame_events = np.zeros((out_h, out_w, 3), dtype=np.float32)
        if mask.any():
            on = mask & (events["p"] > 0)
            off = mask & (events["p"] < 0)
            np.add.at(frame_events[:, :, 1], (ys[on], xs[on]), 210.0)
            np.add.at(frame_events[:, :, 0], (ys[off], xs[off]), 230.0)

        frame_events = cv2.GaussianBlur(frame_events, (0, 0), sigmaX=0.75, sigmaY=0.75)
        cumulative *= 0.72
        cumulative += frame_events
        overlay = np.clip(base.astype(np.float32) + cumulative + frame_events * 1.4, 0, 255).astype(np.uint8)
        if display_mask is not None:
            overlay[~display_mask] = 0
        gif_frames.append(Image.fromarray(overlay))

    gif_frames[0].save(
        path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )
