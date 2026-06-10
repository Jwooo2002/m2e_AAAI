from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .analysis import save_event_analysis
from .events import (
    generate_events_from_flow,
    save_event_preview,
    save_event_stream_gif,
    save_events_csv,
    smooth_foreground_field,
)


EYE_KEYPOINTS = {
    "left_eye": [11, 13],
    "right_eye": [15, 16, 18],
    "left_eyebrow": [1, 11, 13],
    "right_eyebrow": [2, 15, 16],
}
LIP_KEYPOINTS = {"mouth": [6, 12, 14, 17, 19, 20]}
FACE_KEYPOINTS = {
    "skin": [1, 2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "nose": [3, 4, 5, 8, 9],
    "jaw_or_contour": [5, 6, 8, 9, 12, 14, 17, 19, 20],
}
SEMANTIC_KEYPOINT_GROUPS = {**FACE_KEYPOINTS, **EYE_KEYPOINTS, **LIP_KEYPOINTS}


@dataclass
class LivePortraitMotionState:
    kp_source: np.ndarray
    kp_driving: np.ndarray
    displacement: np.ndarray
    expression_delta: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    translation: np.ndarray
    eye_delta: np.ndarray | None = None
    lip_delta: np.ndarray | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kp_source": self.kp_source.tolist(),
            "kp_driving": self.kp_driving.tolist(),
            "displacement": self.displacement.tolist(),
            "expression_delta": self.expression_delta.tolist(),
            "rotation": self.rotation.tolist(),
            "scale": self.scale.tolist(),
            "translation": self.translation.tolist(),
            "eye_delta": None if self.eye_delta is None else self.eye_delta.tolist(),
            "lip_delta": None if self.lip_delta is None else self.lip_delta.tolist(),
        }


def run_liveportrait_adapter(
    source_image_path: Path,
    semantic_mask_path: Path,
    blob_map_path: Path,
    blob_metadata_path: Path,
    output_dir: Path,
    driving_image_path: Path | None = None,
    motion_state_path: Path | None = None,
    liveportrait_repo: Path | None = None,
    duration_ms: float = 140.0,
    contrast_threshold: float = 0.075,
    max_motion_px: float = 18.0,
    steps: int = 16,
    enable_dense_deformation: bool = False,
) -> dict[str, Any]:
    image_rgb = load_image_rgb(source_image_path)
    semantic_mask = np.load(semantic_mask_path)
    blob_map = np.load(blob_map_path)
    blob_metadata = json.loads(blob_metadata_path.read_text(encoding="utf-8"))
    validate_inputs(image_rgb, semantic_mask, blob_map, blob_metadata)

    if enable_dense_deformation:
        raise NotImplementedError("Dense LivePortrait deformation support is intentionally disabled by default")

    if motion_state_path is not None:
        motion_state = load_motion_state(motion_state_path)
        motion_source = str(motion_state_path)
    elif driving_image_path is not None:
        motion_state = extract_liveportrait_motion_state(
            source_image_path,
            driving_image_path,
            liveportrait_repo=liveportrait_repo,
        )
        motion_source = str(driving_image_path)
    else:
        raise ValueError("Pass either --motion-state or --driving-image")

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_points_px = project_keypoints_to_image(motion_state.kp_source, semantic_mask)
    sparse_displacement_px = scale_displacement_to_pixels(
        motion_state.displacement,
        motion_state.kp_source,
        sparse_points_px,
        semantic_mask,
        max_motion_px=max_motion_px,
    )
    blob_motion = sparse_motion_to_blob_motion(
        blob_metadata,
        sparse_points_px,
        sparse_displacement_px,
        duration_ms=duration_ms,
        contrast_threshold=contrast_threshold,
    )

    flow_x, flow_y, threshold_map = blob_motion_to_dense_flow(
        blob_map,
        semantic_mask,
        blob_motion["blobs"],
        default_threshold=contrast_threshold,
    )
    events = generate_events_from_flow(
        image_rgb,
        blob_map,
        blob_metadata,
        flow_x,
        flow_y,
        threshold_map=threshold_map,
        duration_ms=duration_ms,
        steps=steps,
        max_events_per_step=5000,
    )

    blob_motion_payload = {
        "mode": "liveportrait_sparse_motion_prior",
        "source_image_path": str(source_image_path),
        "motion_source": motion_source,
        "dense_deformation_enabled": False,
        "schema": motion_state.to_json_dict(),
        "keypoints_projected_px": sparse_points_px.tolist(),
        "keypoint_displacement_px": sparse_displacement_px.tolist(),
        "blob_motion": blob_motion["blobs"],
        "warnings": blob_motion["warnings"],
    }
    (output_dir / "liveportrait_blob_motion.json").write_text(
        json.dumps(blob_motion_payload, indent=2),
        encoding="utf-8",
    )

    save_sparse_motion_overlay(
        output_dir / "liveportrait_sparse_motion_overlay.png",
        image_rgb,
        semantic_mask,
        sparse_points_px,
        sparse_displacement_px,
    )
    save_blob_motion_overlay(
        output_dir / "liveportrait_blob_motion_overlay.png",
        image_rgb,
        semantic_mask,
        blob_motion["blobs"],
    )
    save_event_stream_gif(output_dir / "liveportrait_direct_event_stream.gif", image_rgb, events, frames=steps)
    save_event_stream_gif(
        output_dir / "liveportrait_direct_event_black.gif",
        image_rgb,
        events,
        background_mode="black",
        frames=steps,
    )
    save_event_preview(output_dir / "liveportrait_direct_event_preview.png", semantic_mask.shape, events)
    np.save(output_dir / "liveportrait_direct_events.npy", events)
    save_events_csv(output_dir / "liveportrait_direct_events.csv", events)

    stats = save_event_analysis(output_dir, image_rgb, semantic_mask, blob_map, blob_metadata, events)
    summary = {
        "mode": "liveportrait_sparse_motion_prior",
        "source_image_path": str(source_image_path),
        "motion_source": motion_source,
        "num_blobs_with_motion": len(blob_motion["blobs"]),
        "num_events": int(len(events)),
        "duration_ms": float(duration_ms),
        "contrast_threshold": float(contrast_threshold),
        "max_motion_px": float(max_motion_px),
        "flow": {
            "mean_magnitude_px": mean_foreground_magnitude(flow_x, flow_y, semantic_mask > 0),
            "max_magnitude_px": float(np.sqrt(flow_x * flow_x + flow_y * flow_y).max()),
        },
        "outputs": [
            "liveportrait_blob_motion.json",
            "liveportrait_sparse_motion_overlay.png",
            "liveportrait_blob_motion_overlay.png",
            "liveportrait_direct_event_stream.gif",
            "liveportrait_direct_event_black.gif",
            "liveportrait_direct_event_statistics.json",
        ],
        "warnings": blob_motion["warnings"],
        "statistics": stats,
    }
    (output_dir / "liveportrait_direct_event_statistics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def load_motion_state(path: Path) -> LivePortraitMotionState:
    raw = load_structured_motion(path)
    if isinstance(raw, dict) and "schema" in raw and isinstance(raw["schema"], dict):
        raw = raw["schema"]
    if isinstance(raw, dict) and "motion" in raw and isinstance(raw["motion"], list):
        raw = liveportrait_template_to_motion_state(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported motion state payload in {path}")

    kp_source = first_available_array(raw, ["kp_source", "x_s", "source", "kp"])
    kp_driving = first_optional_array(raw, ["kp_driving", "x_d", "driving", "x_d_i_new"])
    if kp_driving is None:
        displacement = first_available_array(raw, ["displacement", "delta", "motion"])
        if displacement is None:
            raise ValueError("motion state must contain kp_driving/x_d or displacement")
        kp_driving = kp_source + displacement
    displacement = array_or_default(raw, "displacement", kp_driving - kp_source)
    expression_delta = array_or_default(raw, "expression_delta", raw.get("exp", np.zeros_like(kp_source)))
    rotation = array_or_default(raw, "rotation", raw.get("R", np.eye(3, dtype=np.float32)))
    scale = array_or_default(raw, "scale", np.array([1.0], dtype=np.float32))
    translation = array_or_default(raw, "translation", raw.get("t", np.zeros((3,), dtype=np.float32)))
    eye_delta = optional_array(raw.get("eye_delta"))
    lip_delta = optional_array(raw.get("lip_delta"))

    return normalize_motion_state(
        LivePortraitMotionState(
            kp_source=kp_source,
            kp_driving=kp_driving,
            displacement=displacement,
            expression_delta=expression_delta,
            rotation=rotation,
            scale=scale,
            translation=translation,
            eye_delta=eye_delta,
            lip_delta=lip_delta,
        )
    )


def load_structured_motion(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.shape == () and isinstance(arr.item(), dict):
            return arr.item()
        return {"displacement": arr}
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported motion state extension: {path.suffix}")


def liveportrait_template_to_motion_state(template: dict[str, Any]) -> dict[str, Any]:
    motion = template["motion"]
    if not motion:
        raise ValueError("LivePortrait motion template is empty")
    source = motion[0]
    driving = motion[min(1, len(motion) - 1)]
    kp_source = np.asarray(source.get("x_s", source.get("kp")), dtype=np.float32)
    kp_driving = np.asarray(driving.get("x_s", driving.get("kp")), dtype=np.float32)
    return {
        "kp_source": kp_source,
        "kp_driving": kp_driving,
        "displacement": kp_driving - kp_source,
        "expression_delta": np.asarray(driving.get("exp", np.zeros_like(kp_source)), dtype=np.float32)
        - np.asarray(source.get("exp", np.zeros_like(kp_source)), dtype=np.float32),
        "rotation": np.asarray(driving.get("R", np.eye(3)), dtype=np.float32),
        "scale": np.asarray(driving.get("scale", [1.0]), dtype=np.float32),
        "translation": np.asarray(driving.get("t", [0.0, 0.0, 0.0]), dtype=np.float32),
    }


def extract_liveportrait_motion_state(
    source_image_path: Path,
    driving_image_path: Path,
    liveportrait_repo: Path | None = None,
) -> LivePortraitMotionState:
    repo = liveportrait_repo or Path("/tmp/LivePortrait")
    if not repo.exists():
        raise FileNotFoundError(f"LivePortrait repository not found: {repo}")
    sys.path.insert(0, str(repo))
    try:
        import torch
        import yaml

        from src.config.inference_config import InferenceConfig
        from src.utils.camera import get_rotation_matrix, headpose_pred_to_degree
        from src.utils.helper import load_model
    except Exception as exc:
        raise RuntimeError(f"Could not import LivePortrait from {repo}: {exc}") from exc

    cfg = InferenceConfig(flag_force_cpu=True, flag_use_half_precision=False, flag_pasteback=False)
    model_config = yaml.load(open(cfg.models_config, "r", encoding="utf-8"), Loader=yaml.SafeLoader)
    motion_extractor = load_model(cfg.checkpoint_M, model_config, "cpu", "motion_extractor")

    source_rgb = load_image_rgb(source_image_path)
    driving_rgb = load_image_rgb(driving_image_path)
    source_tensor = prepare_liveportrait_tensor(source_rgb, cfg.input_shape)
    driving_tensor = prepare_liveportrait_tensor(driving_rgb, cfg.input_shape)

    with torch.no_grad():
        source_info = refine_liveportrait_kp_info(motion_extractor(source_tensor), headpose_pred_to_degree)
        driving_info = refine_liveportrait_kp_info(motion_extractor(driving_tensor), headpose_pred_to_degree)
    kp_source_t = transform_liveportrait_keypoint(source_info, get_rotation_matrix)
    kp_driving_t = transform_liveportrait_keypoint(driving_info, get_rotation_matrix)
    kp_source = kp_source_t.detach().cpu().numpy()
    kp_driving = kp_driving_t.detach().cpu().numpy()
    expression_delta = (driving_info["exp"] - source_info["exp"]).detach().cpu().numpy()
    rotation = get_rotation_matrix(
        driving_info["pitch"],
        driving_info["yaw"],
        driving_info["roll"],
    ).detach().cpu().numpy()

    return normalize_motion_state(
        LivePortraitMotionState(
            kp_source=kp_source,
            kp_driving=kp_driving,
            displacement=kp_driving - kp_source,
            expression_delta=expression_delta,
            rotation=rotation,
            scale=driving_info["scale"].detach().cpu().numpy(),
            translation=driving_info["t"].detach().cpu().numpy(),
        )
    )


def prepare_liveportrait_tensor(image_rgb: np.ndarray, input_shape: tuple[int, int]):
    import torch

    resized = cv2.resize(image_rgb, (int(input_shape[1]), int(input_shape[0])))
    tensor = resized[np.newaxis].astype(np.float32) / 255.0
    return torch.from_numpy(tensor).permute(0, 3, 1, 2).contiguous()


def refine_liveportrait_kp_info(kp_info: dict[str, Any], headpose_pred_to_degree_fn) -> dict[str, Any]:
    bs = kp_info["kp"].shape[0]
    out = dict(kp_info)
    out["pitch"] = headpose_pred_to_degree_fn(out["pitch"])[:, None]
    out["yaw"] = headpose_pred_to_degree_fn(out["yaw"])[:, None]
    out["roll"] = headpose_pred_to_degree_fn(out["roll"])[:, None]
    out["kp"] = out["kp"].reshape(bs, -1, 3).float()
    out["exp"] = out["exp"].reshape(bs, -1, 3).float()
    out["t"] = out["t"].float()
    out["scale"] = out["scale"].float()
    return out


def transform_liveportrait_keypoint(kp_info: dict[str, Any], get_rotation_matrix_fn):
    kp = kp_info["kp"]
    rot = get_rotation_matrix_fn(kp_info["pitch"], kp_info["yaw"], kp_info["roll"])
    transformed = kp @ rot + kp_info["exp"]
    transformed *= kp_info["scale"][..., None]
    transformed[:, :, 0:2] += kp_info["t"][:, None, 0:2]
    return transformed


def normalize_motion_state(state: LivePortraitMotionState) -> LivePortraitMotionState:
    state.kp_source = squeeze_keypoints(np.asarray(state.kp_source, dtype=np.float32))
    state.kp_driving = squeeze_keypoints(np.asarray(state.kp_driving, dtype=np.float32))
    state.displacement = squeeze_keypoints(np.asarray(state.displacement, dtype=np.float32))
    state.expression_delta = squeeze_keypoints(np.asarray(state.expression_delta, dtype=np.float32))
    if state.kp_source.shape != state.kp_driving.shape:
        raise ValueError(f"kp_source and kp_driving shape mismatch: {state.kp_source.shape} vs {state.kp_driving.shape}")
    if state.displacement.shape != state.kp_source.shape:
        state.displacement = state.kp_driving - state.kp_source
    if state.expression_delta.shape != state.kp_source.shape:
        state.expression_delta = np.zeros_like(state.kp_source, dtype=np.float32)
    state.rotation = np.asarray(state.rotation, dtype=np.float32).squeeze()
    state.scale = np.asarray(state.scale, dtype=np.float32).squeeze()
    state.translation = np.asarray(state.translation, dtype=np.float32).squeeze()
    if state.eye_delta is not None:
        state.eye_delta = squeeze_keypoints(np.asarray(state.eye_delta, dtype=np.float32))
    if state.lip_delta is not None:
        state.lip_delta = squeeze_keypoints(np.asarray(state.lip_delta, dtype=np.float32))
    return state


def squeeze_keypoints(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1 and arr.size % 3 == 0:
        arr = arr.reshape(-1, 3)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected keypoints with shape (N,2+) or (N,3), got {arr.shape}")
    if arr.shape[1] == 2:
        arr = np.pad(arr, ((0, 0), (0, 1)), mode="constant")
    return arr[:, :3].astype(np.float32)


def project_keypoints_to_image(kp_source: np.ndarray, semantic_mask: np.ndarray) -> np.ndarray:
    h, w = semantic_mask.shape
    xy = kp_source[:, :2].astype(np.float32).copy()
    if xy.size == 0:
        raise ValueError("empty keypoint set")

    x_min, y_min = float(xy[:, 0].min()), float(xy[:, 1].min())
    x_max, y_max = float(xy[:, 0].max()), float(xy[:, 1].max())
    if x_min >= 0 and y_min >= 0 and x_max <= w - 1 and y_max <= h - 1 and max(x_max - x_min, y_max - y_min) > 8:
        return xy

    fg = semantic_mask > 0
    if fg.any():
        ys, xs = np.where(fg)
        left, right = float(xs.min()), float(xs.max())
        top, bottom = float(ys.min()), float(ys.max())
    else:
        left, right = 0.15 * w, 0.85 * w
        top, bottom = 0.15 * h, 0.85 * h

    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    margin_x = 0.08 * max(right - left, 1.0)
    margin_y = 0.08 * max(bottom - top, 1.0)
    out = np.empty_like(xy, dtype=np.float32)
    out[:, 0] = (xy[:, 0] - x_min) / span_x * max(right - left - 2.0 * margin_x, 1.0) + left + margin_x
    out[:, 1] = (xy[:, 1] - y_min) / span_y * max(bottom - top - 2.0 * margin_y, 1.0) + top + margin_y
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return out


def scale_displacement_to_pixels(
    displacement: np.ndarray,
    kp_source: np.ndarray,
    kp_source_px: np.ndarray,
    semantic_mask: np.ndarray,
    max_motion_px: float,
) -> np.ndarray:
    disp_xy = displacement[:, :2].astype(np.float32)
    source_xy = kp_source[:, :2].astype(np.float32)
    source_span = max(float(np.ptp(source_xy[:, 0])), float(np.ptp(source_xy[:, 1])), 1e-6)
    pixel_span = max(float(np.ptp(kp_source_px[:, 0])), float(np.ptp(kp_source_px[:, 1])), 1.0)
    disp_px = disp_xy * (pixel_span / source_span)

    mag = np.sqrt((disp_px * disp_px).sum(axis=1))
    max_mag = float(mag.max()) if mag.size else 0.0
    if max_mag > max_motion_px > 0:
        disp_px *= float(max_motion_px) / max(max_mag, 1e-6)
    if max_mag < 0.75 and np.any(np.abs(disp_xy) > 1e-6):
        fg = semantic_mask > 0
        target = min(float(max_motion_px), max(3.0, min(semantic_mask.shape) * 0.015))
        disp_px *= target / max(max_mag, 1e-6)
    return disp_px.astype(np.float32)


def sparse_motion_to_blob_motion(
    blob_metadata: list[dict],
    keypoints_px: np.ndarray,
    displacement_px: np.ndarray,
    duration_ms: float,
    contrast_threshold: float,
) -> dict[str, Any]:
    warnings: list[str] = []
    rows = []
    for blob in blob_metadata:
        blob_id = int(blob["blob_id"])
        label = str(blob.get("semantic_label", "unknown"))
        centroid = np.asarray(blob["centroid"], dtype=np.float32)
        indices = valid_group_indices(label, len(keypoints_px))
        assignment = "semantic_group" if indices else "nearest"
        if not indices:
            distances = np.linalg.norm(keypoints_px - centroid[None, :], axis=1)
            nearest = int(np.argmin(distances))
            indices = [nearest]
        group_points = keypoints_px[indices]
        distances = np.linalg.norm(group_points - centroid[None, :], axis=1)
        sigma = max(blob_radius(blob), 6.0)
        weights = np.exp(-0.5 * (distances / sigma) ** 2).astype(np.float32)
        if float(weights.sum()) <= 1e-6:
            weights = np.ones(len(indices), dtype=np.float32)
        weights /= float(weights.sum())
        motion = (displacement_px[indices] * weights[:, None]).sum(axis=0)
        magnitude = float(np.linalg.norm(motion))
        direction = [0.0, 0.0] if magnitude <= 1e-6 else [float(motion[0] / magnitude), float(motion[1] / magnitude)]
        rows.append(
            {
                "blob_id": blob_id,
                "semantic_id": int(blob.get("semantic_id", 0)),
                "semantic_label": label,
                "centroid": [float(centroid[0]), float(centroid[1])],
                "assigned_keypoint_indices": [int(i) for i in indices],
                "assignment": assignment,
                "keypoint_weights": [float(v) for v in weights],
                "motion_px": [float(motion[0]), float(motion[1])],
                "motion_context": {
                    "motion_type": "liveportrait_sparse_prior",
                    "direction": direction,
                    "magnitude": magnitude,
                    "duration_ms": float(duration_ms),
                    "temporal_profile": "ease_in_out_cycle",
                    "polarity_hint": "mixed",
                },
                "event_generation_context": {
                    "contrast_threshold": float(contrast_threshold),
                    "noise_level": 0.0,
                    "timestamp_start": 0.0,
                    "timestamp_end": float(duration_ms),
                },
            }
        )
    if not rows:
        warnings.append("no blob metadata rows were available")
    return {"blobs": rows, "warnings": warnings}


def blob_motion_to_dense_flow(
    blob_map: np.ndarray,
    semantic_mask: np.ndarray,
    blob_motion: list[dict[str, Any]],
    default_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow_x = np.zeros(blob_map.shape, dtype=np.float32)
    flow_y = np.zeros(blob_map.shape, dtype=np.float32)
    threshold_map = np.full(blob_map.shape, float(default_threshold), dtype=np.float32)
    threshold_map[semantic_mask == 0] = 1.0
    for row in blob_motion:
        bid = int(row["blob_id"])
        mask = blob_map == bid
        if not mask.any():
            continue
        dx, dy = row["motion_px"]
        flow_x[mask] = float(dx)
        flow_y[mask] = float(dy)
        threshold_map[mask] = float(row["event_generation_context"]["contrast_threshold"])
    foreground = (blob_map > 0).astype(np.float32)
    flow_x = smooth_foreground_field(flow_x, foreground)
    flow_y = smooth_foreground_field(flow_y, foreground)
    flow_x[blob_map == 0] = 0.0
    flow_y[blob_map == 0] = 0.0
    return flow_x, flow_y, threshold_map


def save_sparse_motion_overlay(
    path: Path,
    image_rgb: np.ndarray,
    semantic_mask: np.ndarray,
    keypoints_px: np.ndarray,
    displacement_px: np.ndarray,
) -> None:
    canvas = dim_foreground(image_rgb, semantic_mask)
    h, w = semantic_mask.shape
    for i, (point, disp) in enumerate(zip(keypoints_px, displacement_px)):
        start = (int(round(point[0])), int(round(point[1])))
        end = (
            int(np.clip(round(point[0] + disp[0] * 3.0), 0, w - 1)),
            int(np.clip(round(point[1] + disp[1] * 3.0), 0, h - 1)),
        )
        cv2.circle(canvas, start, 3, (255, 220, 40), -1)
        cv2.arrowedLine(canvas, start, end, (40, 240, 255), 2, tipLength=0.35)
        cv2.putText(canvas, str(i), (start[0] + 4, start[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_blob_motion_overlay(
    path: Path,
    image_rgb: np.ndarray,
    semantic_mask: np.ndarray,
    blob_motion: list[dict[str, Any]],
) -> None:
    canvas = dim_foreground(image_rgb, semantic_mask)
    h, w = semantic_mask.shape
    for row in blob_motion:
        centroid = row["centroid"]
        motion = row["motion_px"]
        start = (int(round(centroid[0])), int(round(centroid[1])))
        end = (
            int(np.clip(round(centroid[0] + motion[0] * 4.0), 0, w - 1)),
            int(np.clip(round(centroid[1] + motion[1] * 4.0), 0, h - 1)),
        )
        color = color_for_label(str(row.get("semantic_label", "")))
        cv2.arrowedLine(canvas, start, end, color, 2, tipLength=0.35)
        cv2.circle(canvas, start, 2, color, -1)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def dim_foreground(image_rgb: np.ndarray, semantic_mask: np.ndarray) -> np.ndarray:
    canvas = (image_rgb.astype(np.float32) * 0.35).astype(np.uint8)
    canvas[semantic_mask == 0] = 0
    return canvas


def color_for_label(label: str) -> tuple[int, int, int]:
    colors = {
        "skin": (230, 190, 90),
        "left_eye": (40, 180, 255),
        "right_eye": (60, 220, 255),
        "nose": (80, 220, 120),
        "mouth": (255, 80, 100),
        "left_eyebrow": (180, 90, 240),
        "right_eyebrow": (210, 120, 255),
        "jaw_or_contour": (250, 210, 70),
    }
    return colors.get(label, (240, 240, 240))


def load_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def validate_inputs(
    image_rgb: np.ndarray,
    semantic_mask: np.ndarray,
    blob_map: np.ndarray,
    blob_metadata: list[dict],
) -> None:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("source image must be HxWx3 RGB")
    if semantic_mask.shape != blob_map.shape:
        raise ValueError(f"semantic_mask and blob_map shape mismatch: {semantic_mask.shape} vs {blob_map.shape}")
    if image_rgb.shape[:2] != semantic_mask.shape:
        raise ValueError(f"image and mask shape mismatch: {image_rgb.shape[:2]} vs {semantic_mask.shape}")
    if not isinstance(blob_metadata, list):
        raise ValueError("blob_metadata must be a JSON list")


def first_available_array(raw: dict[str, Any], keys: list[str]) -> np.ndarray:
    for key in keys:
        if key in raw and raw[key] is not None:
            return squeeze_keypoints(np.asarray(raw[key], dtype=np.float32))
    raise ValueError(f"motion state missing required keys, tried {keys}")


def first_optional_array(raw: dict[str, Any], keys: list[str]) -> np.ndarray | None:
    for key in keys:
        if key in raw and raw[key] is not None:
            return squeeze_keypoints(np.asarray(raw[key], dtype=np.float32))
    return None


def array_or_default(raw: dict[str, Any], key: str, default: Any) -> np.ndarray:
    value = raw.get(key, default)
    return np.asarray(value, dtype=np.float32)


def optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def valid_group_indices(label: str, num_keypoints: int) -> list[int]:
    return [idx for idx in SEMANTIC_KEYPOINT_GROUPS.get(label, []) if idx < num_keypoints]


def blob_radius(blob: dict[str, Any]) -> float:
    bbox = blob.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        width = max(float(bbox[2]) - float(bbox[0]), 1.0)
        height = max(float(bbox[3]) - float(bbox[1]), 1.0)
        return 0.5 * max(width, height)
    return max(float(blob.get("pixel_count", 100)) ** 0.5, 6.0)


def mean_foreground_magnitude(flow_x: np.ndarray, flow_y: np.ndarray, foreground: np.ndarray) -> float:
    mag = np.sqrt(flow_x * flow_x + flow_y * flow_y)
    if foreground.any():
        return float(mag[foreground].mean())
    return float(mag.mean()) if mag.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LivePortrait sparse motion into blob-level direct I2E events.")
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--semantic-mask", type=Path, required=True)
    parser.add_argument("--blob-map", type=Path, required=True)
    parser.add_argument("--blob-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--driving-image", type=Path, default=None)
    parser.add_argument("--motion-state", type=Path, default=None)
    parser.add_argument("--liveportrait-repo", type=Path, default=Path("/tmp/LivePortrait"))
    parser.add_argument("--duration-ms", type=float, default=140.0)
    parser.add_argument("--contrast-threshold", type=float, default=0.075)
    parser.add_argument("--max-motion-px", type=float, default=18.0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--enable-dense-deformation", action="store_true")
    args = parser.parse_args()

    summary = run_liveportrait_adapter(
        source_image_path=args.source_image,
        semantic_mask_path=args.semantic_mask,
        blob_map_path=args.blob_map,
        blob_metadata_path=args.blob_metadata,
        output_dir=args.output_dir,
        driving_image_path=args.driving_image,
        motion_state_path=args.motion_state,
        liveportrait_repo=args.liveportrait_repo,
        duration_ms=args.duration_ms,
        contrast_threshold=args.contrast_threshold,
        max_motion_px=args.max_motion_px,
        steps=args.steps,
        enable_dense_deformation=args.enable_dense_deformation,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "num_blobs_with_motion": summary["num_blobs_with_motion"],
                "num_events": summary["num_events"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
