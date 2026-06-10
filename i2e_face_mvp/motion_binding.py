from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SUPPORTED_PROMPTS = {
    "blink_left_eye",
    "blink_right_eye",
    "blink_both_eyes",
    "mouth_open",
    "mouth_stretch",
    "jaw_shift",
    "custom_region_shift",
}

MOTION_VECTOR_SCHEMA_VERSION = "bound_blob_motion.v1"


def run_motion_binding(
    blob_feature_context_path: Path,
    motion_prompt_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    feature_payload = json.loads(blob_feature_context_path.read_text(encoding="utf-8"))
    prompt_payload = json.loads(motion_prompt_path.read_text(encoding="utf-8"))
    blobs = feature_payload.get("blob_feature_context")
    if not isinstance(blobs, list):
        raise ValueError("blob_feature_context.json must contain a 'blob_feature_context' list")
    prompts = normalize_motion_prompts(prompt_payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    bound = bind_prompts_to_blobs(blobs, prompts)
    output = build_bound_motion_payload(
        blob_feature_context_path=blob_feature_context_path,
        motion_prompt_path=motion_prompt_path,
        prompts=prompts,
        bound_blob_motion=bound["blob_motion"],
        warnings=bound["warnings"],
    )
    (output_dir / "bound_blob_motion.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    image_path = feature_payload.get("source_image_path")
    if image_path:
        save_bound_motion_overlay(output_dir / "bound_motion_overlay.png", Path(image_path), bound["blob_motion"])
    else:
        save_blank_overlay(output_dir / "bound_motion_overlay.png", bound["blob_motion"])

    return {
        "mode": "external_prompt_to_blob_motion",
        "schema_version": MOTION_VECTOR_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "num_prompts": len(prompts),
        "num_bound_blobs": len(bound["blob_motion"]),
        "outputs": ["bound_blob_motion.json", "bound_motion_overlay.png"],
        "warnings": bound["warnings"],
    }


def build_bound_motion_payload(
    blob_feature_context_path: Path,
    motion_prompt_path: Path,
    prompts: list[dict[str, Any]],
    bound_blob_motion: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    summary = summarize_bound_motion(prompts, bound_blob_motion, warnings)
    return {
        "schema_version": MOTION_VECTOR_SCHEMA_VERSION,
        "mode": "external_prompt_to_blob_motion",
        "source": {
            "blob_feature_context_path": str(blob_feature_context_path),
        },
        "motion_prompt": {
            "path": str(motion_prompt_path),
            "prompts": prompts,
            "supported_prompt_types": sorted(SUPPORTED_PROMPTS),
        },
        "blobs": bound_blob_motion,
        "summary": summary,
        "contract": {
            "input": {
                "blob_feature_context_path": str(blob_feature_context_path),
                "motion_prompt_path": str(motion_prompt_path),
                "motion_prompt_format": {
                    "motions": [
                        {
                            "type": sorted(SUPPORTED_PROMPTS),
                            "magnitude_px": "non-negative float displacement over the event window",
                            "strength": "optional float multiplier, default 1.0",
                            "duration_ms": "optional positive float, default 140.0",
                            "direction": "required [dx, dy] for custom_region_shift; optional for jaw_shift",
                            "target": "optional semantic label or region selector for custom_region_shift",
                            "semantic_label": "optional semantic label selector for custom_region_shift",
                            "control_group": "optional control-group selector for custom_region_shift",
                            "region_consistency_group": "optional consistency-group selector for custom_region_shift",
                        }
                    ]
                },
            },
            "output": {
                "bound_blob_motion": [
                    {
                        "blob_id": "int blob id from blob_map.npy",
                        "semantic_id": "int semantic id from semantic_mask.npy",
                        "semantic_label": "string semantic label",
                        "centroid": "[x, y] in source image pixels",
                        "bbox": "[x0, y0, x1, y1] in source image pixels",
                        "motion_px": "[dx, dy] total displacement in pixels over the event window",
                        "motion_context": "normalized direction, magnitude, and polarity hint",
                        "source_prompts": "prompt contributions used to produce this vector",
                    }
                ]
            },
            "downstream_consumer": "i2e_face_mvp.bound_motion_events",
        },
        "blob_feature_context_path": str(blob_feature_context_path),
        "motion_prompt_path": str(motion_prompt_path),
        "supported_prompts": sorted(SUPPORTED_PROMPTS),
        "input_prompts": prompts,
        "bound_blob_motion": bound_blob_motion,
        "warnings": warnings,
        "notes": [
            "This module only binds external prompt motion to blob-level vectors.",
            "No events are generated here.",
            "No video is created here.",
        ],
    }


def summarize_bound_motion(
    prompts: list[dict[str, Any]],
    bound_blob_motion: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    magnitudes = [float(row.get("motion_context", {}).get("magnitude", 0.0)) for row in bound_blob_motion]
    moving = [value for value in magnitudes if value > 1e-6]
    semantic_counts: dict[str, int] = {}
    for row in bound_blob_motion:
        label = str(row.get("semantic_label", "unknown"))
        semantic_counts[label] = semantic_counts.get(label, 0) + 1
    return {
        "num_prompts": int(len(prompts)),
        "num_bound_blobs": int(len(bound_blob_motion)),
        "num_moving_blobs": int(len(moving)),
        "max_motion_px": float(max(magnitudes)) if magnitudes else 0.0,
        "mean_motion_px": float(np.mean(magnitudes)) if magnitudes else 0.0,
        "semantic_counts": dict(sorted(semantic_counts.items())),
        "warnings": warnings,
    }


def normalize_motion_prompts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_prompts = payload.get("motions", payload.get("prompts"))
    if raw_prompts is None:
        raw_prompts = [payload]
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError("motion_prompt.json must contain a prompt object or a non-empty 'motions' list")

    prompts = []
    for i, raw in enumerate(raw_prompts):
        if not isinstance(raw, dict):
            raise ValueError(f"motion prompt {i} must be an object")
        motion_type = str(raw.get("type", raw.get("motion_type", "")))
        if motion_type not in SUPPORTED_PROMPTS:
            raise ValueError(f"unsupported motion prompt {motion_type!r}; supported: {sorted(SUPPORTED_PROMPTS)}")
        magnitude = float(raw.get("magnitude_px", raw.get("magnitude", default_magnitude(motion_type))))
        if magnitude < 0.0:
            raise ValueError(f"motion prompt {i} magnitude must be non-negative")
        prompt = {
            "type": motion_type,
            "magnitude_px": magnitude,
            "strength": float(raw.get("strength", 1.0)),
            "duration_ms": float(raw.get("duration_ms", 140.0)),
            "target": raw.get("target"),
            "semantic_label": raw.get("semantic_label"),
            "control_group": raw.get("control_group"),
            "region_consistency_group": raw.get("region_consistency_group"),
            "direction": normalize_direction(raw.get("direction")),
        }
        if motion_type == "custom_region_shift" and prompt["direction"] is None:
            raise ValueError("custom_region_shift requires direction: [dx, dy]")
        prompts.append(prompt)
    return prompts


def bind_prompts_to_blobs(blobs: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    accum: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    for prompt in prompts:
        matched = 0
        for blob in blobs:
            contribution = prompt_blob_contribution(blob, prompt, blobs)
            if contribution is None:
                continue
            matched += 1
            blob_id = int(blob["blob_id"])
            if blob_id not in accum:
                accum[blob_id] = base_bound_blob(blob)
            row = accum[blob_id]
            row["motion_px"][0] += contribution["vector"][0]
            row["motion_px"][1] += contribution["vector"][1]
            row["source_prompts"].append(
                {
                    "type": prompt["type"],
                    "weight": contribution["weight"],
                    "vector_px": [float(contribution["vector"][0]), float(contribution["vector"][1])],
                }
            )
        if matched == 0:
            warnings.append(f"prompt {prompt['type']!r} did not match any blob")

    rows = []
    for row in accum.values():
        dx, dy = row["motion_px"]
        magnitude = float(np.hypot(dx, dy))
        row["motion_px"] = [float(dx), float(dy)]
        row["motion_context"] = {
            "motion_type": "prompt_bound_blob_motion",
            "direction": [0.0, 0.0] if magnitude <= 1e-6 else [float(dx / magnitude), float(dy / magnitude)],
            "magnitude": magnitude,
            "polarity_hint": "mixed",
        }
        rows.append(row)
    rows.sort(key=lambda item: int(item["blob_id"]))
    return {"blob_motion": rows, "warnings": warnings}


def prompt_blob_contribution(
    blob: dict[str, Any],
    prompt: dict[str, Any],
    all_blobs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    motion_type = prompt["type"]
    magnitude = float(prompt["magnitude_px"]) * float(prompt["strength"])
    if magnitude <= 0.0:
        return None

    if motion_type == "blink_left_eye":
        return eye_blink_contribution(blob, "left_eye", magnitude)
    if motion_type == "blink_right_eye":
        return eye_blink_contribution(blob, "right_eye", magnitude)
    if motion_type == "blink_both_eyes":
        left = eye_blink_contribution(blob, "left_eye", magnitude)
        right = eye_blink_contribution(blob, "right_eye", magnitude)
        return stronger_contribution(left, right)
    if motion_type == "mouth_open":
        return label_weighted_contribution(blob, ["mouth"], [0.0, magnitude], anchor_key="lip")
    if motion_type == "mouth_stretch":
        if str(blob.get("semantic_label")) != "mouth":
            return None
        mouth_center = group_centroid(all_blobs, ["mouth"])
        cx = float(blob["centroid"][0])
        direction_x = -1.0 if cx < mouth_center[0] else 1.0
        return {"vector": np.array([direction_x * magnitude, 0.0], dtype=np.float32), "weight": 1.0}
    if motion_type == "jaw_shift":
        direction = prompt["direction"] or [1.0, 0.15]
        return group_weighted_contribution(blob, ["jaw"], direction, magnitude)
    if motion_type == "custom_region_shift":
        if not custom_prompt_matches(blob, prompt):
            return None
        return {"vector": np.array(prompt["direction"], dtype=np.float32) * magnitude, "weight": 1.0}
    return None


def eye_blink_contribution(blob: dict[str, Any], eye_label: str, magnitude: float) -> dict[str, Any] | None:
    semantic = str(blob.get("semantic_label", ""))
    if semantic == eye_label:
        weight = inverse_anchor_weight(blob, eye_label)
        return {"vector": np.array([0.0, magnitude * weight], dtype=np.float32), "weight": weight}
    eyebrow_label = "left_eyebrow" if eye_label == "left_eye" else "right_eyebrow"
    if semantic == eyebrow_label:
        weight = 0.35 * inverse_anchor_weight(blob, eye_label)
        return {"vector": np.array([0.0, -0.45 * magnitude * weight], dtype=np.float32), "weight": weight}
    return None


def label_weighted_contribution(
    blob: dict[str, Any],
    semantic_labels: list[str],
    vector: list[float],
    anchor_key: str | None = None,
) -> dict[str, Any] | None:
    if str(blob.get("semantic_label", "")) not in semantic_labels:
        return None
    weight = inverse_anchor_weight(blob, anchor_key) if anchor_key else 1.0
    return {"vector": np.array(vector, dtype=np.float32) * weight, "weight": weight}


def group_weighted_contribution(
    blob: dict[str, Any],
    control_groups: list[str],
    direction: list[float],
    magnitude: float,
) -> dict[str, Any] | None:
    blob_groups = set(blob.get("control_groups", []))
    if not blob_groups.intersection(control_groups):
        return None
    direction_arr = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction_arr))
    if norm <= 1e-6:
        return None
    direction_arr /= norm
    weight = 1.0
    if str(blob.get("semantic_label")) == "skin":
        weight = 0.45
    return {"vector": direction_arr * magnitude * weight, "weight": weight}


def custom_prompt_matches(blob: dict[str, Any], prompt: dict[str, Any]) -> bool:
    semantic_targets = [prompt.get("target"), prompt.get("semantic_label")]
    if str(blob.get("semantic_label")) in {str(v) for v in semantic_targets if v is not None}:
        return True
    control_group = prompt.get("control_group")
    if control_group is not None and str(control_group) in set(blob.get("control_groups", [])):
        return True
    consistency = prompt.get("region_consistency_group")
    if consistency is not None and str(consistency) in set(blob.get("region_consistency_groups", [])):
        return True
    return False


def base_bound_blob(blob: dict[str, Any]) -> dict[str, Any]:
    return {
        "blob_id": int(blob["blob_id"]),
        "semantic_id": int(blob.get("semantic_id", 0)),
        "semantic_label": str(blob.get("semantic_label", "unknown")),
        "centroid": [float(blob["centroid"][0]), float(blob["centroid"][1])],
        "bbox": blob.get("bbox"),
        "control_groups": list(blob.get("control_groups", [])),
        "region_consistency_groups": list(blob.get("region_consistency_groups", [])),
        "motion_px": [0.0, 0.0],
        "source_prompts": [],
    }


def stronger_contribution(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    if a is None:
        return b
    if b is None:
        return a
    return a if float(a["weight"]) >= float(b["weight"]) else b


def inverse_anchor_weight(blob: dict[str, Any], anchor_key: str | None) -> float:
    if anchor_key is None:
        return 1.0
    distances = blob.get("anchor_distances", {})
    value = distances.get(anchor_key)
    if value is None:
        return 1.0
    return float(1.0 / (1.0 + max(float(value), 0.0) / 24.0))


def group_centroid(blobs: list[dict[str, Any]], labels: list[str]) -> tuple[float, float]:
    pts = [
        (float(blob["centroid"][0]), float(blob["centroid"][1]))
        for blob in blobs
        if str(blob.get("semantic_label")) in labels
    ]
    if not pts:
        return 0.0, 0.0
    arr = np.asarray(pts, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def normalize_direction(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("direction must be [dx, dy]")
    dx, dy = float(value[0]), float(value[1])
    norm = float(np.hypot(dx, dy))
    if norm <= 1e-6:
        raise ValueError("direction vector must be non-zero")
    return [dx / norm, dy / norm]


def default_magnitude(motion_type: str) -> float:
    defaults = {
        "blink_left_eye": 4.0,
        "blink_right_eye": 4.0,
        "blink_both_eyes": 4.0,
        "mouth_open": 5.0,
        "mouth_stretch": 4.0,
        "jaw_shift": 3.5,
        "custom_region_shift": 3.0,
    }
    return defaults[motion_type]


def save_bound_motion_overlay(path: Path, image_path: Path, blob_motion: list[dict[str, Any]]) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        save_blank_overlay(path, blob_motion)
        return
    canvas = (cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) * 0.55).astype(np.uint8)
    draw_motion_rows(canvas, blob_motion)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_blank_overlay(path: Path, blob_motion: list[dict[str, Any]]) -> None:
    if blob_motion:
        max_x = max(float(row["centroid"][0]) for row in blob_motion)
        max_y = max(float(row["centroid"][1]) for row in blob_motion)
        w = max(64, int(max_x + 32))
        h = max(64, int(max_y + 32))
    else:
        w, h = 256, 256
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    draw_motion_rows(canvas, blob_motion)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def draw_motion_rows(canvas: np.ndarray, blob_motion: list[dict[str, Any]]) -> None:
    h, w = canvas.shape[:2]
    for row in blob_motion:
        cx, cy = row["centroid"]
        dx, dy = row["motion_px"]
        start = (int(np.clip(round(cx), 0, w - 1)), int(np.clip(round(cy), 0, h - 1)))
        end = (
            int(np.clip(round(cx + dx * 4.0), 0, w - 1)),
            int(np.clip(round(cy + dy * 4.0), 0, h - 1)),
        )
        color = color_for_label(str(row.get("semantic_label", "")))
        cv2.circle(canvas, start, 3, color, -1)
        cv2.arrowedLine(canvas, start, end, color, 2, tipLength=0.35)
        cv2.putText(canvas, str(row["blob_id"]), (start[0] + 4, start[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind external face motion prompts to blob-level motion vectors.")
    parser.add_argument("--blob-feature-context", type=Path, required=True)
    parser.add_argument("--motion-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = run_motion_binding(
        blob_feature_context_path=args.blob_feature_context,
        motion_prompt_path=args.motion_prompt,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
