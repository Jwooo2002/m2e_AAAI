from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .analysis import load_image_for_output, save_event_analysis
from .events import generate_events, save_event_preview, save_event_stream_gif, save_events_csv, save_motion_preview_gif
from .semantics import REGION_LABELS


def run_debug_controlled_motion(
    output_dir: Path,
    image_rgb: np.ndarray,
    semantic_mask: np.ndarray,
    blob_map: np.ndarray,
    blob_metadata: list[dict],
    prompt_path: Path,
    debug_subdir_name: str = "debug_controlled_motion",
) -> dict[str, Any]:
    prompt = load_debug_motion_prompt(prompt_path)
    debug_dir = output_dir / debug_subdir_name
    debug_dir.mkdir(parents=True, exist_ok=True)

    prompts, motion_by_label, warnings = build_controlled_blob_prompts(blob_metadata, prompt)
    save_motion_preview_gif(debug_dir / "motion_preview.gif", image_rgb, blob_map, prompts)
    save_motion_preview_gif(
        debug_dir / "motion_preview_foreground_rgb.gif",
        image_rgb,
        blob_map,
        prompts,
        foreground_mask=semantic_mask > 0,
        background_mode="foreground_rgb",
    )
    events = generate_events(image_rgb, blob_map, blob_metadata, prompts)

    np.save(debug_dir / "events.npy", events)
    save_events_csv(debug_dir / "events.csv", events)
    save_event_preview(debug_dir / "event_frame_preview.png", semantic_mask.shape, events)
    save_event_stream_gif(debug_dir / "event_stream.gif", image_rgb, events)
    save_event_stream_gif(
        debug_dir / "event_stream_foreground_rgb.gif",
        image_rgb,
        events,
        foreground_mask=semantic_mask > 0,
        background_mode="foreground_rgb",
    )
    save_event_stream_gif(debug_dir / "event_stream_black.gif", image_rgb, events, background_mode="black")
    save_dense_flow_overlay(debug_dir / "dense_flow_overlay.png", image_rgb, semantic_mask, motion_by_label)

    stats = save_event_analysis(debug_dir, image_rgb, semantic_mask, blob_map, blob_metadata, events)
    debug_summary = {
        "mode": "debug_controlled_motion",
        "prompt_path": str(prompt_path),
        "requested_motions": prompt["motions"],
        "matched_motion_targets": sorted(motion_by_label),
        "num_controlled_blob_prompts": len(prompts),
        "num_events": int(len(events)),
        "warnings": warnings,
        "statistics": stats,
    }
    (debug_dir / "debug_summary.json").write_text(json.dumps(debug_summary, indent=2), encoding="utf-8")
    (debug_dir / "debug_motion_prompt.json").write_text(json.dumps(prompt, indent=2), encoding="utf-8")
    return debug_summary


def load_debug_motion_prompt(path: Path) -> dict[str, Any]:
    prompt = json.loads(path.read_text(encoding="utf-8"))
    motions = prompt.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ValueError("debug motion prompt must contain a non-empty 'motions' list")

    valid_labels = set(REGION_LABELS.values())
    normalized = []
    for i, motion in enumerate(motions):
        if not isinstance(motion, dict):
            raise ValueError(f"motion {i} must be an object")
        target = motion.get("target")
        if target not in valid_labels:
            raise ValueError(f"motion {i} has unknown target {target!r}; valid targets: {sorted(valid_labels)}")
        direction = motion.get("direction")
        if not isinstance(direction, list) or len(direction) != 2:
            raise ValueError(f"motion {i} must define direction as [dx, dy]")
        dx, dy = float(direction[0]), float(direction[1])
        magnitude = float(motion.get("magnitude_px", 0.0))
        duration = float(motion.get("duration_ms", 0.0))
        if magnitude < 0.0:
            raise ValueError(f"motion {i} magnitude_px must be non-negative")
        if duration <= 0.0:
            raise ValueError(f"motion {i} duration_ms must be positive")
        normalized.append(
            {
                "target": target,
                "direction": [dx, dy],
                "magnitude_px": magnitude,
                "duration_ms": duration,
                "temporal_profile": str(motion.get("temporal_profile", "linear")),
                "contrast_threshold": float(motion.get("contrast_threshold", 0.08)),
            }
        )
    return {"motions": normalized}


def build_controlled_blob_prompts(blob_metadata: list[dict], prompt: dict[str, Any]) -> tuple[list[dict], dict[str, dict], list[str]]:
    warnings: list[str] = []
    motion_by_label: dict[str, dict] = {}
    for motion in prompt["motions"]:
        target = motion["target"]
        if target in motion_by_label:
            warnings.append(f"duplicate target {target!r}: using the last motion entry")
        motion_by_label[target] = motion

    prompts = []
    matched_labels = set()
    for blob in blob_metadata:
        semantic = blob["semantic_label"]
        motion = motion_by_label.get(semantic)
        if motion is None:
            continue
        matched_labels.add(semantic)
        prompts.append(
            {
                "blob_id": int(blob["blob_id"]),
                "semantic_label": semantic,
                "appearance_context": {
                    "bbox": blob["bbox"],
                    "centroid": blob["centroid"],
                    "area": blob["pixel_count"],
                    "mean_rgb": blob["mean_rgb"],
                    "local_texture_strength": blob["local_texture_strength"],
                },
                "motion_context": {
                    "motion_type": "debug_controlled",
                    "direction": motion["direction"],
                    "magnitude": motion["magnitude_px"],
                    "duration_ms": motion["duration_ms"],
                    "temporal_profile": motion["temporal_profile"],
                    "polarity_hint": "mixed",
                },
                "event_generation_context": {
                    "contrast_threshold": motion["contrast_threshold"],
                    "noise_level": 0.0,
                    "timestamp_start": 0.0,
                    "timestamp_end": float(motion["duration_ms"]),
                },
            }
        )

    for target in sorted(motion_by_label):
        if target not in matched_labels:
            warnings.append(f"target {target!r} did not match any generated blob")
    return prompts, motion_by_label, warnings


def save_dense_flow_overlay(
    path: Path,
    image_rgb: np.ndarray,
    semantic_mask: np.ndarray,
    motion_by_label: dict[str, dict],
) -> None:
    overlay = image_rgb.astype(np.float32).copy()
    tint = np.zeros_like(overlay)
    active_mask = np.zeros(semantic_mask.shape, dtype=bool)

    for region_id, label in REGION_LABELS.items():
        if label not in motion_by_label:
            continue
        region_mask = semantic_mask == region_id
        if not region_mask.any():
            continue
        active_mask |= region_mask
        color = color_for_label(region_id).astype(np.float32)
        tint[region_mask] = color

    overlay[active_mask] = overlay[active_mask] * 0.55 + tint[active_mask] * 0.45
    canvas = np.clip(overlay, 0, 255).astype(np.uint8)

    h, w = semantic_mask.shape
    for region_id, label in REGION_LABELS.items():
        motion = motion_by_label.get(label)
        if motion is None:
            continue
        ys, xs = np.where(semantic_mask == region_id)
        if len(xs) == 0:
            continue
        direction = np.array(motion["direction"], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction /= norm
        displacement = direction * float(motion["magnitude_px"])
        color = tuple(int(v) for v in color_for_label(region_id).tolist())

        stride = max(18, min(h, w) // 24)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        for y in range(y0, y1 + 1, stride):
            for x in range(x0, x1 + 1, stride):
                if semantic_mask[y, x] != region_id:
                    continue
                start = (int(x), int(y))
                end = (
                    int(np.clip(round(x + displacement[0] * 3.0), 0, w - 1)),
                    int(np.clip(round(y + displacement[1] * 3.0), 0, h - 1)),
                )
                cv2.arrowedLine(canvas, start, end, color, 2, tipLength=0.35)

        centroid = (int(round(float(xs.mean()))), int(round(float(ys.mean()))))
        end = (
            int(np.clip(round(centroid[0] + displacement[0] * 4.0), 0, w - 1)),
            int(np.clip(round(centroid[1] + displacement[1] * 4.0), 0, h - 1)),
        )
        cv2.arrowedLine(canvas, centroid, end, color, 4, tipLength=0.28)

    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def color_for_label(region_id: int) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [244, 161, 97],
            [42, 157, 143],
            [38, 70, 83],
            [233, 196, 106],
            [230, 57, 70],
            [131, 56, 236],
            [255, 0, 110],
            [58, 134, 255],
        ],
        dtype=np.uint8,
    )
    return palette[int(region_id) % len(palette)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run researcher-controlled semantic motion debug generation.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=Path("debug_motion_prompt.json"))
    parser.add_argument("--image-path", type=Path, default=None)
    args = parser.parse_args()

    image_rgb = load_image_for_output(args.output_dir, args.image_path)
    semantic_mask = np.load(args.output_dir / "semantic_mask.npy")
    blob_map = np.load(args.output_dir / "blob_map.npy")
    blob_metadata = json.loads((args.output_dir / "blob_metadata.json").read_text(encoding="utf-8"))
    summary = run_debug_controlled_motion(args.output_dir, image_rgb, semantic_mask, blob_map, blob_metadata, args.prompt)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir / "debug_controlled_motion"),
                "num_controlled_blob_prompts": summary["num_controlled_blob_prompts"],
                "num_events": summary["num_events"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
