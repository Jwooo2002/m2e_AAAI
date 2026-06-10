from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from i2e_face_mvp.bound_motion_events import run_bound_motion_event_generation
from i2e_face_mvp.face_feature_adapter import run_face_feature_adapter
from i2e_face_mvp.motion_binding import run_motion_binding
from i2e_face_mvp.pipeline import load_sample_at_or_after
from i2e_face_mvp.semantics import build_semantic_mask, save_semantic_png
from i2e_face_mvp.superpixels import blob_metadata, save_blob_map_png, slic_superpixels
from i2e_face_mvp.validate_bound_motion_events import run_validation
from i2e_face_mvp.wflw import save_landmarks_json, search_wflw_annotations


def run_prompt_to_event(
    sample_index: int,
    motion_prompt: Path,
    output_dir: Path,
    threshold: float,
    seed: int,
    n_segments: int = 180,
    fixture: bool = False,
) -> dict:
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fixture:
        source_info = prepare_fixture_source(output_dir, seed)
    else:
        source_info = prepare_wflw_source(output_dir, sample_index)

    image_rgb = source_info["image_rgb"]
    h, w = image_rgb.shape[:2]
    copied_image_path = source_info["image_path"]
    landmarks_path = source_info["landmarks_path"]

    semantic_mask_path = output_dir / "semantic_mask.npy"
    blob_map_path = output_dir / "blob_map.npy"
    blob_metadata_path = output_dir / "blob_metadata.json"

    if semantic_mask_path.exists():
        semantic_mask = np.load(semantic_mask_path)
    else:
        semantic_mask, semantic_stats, semantic_warnings = build_semantic_mask((h, w), source_info["landmarks"])
        np.save(semantic_mask_path, semantic_mask)
        save_semantic_png(output_dir / "semantic_mask.png", semantic_mask)
        (output_dir / "semantic_summary.json").write_text(
            json.dumps({"stats": semantic_stats, "warnings": semantic_warnings}, indent=2),
            encoding="utf-8",
        )

    if blob_map_path.exists() and blob_metadata_path.exists():
        blob_map = np.load(blob_map_path)
        blob_meta = json.loads(blob_metadata_path.read_text(encoding="utf-8"))
    else:
        foreground = semantic_mask > 0
        raw_blob_map = slic_superpixels(image_rgb, foreground, n_segments=n_segments)
        blob_map, blob_meta, blob_warnings = blob_metadata(image_rgb, raw_blob_map, semantic_mask)
        np.save(blob_map_path, blob_map)
        save_blob_map_png(output_dir / "blob_map.png", blob_map)
        blob_metadata_path.write_text(json.dumps(blob_meta, indent=2), encoding="utf-8")
        (output_dir / "blob_summary.json").write_text(
            json.dumps({"num_blobs": len(blob_meta), "warnings": blob_warnings}, indent=2),
            encoding="utf-8",
        )

    feature_summary = run_face_feature_adapter(
        image_path=copied_image_path,
        landmarks_path=landmarks_path,
        semantic_mask_path=semantic_mask_path,
        blob_map_path=blob_map_path,
        blob_metadata_path=blob_metadata_path,
        output_dir=output_dir,
    )
    binding_summary = run_motion_binding(
        blob_feature_context_path=output_dir / "blob_feature_context.json",
        motion_prompt_path=motion_prompt,
        output_dir=output_dir,
    )
    event_stats = run_bound_motion_event_generation(
        image_path=copied_image_path,
        semantic_mask_path=semantic_mask_path,
        blob_map_path=blob_map_path,
        bound_blob_motion_path=output_dir / "bound_blob_motion.json",
        output_dir=output_dir,
        contrast_threshold=threshold,
    )
    validation_report = run_validation(
        events_path=output_dir / "events.npy",
        semantic_mask_path=semantic_mask_path,
        blob_map_path=blob_map_path,
        output_dir=output_dir,
        image_path=copied_image_path,
        bound_blob_motion_path=output_dir / "bound_blob_motion.json",
        thresholds=[threshold * 0.5, threshold, threshold * 1.5],
    )

    summary = {
        "mode": "prompt_to_direct_event",
        "status": "ok",
        "fixture": bool(fixture),
        "sample_index": int(sample_index),
        "annotation_file": source_info.get("annotation_file"),
        "annotation_line_index": source_info.get("annotation_line_index"),
        "image_path": source_info.get("original_image_path"),
        "copied_image_path": str(copied_image_path),
        "motion_prompt": str(motion_prompt),
        "threshold": float(threshold),
        "seed": int(seed),
        "num_blobs": len(blob_meta),
        "num_events": int(event_stats["total_events"]),
        "validation_passed": bool(validation_report["passed"]),
        "summaries": {
            "face_feature_adapter": feature_summary,
            "motion_binding": binding_summary,
            "event_generation": {
                "total_events": int(event_stats["total_events"]),
                "polarity_distribution": event_stats["polarity_distribution"],
            },
            "validation": {
                "passed": bool(validation_report["passed"]),
                "polarity_counts": validation_report["polarity_counts"],
            },
        },
        "required_outputs": [
            "blob_feature_context.json",
            "bound_blob_motion.json",
            "events.npy",
            "events.csv",
            "event_preview.png",
            "validation_report.json",
            "run_manifest.json",
        ],
        "notes": [
            "LivePortrait is not used.",
            "No video is generated.",
            "Events are generated directly from image gradients and bound blob motion.",
        ],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = build_run_manifest(
        sample_index=sample_index,
        motion_prompt=motion_prompt,
        output_dir=output_dir,
        threshold=threshold,
        seed=seed,
        n_segments=n_segments,
        fixture=fixture,
        source_info=source_info,
        copied_image_path=copied_image_path,
        landmarks_path=landmarks_path,
        feature_summary=feature_summary,
        binding_summary=binding_summary,
        event_stats=event_stats,
        validation_report=validation_report,
        num_blobs=len(blob_meta),
    )
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def build_run_manifest(
    sample_index: int,
    motion_prompt: Path,
    output_dir: Path,
    threshold: float,
    seed: int,
    n_segments: int,
    fixture: bool,
    source_info: dict,
    copied_image_path: Path,
    landmarks_path: Path,
    feature_summary: dict,
    binding_summary: dict,
    event_stats: dict,
    validation_report: dict,
    num_blobs: int,
) -> dict:
    bound_motion_path = output_dir / "bound_blob_motion.json"
    bound_motion_payload = json.loads(bound_motion_path.read_text(encoding="utf-8"))
    schema_version = bound_motion_payload.get("schema_version")
    bound_vectors = bound_motion_payload.get("blobs", bound_motion_payload.get("bound_blob_motion", []))
    output_files = {
        "source_rgb": output_dir / "source_rgb.png",
        "landmarks": landmarks_path,
        "semantic_mask": output_dir / "semantic_mask.npy",
        "semantic_mask_png": output_dir / "semantic_mask.png",
        "blob_map": output_dir / "blob_map.npy",
        "blob_map_png": output_dir / "blob_map.png",
        "blob_metadata": output_dir / "blob_metadata.json",
        "blob_feature_context": output_dir / "blob_feature_context.json",
        "bound_blob_motion": bound_motion_path,
        "events_npy": output_dir / "events.npy",
        "events_csv": output_dir / "events.csv",
        "event_preview": output_dir / "event_preview.png",
        "event_statistics": output_dir / "event_statistics.json",
        "validation_report": output_dir / "validation_report.json",
        "run_summary": output_dir / "run_summary.json",
        "run_manifest": output_dir / "run_manifest.json",
    }
    return {
        "mode": "prompt_to_direct_event",
        "command": {
            "entrypoint": "run_prompt_to_event.py",
            "args": {
                "fixture": bool(fixture),
                "sample_index": int(sample_index),
                "motion_prompt": str(motion_prompt),
                "output_dir": str(output_dir),
                "threshold": float(threshold),
                "seed": int(seed),
                "n_segments": int(n_segments),
            },
        },
        "source": {
            "fixture": bool(fixture),
            "original_image_path": source_info.get("original_image_path"),
            "copied_image_path": str(copied_image_path),
            "landmarks_path": str(landmarks_path),
            "annotation_file": source_info.get("annotation_file"),
            "annotation_line_index": source_info.get("annotation_line_index"),
        },
        "config": {
            "motion_prompt_path": str(motion_prompt),
            "threshold": float(threshold),
            "seed": int(seed),
            "n_segments": int(n_segments),
        },
        "outputs": {name: str(path) for name, path in output_files.items()},
        "summary": {
            "schema_version": schema_version,
            "num_blobs": int(num_blobs),
            "num_bound_vectors": int(len(bound_vectors)) if isinstance(bound_vectors, list) else 0,
            "num_events": int(event_stats["total_events"]),
            "validation_passed": bool(validation_report["passed"]),
            "polarity_distribution": event_stats["polarity_distribution"],
            "motion_binding": binding_summary,
            "face_feature_adapter": feature_summary,
        },
        "notes": [
            "Direct image-to-event path.",
            "No RGB video generation.",
            "No LivePortrait motion source.",
            "No image-to-video-to-event logic.",
        ],
    }


def prepare_wflw_source(output_dir: Path, sample_index: int) -> dict:
    train_annotations, _test_annotations = search_wflw_annotations()
    sample = load_sample_at_or_after(train_annotations, sample_index)
    image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {sample.image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    copied_image_path = output_dir / "source_rgb.png"
    cv2.imwrite(str(copied_image_path), image_bgr)

    landmarks_path = output_dir / "landmarks.json"
    if not landmarks_path.exists():
        save_landmarks_json(landmarks_path, sample, (h, w))
    return {
        "image_rgb": image_rgb,
        "image_path": copied_image_path,
        "landmarks_path": landmarks_path,
        "landmarks": sample.landmarks,
        "annotation_file": str(train_annotations),
        "annotation_line_index": int(sample.line_index),
        "original_image_path": str(sample.image_path),
    }


def prepare_fixture_source(output_dir: Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    h, w = 256, 256
    image_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    image_rgb[:] = [28, 30, 34]

    skin = np.array([190, 150, 122], dtype=np.uint8)
    cv2.ellipse(image_rgb, (128, 132), (78, 92), 0, 0, 360, skin.tolist(), -1)
    cv2.ellipse(image_rgb, (128, 122), (54, 50), 0, 0, 360, [202, 162, 132], -1)
    cv2.circle(image_rgb, (96, 104), 9, [238, 238, 232], -1)
    cv2.circle(image_rgb, (160, 104), 9, [238, 238, 232], -1)
    cv2.circle(image_rgb, (97, 105), 4, [32, 40, 48], -1)
    cv2.circle(image_rgb, (159, 105), 4, [32, 40, 48], -1)
    cv2.ellipse(image_rgb, (128, 154), (13, 22), 0, 0, 360, [154, 112, 92], -1)
    cv2.ellipse(image_rgb, (128, 178), (33, 12), 0, 0, 360, [158, 45, 62], -1)
    cv2.ellipse(image_rgb, (128, 178), (23, 5), 0, 0, 360, [54, 20, 28], -1)
    cv2.rectangle(image_rgb, (78, 79), (112, 85), [72, 48, 42], -1)
    cv2.rectangle(image_rgb, (146, 79), (181, 85), [72, 48, 42], -1)

    noise = rng.normal(0, 3, image_rgb.shape).astype(np.int16)
    image_rgb = np.clip(image_rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    image_path = output_dir / "source_rgb.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    landmarks = synthetic_wflw_landmarks()
    landmarks_path = output_dir / "landmarks.json"
    landmarks_payload = {
        "dataset_name": "synthetic_fixture_wflw_shape",
        "image_path": str(image_path),
        "image_height": h,
        "image_width": w,
        "coordinate_format": "xy",
        "num_keypoints": 98,
        "landmarks": landmarks,
    }
    landmarks_path.write_text(json.dumps(landmarks_payload, indent=2), encoding="utf-8")
    return {
        "image_rgb": image_rgb,
        "image_path": image_path,
        "landmarks_path": landmarks_path,
        "landmarks": landmarks,
        "annotation_file": None,
        "annotation_line_index": None,
        "original_image_path": str(image_path),
    }


def synthetic_wflw_landmarks() -> list[dict]:
    points = [{"id": i, "x": 128.0, "y": 132.0, "visibility": True} for i in range(98)]

    for j, idx in enumerate(range(0, 33)):
        theta = np.pi + (j / 32.0) * np.pi
        points[idx]["x"] = float(128 + 78 * np.cos(theta))
        points[idx]["y"] = float(132 + 92 * np.sin(theta))

    for j, idx in enumerate(range(33, 42)):
        x = 78 + j * 4.2
        points[idx]["x"] = float(x)
        points[idx]["y"] = float(82 + 3.0 * np.sin(j / 8.0 * np.pi))

    for j, idx in enumerate(range(42, 51)):
        x = 146 + j * 4.2
        points[idx]["x"] = float(x)
        points[idx]["y"] = float(82 + 3.0 * np.sin(j / 8.0 * np.pi))

    nose_grid = [
        (119, 118),
        (128, 115),
        (137, 118),
        (117, 137),
        (128, 134),
        (139, 137),
        (113, 154),
        (128, 158),
        (143, 154),
    ]
    for idx, (x, y) in zip(range(51, 60), nose_grid):
        points[idx]["x"] = float(x)
        points[idx]["y"] = float(y)

    add_ellipse_points(points, range(60, 68), center=(96, 104), radius=(14, 8))
    add_ellipse_points(points, range(68, 76), center=(160, 104), radius=(14, 8))
    add_ellipse_points(points, range(76, 96), center=(128, 178), radius=(35, 14))

    for j, idx in enumerate(range(96, 98)):
        points[idx]["x"] = float(112 + j * 32)
        points[idx]["y"] = 199.0
    return points


def add_ellipse_points(points: list[dict], indices: range, center: tuple[float, float], radius: tuple[float, float]) -> None:
    indices_list = list(indices)
    for j, idx in enumerate(indices_list):
        theta = (j / len(indices_list)) * 2.0 * np.pi
        points[idx]["x"] = float(center[0] + radius[0] * np.cos(theta))
        points[idx]["y"] = float(center[1] + radius[1] * np.sin(theta))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WFLW prompt-to-direct-event pipeline.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--motion-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-segments", type=int, default=180)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()

    summary = run_prompt_to_event(
        sample_index=args.sample_index,
        motion_prompt=args.motion_prompt,
        output_dir=args.output_dir,
        threshold=args.threshold,
        seed=args.seed,
        n_segments=args.n_segments,
        fixture=args.fixture,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "num_blobs": summary["num_blobs"],
                "num_events": summary["num_events"],
                "validation_passed": summary["validation_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
