from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .analysis import save_event_analysis
from .debug_motion import run_debug_controlled_motion
from .events import generate_events, save_event_preview, save_event_stream_gif, save_events_csv, save_motion_preview_gif
from .prompts import build_blob_prompts
from .semantics import REGION_LABELS, build_semantic_mask, save_semantic_png
from .superpixels import blob_metadata, save_blob_map_png, slic_superpixels
from .wflw import WFLWSample, load_first_valid_sample, save_landmarks_json, search_wflw_annotations


def run_pipeline(
    sample: WFLWSample,
    output_dir: Path,
    n_segments: int = 180,
    seed: int = 0,
    debug_motion_prompt: Path | None = None,
) -> dict:
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {sample.image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    warnings: list[str] = []
    save_landmarks_json(output_dir / "landmarks.json", sample, (h, w))
    anchors = json.loads((output_dir / "landmarks.json").read_text(encoding="utf-8"))["semantic_anchors"]
    (output_dir / "semantic_anchors.json").write_text(json.dumps(anchors, indent=2), encoding="utf-8")

    semantic_mask, semantic_stats, semantic_warnings = build_semantic_mask((h, w), sample.landmarks)
    warnings.extend(semantic_warnings)
    np.save(output_dir / "semantic_mask.npy", semantic_mask)
    save_semantic_png(output_dir / "semantic_mask.png", semantic_mask)

    foreground = semantic_mask > 0
    raw_blob_map = slic_superpixels(image_rgb, foreground, n_segments=n_segments)
    blob_map, metadata, blob_warnings = blob_metadata(image_rgb, raw_blob_map, semantic_mask)
    warnings.extend(blob_warnings)
    np.save(output_dir / "blob_map.npy", blob_map)
    save_blob_map_png(output_dir / "blob_map.png", blob_map)
    (output_dir / "blob_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    prompts = build_blob_prompts(metadata)
    (output_dir / "blob_prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    save_motion_preview_gif(output_dir / "motion_preview.gif", image_rgb, blob_map, prompts)
    save_motion_preview_gif(
        output_dir / "motion_preview_foreground_rgb.gif",
        image_rgb,
        blob_map,
        prompts,
        foreground_mask=foreground,
        background_mode="foreground_rgb",
    )

    events = generate_events(image_rgb, blob_map, metadata, prompts)
    if len(events) == 0:
        warnings.append("zero-event output")
    np.save(output_dir / "events.npy", events)
    save_events_csv(output_dir / "events.csv", events)
    save_event_preview(output_dir / "event_frame_preview.png", (h, w), events)
    save_event_stream_gif(output_dir / "event_stream.gif", image_rgb, events)
    save_event_stream_gif(
        output_dir / "event_stream_foreground_rgb.gif",
        image_rgb,
        events,
        foreground_mask=foreground,
        background_mode="foreground_rgb",
    )
    save_event_stream_gif(output_dir / "event_stream_black.gif", image_rgb, events, background_mode="black")
    event_stats = save_event_analysis(output_dir, image_rgb, semantic_mask, blob_map, metadata, events)
    debug_motion_summary = None
    if debug_motion_prompt is not None:
        debug_motion_summary = run_debug_controlled_motion(
            output_dir,
            image_rgb,
            semantic_mask,
            blob_map,
            metadata,
            debug_motion_prompt,
        )

    semantic_counts = {
        REGION_LABELS[int(i)]: int((semantic_mask == i).sum())
        for i in np.unique(semantic_mask)
        if int(i) in REGION_LABELS
    }
    summary = {
        "status": "ok",
        "dataset": "WFLW",
        "annotation_file": str(sample.annotation_file),
        "annotation_line_index": sample.line_index,
        "image_path": str(sample.image_path),
        "image_shape": [h, w, 3],
        "num_landmarks": len(sample.landmarks),
        "num_semantic_regions_present": int(sum(1 for s in semantic_stats if s["area"] > 0)),
        "semantic_region_stats": semantic_stats,
        "semantic_pixel_counts": semantic_counts,
        "num_blobs": len(metadata),
        "num_events": int(len(events)),
        "event_statistics": event_stats["meaningful_region_check"],
        "debug_controlled_motion": None
        if debug_motion_summary is None
        else {
            "output_dir": str(output_dir / "debug_controlled_motion"),
            "num_controlled_blob_prompts": debug_motion_summary["num_controlled_blob_prompts"],
            "num_events": debug_motion_summary["num_events"],
            "warnings": debug_motion_summary["warnings"],
        },
        "output_dir": str(output_dir),
        "warnings": warnings,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WFLW context-prompted image-to-event MVP sample.")
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wflw_mvp_sample"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--n-segments", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-motion-prompt", type=Path, default=None)
    args = parser.parse_args()

    annotation_file = args.annotation_file
    if annotation_file is None:
        train, _test = search_wflw_annotations()
        annotation_file = train

    sample = load_sample_at_or_after(annotation_file, args.sample_index)
    summary = run_pipeline(
        sample,
        args.output_dir,
        n_segments=args.n_segments,
        seed=args.seed,
        debug_motion_prompt=args.debug_motion_prompt,
    )
    print(json.dumps({"output_dir": summary["output_dir"], "num_blobs": summary["num_blobs"], "num_events": summary["num_events"]}, indent=2))


def load_sample_at_or_after(annotation_file: Path, sample_index: int) -> WFLWSample:
    with annotation_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < sample_index:
                continue
            from .wflw import parse_wflw_line

            sample = parse_wflw_line(annotation_file, line, i)
            if sample.image_path.exists():
                return sample
    return load_first_valid_sample(annotation_file)


if __name__ == "__main__":
    main()
