from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .semantics import REGION_LABELS


TARGET_REGION_LABELS = {"left_eye", "right_eye", "mouth", "jaw_or_contour"}


def save_event_analysis(
    output_dir: Path,
    image_rgb: np.ndarray,
    semantic_mask: np.ndarray,
    blob_map: np.ndarray,
    blob_metadata: list[dict],
    events: np.ndarray,
    temporal_bins: int = 12,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    density = event_density_map(semantic_mask.shape, events)
    density_vis = render_density_png(density)
    cv2.imwrite(str(output_dir / "event_density.png"), cv2.cvtColor(density_vis, cv2.COLOR_RGB2BGR))

    overlay = render_heatmap_overlay(image_rgb, density)
    cv2.imwrite(str(output_dir / "event_heatmap_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    stats = build_event_statistics(semantic_mask, blob_map, blob_metadata, events, density, temporal_bins)
    (output_dir / "event_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def event_density_map(image_shape: tuple[int, int], events: np.ndarray) -> np.ndarray:
    h, w = image_shape
    density = np.zeros((h, w), dtype=np.float32)
    if len(events) == 0:
        return density
    ys = np.clip(events["y"].astype(np.int64), 0, h - 1)
    xs = np.clip(events["x"].astype(np.int64), 0, w - 1)
    np.add.at(density, (ys, xs), 1.0)
    return density


def render_density_png(density: np.ndarray) -> np.ndarray:
    normalized = normalize_density(density)
    heat = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat[normalized <= 0.0] = 0
    return heat


def render_heatmap_overlay(image_rgb: np.ndarray, density: np.ndarray) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[:2] != density.shape:
        raise ValueError("image_rgb and density must share the same height and width")
    base = image_rgb.astype(np.float32)
    normalized = normalize_density(density)
    heat_bgr = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = (normalized ** 0.7)[:, :, None] * 0.7
    overlay = base * (1.0 - alpha) + heat * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def normalize_density(density: np.ndarray) -> np.ndarray:
    if density.size == 0 or float(density.max()) <= 0.0:
        return np.zeros_like(density, dtype=np.float32)
    blurred = cv2.GaussianBlur(density.astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0)
    positive = blurred[blurred > 0]
    scale = float(np.percentile(positive, 99.0)) if positive.size else float(blurred.max())
    if scale <= 0.0:
        scale = float(blurred.max())
    return np.clip(blurred / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


def build_event_statistics(
    semantic_mask: np.ndarray,
    blob_map: np.ndarray,
    blob_metadata: list[dict],
    events: np.ndarray,
    density: np.ndarray,
    temporal_bins: int,
) -> dict[str, Any]:
    total = int(len(events))
    h, w = semantic_mask.shape
    on_count = int((events["p"] > 0).sum()) if total else 0
    off_count = int((events["p"] < 0).sum()) if total else 0
    region_stats = semantic_region_statistics(semantic_mask, events)
    blob_stats = blob_event_statistics(blob_map, blob_metadata, events)
    temporal_hist = temporal_histogram(events, temporal_bins)

    target_region_ids = [
        int(region_id)
        for region_id, label in REGION_LABELS.items()
        if label in TARGET_REGION_LABELS
    ]
    target_event_count = int(sum(s["event_count"] for s in region_stats if s["region_id"] in target_region_ids))
    target_pixel_count = int(sum(s["pixel_count"] for s in region_stats if s["region_id"] in target_region_ids))
    foreground_pixel_count = int((semantic_mask > 0).sum())
    target_event_fraction = safe_div(target_event_count, total)
    target_foreground_area_fraction = safe_div(target_pixel_count, foreground_pixel_count)
    target_enrichment = safe_div(target_event_fraction, target_foreground_area_fraction)

    return {
        "image_shape": [int(h), int(w)],
        "total_events": total,
        "density": {
            "max_events_per_pixel": int(density.max()) if density.size else 0,
            "active_pixels": int((density > 0).sum()),
            "active_pixel_fraction": safe_div(int((density > 0).sum()), int(h * w)),
            "mean_events_per_active_pixel": safe_div(total, int((density > 0).sum())),
        },
        "polarity_distribution": {
            "on": on_count,
            "off": off_count,
            "on_fraction": safe_div(on_count, total),
            "off_fraction": safe_div(off_count, total),
        },
        "semantic_region_event_counts": region_stats,
        "blob_event_counts": blob_stats,
        "temporal_event_histogram": temporal_hist,
        "meaningful_region_check": {
            "target_regions": sorted(TARGET_REGION_LABELS),
            "target_event_count": target_event_count,
            "target_event_fraction": target_event_fraction,
            "target_foreground_area_fraction": target_foreground_area_fraction,
            "target_enrichment_vs_foreground_area": target_enrichment,
            "background_event_count": int(next((s["event_count"] for s in region_stats if s["region_id"] == 0), 0)),
        },
    }


def semantic_region_statistics(semantic_mask: np.ndarray, events: np.ndarray) -> list[dict[str, Any]]:
    total = int(len(events))
    stats = []
    event_semantic_ids = events["semantic_id"].astype(np.int64) if total else np.empty((0,), dtype=np.int64)
    event_polarities = events["p"] if total else np.empty((0,), dtype=np.int8)
    for region_id, label in REGION_LABELS.items():
        pixel_count = int((semantic_mask == region_id).sum())
        if total:
            region_events = event_semantic_ids == region_id
            event_count = int(region_events.sum())
            on_count = int(((event_polarities > 0) & region_events).sum())
            off_count = int(((event_polarities < 0) & region_events).sum())
        else:
            event_count = 0
            on_count = 0
            off_count = 0
        stats.append(
            {
                "region_id": int(region_id),
                "label": label,
                "pixel_count": pixel_count,
                "event_count": event_count,
                "events_per_1000_pixels": safe_div(event_count * 1000.0, pixel_count),
                "event_fraction": safe_div(event_count, total),
                "on": on_count,
                "off": off_count,
            }
        )
    return stats


def blob_event_statistics(blob_map: np.ndarray, blob_metadata: list[dict], events: np.ndarray) -> list[dict[str, Any]]:
    total = int(len(events))
    blob_ids = events["blob_id"].astype(np.int64) if total else np.empty((0,), dtype=np.int64)
    polarities = events["p"] if total else np.empty((0,), dtype=np.int8)
    rows = []
    for meta in blob_metadata:
        blob_id = int(meta["blob_id"])
        pixel_count = int(meta.get("pixel_count", int((blob_map == blob_id).sum())))
        if total:
            blob_events = blob_ids == blob_id
            event_count = int(blob_events.sum())
            on_count = int(((polarities > 0) & blob_events).sum())
            off_count = int(((polarities < 0) & blob_events).sum())
        else:
            event_count = 0
            on_count = 0
            off_count = 0
        rows.append(
            {
                "blob_id": blob_id,
                "semantic_id": int(meta.get("semantic_id", -1)),
                "semantic_label": meta.get("semantic_label", "unknown"),
                "pixel_count": pixel_count,
                "event_count": event_count,
                "events_per_1000_pixels": safe_div(event_count * 1000.0, pixel_count),
                "event_fraction": safe_div(event_count, total),
                "on": on_count,
                "off": off_count,
                "centroid": meta.get("centroid"),
                "bbox": meta.get("bbox"),
            }
        )
    rows.sort(key=lambda row: (-row["event_count"], row["blob_id"]))
    return rows


def temporal_histogram(events: np.ndarray, bins: int) -> list[dict[str, Any]]:
    total = int(len(events))
    bins = max(1, int(bins))
    if total == 0:
        return []
    t = events["t"].astype(np.float32)
    start = float(t.min())
    end = float(t.max())
    if start == end:
        edges = np.linspace(start, start + 1.0, bins + 1, dtype=np.float32)
    else:
        edges = np.linspace(start, end, bins + 1, dtype=np.float32)
    rows = []
    for i in range(bins):
        if i == bins - 1:
            mask = (t >= edges[i]) & (t <= edges[i + 1])
        else:
            mask = (t >= edges[i]) & (t < edges[i + 1])
        count = int(mask.sum())
        rows.append(
            {
                "bin_index": int(i),
                "t_start_ms": round(float(edges[i]), 4),
                "t_end_ms": round(float(edges[i + 1]), 4),
                "event_count": count,
                "event_fraction": safe_div(count, total),
                "on": int(((events["p"] > 0) & mask).sum()),
                "off": int(((events["p"] < 0) & mask).sum()),
            }
        )
    return rows


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def load_image_for_output(output_dir: Path, image_path: Path | None) -> np.ndarray:
    if image_path is None:
        summary_path = output_dir / "run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError("Pass --image-path or provide run_summary.json in the output directory")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        image_path = Path(summary["image_path"])
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated facial events.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--temporal-bins", type=int, default=12)
    args = parser.parse_args()

    image_rgb = load_image_for_output(args.output_dir, args.image_path)
    semantic_mask = np.load(args.output_dir / "semantic_mask.npy")
    blob_map = np.load(args.output_dir / "blob_map.npy")
    events = np.load(args.output_dir / "events.npy")
    blob_metadata = json.loads((args.output_dir / "blob_metadata.json").read_text(encoding="utf-8"))
    stats = save_event_analysis(
        args.output_dir,
        image_rgb,
        semantic_mask,
        blob_map,
        blob_metadata,
        events,
        temporal_bins=args.temporal_bins,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "total_events": stats["total_events"]}, indent=2))


if __name__ == "__main__":
    main()
