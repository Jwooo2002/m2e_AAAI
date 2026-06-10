from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .events import EVENT_DTYPE
except ImportError:  # Allows `python i2e_face_mvp/validate_bound_motion_events.py ...`.
    from events import EVENT_DTYPE


EXPECTED_COLUMNS = ["x", "y", "t", "p", "blob_id", "semantic_id"]


def run_validation(
    events_path: Path,
    semantic_mask_path: Path,
    blob_map_path: Path,
    output_dir: Path,
    allow_background_events: bool = False,
    image_path: Path | None = None,
    bound_blob_motion_path: Path | None = None,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    events = load_events(events_path)
    semantic_mask = np.load(semantic_mask_path)
    blob_map = np.load(blob_map_path)
    validate_mask_shapes(semantic_mask, blob_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    column_check = check_event_columns(events)
    bounds_check = check_bounds(events, semantic_mask.shape)
    background_check = check_background_events(events, semantic_mask, allow_background_events)
    polarity = polarity_counts(events)
    blob_counts = event_count_by_blob(events, blob_map)
    semantic_counts = event_count_by_semantic(events, semantic_mask)
    threshold_sensitivity = run_threshold_sensitivity(
        image_path=image_path,
        semantic_mask_path=semantic_mask_path,
        blob_map_path=blob_map_path,
        bound_blob_motion_path=bound_blob_motion_path,
        output_dir=output_dir,
        thresholds=thresholds or [0.04, 0.08, 0.12],
    )

    write_count_csv(output_dir / "event_count_by_blob.csv", blob_counts, ["blob_id", "pixel_count", "event_count", "on", "off"])
    write_count_csv(
        output_dir / "event_count_by_semantic.csv",
        semantic_counts,
        ["semantic_id", "pixel_count", "event_count", "on", "off"],
    )

    report = {
        "mode": "validate_bound_motion_events",
        "events_path": str(events_path),
        "semantic_mask_path": str(semantic_mask_path),
        "blob_map_path": str(blob_map_path),
        "image_shape": [int(semantic_mask.shape[0]), int(semantic_mask.shape[1])],
        "total_events": int(len(events)),
        "checks": {
            "columns": column_check,
            "bounds": bounds_check,
            "background": background_check,
        },
        "polarity_counts": polarity,
        "event_counts_per_blob": blob_counts,
        "event_counts_per_semantic_region": semantic_counts,
        "threshold_sensitivity": threshold_sensitivity,
        "passed": bool(column_check["passed"] and bounds_check["passed"] and background_check["passed"]),
        "outputs": ["validation_report.json", "event_count_by_blob.csv", "event_count_by_semantic.csv"],
    }
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_events(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        events = np.load(path, allow_pickle=False)
    elif suffix == ".csv":
        events = load_events_csv(path)
    else:
        raise ValueError(f"Unsupported events extension: {path.suffix}")
    if events.dtype.names is None:
        raise ValueError("events must be a structured array with named columns")
    return events


def load_events_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.size == 0:
        return np.empty((0,), dtype=EVENT_DTYPE)
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)
    out = np.empty((len(data),), dtype=EVENT_DTYPE)
    for name in EXPECTED_COLUMNS:
        out[name] = data[name]
    return out


def check_event_columns(events: np.ndarray) -> dict[str, Any]:
    names = list(events.dtype.names or [])
    missing = [name for name in EXPECTED_COLUMNS if name not in names]
    extra = [name for name in names if name not in EXPECTED_COLUMNS]
    dtype_matches = events.dtype == EVENT_DTYPE
    return {
        "passed": len(missing) == 0,
        "columns": names,
        "expected_columns": EXPECTED_COLUMNS,
        "missing": missing,
        "extra": extra,
        "dtype_matches_expected": bool(dtype_matches),
    }


def check_bounds(events: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
    h, w = image_shape
    if len(events) == 0:
        return {"passed": True, "out_of_bounds_count": 0, "examples": []}
    valid = (events["x"] >= 0) & (events["x"] < w) & (events["y"] >= 0) & (events["y"] < h)
    bad = np.where(~valid)[0]
    return {
        "passed": int(len(bad)) == 0,
        "out_of_bounds_count": int(len(bad)),
        "examples": event_examples(events, bad[:10]),
    }


def check_background_events(
    events: np.ndarray,
    semantic_mask: np.ndarray,
    allow_background_events: bool,
) -> dict[str, Any]:
    if len(events) == 0:
        return {"passed": True, "background_event_count": 0, "examples": [], "allow_background_events": bool(allow_background_events)}
    h, w = semantic_mask.shape
    in_bounds = (events["x"] >= 0) & (events["x"] < w) & (events["y"] >= 0) & (events["y"] < h)
    background = np.zeros(len(events), dtype=bool)
    idx = np.where(in_bounds)[0]
    background[idx] = semantic_mask[events["y"][idx], events["x"][idx]] == 0
    bad = np.where(background)[0]
    return {
        "passed": bool(allow_background_events or len(bad) == 0),
        "allow_background_events": bool(allow_background_events),
        "background_event_count": int(len(bad)),
        "examples": event_examples(events, bad[:10]),
    }


def polarity_counts(events: np.ndarray) -> dict[str, Any]:
    if len(events) == 0:
        return {"on": 0, "off": 0, "other": 0, "on_fraction": 0.0, "off_fraction": 0.0}
    on = int((events["p"] > 0).sum())
    off = int((events["p"] < 0).sum())
    other = int((events["p"] == 0).sum())
    total = int(len(events))
    return {
        "on": on,
        "off": off,
        "other": other,
        "on_fraction": safe_div(on, total),
        "off_fraction": safe_div(off, total),
    }


def event_count_by_blob(events: np.ndarray, blob_map: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    event_blob = events["blob_id"].astype(np.int32) if len(events) else np.empty((0,), dtype=np.int32)
    polarity = events["p"] if len(events) else np.empty((0,), dtype=np.int8)
    for blob_id in sorted(int(v) for v in np.unique(blob_map) if int(v) > 0):
        mask = event_blob == blob_id
        rows.append(
            {
                "blob_id": blob_id,
                "pixel_count": int((blob_map == blob_id).sum()),
                "event_count": int(mask.sum()),
                "on": int(((polarity > 0) & mask).sum()),
                "off": int(((polarity < 0) & mask).sum()),
            }
        )
    rows.sort(key=lambda row: (-row["event_count"], row["blob_id"]))
    return rows


def event_count_by_semantic(events: np.ndarray, semantic_mask: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    event_sem = events["semantic_id"].astype(np.int32) if len(events) else np.empty((0,), dtype=np.int32)
    polarity = events["p"] if len(events) else np.empty((0,), dtype=np.int8)
    for semantic_id in sorted(int(v) for v in np.unique(semantic_mask)):
        mask = event_sem == semantic_id
        rows.append(
            {
                "semantic_id": semantic_id,
                "pixel_count": int((semantic_mask == semantic_id).sum()),
                "event_count": int(mask.sum()),
                "on": int(((polarity > 0) & mask).sum()),
                "off": int(((polarity < 0) & mask).sum()),
            }
        )
    rows.sort(key=lambda row: (-row["event_count"], row["semantic_id"]))
    return rows


def run_threshold_sensitivity(
    image_path: Path | None,
    semantic_mask_path: Path,
    blob_map_path: Path,
    bound_blob_motion_path: Path | None,
    output_dir: Path,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    if image_path is None or bound_blob_motion_path is None:
        return [
            {
                "status": "skipped",
                "reason": "pass --image and --bound-blob-motion to test threshold sensitivity",
                "thresholds": [float(v) for v in thresholds],
            }
        ]
    try:
        from .bound_motion_events import run_bound_motion_event_generation
    except ImportError:  # Allows direct script execution.
        from bound_motion_events import run_bound_motion_event_generation

    rows = []
    for threshold in thresholds[:3]:
        subdir = output_dir / f"threshold_{float(threshold):.4f}".replace(".", "p")
        stats = run_bound_motion_event_generation(
            image_path=image_path,
            semantic_mask_path=semantic_mask_path,
            blob_map_path=blob_map_path,
            bound_blob_motion_path=bound_blob_motion_path,
            output_dir=subdir,
            contrast_threshold=float(threshold),
        )
        rows.append(
            {
                "threshold": float(threshold),
                "total_events": int(stats["total_events"]),
                "on": int(stats["polarity_distribution"]["on"]),
                "off": int(stats["polarity_distribution"]["off"]),
                "output_dir": str(subdir),
            }
        )
    return rows


def write_count_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def event_examples(events: np.ndarray, indices: np.ndarray) -> list[dict[str, Any]]:
    examples = []
    for idx in indices:
        event = events[int(idx)]
        examples.append({name: scalar_to_python(event[name]) for name in EXPECTED_COLUMNS if name in events.dtype.names})
    return examples


def scalar_to_python(value: Any) -> int | float:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def validate_mask_shapes(semantic_mask: np.ndarray, blob_map: np.ndarray) -> None:
    if semantic_mask.shape != blob_map.shape:
        raise ValueError(f"semantic_mask and blob_map shape mismatch: {semantic_mask.shape} vs {blob_map.shape}")


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(v.strip()) for v in value.split(",") if v.strip()]
    if len(thresholds) != 3:
        raise ValueError("--thresholds must contain exactly 3 comma-separated values")
    if any(v <= 0.0 for v in thresholds):
        raise ValueError("threshold values must be positive")
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate direct bound-motion event outputs.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--semantic-mask", type=Path, required=True)
    parser.add_argument("--blob-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-background-events", action="store_true")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--bound-blob-motion", type=Path, default=None)
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("0.04,0.08,0.12"))
    args = parser.parse_args()

    report = run_validation(
        events_path=args.events,
        semantic_mask_path=args.semantic_mask,
        blob_map_path=args.blob_map,
        output_dir=args.output_dir,
        allow_background_events=args.allow_background_events,
        image_path=args.image,
        bound_blob_motion_path=args.bound_blob_motion,
        thresholds=args.thresholds,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "passed": report["passed"], "total_events": report["total_events"]}, indent=2))


if __name__ == "__main__":
    main()
