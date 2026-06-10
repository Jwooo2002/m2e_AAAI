from __future__ import annotations


def build_blob_prompts(blob_metadata: list[dict], duration_ms: int = 50) -> list[dict]:
    prompts = []
    for blob in blob_metadata:
        semantic = blob["semantic_label"]
        motion = motion_for_blob(blob)
        prompts.append(
            {
                "blob_id": blob["blob_id"],
                "semantic_label": semantic,
                "appearance_context": {
                    "bbox": blob["bbox"],
                    "centroid": blob["centroid"],
                    "area": blob["pixel_count"],
                    "mean_rgb": blob["mean_rgb"],
                    "local_texture_strength": blob["local_texture_strength"],
                },
                "motion_context": {
                    **motion,
                    "duration_ms": duration_ms,
                },
                "event_generation_context": {
                    "contrast_threshold": 0.08,
                    "noise_level": 0.0,
                    "timestamp_start": 0.0,
                    "timestamp_end": float(duration_ms),
                },
            }
        )
    return prompts


def motion_for_blob(blob: dict) -> dict:
    label = blob["semantic_label"]
    cx, cy = blob["centroid"]
    phase = ((blob["blob_id"] * 17) % 11 - 5) / 20.0
    if label in {"left_eye", "right_eye"}:
        return {
            "motion_type": "blink",
            "direction": [0.0, 1.0],
            "magnitude": 1.6 + phase,
            "temporal_profile": "ease_in_out",
            "polarity_hint": "mixed",
        }
    if label == "mouth":
        return {
            "motion_type": "mouth_open",
            "direction": [0.0, 1.0 if cy >= 0 else 0.8],
            "magnitude": 2.0 + phase,
            "temporal_profile": "sinusoidal",
            "polarity_hint": "mixed",
        }
    if label == "nose":
        return {
            "motion_type": "rigid",
            "direction": [0.35, 0.05],
            "magnitude": 0.45,
            "temporal_profile": "linear",
            "polarity_hint": "mixed",
        }
    if label in {"jaw_or_contour"}:
        return {
            "motion_type": "contour_shift",
            "direction": [0.55, 0.25],
            "magnitude": 1.1 + phase,
            "temporal_profile": "linear",
            "polarity_hint": "mixed",
        }
    if label in {"left_eyebrow", "right_eyebrow"}:
        return {
            "motion_type": "residual",
            "direction": [0.05, -1.0],
            "magnitude": 0.9 + phase,
            "temporal_profile": "ease_in_out",
            "polarity_hint": "mixed",
        }
    return {
        "motion_type": "residual",
        "direction": [0.25, 0.1],
        "magnitude": 0.45 + abs(phase) * 0.5,
        "temporal_profile": "sinusoidal",
        "polarity_hint": "mixed",
    }
