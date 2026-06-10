# Active Schemas

This document covers only the active pipeline artifacts.

## `blob_feature_context.json`

Produced by `face_feature_adapter.py`.

Top-level fields:

- `mode`: `blob_feature_context_only`
- `source_image_path`: source RGB image used by the run
- `num_blobs`: number of blob rows
- `control_groups`: semantic labels grouped for prompting
- `region_consistency_groups`: semantic labels grouped for consistency
- `blob_feature_context`: list of blob feature rows

Each blob row contains:

- `blob_id`: integer id matching `blob_map.npy`
- `semantic_id`: integer id matching `semantic_mask.npy`
- `semantic_label`: semantic region label
- `pixel_count`: number of pixels in the blob
- `centroid`: `[x, y]` in source image pixels
- `bbox`: `[x0, y0, x1, y1]` in source image pixels
- `area_fraction_of_image`: blob area divided by image area
- `appearance_stats`: RGB/luminance/texture summary
- `anchor_distances`: distances to face anchors
- `control_groups`: groups used by motion prompts
- `region_consistency_groups`: consistency groups used by motion prompts
- `neighbor_blob_ids`: neighboring blob ids when available

## `bound_blob_motion.json`

Produced by `motion_binding.py`.

Schema version: `bound_blob_motion.v1`

Canonical top-level fields:

- `schema_version`: `bound_blob_motion.v1`
- `mode`: `external_prompt_to_blob_motion`
- `source`: input feature-context paths
- `motion_prompt`: prompt path, normalized prompts, and supported prompt types
- `blobs`: canonical list of bound motion rows
- `summary`: prompt count, bound vector count, motion magnitude summary, semantic counts, warnings

Compatibility fields:

- `bound_blob_motion`: legacy alias of `blobs`
- `input_prompts`: legacy alias of normalized prompt rows
- `supported_prompts`: legacy supported prompt list
- `blob_feature_context_path`
- `motion_prompt_path`
- `warnings`

Each canonical `blobs` row contains:

- `blob_id`: integer id from `blob_map.npy`
- `semantic_id`: integer id from `semantic_mask.npy`
- `semantic_label`: semantic label
- `centroid`: `[x, y]` in source image pixels
- `bbox`: `[x0, y0, x1, y1]`
- `control_groups`: groups matched by prompts
- `region_consistency_groups`: consistency groups matched by prompts
- `motion_px`: `[dx, dy]`, total displacement in pixels over the event window
- `source_prompts`: prompt contributions that produced the vector
- `motion_context`: normalized direction, magnitude, and polarity hint

Downstream code should read `blobs` first and fall back to `bound_blob_motion` for older files.

## `run_manifest.json`

Produced by `run_prompt_to_event.py` in every run directory.

Top-level fields:

- `mode`: `prompt_to_direct_event`
- `command`: entrypoint and CLI-equivalent args
- `source`: image, landmark, fixture, and annotation metadata
- `config`: motion prompt path, threshold, seed, and segmentation count
- `outputs`: stable paths for generated artifacts
- `summary`: schema version, blob count, bound-vector count, event count, validation result, polarity distribution, and module summaries
- `notes`: explicit direct-path constraints

Use this file as the run index for regression checks.

## `validation_report.json`

Produced by `validate_bound_motion_events.py`.

Top-level fields:

- `mode`: `validate_bound_motion_events`
- `events_path`
- `semantic_mask_path`
- `blob_map_path`
- `image_shape`
- `total_events`
- `checks`: column, bounds, and background checks
- `polarity_counts`: ON/OFF/other counts and fractions
- `event_counts_per_blob`
- `event_counts_per_semantic_region`
- `threshold_sensitivity`
- `passed`: final validation boolean
- `outputs`: validation output file names
