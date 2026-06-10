# Active Pipeline

The active path is direct image-to-event generation from a single RGB face image, WFLW landmarks, and externally provided motion prompts.

```text
RGB face image + WFLW landmarks
-> semantic mask / blob map
-> face_feature_adapter.py
-> blob_feature_context.json
-> motion_binding.py
-> bound_blob_motion.json
-> bound_motion_events.py
-> events.npy / events.csv / event_preview.png
-> validate_bound_motion_events.py
-> validation_report.json
```

## Modules

### `i2e_face_mvp/face_feature_adapter.py`

Inputs:

- RGB source image
- WFLW landmark JSON
- `semantic_mask.npy`
- `blob_map.npy`
- `blob_metadata.json`

Outputs:

- `face_feature_state.json`
- `blob_feature_context.json`
- feature overlays for inspection

Responsibility: convert landmarks, semantic regions, blob metadata, and local appearance into a stable face/blob feature context. It does not generate motion or events.

### `i2e_face_mvp/motion_binding.py`

Inputs:

- `blob_feature_context.json`
- external motion prompt JSON

Outputs:

- `bound_blob_motion.json`
- `bound_motion_overlay.png`

Responsibility: bind externally supplied motion prompts to blob-level motion vectors. Canonical motion rows are stored under `blobs`; the legacy `bound_blob_motion` alias is kept for compatibility.

### `i2e_face_mvp/bound_motion_events.py`

Inputs:

- RGB source image
- `semantic_mask.npy`
- `blob_map.npy`
- `bound_blob_motion.json`

Outputs:

- `events.npy`
- `events.csv`
- `event_preview.png`
- `event_statistics.json`

Responsibility: convert blob motion vectors into dense flow, compute directional log-intensity changes from image gradients, accumulate contrast, and emit ON/OFF events.

### `i2e_face_mvp/validate_bound_motion_events.py`

Inputs:

- `events.npy` or `events.csv`
- `semantic_mask.npy`
- `blob_map.npy`
- optional source image and bound motion JSON for threshold sensitivity checks

Outputs:

- `validation_report.json`
- event count CSV summaries

Responsibility: validate event columns, image bounds, background-event policy, polarity counts, per-blob counts, per-semantic counts, and optional threshold sensitivity.

## Explicit Non-Goals

The active path does not use:

- RGB video generation
- image-to-video-to-event conversion
- LivePortrait as a motion source
- learning modules or training loops
