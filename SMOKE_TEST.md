# Main Pipeline Smoke Test

Run the clean direct image-to-event fixture path:

```bash
python3 run_prompt_to_event.py --fixture --motion-prompt motion_prompt.json --output-dir outputs/prompt_to_event_fixture --threshold 0.08 --seed 0
```

This command verifies the main handoff:

```text
RGB fixture image + WFLW fixture landmarks
-> semantic mask / blob map
-> face_feature_adapter.py
-> blob_feature_context.json
-> motion_binding.py
-> bound_blob_motion.json
-> bound_motion_events.py
-> events.npy / events.csv / event_preview.png
-> validate_bound_motion_events.py
```

Expected required outputs:

```text
blob_feature_context.json
bound_blob_motion.json
events.npy
events.csv
event_preview.png
validation_report.json
```

The motion-vector JSON uses `schema_version: "bound_blob_motion.v1"` and keeps downstream vectors under the stable `bound_blob_motion` list. Each row contains `blob_id`, `semantic_id`, `semantic_label`, `centroid`, `bbox`, `motion_px`, `motion_context`, and `source_prompts`.

Run every supported prompt primitive through the same clean path:

```bash
bash scripts/smoke_prompt_primitives.sh
```

The primitive prompt fixtures live in `prompt_fixtures/`, and outputs are written under `outputs/prompt_primitive_smoke/` by default. Override `THRESHOLD`, `SEED`, or `OUTPUT_ROOT` in the environment when needed.

The canonical motion-vector schema fields are `source`, `motion_prompt`, `blobs`, and `summary`. The legacy `bound_blob_motion` list remains present as a compatibility alias for older downstream readers.
