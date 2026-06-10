# Project Status

## Verified Active Path

The clean fixture pipeline is currently verified:

```bash
python3 run_prompt_to_event.py --fixture --motion-prompt motion_prompt.json --output-dir outputs/prompt_to_event_fixture --threshold 0.08 --seed 0
```

The supported motion primitives are covered by prompt fixtures in `prompt_fixtures/` and the smoke runner:

```bash
bash scripts/smoke_prompt_primitives.sh
```

Current verified artifacts:

- `blob_feature_context.json`
- `bound_blob_motion.json` using `schema_version: "bound_blob_motion.v1"`
- `events.npy`
- `events.csv`
- `event_preview.png`
- `validation_report.json`
- `run_manifest.json`

## Current Scope

The active pipeline is direct image-to-event generation from source image gradients and externally supplied prompt-bound blob motion vectors.

Out of scope for the active path:

- RGB video generation
- image-to-video-to-event conversion
- LivePortrait motion sourcing
- learning or training modules

## Immediate Next Steps

- Keep schema-compatible smoke tests passing for every prompt primitive.
- Add targeted unit tests around prompt normalization and bound motion schema loading.
- Add small regression checks that compare manifest counts and required output presence.
- Keep archived code under `old/` out of active imports and documentation except as historical reference.
