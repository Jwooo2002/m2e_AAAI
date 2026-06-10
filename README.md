# M2E Direct Face Motion-to-Event Pipeline

This repository's active path generates event streams directly from one RGB face image, WFLW-style landmarks, and an external motion prompt/vector input.

The active pipeline is:

```text
run_prompt_to_event.py
-> i2e_face_mvp/face_feature_adapter.py
-> i2e_face_mvp/motion_binding.py
-> i2e_face_mvp/bound_motion_events.py
-> i2e_face_mvp/validate_bound_motion_events.py
```

It does not use RGB video generation, image-to-video-to-event conversion, or LivePortrait motion sourcing.

## Quick Smoke Test

Run the fixture path:

```bash
python3 run_prompt_to_event.py --fixture --motion-prompt motion_prompt.json --output-dir outputs/prompt_to_event_fixture --threshold 0.08 --seed 0
```

Run every supported motion primitive:

```bash
bash scripts/smoke_prompt_primitives.sh
```

Primitive prompt fixtures live in `prompt_fixtures/`. Smoke outputs are written under `outputs/`.

## Active Outputs

Each run directory contains:

- `blob_feature_context.json`
- `bound_blob_motion.json`
- `events.npy`
- `events.csv`
- `event_preview.png`
- `validation_report.json`
- `run_manifest.json`

## Documentation

- `docs/PIPELINE.md`: active module path and responsibilities.
- `docs/SCHEMA.md`: JSON contracts for the main artifacts.
- `docs/PROJECT_STATUS.md`: verified status and immediate next steps.
- `SMOKE_TEST.md`: exact smoke-test commands.

## Archived Code

`old/` is archived experimental material. It is not part of the active pipeline and should not be used for new work unless explicitly requested.
