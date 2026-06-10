# Archived Experimental Code

This folder is archived. Nothing under `old/` is part of the active pipeline, and new work should not depend on it unless explicitly requested.

This folder keeps earlier exploratory code paths so they can be recovered later without keeping them in the main prompt-to-event path.

Archived here:

- `liveportrait_code_inspection.md`: notes from inspecting LivePortrait motion representations.
- `i2e_face_mvp/liveportrait_adapter.py`: earlier LivePortrait-inspired sparse motion adapter. This is no longer the main direction because motion now comes from external prompts/vectors.
- `i2e_face_mvp/arbitrary_flow.py`: older synthetic/arbitrary flow experiments with GIF previews.
- `i2e_face_mvp/debug_motion.py`: older researcher-controlled debug motion path with GIF previews. A copy remains in the package because `pipeline.py` still imports it.
- `i2e_face_mvp/uploaded_flow.py`: older uploaded optical-flow event path with GIF previews.
- `i2e_face_mvp/pipeline.py` and `i2e_face_mvp/prompts.py`: earlier context-prompted WFLW MVP pipeline code. Copies remain in the package where still needed.

The current main path is:

1. `face_feature_adapter.py`
2. `motion_binding.py`
3. `bound_motion_events.py`
4. `validate_bound_motion_events.py`
5. `run_prompt_to_event.py`

The current main path does not use LivePortrait, RGB video generation, or image-to-video-to-event processing.
