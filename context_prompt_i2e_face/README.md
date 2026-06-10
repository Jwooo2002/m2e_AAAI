# Context-Prompted Image-to-Event Face Synthesis

This folder contains split JSON context prompts for a Codex-oriented MVP implementation.

Core idea:
- Use an existing face keypoint dataset.
- Build coarse semantic parsing from provided keypoints.
- Group face regions with superpixels.
- Create blob-level context prompts.
- Generate synthetic event-like data from blob-level heuristic motion.

The MVP does not train a model, does not require video, and does not run a separate face landmark detector.
