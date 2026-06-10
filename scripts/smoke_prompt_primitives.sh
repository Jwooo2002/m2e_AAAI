#!/usr/bin/env bash
set -euo pipefail

threshold="${THRESHOLD:-0.08}"
seed="${SEED:-0}"
output_root="${OUTPUT_ROOT:-outputs/prompt_primitive_smoke}"

for prompt_path in prompt_fixtures/*.json; do
  name="$(basename "${prompt_path}" .json)"
  python3 run_prompt_to_event.py \
    --fixture \
    --motion-prompt "${prompt_path}" \
    --output-dir "${output_root}/${name}" \
    --threshold "${threshold}" \
    --seed "${seed}"
done
