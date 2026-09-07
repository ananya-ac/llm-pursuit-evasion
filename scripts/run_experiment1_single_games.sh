#!/usr/bin/env bash
# Runs one 4-blue vs 2-red LLM-planner game (seed=2000, 1 Hz) for each of the
# 6 models used in Experiment I's full sweep (run_full_sweep.sh), saving
# results.csv, reasoning.csv, and the overview/blue-POV/red-POV GIFs into a
# directory named after each model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODELS=(
  "deepseek/deepseek-chat"
  "qwen/qwen3-30b-a3b"
  "anthropic/claude-haiku-4.5"
  "openai/gpt-5-nano"
  "meta-llama/llama-3.3-70b-instruct"
  "google/gemini-2.5-flash"
)

for model in "${MODELS[@]}"; do
  model_slug="${model//\//_}"
  model_slug="${model_slug//:/_}"

  echo "=== model=${model} ==="
  python3 -m scripts.run_single_model_demo \
    --model "$model" \
    --output-dir "$model_slug"
done

echo "All single-game demos complete."
