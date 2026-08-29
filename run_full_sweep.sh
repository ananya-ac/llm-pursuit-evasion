#!/usr/bin/env bash
# Runs the full model x planning-interval x red-count x seed sweep via
# run_red_count_grid.py: 6 models x 2 planning cadences, each internally
# covering red-counts 1-4 x 10 seeds (40 games per combination, 480 total).
# Results and reasoning CSVs are written to separate directories.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="sweep_results/results"
REASONING_DIR="sweep_results/reasoning"
mkdir -p "$RESULTS_DIR" "$REASONING_DIR"

RED_COUNTS="1,2,3,4"
N_SEEDS=10

MODELS=(
  "deepseek/deepseek-chat"
  "qwen/qwen3-30b-a3b"
  "anthropic/claude-haiku-4.5"
  "openai/gpt-5-nano"
  "meta-llama/llama-3.3-70b-instruct"
  "google/gemini-2.5-flash"
)
INTERVALS=(1 5)

for model in "${MODELS[@]}"; do
  model_slug="${model//\//_}"
  model_slug="${model_slug//:/_}"
  for interval in "${INTERVALS[@]}"; do
    stem="red_count_grid_${N_SEEDS}seeds_llm_${model_slug}_${interval}s"
    results_csv="${RESULTS_DIR}/${stem}_results.csv"
    reasoning_csv="${REASONING_DIR}/${stem}_reasoning.csv"

    echo "=== model=${model} interval=${interval}s ==="
    python3 run_red_count_grid.py \
      --planner llm \
      --model "$model" \
      --planning-interval-seconds "$interval" \
      --red-counts "$RED_COUNTS" \
      --n-seeds "$N_SEEDS" \
      --output-results-csv "$results_csv" \
      --output-reasoning-csv "$reasoning_csv"
  done
done

echo "Sweep complete."
echo "Results CSVs:   $RESULTS_DIR"
echo "Reasoning CSVs: $REASONING_DIR"
