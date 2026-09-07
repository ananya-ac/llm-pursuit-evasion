#!/usr/bin/env bash
# Runs the Experiment II (vision-augmented blue planner) sweep via
# run_vision_experiment.py: 3 vision-capable models x 2 planning cadences,
# each internally covering contact-slot-counts 1-4 x 10 seeds (40 games per
# combination, 240 total). Results and reasoning CSVs are written to the
# same sweep_results directories as Experiment I.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

RESULTS_DIR="sweep_results/results"
REASONING_DIR="sweep_results/reasoning"
mkdir -p "$RESULTS_DIR" "$REASONING_DIR"

CONTACT_SLOT_COUNTS="1,2,3,4"
N_SEEDS=10
P_BIRD=0.15
P_CIVILIAN=0.15

MODELS=(
  "anthropic/claude-haiku-4.5"
  "openai/gpt-5-nano"
  "google/gemini-2.5-flash"
)
INTERVALS=(3 5)

for model in "${MODELS[@]}"; do
  model_slug="${model//\//_}"
  model_slug="${model_slug//:/_}"
  for interval in "${INTERVALS[@]}"; do
    stem="vision_experiment_${N_SEEDS}seeds_llm_${model_slug}_${interval}s"
    results_csv="${RESULTS_DIR}/${stem}_results.csv"
    reasoning_csv="${REASONING_DIR}/${stem}_reasoning.csv"

    echo "=== model=${model} interval=${interval}s ==="
    python3 -m scripts.run_vision_experiment \
      --model "$model" \
      --planning-interval-seconds "$interval" \
      --contact-slot-counts "$CONTACT_SLOT_COUNTS" \
      --n-seeds "$N_SEEDS" \
      --p-bird "$P_BIRD" \
      --p-civilian "$P_CIVILIAN" \
      --output-results-csv "$results_csv" \
      --output-reasoning-csv "$reasoning_csv"
  done
done

echo "Sweep complete."
echo "Results CSVs:   $RESULTS_DIR"
echo "Reasoning CSVs: $REASONING_DIR"
