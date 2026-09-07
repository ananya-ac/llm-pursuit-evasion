#!/usr/bin/env bash
# Runs setup_env.sh to create the "multi_agent_mpc" conda environment on a
# fresh machine (this assumes it does not already exist), then runs a
# minimal, free smoke test against the rule-based planner (no
# OPENROUTER_API_KEY required, no network calls) to confirm the
# environment, CasADi/OSQP solver stack, and simulation code all work
# end-to-end right after cloning this repo.
set -euo pipefail

ENV_NAME="multi_agent_mpc"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/setup_env.sh"

echo
echo "Running smoke test: rule-based planner, 1 red, 1 seed, 100 steps (free, no API calls)..."
conda run -n "$ENV_NAME" --cwd "$SCRIPT_DIR" python3 -m scripts.run_red_count_grid \
    --planner rule \
    --red-counts 1 \
    --n-seeds 1 \
    --sim-steps 100 \
    --output-results-csv /tmp/smoke_test_results.csv \
    --output-reasoning-csv /tmp/smoke_test_reasoning.csv

echo
echo "Smoke test passed. Environment '${ENV_NAME}' is ready."
echo "Results written to /tmp/smoke_test_results.csv"
