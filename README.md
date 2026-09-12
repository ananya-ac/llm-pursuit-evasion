# LLM Pursuit-Evasion

A multi-agent pursuit-evasion / perimeter-defense simulation testbed where a
"blue" defender swarm is coordinated by either rule-based logic or an
LLM-backed role planner (via OpenRouter), against a "red" attacker team that
tries to breach a defended perimeter or escape the arena. Low-level motion is
solved with CasADi/OSQP via a decentralized per-agent MPC solver; role
assignment (RECON / NEUTRALIZE, etc.) is decided by the pluggable planner
layer on top.

Two experiments are supported:

- **Experiment I** — text-only LLM role planning: the planner sees numeric
  agent states (positions, detected reds, coverage) and assigns roles/targets.
- **Experiment II** — vision-augmented planning: red agents can be
  substituted with harmless decoys (birds, civilian drones); the planner is
  shown an image per detected contact and must decide what to engage.

## Repository layout

- `agents/` — agent dynamics (`Agent`, state/dynamics integration).
- `control/` — low-level controllers and solvers (decentralized per-agent MPC,
  CBF safety filters).
- `perception/` — field-of-view/detection geometry (`sensing.py`) and
  coverage tracking for RECON (`coverage.py`).
- `planning/` — role planners: `RuleBasedBlueRolePlanner`/`LLMRolePlanner` (blue)
  and their red-team counterparts, plus role definitions (`roles.py`).
- `simulation/` — `testbed.py` (the `Simulation` class driving one episode)
  and `decentralized_solver.py`.
- `assets/` — placeholder and sourced images (drones, birds, civilians) used
  by the vision-augmented planner.
- `scripts/` — experiment drivers and sweeps: single-game demos, red-count/
  vision-experiment grids, their parallel/sequential sweep wrappers, and the
  after-the-fact GIF renderer (see below). Run as modules
  (`python3 -m scripts.<name>`) since they're a package; the `.sh` wrappers
  can be run directly.
- `verification/` — standalone, free (no API calls) `verify_*.py` scripts
  that sanity-check geometry/mechanics in isolation and render GIFs for
  visual inspection. Run as modules (`python3 -m verification.<name>`) since
  they're a package.

## Setup

Requires [conda](https://docs.conda.io/) and Python 3.11.

```bash
./setup_env.sh
conda activate multi_agent_mpc
```

This creates a `multi_agent_mpc` conda environment and installs
`requirements.txt` (numpy, scipy, casadi, osqp, matplotlib, openai, pydantic)
into it.

LLM-backed planners call OpenRouter, so set an API key before running
anything with `--planner llm` (the default for most drivers) or the vision
experiment:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

Rule-based planners (`--planner rule`) and all `verify_*.py` scripts need no
API key and make no network calls.

## Reproduce: quickest sanity check

```bash
./smoke_test.sh
```

Creates the conda environment (if needed) and runs one free, rule-based game
(1 red, 1 seed, 100 steps) to confirm the environment, CasADi/OSQP stack, and
simulation code work end-to-end. Writes `/tmp/smoke_test_results.csv`.

## Reproduce: a minimal simulation run

```bash
python3 main.py
```

Runs `Simulation()` with its defaults and calls `generate_animation()`.

## Reproduce: verification scripts (no API key needed)

Each renders GIF(s) into the repo root for visual inspection:

```bash
python3 -m verification.verify_bearing_fov          # bearing-as-state + FOV cone geometry
python3 -m verification.verify_recon_fov_pov        # blue-POV / red-POV FOV/detection GIFs
python3 -m verification.verify_rule_based_recon     # coverage-driven RECON behavior
python3 -m verification.verify_contact_substitution # Experiment II decoy substitution logic
```

## Reproduce: a single LLM-planner game

```bash
export OPENROUTER_API_KEY=sk-or-...
python3 -m scripts.run_single_model_demo \
  --model anthropic/claude-haiku-4.5 \
  --output-dir demo_haiku
```

Runs one 4-blue vs 2-red game (seed 2000, 1 Hz planning) and writes
`results.csv`, `reasoning.csv`, and overview/blue-POV/red-POV GIFs into
`demo_haiku/`.

`python3 -m scripts.run_role_scenarios` runs a free (rule-based) 1v1
RECON→NEUTRALIZE demo and needs no arguments or API key.

## Reproduce: Experiment I sweep (red-count grid)

Single (model, planning-interval) combination:

```bash
python3 -m scripts.run_red_count_grid \
  --planner llm \
  --model anthropic/claude-haiku-4.5 \
  --planning-interval-seconds 1 \
  --red-counts 1,2,3,4 \
  --n-seeds 10 \
  --output-results-csv results.csv \
  --output-reasoning-csv reasoning.csv
```

Use `--planner rule` to skip the API entirely. Full 6-model x 2-interval
sweep (480 games total; sequential) — run from the repo root:

```bash
./scripts/run_full_sweep.sh
```

or the parallel driver (flattens every individual game into one job pool for
better load balancing):

```bash
python3 -m scripts.run_sweep_parallel
```

Both write per-(model, interval) CSVs to `sweep_results/results/` and
`sweep_results/reasoning/`.

## Reproduce: Experiment II sweep (vision-augmented, contact substitution)

```bash
./scripts/run_vision_sweep.sh
```

or in parallel:

```bash
python3 -m scripts.run_vision_sweep_parallel
```

Sweeps 3 vision-capable models (`anthropic/claude-haiku-4.5`,
`openai/gpt-5-nano`, `google/gemini-2.5-flash`) x 2 planning cadences x
contact-slot-counts 1-4 x 10 seeds (240 games), writing to the same
`sweep_results/` directories as Experiment I.

## Reproduce: blue-count / velocity grids

```bash
python3 -m scripts.run_blue_count_grid --n-games 3 --output-csv blue_count.csv --output-heatmap blue_count.png
python3 -m scripts.run_velocity_grid --n-games 100 --output-csv velocity.csv --output-heatmap velocity.png
```

Both use the rule-based decentralized solver (no LLM, no API key) and produce a
heatmap image plus a per-game CSV.

## Reproduce: render a game as GIFs after the fact

```bash
python3 -m scripts.render_llm_game_gif
```

Re-runs the fixed seed=2000, 4-blue vs 2-red LLM game (since
`run_red_count_grid.py` only logs CSVs, not the `Simulation` object) and
emits the overview/blue-POV/red-POV GIFs alongside matching results/reasoning
CSVs.

## Notes on reproducibility

- All simulations are seeded (`--seed`/`--base-seed`/`--n-seeds`), but LLM
  planner calls are not deterministic (a live API call each planning cycle),
  so `--planner llm` runs will vary between invocations even at a fixed seed.
- `--planner rule` runs are fully deterministic given a seed.
