"""Parallel driver for the Experiment II (vision-augmented blue planner)
model x planning-interval x contact-slot-count x seed sweep, replacing
run_vision_sweep.sh's strictly sequential loop -- mirrors
run_sweep_parallel.py's design exactly, one level down (contact_slot_count
in place of red_count).

Every game is fully independent -- run_single_game() builds a fresh
Simulation (fresh CasADi problems, fresh planner instances, fresh per-slot
drone/bird/civilian draw) from scratch each call, with no shared state
across games -- so every individual game across every (model, interval,
contact_slot_count, seed) combination is flattened into one job list and
scheduled independently through a single worker pool for better load
balancing than parallelizing only the 6 (model, interval) combinations.

Output format matches run_vision_sweep.sh exactly: one {results,reasoning}.csv
pair per (model, interval) combination, written to the same
sweep_results/{results,reasoning} directories used by Experiment I, with
identical seeds across models/intervals for the same
(contact_slot_count, local_idx) position so every model is evaluated against
the same set of per-slot draws.
"""

import argparse
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from scripts.run_red_count_grid import save_csv
from scripts.run_vision_experiment import (
    DEFAULT_P_BIRD,
    DEFAULT_P_CIVILIAN,
    REASONING_FIELDNAMES,
    RESULT_FIELDNAMES,
    VISION_CAPABLE_MODELS,
    run_single_game,
)

RESULTS_DIR = "sweep_results/results"
REASONING_DIR = "sweep_results/reasoning"

CONTACT_SLOT_COUNTS = [1, 2, 3, 4]
N_SEEDS = 10
BASE_SEED = 1000
SIM_STEPS = 600
HORIZON = 20
NUM_BLUES = 4

MODELS = VISION_CAPABLE_MODELS
DEFAULT_INTERVALS = [3, 5]

_JOB_DERIVED_FIELDS = (
    "game_idx",
    "seed",
    "num_contact_slots",
    "num_blues",
    "sim_steps_limit",
    "model",
    "planning_interval_seconds",
    "p_bird",
    "p_civilian",
    "outcome",
    "error_message",
)
_ERROR_ROW_DEFAULTS = {
    field: None for field in RESULT_FIELDNAMES if field not in _JOB_DERIVED_FIELDS
}


def model_slug(model):
    return model.replace("/", "_").replace(":", "_")


def build_jobs(intervals, p_bird, p_civilian):
    """One job per (model, interval, contact_slot_count, seed). Seed
    assignment mirrors run_vision_experiment.py's own sequential scheme
    exactly (seed = base_seed + slot_count_rank * n_seeds + local_idx) so
    the seed used for a given (contact_slot_count, local_idx) is identical
    across every model/interval."""
    jobs = []
    for model in MODELS:
        for interval in intervals:
            game_idx = 0
            for slot_idx, num_contact_slots in enumerate(CONTACT_SLOT_COUNTS):
                for local_idx in range(N_SEEDS):
                    seed = BASE_SEED + slot_idx * N_SEEDS + local_idx
                    jobs.append(
                        dict(
                            game_idx=game_idx,
                            seed=seed,
                            num_contact_slots=num_contact_slots,
                            num_blues=NUM_BLUES,
                            sim_steps=SIM_STEPS,
                            horizon=HORIZON,
                            model=model,
                            planning_interval_seconds=float(interval),
                            p_bird=p_bird,
                            p_civilian=p_civilian,
                            verbose_games=False,
                        )
                    )
                    game_idx += 1
    return jobs


def _run_job(job):
    """See run_sweep_parallel.py's _run_job docstring -- same OSQP/IPOPT
    fd-redirect rationale applies unchanged here."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        row, reasoning_rows = run_single_game(**job)
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)
    return job["model"], job["planning_interval_seconds"], row, reasoning_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent games. This workload is >99%% blocked on network I/O "
        "(OpenRouter multi-modal calls) -- start conservative and raise it if "
        "no 429s show up.",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default=",".join(str(i) for i in DEFAULT_INTERVALS),
        help="Comma-separated planning-interval-seconds values, e.g. '3' or '3,5'.",
    )
    parser.add_argument("--p-bird", type=float, default=DEFAULT_P_BIRD)
    parser.add_argument("--p-civilian", type=float, default=DEFAULT_P_CIVILIAN)
    args = parser.parse_args()
    intervals = [float(x) for x in args.intervals.split(",") if x.strip()]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REASONING_DIR, exist_ok=True)

    jobs = build_jobs(intervals, args.p_bird, args.p_civilian)
    total = len(jobs)
    print(
        f"Submitting {total} games across {len(MODELS)} models x {len(intervals)} "
        f"intervals x {len(CONTACT_SLOT_COUNTS)} contact-slot-counts x {N_SEEDS} seeds, "
        f"p_bird={args.p_bird}, p_civilian={args.p_civilian}, {args.workers} workers."
    )

    results_by_combo = defaultdict(list)
    reasoning_by_combo = defaultdict(list)

    start = time.monotonic()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                _, _, row, reasoning_rows = future.result()
            except Exception as exc:  # noqa: BLE001 -- one job's failure must not kill the pool
                print(
                    f"[job failed] model={job['model']} interval="
                    f"{job['planning_interval_seconds']} seed={job['seed']} "
                    f"num_contact_slots={job['num_contact_slots']}: {type(exc).__name__}: {exc}"
                )
                row = {
                    "game_idx": job["game_idx"],
                    "seed": job["seed"],
                    "num_contact_slots": job["num_contact_slots"],
                    "num_blues": job["num_blues"],
                    "sim_steps_limit": job["sim_steps"],
                    "model": job["model"],
                    "planning_interval_seconds": job["planning_interval_seconds"],
                    "p_bird": job["p_bird"],
                    "p_civilian": job["p_civilian"],
                    "outcome": "error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    **_ERROR_ROW_DEFAULTS,
                }
                reasoning_rows = []

            key = (job["model"], job["planning_interval_seconds"])
            results_by_combo[key].append(row)
            reasoning_by_combo[key].extend(reasoning_rows)
            completed += 1
            elapsed = time.monotonic() - start
            print(
                f"[{completed}/{total}] model={job['model']} "
                f"interval={job['planning_interval_seconds']:g}s seed={job['seed']} "
                f"num_contact_slots={job['num_contact_slots']} -> outcome={row.get('outcome')} "
                f"({elapsed:.0f}s elapsed)"
            )

    for (model, interval), rows in results_by_combo.items():
        rows.sort(key=lambda r: r["game_idx"])
        reasoning_rows = reasoning_by_combo[(model, interval)]
        reasoning_rows.sort(key=lambda r: (r["sim_step"], r["team"], r["agent_id"]))
        stem = f"vision_experiment_{N_SEEDS}seeds_llm_{model_slug(model)}_{interval:g}s"
        save_csv(rows, RESULT_FIELDNAMES, os.path.join(RESULTS_DIR, f"{stem}_results.csv"))
        save_csv(
            reasoning_rows,
            REASONING_FIELDNAMES,
            os.path.join(REASONING_DIR, f"{stem}_reasoning.csv"),
        )
        print(f"Wrote {stem}_{{results,reasoning}}.csv ({len(rows)} games)")

    print(f"Sweep complete in {time.monotonic() - start:.0f}s.")


if __name__ == "__main__":
    main()
