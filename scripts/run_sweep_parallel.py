"""Parallel driver for the full model x planning-interval x red-count x seed
sweep, replacing run_full_sweep.sh's strictly sequential loop.

Every game is fully independent -- run_single_game() builds a fresh
Simulation (fresh CasADi problems, fresh planner instances) from scratch each
call, with no shared state across games -- so rather than parallelizing only
the 12 model x interval combinations (coarse, unevenly loaded: one slow
combination blocks a whole worker while others finish early and idle), every
individual game across every combination is flattened into one job list and
scheduled independently through a single worker pool for better load
balancing.

Output format matches run_full_sweep.sh exactly: one {results,reasoning}.csv
pair per (model, interval) combination, written to the same
sweep_results/{results,reasoning} directories, with identical seeds across
models/intervals for the same (red_count, local_idx) position so every model
is evaluated against the same set of initial conditions.
"""

import argparse
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from scripts.run_red_count_grid import (
    REASONING_FIELDNAMES,
    RESULT_FIELDNAMES,
    run_single_game,
    save_csv,
)

RESULTS_DIR = "sweep_results/results"
REASONING_DIR = "sweep_results/reasoning"

RED_COUNTS = [1, 2, 3, 4]
N_SEEDS = 10
BASE_SEED = 1000
SIM_STEPS = 600
HORIZON = 20
NUM_BLUES = 4

MODELS = [
    "deepseek/deepseek-chat",
    "qwen/qwen3-30b-a3b",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5-nano",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-flash",
]
DEFAULT_INTERVALS = [1, 5]
DEFAULT_SPEED = 2.0  # matches Simulation's own blue/red v_max=a_max default

_JOB_DERIVED_FIELDS = (
    "game_idx",
    "seed",
    "num_reds",
    "num_blues",
    "sim_steps_limit",
    "planner_type",
    "model",
    "planning_interval_seconds",
    "outcome",
    "error_message",
    "blue_v_max",
    "blue_a_max",
    "red_v_max",
    "red_a_max",
)
_ERROR_ROW_DEFAULTS = {
    field: None for field in RESULT_FIELDNAMES if field not in _JOB_DERIVED_FIELDS
}


def model_slug(model):
    return model.replace("/", "_").replace(":", "_")


def build_jobs(intervals, speed):
    """One job per (model, interval, red_count, seed). Seed assignment
    mirrors run_red_count_grid.py's own sequential scheme exactly (seed =
    base_seed + red_count_rank * n_seeds + local_idx) so the seed used for a
    given (red_count, local_idx) is identical across every model/interval --
    matching what running run_full_sweep.sh's unmodified base-seed default
    for each combination already produced.

    speed is applied uniformly to blue_v_max/blue_a_max/red_v_max/red_a_max
    -- kept symmetric across teams unless a caller edits the job dicts
    afterward."""
    jobs = []
    for model in MODELS:
        for interval in intervals:
            game_idx = 0
            for red_idx, num_reds in enumerate(RED_COUNTS):
                for local_idx in range(N_SEEDS):
                    seed = BASE_SEED + red_idx * N_SEEDS + local_idx
                    jobs.append(
                        dict(
                            game_idx=game_idx,
                            seed=seed,
                            num_reds=num_reds,
                            num_blues=NUM_BLUES,
                            sim_steps=SIM_STEPS,
                            horizon=HORIZON,
                            planner_type="llm",
                            model=model,
                            planning_interval_seconds=float(interval),
                            verbose_games=False,
                            blue_v_max=speed,
                            blue_a_max=speed,
                            red_v_max=speed,
                            red_a_max=speed,
                        )
                    )
                    game_idx += 1
    return jobs


def _run_job(job):
    """run_single_game already silences its own Python-level prints via
    contextlib.redirect_stdout when verbose_games=False, but OSQP/IPOPT's
    solver banners are C-level writes straight to the process's real stdout
    fd -- they bypass sys.stdout entirely (neither a flat {"verbose": False}
    nor a nested {"osqp": {"verbose": False}} suppresses it in this CasADi
    build, verified directly). With many concurrent workers those banners
    interleave into unreadable noise, so redirect the real fd for this
    worker process only -- doesn't touch the parent process's own progress
    output, which is printed after future.result() returns, not in here."""
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
        help="Concurrent games. Local CPU/RAM aren't the bottleneck here (this "
        "workload is >99%% blocked on network I/O) -- OpenRouter's rate limit "
        "is. Start conservative and raise it if no 429s show up.",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default=",".join(str(i) for i in DEFAULT_INTERVALS),
        help="Comma-separated planning-interval-seconds values, e.g. '3' or '1,5'.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help="Uniform blue_v_max=blue_a_max=red_v_max=red_a_max for this run.",
    )
    args = parser.parse_args()
    intervals = [float(x) for x in args.intervals.split(",") if x.strip()]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REASONING_DIR, exist_ok=True)

    jobs = build_jobs(intervals, args.speed)
    total = len(jobs)
    print(
        f"Submitting {total} games across {len(MODELS)} models x {len(intervals)} "
        f"intervals x {len(RED_COUNTS)} red-counts x {N_SEEDS} seeds, speed={args.speed:g}, "
        f"{args.workers} workers."
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
                    f"num_reds={job['num_reds']}: {type(exc).__name__}: {exc}"
                )
                row = {
                    "game_idx": job["game_idx"],
                    "seed": job["seed"],
                    "num_reds": job["num_reds"],
                    "num_blues": job["num_blues"],
                    "sim_steps_limit": job["sim_steps"],
                    "planner_type": job["planner_type"],
                    "model": job["model"],
                    "planning_interval_seconds": job["planning_interval_seconds"],
                    "blue_v_max": job["blue_v_max"],
                    "blue_a_max": job["blue_a_max"],
                    "red_v_max": job["red_v_max"],
                    "red_a_max": job["red_a_max"],
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
                f"num_reds={job['num_reds']} -> outcome={row.get('outcome')} "
                f"({elapsed:.0f}s elapsed)"
            )

    for (model, interval), rows in results_by_combo.items():
        rows.sort(key=lambda r: r["game_idx"])
        reasoning_rows = reasoning_by_combo[(model, interval)]
        reasoning_rows.sort(key=lambda r: (r["sim_step"], r["team"], r["agent_id"]))
        speed_tag = "" if args.speed == DEFAULT_SPEED else f"_speed{args.speed:g}"
        stem = f"red_count_grid_{N_SEEDS}seeds_llm_{model_slug(model)}_{interval:g}s{speed_tag}"
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
