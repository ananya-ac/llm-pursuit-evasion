"""Runs one 4-blue vs 2-red LLM-planner game for a given model and saves
results.csv, reasoning.csv, and the three GIFs (overview, blue-POV, red-POV)
into --output-dir. Used by run_experiment1_single_games.sh to produce one
labeled directory per model for Experiment I's model comparison.
"""

import argparse
import contextlib
import io
import os

from run_red_count_grid import (
    REASONING_FIELDNAMES,
    RESULT_FIELDNAMES,
    build_planners,
    determine_outcome,
    save_csv,
)
from simulation import Simulation
from verify_recon_fov_pov import render_pov_gif

SEED = 2000
NUM_BLUES = 4
NUM_REDS = 2
SIM_STEPS = 1000
HORIZON = 20
PLANNING_INTERVAL_SECONDS = 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="OpenRouter model slug.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    role_planner, red_role_planner = build_planners("llm", args.model)

    sim = Simulation(
        solver_mode="decentralized",
        num_blues=NUM_BLUES,
        num_reds=NUM_REDS,
        sim_steps=SIM_STEPS,
        horizon=HORIZON,
        seed=args.seed,
        enable_blue_blue_cbf=True,
        enable_blue_red_cbf=True,
        role_planner=role_planner,
        red_role_planner=red_role_planner,
        planning_interval_seconds=PLANNING_INTERVAL_SECONDS,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()
        sim.generate_animation(video_filename=os.path.join(args.output_dir, "overview.gif"))
        render_pov_gif(sim, "blue", os.path.join(args.output_dir, "blue_pov.gif"))
        render_pov_gif(sim, "red", os.path.join(args.output_dir, "red_pov.gif"))

    row = {
        "game_idx": 0,
        "seed": args.seed,
        "num_reds": NUM_REDS,
        "num_blues": NUM_BLUES,
        "sim_steps_limit": SIM_STEPS,
        "planner_type": "llm",
        "model": args.model,
        "planning_interval_seconds": PLANNING_INTERVAL_SECONDS,
        "error_message": "",
        "outcome": determine_outcome(sim),
        "steps_completed": max(len(sim.history_red) - 1, 0),
        "num_captured": sim.num_captured,
        "num_escaped": sim.num_escaped,
        "num_red_active_at_end": sim.red_outcomes.count("active"),
        "red_outcomes": ",".join(sim.red_outcomes),
        "blue_score": sim.blue_score,
        "blue_planner_fallback_count": len(
            getattr(role_planner.inner, "fallback_events", [])
        ),
        "red_planner_fallback_count": len(
            getattr(red_role_planner.inner, "fallback_events", [])
        ),
        "final_num_blue_disabled": sum(sim.blue_disabled),
        "touchdown": int(sim.touchdown),
        "touchdown_step": sim.touchdown_step,
        "touchdown_red_index": sim.touchdown_red_index,
        "collided": int(sim.collided),
        "collision_step": sim.collision_step,
        "collision_kind": sim.collision_kind,
        "collision_agents": sim.collision_agents,
        "defenders_won": int(sim.defenders_won),
        "defenders_win_step": sim.defenders_win_step,
        "solver_failed": int(sim.solver_failed),
        "solver_failure_step": sim.solver_failure_step,
    }
    reasoning_rows = []
    for entry in role_planner.log + red_role_planner.log:
        reasoning_rows.append(
            {
                "game_idx": 0,
                "seed": args.seed,
                "num_reds": NUM_REDS,
                "num_blues": NUM_BLUES,
                "model": args.model,
                "planning_interval_seconds": PLANNING_INTERVAL_SECONDS,
                **entry,
            }
        )
    reasoning_rows.sort(key=lambda r: (r["sim_step"], r["team"], r["agent_id"]))

    save_csv([row], RESULT_FIELDNAMES, os.path.join(args.output_dir, "results.csv"))
    save_csv(reasoning_rows, REASONING_FIELDNAMES, os.path.join(args.output_dir, "reasoning.csv"))

    print(
        f"[{args.model}] outcome={row['outcome']} steps={row['steps_completed']} "
        f"blue_score={row['blue_score']} red_outcomes={row['red_outcomes']} "
        f"fallback_count={row['blue_planner_fallback_count']}"
    )


if __name__ == "__main__":
    main()
