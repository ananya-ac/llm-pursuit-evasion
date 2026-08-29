"""Renders the 4-blue vs 2-red LLM-planner game (seed=2000, 1 Hz) as GIFs,
matching the established convention: a plain top-down overview (role labels,
no FOV) plus blue-POV and red-POV renders (FOV wedges + role labels for the
POV team, detection-colored markers for the opposing team).

run_red_count_grid.py only logs CSVs -- it never keeps the Simulation object
around to render from -- so this re-runs the same seeded game (a fresh LLM
call each cycle, since the API isn't deterministic) and re-emits the paired
results/reasoning CSVs alongside the GIFs so all three describe the same run.
"""

import contextlib
import io

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
MODEL = "deepseek/deepseek-chat"
PLANNING_INTERVAL_SECONDS = 1.0
STEM = "blue4_red2_llm_1hz"


def main():
    role_planner, red_role_planner = build_planners("llm", MODEL)

    sim = Simulation(
        solver_mode="decentralized",
        num_blues=NUM_BLUES,
        num_reds=NUM_REDS,
        sim_steps=SIM_STEPS,
        horizon=HORIZON,
        seed=SEED,
        enable_blue_blue_cbf=True,
        enable_blue_red_cbf=True,
        role_planner=role_planner,
        red_role_planner=red_role_planner,
        planning_interval_seconds=PLANNING_INTERVAL_SECONDS,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()
        sim.generate_animation(video_filename=f"{STEM}_overview.gif")
        render_pov_gif(sim, "blue", f"{STEM}_blue_pov.gif")
        render_pov_gif(sim, "red", f"{STEM}_red_pov.gif")

    row = {
        "game_idx": 0,
        "seed": SEED,
        "num_reds": NUM_REDS,
        "num_blues": NUM_BLUES,
        "sim_steps_limit": SIM_STEPS,
        "planner_type": "llm",
        "model": MODEL,
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
                "seed": SEED,
                "num_reds": NUM_REDS,
                "num_blues": NUM_BLUES,
                "model": MODEL,
                "planning_interval_seconds": PLANNING_INTERVAL_SECONDS,
                **entry,
            }
        )
    reasoning_rows.sort(key=lambda r: (r["sim_step"], r["team"], r["agent_id"]))

    save_csv([row], RESULT_FIELDNAMES, f"{STEM}_results.csv")
    save_csv(reasoning_rows, REASONING_FIELDNAMES, f"{STEM}_reasoning.csv")

    print(
        f"outcome={row['outcome']} steps={row['steps_completed']} "
        f"blue_score={row['blue_score']} red_outcomes={row['red_outcomes']}"
    )
    print(f"Saved {STEM}_overview.gif, {STEM}_blue_pov.gif, {STEM}_red_pov.gif")


if __name__ == "__main__":
    main()
