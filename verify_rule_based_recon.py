"""Renders the new coverage-driven RECON behavior in RuleBasedRolePlanner (see
coverage.CoverageTracker) as GIFs, for visual inspection: a plain top-down
overview (role labels, no FOV) plus a blue-POV render (FOV wedges + role
labels for blue, detection-colored markers for red) -- reusing
verify_recon_fov_pov.render_pov_gif so it matches the established convention.

Unlike verify_recon_fov_pov.py's MixedPlanner (which force-assigns RECON to
every even-indexed blue regardless of tactical picture), this uses the real
RuleBasedRolePlanner with recon_fraction/min_recon enabled, so the RECON
assignment -- and which region it targets -- comes from actual coverage
staleness, not a hardcoded rule.

Produces rule_based_recon_overview.gif and rule_based_recon_blue_pov.gif.
"""

import contextlib
import io

from planner import RuleBasedRolePlanner
from simulation import Simulation
from verify_recon_fov_pov import render_pov_gif

SEED = 7
NUM_BLUES = 5
NUM_REDS = 2
SIM_STEPS = 150
HORIZON = 10
PLANNING_INTERVAL_SECONDS = 1.0


def run_simulation():
    role_planner = RuleBasedRolePlanner(
        neutralize_fraction=0.4,
        capture_fraction=0.5,
        recon_fraction=0.4,
        min_recon=1,
    )
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
        planning_interval_seconds=PLANNING_INTERVAL_SECONDS,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()
    return sim, role_planner


def main():
    sim, role_planner = run_simulation()
    with contextlib.redirect_stdout(io.StringIO()):
        sim.generate_animation(video_filename="rule_based_recon_overview.gif")
        render_pov_gif(sim, "blue", "rule_based_recon_blue_pov.gif")

    recon_steps = [
        h for h in sim.role_assignment_history
        if any(r.value == "recon" for r in h.values())
    ]
    print(f"steps_completed={max(len(sim.history_red) - 1, 0)}")
    print(f"planning cycles with >=1 RECON assignment: {len(recon_steps)}/{len(sim.role_assignment_history)}")
    print(f"last recon target regions: {role_planner.last_recon_target_regions}")
    print("Saved rule_based_recon_overview.gif, rule_based_recon_blue_pov.gif")


if __name__ == "__main__":
    main()
