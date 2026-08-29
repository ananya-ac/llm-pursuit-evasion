"""Single 1v1 demo: blue starts on RECON (random-waypoint patrol + bearing
sweep) and switches to NEUTRALIZE, with the red as target_red_id, the moment
its own FOV actually detects the red -- never before. Renders one GIF from
blue's POV (FOV wedge + role label + detection-colored red) so the RECON ->
NEUTRALIZE transition and the detection event that triggers it are both
directly visible.
"""

import contextlib
import io

import numpy as np

from planner import RolePlanner, _infer_stride
from roles import Role
from simulation import Simulation
from verify_recon_fov_pov import render_pov_gif

SEED = 21
NUM_BLUES = 1
NUM_REDS = 1
SIM_STEPS = 600
HORIZON = 10
PLANNING_INTERVAL_SECONDS = 1.0


class ReconThenNeutralizePlanner(RolePlanner):
    """RECON until the red is actually detected by blue's own FOV, then
    NEUTRALIZE with target_red_id set to the (last) detected red. Once
    spotted, stays committed to NEUTRALIZE even if the red later drops out of
    FOV -- it doesn't forget a threat it already confirmed -- reusing the
    last known target_red_id in that case."""

    def __init__(self):
        self.spotted = False
        self.last_target_red_ids = {}

    def plan(self, obs):
        disabled = set(obs.disabled_blues or [])
        active_blues = [i for i in range(obs.num_blues) if i not in disabled]

        detected_reds = (
            list(range(obs.num_reds))
            if obs.detected_red_indices is None
            else list(obs.detected_red_indices)
        )
        if detected_reds:
            self.spotted = True

        if not self.spotted:
            self.last_target_red_ids = {}
            return {i: Role.RECON for i in active_blues}

        if detected_reds:
            red_nx = _infer_stride(obs.red_states, obs.num_reds)
            blue_nx = _infer_stride(obs.x_current, obs.num_blues)
            red_states = np.asarray(obs.red_states, dtype=float).reshape(red_nx * obs.num_reds)
            x_current = np.asarray(obs.x_current, dtype=float).reshape(blue_nx * obs.num_blues)
            targets = {}
            for i in active_blues:
                blue_pos = x_current[i * blue_nx: i * blue_nx + 2]
                dists = {
                    j: np.linalg.norm(blue_pos - red_states[j * red_nx: j * red_nx + 2])
                    for j in detected_reds
                }
                targets[i] = min(dists, key=dists.get)
            self.last_target_red_ids = targets
        # else: red temporarily out of FOV again -- keep last known target so
        # solve_decentralized's target_override stays populated rather than
        # silently falling back to the ground-truth nearest-red heuristic.
        return {i: Role.NEUTRALIZE for i in active_blues}


if __name__ == "__main__":
    sim = Simulation(
        solver_mode="decentralized",
        num_blues=NUM_BLUES,
        num_reds=NUM_REDS,
        sim_steps=SIM_STEPS,
        horizon=HORIZON,
        seed=SEED,
        enable_blue_blue_cbf=True,
        enable_blue_red_cbf=True,
        role_planner=ReconThenNeutralizePlanner(),
        planning_interval_seconds=PLANNING_INTERVAL_SECONDS,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()
        render_pov_gif(sim, "blue", "recon_then_neutralize.gif")

    print(
        f"steps={len(sim.history_red) - 1} solver_failed={sim.solver_failed} "
        f"blue_score={sim.blue_score} red_outcomes={sim.red_outcomes}"
    )
