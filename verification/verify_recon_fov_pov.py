"""Verification script: renders a single decentralized simulation run twice,
once from each team's point of view. Every agent on the POV team gets a
rendered FOV wedge (oriented along its persisted bearing state, sensing.py's
geometry), and the OPPOSING team's markers are colored by whether they're
currently detected by the POV team's combined field of view -- a direct
visual check that FOV cones track bearing correctly and that detection
behaves as expected, using the same decentralized machinery (including the
RECON gap-filling controller) as the rest of the simulation.

Produces fov_blue_pov.gif and fov_red_pov.gif.
"""

import contextlib
import io

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import perception.sensing as sensing
from planning.planner import BlueRolePlanner
from planning.roles import BlueRole
from simulation.testbed import Simulation


class MixedPlanner(BlueRolePlanner):
    """Some blues DEFEND, some RECON -- gives the FOV/RECON dynamic something
    to actually show, rather than an all-DEFEND stalemate."""

    def plan(self, obs):
        return {i: (BlueRole.RECON if i % 2 == 0 else BlueRole.DEFEND) for i in range(obs.num_blues)}


def run_simulation():
    sim = Simulation(
        solver_mode="decentralized", num_blues=4, num_reds=2, sim_steps=100,
        horizon=10, seed=13, enable_blue_blue_cbf=True, enable_blue_red_cbf=True,
        role_planner=MixedPlanner(), planning_interval_seconds=1.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()
    return sim


def render_pov_gif(sim, pov, output_path):
    """pov: 'blue' or 'red'. Draws the POV team's FOV wedges (one per agent,
    oriented along its bearing) plus its own role labels, and colors the
    OPPOSING team's markers green (detected) or gray (hidden) based on the
    POV team's combined FOV, also labeled with its (ground-truth, for our
    inspection only -- not something the POV team actually knows) role."""
    is_blue_pov = pov == "blue"
    own_history = sim.history_blues if is_blue_pov else sim.history_red
    opp_history = sim.history_red if is_blue_pov else sim.history_blues
    own_role_history = sim.role_assignment_history if is_blue_pov else sim.red_role_assignment_history
    opp_role_history = sim.red_role_assignment_history if is_blue_pov else sim.role_assignment_history
    own_disabled_history = sim.blue_disabled_history if is_blue_pov else None
    num_own = sim.num_blues if is_blue_pov else sim.num_reds
    num_opp = sim.num_reds if is_blue_pov else sim.num_blues
    own_nx = sim.blue_nx if is_blue_pov else sim.red_nx
    opp_nx = sim.red_nx if is_blue_pov else sim.blue_nx

    plot_max = max(sim.arena_size, 105.0)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlim(0, plot_max)
    ax.set_ylim(0, plot_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{pov.upper()} POV -- field of view + detection")

    arena_border = plt.Rectangle(
        (0, 0), sim.arena_size, sim.arena_size, fill=False, edgecolor="k",
        linewidth=2, alpha=0.7,
    )
    ax.add_patch(arena_border)

    if sim.defense_polygon is not None:
        poly = np.vstack([sim.defense_polygon, sim.defense_polygon[0]])
        ax.plot(poly[:, 0], poly[:, 1], "k--", linewidth=2, alpha=0.7, label="Defense Polygon")

    own_color = "tab:blue" if is_blue_pov else "tab:red"
    own_label = "Blue" if is_blue_pov else "Red"
    opp_label = "Red" if is_blue_pov else "Blue"
    detected_color = "tab:green"
    hidden_color = "0.6"

    own_points = [ax.plot([], [], "o", color=own_color, markersize=10)[0] for _ in range(num_own)]
    own_wedges = [
        patches.Wedge((0, 0), sim.fov_range, 0, 0, alpha=0.15, color=own_color)
        for _ in range(num_own)
    ]
    for w in own_wedges:
        ax.add_patch(w)
    opp_points = [ax.plot([], [], "s", markersize=10, color=hidden_color)[0] for _ in range(num_opp)]
    own_role_texts = [
        ax.text(0, 0, "", fontsize=8, fontweight="bold", color=own_color) for _ in range(num_own)
    ]
    opp_role_texts = [
        ax.text(0, 0, "", fontsize=8, fontweight="bold", color="0.3") for _ in range(num_opp)
    ]

    ax.plot([], [], "o", color=own_color, label=f"{own_label} (POV)")
    ax.plot([], [], "s", color=detected_color, label=f"{opp_label} (detected)")
    ax.plot([], [], "s", color=hidden_color, label=f"{opp_label} (hidden)")
    ax.legend(loc="upper right", fontsize=9)

    half_angle = sim.fov_half_angle_rad
    fov_range = sim.fov_range
    label_offset = 0.015 * sim.arena_size

    def update(frame):
        own_frame = min(frame, len(own_history) - 1)
        opp_frame = min(frame, len(opp_history) - 1)
        own_state = own_history[own_frame]
        opp_state = opp_history[opp_frame]
        own_roles_now = own_role_history[min(frame, len(own_role_history) - 1)]
        opp_roles_now = opp_role_history[min(frame, len(opp_role_history) - 1)]
        own_disabled_now = (
            own_disabled_history[min(frame, len(own_disabled_history) - 1)]
            if own_disabled_history is not None else None
        )

        own_positions = []
        own_bearings = []
        for i in range(num_own):
            p = own_state[i * own_nx: i * own_nx + 2]
            theta = own_state[i * own_nx + 4]
            disabled = bool(own_disabled_now[i]) if own_disabled_now is not None else False
            own_points[i].set_data([p[0]], [p[1]])
            own_role_texts[i].set_position((p[0] + label_offset, p[1] + label_offset))
            if disabled:
                own_role_texts[i].set_text("DISABLED")
                own_points[i].set_alpha(0.3)
                own_wedges[i].set_visible(False)
            else:
                role = own_roles_now.get(i)
                own_role_texts[i].set_text(role.value.upper() if role is not None else "")
                own_points[i].set_alpha(1.0)
                own_wedges[i].set_visible(True)
                own_wedges[i].set_center((p[0], p[1]))
                theta_deg = np.degrees(theta)
                own_wedges[i].set_theta1(theta_deg - np.degrees(half_angle))
                own_wedges[i].set_theta2(theta_deg + np.degrees(half_angle))
            # A disabled agent's sensor is down too -- exclude it as an
            # observer for the opposing team's detection check below.
            own_positions.append(p if not disabled else None)
            own_bearings.append(theta)

        for j in range(num_opp):
            p = opp_state[j * opp_nx: j * opp_nx + 2]
            detected = any(
                sensing.is_detected(own_positions[i], own_bearings[i], p, half_angle, fov_range)
                for i in range(num_own)
                if own_positions[i] is not None
            )
            opp_points[j].set_data([p[0]], [p[1]])
            opp_points[j].set_color(detected_color if detected else hidden_color)
            opp_role_texts[j].set_position((p[0] + label_offset, p[1] + label_offset))
            opp_role = opp_roles_now.get(j)
            opp_role_texts[j].set_text(opp_role.value.upper() if opp_role is not None else "")

        return own_points + own_wedges + opp_points + own_role_texts + opp_role_texts

    anim = animation.FuncAnimation(fig, update, frames=len(own_history), interval=50, blit=False)
    anim.save(output_path, writer="pillow", fps=20)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    simulation = run_simulation()
    render_pov_gif(simulation, "blue", "fov_blue_pov.gif")
    render_pov_gif(simulation, "red", "fov_red_pov.gif")
