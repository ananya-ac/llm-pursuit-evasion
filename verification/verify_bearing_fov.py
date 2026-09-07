"""Verification script for bearing-as-state + FOV geometry.

Bypasses the full Simulation/solver/planner stack on purpose -- this only
exercises Agent.step_single (agent.py) and the FOV/detection geometry
(sensing.py) directly, to isolate two claims:

1. Bearing is a genuinely independent state component: holding a=[0,0] and
   commanding a constant yaw rate omega leaves position/velocity exactly
   fixed while theta advances linearly -- proving the position/velocity
   dynamics are untouched by heading, and that heading doesn't need motion
   to change (see agent.py's Agent(include_heading=True)).
2. The FOV cone is oriented along the bearing direction: as theta sweeps
   0 -> 2*pi, static probe points around the agent should light up
   (detected=True) only while the cone is actually pointing at them.

Produces bearing_fov_verification.gif: a fixed blue agent with a rotating
FOV wedge and bearing ray, surrounded by probe points that turn green when
inside the current field of view and gray otherwise.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from agents.dynamics import Agent
import perception.sensing as sensing

dt = 0.1
sim_steps = 200  # 20s of sim time
omega = (2 * np.pi) / (sim_steps * dt)  # constant yaw rate -> exactly one full sweep

agent = Agent(agent_type=Agent.BLUE, dt=dt, count=1, v_max=2.0, a_max=2.0, include_heading=True)

position = np.array([50.0, 50.0])
state = np.array([position[0], position[1], 0.0, 0.0, 0.0])  # theta_0 = 0

history = [state.copy()]
for _ in range(sim_steps):
    state = agent.step_single(state, [0.0, 0.0, omega])
    history.append(state.copy())
history = np.asarray(history)

# --- Numeric verification: position/velocity held fixed, theta sweeps 0->2pi ---
assert np.allclose(history[:, 0:2], position, atol=1e-9), "position drifted -- heading is not independent!"
assert np.allclose(history[:, 2:4], 0.0, atol=1e-9), "velocity drifted -- heading is not independent!"
print(f"Position held fixed at {position} across all {sim_steps} steps: PASS")
print(f"Velocity held at zero across all {sim_steps} steps: PASS")
print(f"theta: start={history[0, 4]:.4f} rad, end={history[-1, 4]:.4f} rad (expected {2*np.pi:.4f})")

# --- Probe points + detection sanity check ---
half_angle = np.deg2rad(sensing.DEFAULT_FOV_HALF_ANGLE_DEG)
fov_range = sensing.DEFAULT_FOV_RANGE
probe_radius = 0.6 * fov_range
num_probes = 12
probe_angles = np.linspace(0, 2 * np.pi, num_probes, endpoint=False)
probes = [position + probe_radius * np.array([np.cos(a), np.sin(a)]) for a in probe_angles]

# Sanity: at theta=0, only the probe(s) near angle 0 should be detected.
detected_at_zero = [
    i for i, p in enumerate(probes) if sensing.is_detected(position, 0.0, p, half_angle, fov_range)
]
print(f"At theta=0, detected probe indices (should cluster near angle 0): {detected_at_zero}")
for i in detected_at_zero:
    print(f"  probe {i} at {np.degrees(probe_angles[i]):.1f} deg")

# --- Render animation ---
fig, ax = plt.subplots(figsize=(7, 7))
half_extent = probe_radius + 5.0
ax.set_xlim(position[0] - half_extent, position[0] + half_extent)
ax.set_ylim(position[1] - half_extent, position[1] + half_extent)
ax.set_aspect("equal")
ax.grid(True)
ax.set_title("Bearing-as-state verification: position fixed, FOV follows theta")

fov_wedge = patches.Wedge(position, fov_range, 0, 0, alpha=0.2, color="tab:blue", label="FOV cone")
ax.add_patch(fov_wedge)
bearing_line, = ax.plot([], [], "b-", linewidth=2, label="bearing")
agent_point, = ax.plot([position[0]], [position[1]], "bo", markersize=14, label="blue agent")
probe_points = [ax.plot([], [], "o", markersize=10)[0] for _ in probes]
theta_text = ax.text(
    position[0] - half_extent + 1, position[1] + half_extent - 2, "", fontsize=12
)
ax.legend(loc="upper right", fontsize=9)


def update(frame):
    theta = history[frame, 4]
    theta_deg = np.degrees(theta)

    bearing_line.set_data(
        [position[0], position[0] + fov_range * np.cos(theta)],
        [position[1], position[1] + fov_range * np.sin(theta)],
    )
    fov_wedge.set_center(position)
    fov_wedge.set_theta1(theta_deg - np.degrees(half_angle))
    fov_wedge.set_theta2(theta_deg + np.degrees(half_angle))

    for p_artist, probe_pos in zip(probe_points, probes):
        detected = sensing.is_detected(position, theta, probe_pos, half_angle, fov_range)
        p_artist.set_data([probe_pos[0]], [probe_pos[1]])
        p_artist.set_color("tab:green" if detected else "0.6")

    theta_text.set_text(f"theta = {theta_deg % 360:.1f} deg")
    return [fov_wedge, bearing_line, agent_point, theta_text] + probe_points


anim = animation.FuncAnimation(fig, update, frames=len(history), interval=50, blit=False)
output_path = "bearing_fov_verification.gif"
anim.save(output_path, writer="pillow", fps=20)
plt.close(fig)
print(f"Saved {output_path}")
