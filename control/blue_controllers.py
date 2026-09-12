"""Blue-team decentralized MPC role controllers: DEFEND, NEUTRALIZE, and
RECON. Each implements BaseBlueRoleController's three hooks; dynamics, box
constraints, and the shared plan() are inherited unchanged.
"""

import casadi as ca
import numpy as np

from control.base import BaseBlueRoleController


class DefendController(BaseBlueRoleController):
    """Holds a guard station on the defended perimeter closest to the red.

    Falls back to simply holding the blue's current position when no
    perimeter polygon is available (defensive coding -- not the expected path
    once the hybrid arena+perimeter-defense game geometry is wired up).
    """

    def __init__(
        self,
        horizon,
        dt,
        a_max,
        v_max,
        defense_polygon=None,
        arena_size=100.0,
        w_hold=1.0,
        w_u=0.5,
        include_heading=False,
        omega_max=2.0 * np.pi,
        w_omega=0.1,
    ):
        self.defense_polygon = (
            None
            if defense_polygon is None
            else np.asarray(defense_polygon, dtype=float).reshape(-1, 2)
        )
        del arena_size
        self.w_hold = float(w_hold)
        super().__init__(
            horizon=horizon, dt=dt, a_max=a_max, v_max=v_max, w_u=w_u,
            include_heading=include_heading, omega_max=omega_max, w_omega=w_omega,
        )

    def _declare_extra_params(self):
        self.target_param = self.opti.parameter(2)

    def _build_role_cost(self):
        J = 0
        for k in range(1, self.N + 1):
            J += self.w_hold * ca.sumsqr(self.X[0:2, k] - self.target_param)
        return J

    @staticmethod
    def _closest_point_on_segment(point, start, end):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            return start.copy()
        alpha = np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0)
        return start + alpha * segment

    def _closest_point_on_polygon(self, point):
        polygon = self.defense_polygon
        closed_polygon = np.vstack([polygon, polygon[0]])
        best_point = polygon[0]
        best_dist_sq = np.inf
        for edge_idx in range(len(polygon)):
            candidate = self._closest_point_on_segment(
                point, closed_polygon[edge_idx], closed_polygon[edge_idx + 1]
            )
            dist_sq = float(np.sum((candidate - point) ** 2))
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_point = candidate
        return best_point

    def _set_extra_values(self, own_state, red_state, other_blue_states, agent_id=None,
                           directed_target=None):
        del other_blue_states, agent_id, directed_target
        if self.defense_polygon is not None and len(self.defense_polygon) >= 3:
            target = self._closest_point_on_polygon(red_state[0:2])
        else:
            target = own_state[0:2].copy()
        self.opti.set_value(self.target_param, target)


class NeutralizeController(BaseBlueRoleController):
    """Intercepts the red using a constant-velocity position prediction.

    Adapted from the blue-side terms of the earlier joint-minimax solver's
    shared cost. The velocity-cutoff term is dropped: without a jointly
    optimized red trajectory, it was matching velocity toward a stale linear
    extrapolation more aggressively than the position term itself pulled
    toward the red's actual predicted position.
    """

    def __init__(self, horizon, dt, a_max, v_max, w_e=5.0, w_u=0.5,
                 include_heading=False, omega_max=2.0 * np.pi, w_omega=0.1):
        self.w_e = float(w_e)
        super().__init__(
            horizon=horizon, dt=dt, a_max=a_max, v_max=v_max, w_u=w_u,
            include_heading=include_heading, omega_max=omega_max, w_omega=w_omega,
        )

    def _declare_extra_params(self):
        self.red_traj_param = self.opti.parameter(2, self.N + 1)

    def _build_role_cost(self):
        J = 0
        for k in range(1, self.N + 1):
            p_i = self.X[0:2, k]
            p_e = self.red_traj_param[:, k]
            J += self.w_e * ca.sumsqr(p_i - p_e)
        return J

    def _set_extra_values(self, own_state, red_state, other_blue_states, agent_id=None,
                           directed_target=None):
        del own_state, other_blue_states, agent_id, directed_target
        p_e0, v_e = red_state[0:2], red_state[2:4]
        traj = np.zeros((2, self.N + 1))
        for k in range(self.N + 1):
            traj[:, k] = p_e0 + k * self.dt * v_e
        self.opti.set_value(self.red_traj_param, traj)


class ReconController(BaseBlueRoleController):
    """RECON controller: patrols toward a random waypoint inside the arena,
    while independently pointing its bearing at that same target -- the
    sensor stays aimed at the point the agent is scouting rather than
    wherever the current thrust happens to be headed. Translation and
    heading are controlled independently (see BaseBlueRoleController), so
    this is a real rotation, not just "which way am I walking."

    Waypoints are sampled uniformly within [0, arena_size]^2, so the patrol
    never needs to be clipped to stay in bounds -- a new one is drawn once
    the agent gets within waypoint_reach_radius of the current one. The
    bearing target is the current bearing-to-waypoint, recomputed each
    solve; once the agent reaches the waypoint and holds position there, the
    bearing holds on it too.

    plan(..., directed_target=...) overrides the random-patrol waypoint with
    a role planner's chosen point (e.g. a stale coverage region's center --
    see coverage.CoverageTracker / RuleBasedBlueRolePlanner) and holds there once
    reached instead of resampling, so an assigned scout actually covers the
    intended region rather than wandering the whole arena -- bearing follows
    along, so the agent keeps facing the assigned region once it arrives.

    Patrol waypoint is tracked per-agent (keyed by agent_id), since a single
    ReconController instance is shared by every blue currently assigned
    RECON -- without per-agent state, multiple simultaneous RECON agents
    would patrol in lockstep. Falls back to a plain hold-position-toward-
    waypoint (no bearing term) when heading isn't part of the state at all.

    No blue-blue avoidance term here, same as every other role -- collision
    safety is handled uniformly downstream by the one-step CBF filter.
    """

    def __init__(self, horizon, dt, a_max, v_max, arena_size=100.0,
                 waypoint_reach_radius=5.0,
                 w_hold=1.0, w_point=1.0, w_u=0.5,
                 include_heading=False, omega_max=2.0 * np.pi, w_omega=0.1):
        self.arena_size = float(arena_size)
        self.waypoint_reach_radius = float(waypoint_reach_radius)
        self.w_hold = float(w_hold)
        self.w_point = float(w_point)
        self.patrol_targets = {}  # agent_id -> np.array([x, y])
        super().__init__(
            horizon=horizon, dt=dt, a_max=a_max, v_max=v_max, w_u=w_u,
            include_heading=include_heading, omega_max=omega_max, w_omega=w_omega,
        )

    def _declare_extra_params(self):
        self.target_param = self.opti.parameter(2)
        if self.include_heading:
            self.theta_target_param = self.opti.parameter(1)

    def _build_role_cost(self):
        J = 0
        for k in range(1, self.N + 1):
            J += self.w_hold * ca.sumsqr(self.X[0:2, k] - self.target_param)
        if self.include_heading:
            for k in range(1, self.N + 1):
                J += self.w_point * ca.sumsqr(self.X[4, k] - self.theta_target_param)
        return J

    def _random_waypoint(self):
        return np.random.uniform(0.0, self.arena_size, size=2)

    def _set_extra_values(self, own_state, red_state, other_blue_states, agent_id=None,
                           directed_target=None):
        del red_state, other_blue_states
        own_pos = own_state[0:2]
        key = agent_id if agent_id is not None else 0

        if directed_target is not None:
            # A role planner (rule-based staleness heuristic or LLM) has
            # picked a specific coverage region for this agent -- pin the
            # patrol target there instead of the uniform-random fallback
            # below. Held in place once reached (no auto-resample) so the
            # agent actually scouts the assigned region rather than
            # wandering off it; this call happens every solve, so a new
            # planning-cycle assignment naturally overrides the old one.
            self.patrol_targets[key] = np.asarray(directed_target, dtype=float).reshape(2)
        else:
            if key not in self.patrol_targets:
                self.patrol_targets[key] = self._random_waypoint()
            if np.linalg.norm(own_pos - self.patrol_targets[key]) < self.waypoint_reach_radius:
                self.patrol_targets[key] = self._random_waypoint()

        target = self.patrol_targets[key]
        self.opti.set_value(self.target_param, target)

        if self.include_heading:
            direction = target - own_pos
            theta_target = (
                float(np.arctan2(direction[1], direction[0]))
                if np.linalg.norm(direction) > 1e-6
                else float(own_state[4])
            )
            self.opti.set_value(self.theta_target_param, theta_target)

