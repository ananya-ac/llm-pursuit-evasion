"""Decentralized per-blue MPC controllers, one per tactical Role.

Each controller builds its small single-agent QP once at construction (mirroring
the pattern in solver.BaseMinimaxSolver) and re-solves it every timestep via
set_value/opti.solve() -- the CasADi graph is never rebuilt per call. Blue-
blue/blue-red safety coordination is handled downstream by the existing
CBF filters (solver.BaseMinimaxSolver.one_step_cbf_filter), not here; these
controllers only produce each agent's nominal (unfiltered) plan.

The DEFEND/NEUTRALIZE/red cost formulations below are a starting point meant
to be refined once real tactics are worked out, not a final spec.
"""

import casadi as ca
import numpy as np

import sensing


class BaseBlueRoleController:
    """Single-blue decentralized MPC controller for one tactical role.

    include_heading=True appends a 5th state (bearing theta) and 3rd control
    (yaw rate omega), with theta_{k+1} = theta_k + dt*omega_k as a genuinely
    independent integrator -- it never enters, and is never driven by, the
    position/velocity block above, which is unchanged either way. Subclasses
    that don't give omega a role-specific cost (i.e. all of them, currently)
    get it regularized toward 0 via w_omega, so heading simply holds at
    whatever it was initialized to.
    """

    def __init__(self, horizon, dt, a_max, v_max, w_u=0.5, include_heading=False,
                 omega_max=2.0 * np.pi, w_omega=0.1):
        self.N = int(horizon)
        self.dt = float(dt)
        self.a_max = float(a_max)
        self.v_max = float(v_max)
        self.w_u = float(w_u)
        self.include_heading = bool(include_heading)
        self.omega_max = float(omega_max)
        self.w_omega = float(w_omega)
        self.nx = 5 if self.include_heading else 4
        self.nu = 3 if self.include_heading else 2

        self.opti = ca.Opti("conic")
        self.X = self.opti.variable(self.nx, self.N + 1)
        self.U = self.opti.variable(self.nu, self.N)
        self.x0_param = self.opti.parameter(self.nx)

        self.opti.subject_to(self.X[:, 0] == self.x0_param)
        for k in range(self.N):
            px, py = self.X[0, k], self.X[1, k]
            vx, vy = self.X[2, k], self.X[3, k]
            ax, ay = self.U[0, k], self.U[1, k]

            vx_next = vx + self.dt * ax
            vy_next = vy + self.dt * ay
            px_next = px + self.dt * vx_next
            py_next = py + self.dt * vy_next

            if self.include_heading:
                theta_next = self.X[4, k] + self.dt * self.U[2, k]
                self.opti.subject_to(
                    self.X[:, k + 1] == ca.vertcat(px_next, py_next, vx_next, vy_next, theta_next)
                )
                self.opti.subject_to(self.opti.bounded(-self.omega_max, self.U[2, k], self.omega_max))
            else:
                self.opti.subject_to(
                    self.X[:, k + 1] == ca.vertcat(px_next, py_next, vx_next, vy_next)
                )
            self.opti.subject_to(self.opti.bounded(-self.a_max, ax, self.a_max))
            self.opti.subject_to(self.opti.bounded(-self.a_max, ay, self.a_max))
            self.opti.subject_to(self.opti.bounded(-self.v_max, vx_next, self.v_max))
            self.opti.subject_to(self.opti.bounded(-self.v_max, vy_next, self.v_max))

        self._declare_extra_params()

        J = self.w_u * sum(ca.sumsqr(self.U[0:2, k]) for k in range(self.N))
        if self.include_heading:
            J += self.w_omega * sum(ca.sumsqr(self.U[2, k]) for k in range(self.N))
        J += self._build_role_cost()
        self.opti.minimize(J)
        self.opti.solver("osqp", {"verbose": False})

    def _declare_extra_params(self):
        raise NotImplementedError

    def _build_role_cost(self):
        raise NotImplementedError

    def _set_extra_values(self, own_state, red_state, other_blue_states, agent_id=None,
                           directed_target=None):
        raise NotImplementedError

    def plan(self, own_state, red_state, other_blue_states=None, agent_id=None,
             directed_target=None):
        """Returns (u0, X_plan, U_plan, solved_ok).

        agent_id: the calling blue's index. Most roles don't need it (the QP
        only ever depends on this agent's own state), but a single
        controller instance is shared by every blue currently assigned the
        same role, so any role that keeps state *across* solves (e.g.
        ReconController's patrol waypoint/sweep phase) needs it to avoid
        multiple simultaneous agents sharing one patrol in lockstep.
        Collision avoidance is not this layer's concern regardless of role
        -- it's handled uniformly downstream by the one-step CBF filter
        (Simulation.run()) on top of whatever nominal plan any role
        produces, using full ground-truth state.

        directed_target: optional 2-vector, currently only consumed by
        ReconController (steers its patrol waypoint toward a role planner's
        chosen coverage region instead of a uniformly random one). Every
        other role ignores it -- accepted here so callers (solve_decentralized)
        don't need to special-case which controller they're calling.
        """
        own_state = np.asarray(own_state, dtype=float).reshape(self.nx)
        # red_state may be wider than 4 (e.g. also carries heading) -- only
        # position [0:2] / velocity [2:4] are ever read downstream.
        red_state = np.asarray(red_state, dtype=float).flatten()

        self.opti.set_value(self.x0_param, own_state)
        self._set_extra_values(
            own_state, red_state, other_blue_states, agent_id=agent_id,
            directed_target=directed_target,
        )

        try:
            sol = self.opti.solve()
        except RuntimeError:
            return None, None, None, False

        X_plan = sol.value(self.X)
        U_plan = sol.value(self.U)
        return U_plan[:, 0], X_plan, U_plan, True


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

    Adapted from the blue-side terms of PursuitEvasionMinimaxSolver's shared
    cost (solver.py). The velocity-cutoff term is dropped: without a jointly
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
    while independently sweeping its bearing back and forth across a
    2*pi/3-radian arc centered on its current direction of travel -- like a
    scout that keeps looking around while walking a patrol route, rather
    than committing its sensor to a single fixed heading. Translation and
    heading are controlled independently (see BaseBlueRoleController), so
    the sweep is a real rotation, not just "which way am I walking."

    Waypoints are sampled uniformly within [0, arena_size]^2, so the patrol
    never needs to be clipped to stay in bounds -- a new one is drawn once
    the agent gets within waypoint_reach_radius of the current one. The
    sweep target is theta_center +- pi/3 (a total range of 2*pi/3 radians),
    oscillating sinusoidally over sweep_period seconds, where theta_center
    is the current bearing-to-waypoint (recomputed each solve).

    plan(..., directed_target=...) overrides the random-patrol waypoint with
    a role planner's chosen point (e.g. a stale coverage region's center --
    see coverage.CoverageTracker / RuleBasedRolePlanner) and holds there once
    reached instead of resampling, so an assigned scout actually covers the
    intended region rather than wandering the whole arena.

    Patrol waypoint and sweep phase are tracked per-agent (keyed by
    agent_id), since a single ReconController instance is shared by every
    blue currently assigned RECON -- without per-agent state, multiple
    simultaneous RECON agents would patrol in lockstep. Falls back to a
    plain hold-position-toward-waypoint (no sweep) when heading isn't part
    of the state at all.

    No blue-blue avoidance term here, same as every other role -- collision
    safety is handled uniformly downstream by the one-step CBF filter.
    """

    def __init__(self, horizon, dt, a_max, v_max, arena_size=100.0,
                 waypoint_reach_radius=5.0, sweep_period=10.0,
                 w_hold=1.0, w_sweep=1.0, w_u=0.5,
                 include_heading=False, omega_max=2.0 * np.pi, w_omega=0.1):
        self.arena_size = float(arena_size)
        self.waypoint_reach_radius = float(waypoint_reach_radius)
        self.sweep_period = float(sweep_period)
        self.w_hold = float(w_hold)
        self.w_sweep = float(w_sweep)
        self.patrol_targets = {}  # agent_id -> np.array([x, y])
        self.elapsed_times = {}   # agent_id -> seconds this agent has been sweeping
        super().__init__(
            horizon=horizon, dt=dt, a_max=a_max, v_max=v_max, w_u=w_u,
            include_heading=include_heading, omega_max=omega_max, w_omega=w_omega,
        )

    def _declare_extra_params(self):
        self.target_param = self.opti.parameter(2)
        if self.include_heading:
            self.theta_target_param = self.opti.parameter(self.N)

    def _build_role_cost(self):
        J = 0
        for k in range(1, self.N + 1):
            J += self.w_hold * ca.sumsqr(self.X[0:2, k] - self.target_param)
        if self.include_heading:
            for k in range(1, self.N + 1):
                J += self.w_sweep * ca.sumsqr(self.X[4, k] - self.theta_target_param[k - 1])
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
            self.elapsed_times.setdefault(key, 0.0)
        else:
            if key not in self.patrol_targets:
                self.patrol_targets[key] = self._random_waypoint()
                self.elapsed_times[key] = 0.0
            if np.linalg.norm(own_pos - self.patrol_targets[key]) < self.waypoint_reach_radius:
                self.patrol_targets[key] = self._random_waypoint()

        target = self.patrol_targets[key]
        self.opti.set_value(self.target_param, target)

        if self.include_heading:
            direction = target - own_pos
            theta_center = (
                float(np.arctan2(direction[1], direction[0]))
                if np.linalg.norm(direction) > 1e-6
                else float(own_state[4])
            )
            t0 = self.elapsed_times[key]
            theta_targets = np.array([
                theta_center
                + (np.pi / 3.0) * np.sin(2.0 * np.pi * (t0 + k * self.dt) / self.sweep_period)
                for k in range(1, self.N + 1)
            ])
            self.opti.set_value(self.theta_target_param, theta_targets)
            self.elapsed_times[key] = t0 + self.dt


class EvadeController:
    """Red's EVADE-role local MPC: flees the decentralized blues' intended plans.

    Only reacts to blues red has actually detected (see plan()'s
    detected_blue_indices) -- each blue's contribution to the avoidance term
    is gated by a per-blue weight parameter (1 if detected, 0 otherwise),
    set fresh every solve, rather than the avoidance term always seeing
    every blue's true plan regardless of fog of war.

    Replaces the old shared-cost red NLP, which no longer makes sense once
    blues are decentralized (there is no single joint cost left to maximize
    against). Kept on IPOPT/nonlinear Opti (like the previous red solve)
    since the avoidance term below is a concave quadratic, not convex-QP-safe.
    """

    def __init__(self, horizon, dt, a_max, v_max, n_blues, w_ue=0.5, w_avoid=5.0,
                 include_heading=False, blue_nx=4, omega_max=2.0 * np.pi, w_omega=0.1):
        self.N = int(horizon)
        self.dt = float(dt)
        self.a_max = float(a_max)
        self.v_max = float(v_max)
        self.n_blues = int(n_blues)
        self.w_ue = float(w_ue)
        self.w_avoid = float(w_avoid)
        self.include_heading = bool(include_heading)
        self.omega_max = float(omega_max)
        self.w_omega = float(w_omega)
        self.nx = 5 if self.include_heading else 4
        self.nu = 3 if self.include_heading else 2
        # blue_nx: stride of the incoming blue_state_plans, which may differ
        # in width from this (red) agent's own state -- only p_i=[0:2] is
        # ever read from it, so its own heading/omega (if any) is irrelevant.
        self.blue_nx = int(blue_nx)

        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx, self.N + 1)
        self.U = self.opti.variable(self.nu, self.N)
        self.x0_param = self.opti.parameter(self.nx)
        self.blue_traj_param = self.opti.parameter(self.blue_nx * self.n_blues, self.N + 1)
        # 1.0 for a detected blue, 0.0 for an undetected one -- set fresh
        # every solve in plan(), so the avoidance term below only ever
        # reacts to blues red has actually seen.
        self.detected_weight_param = self.opti.parameter(self.n_blues)

        self.opti.subject_to(self.X[:, 0] == self.x0_param)
        for k in range(self.N):
            px, py = self.X[0, k], self.X[1, k]
            vx, vy = self.X[2, k], self.X[3, k]
            ax, ay = self.U[0, k], self.U[1, k]

            vx_next = vx + self.dt * ax
            vy_next = vy + self.dt * ay
            px_next = px + self.dt * vx_next
            py_next = py + self.dt * vy_next

            if self.include_heading:
                theta_next = self.X[4, k] + self.dt * self.U[2, k]
                self.opti.subject_to(
                    self.X[:, k + 1] == ca.vertcat(px_next, py_next, vx_next, vy_next, theta_next)
                )
                self.opti.subject_to(self.opti.bounded(-self.omega_max, self.U[2, k], self.omega_max))
            else:
                self.opti.subject_to(
                    self.X[:, k + 1] == ca.vertcat(px_next, py_next, vx_next, vy_next)
                )
            self.opti.subject_to(self.opti.bounded(-self.a_max, ax, self.a_max))
            self.opti.subject_to(self.opti.bounded(-self.a_max, ay, self.a_max))
            self.opti.subject_to(self.opti.bounded(-self.v_max, vx_next, self.v_max))
            self.opti.subject_to(self.opti.bounded(-self.v_max, vy_next, self.v_max))

        J = 0
        for k in range(self.N):
            J += self.w_ue * ca.sumsqr(self.U[0:2, k])
            if self.include_heading:
                J += self.w_omega * ca.sumsqr(self.U[2, k])
        for k in range(1, self.N + 1):
            p_e = self.X[0:2, k]
            for i in range(self.n_blues):
                p_i = self.blue_traj_param[i * self.blue_nx : i * self.blue_nx + 2, k]
                J -= self.detected_weight_param[i] * self.w_avoid * ca.sumsqr(p_e - p_i)
        self.opti.minimize(J)

        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 500, "print_level": 0, "sb": "yes"}
        self.opti.solver("ipopt", p_opts, s_opts)

    def plan(self, red_state, blue_state_plans, detected_blue_indices=None):
        """Returns (u0, X_plan, solved_ok). blue_state_plans: (blue_nx*n_blues, N+1).

        detected_blue_indices: iterable of blue indices red has actually
        detected this step; undetected blues are weighted out of the
        avoidance term entirely. None means "no restriction" (all detected)
        -- back-compat default for direct/isolated calls.
        """
        red_state = np.asarray(red_state, dtype=float).reshape(self.nx)
        blue_state_plans = np.asarray(blue_state_plans, dtype=float).reshape(
            self.blue_nx * self.n_blues, self.N + 1
        )

        self.opti.set_value(self.x0_param, red_state)
        self.opti.set_value(self.blue_traj_param, blue_state_plans)
        if detected_blue_indices is None:
            weights = np.ones(self.n_blues)
        else:
            detected = set(detected_blue_indices)
            weights = np.array([1.0 if i in detected else 0.0 for i in range(self.n_blues)])
        self.opti.set_value(self.detected_weight_param, weights)

        x_guess = np.tile(red_state[:, None], (1, self.N + 1))
        self.opti.set_initial(self.X, x_guess)
        self.opti.set_initial(self.U, np.zeros((self.nu, self.N)))

        try:
            sol = self.opti.solve()
        except RuntimeError:
            return None, None, False

        return sol.value(self.U)[:, 0], sol.value(self.X), True


class AttackController:
    """Red's ATTACK-role local MPC: homes in on the defended region.

    No blue-avoidance term in the nominal cost -- pursuer-pursuer/pursuer-evader
    safety coordination is handled downstream by the CBF filter, same as
    DefendController/NeutralizeController on the blue side. Pure convex
    quadratic cost, so (unlike EvadeController) this stays on the conic/OSQP
    solver like the blue role controllers.
    """

    def __init__(
        self,
        horizon,
        dt,
        a_max,
        v_max,
        defense_center=None,
        arena_size=100.0,
        w_ue=0.5,
        w_progress=1.0,
        include_heading=False,
        omega_max=2.0 * np.pi,
        w_omega=0.1,
    ):
        self.N = int(horizon)
        self.dt = float(dt)
        self.a_max = float(a_max)
        self.v_max = float(v_max)
        self.defense_center = (
            np.array([0.5 * arena_size, 0.5 * arena_size], dtype=float)
            if defense_center is None
            else np.asarray(defense_center, dtype=float).reshape(2)
        )
        self.w_ue = float(w_ue)
        self.w_progress = float(w_progress)
        self.include_heading = bool(include_heading)
        self.omega_max = float(omega_max)
        self.w_omega = float(w_omega)
        self.nx = 5 if self.include_heading else 4
        self.nu = 3 if self.include_heading else 2

        self.opti = ca.Opti("conic")
        self.X = self.opti.variable(self.nx, self.N + 1)
        self.U = self.opti.variable(self.nu, self.N)
        self.x0_param = self.opti.parameter(self.nx)

        self.opti.subject_to(self.X[:, 0] == self.x0_param)
        for k in range(self.N):
            px, py = self.X[0, k], self.X[1, k]
            vx, vy = self.X[2, k], self.X[3, k]
            ax, ay = self.U[0, k], self.U[1, k]

            vx_next = vx + self.dt * ax
            vy_next = vy + self.dt * ay
            px_next = px + self.dt * vx_next
            py_next = py + self.dt * vy_next

            if self.include_heading:
                theta_next = self.X[4, k] + self.dt * self.U[2, k]
                self.opti.subject_to(
                    self.X[:, k + 1] == ca.vertcat(px_next, py_next, vx_next, vy_next, theta_next)
                )
                self.opti.subject_to(self.opti.bounded(-self.omega_max, self.U[2, k], self.omega_max))
            else:
                self.opti.subject_to(
                    self.X[:, k + 1] == ca.vertcat(px_next, py_next, vx_next, vy_next)
                )
            self.opti.subject_to(self.opti.bounded(-self.a_max, ax, self.a_max))
            self.opti.subject_to(self.opti.bounded(-self.a_max, ay, self.a_max))
            self.opti.subject_to(self.opti.bounded(-self.v_max, vx_next, self.v_max))
            self.opti.subject_to(self.opti.bounded(-self.v_max, vy_next, self.v_max))

        J = self.w_ue * sum(ca.sumsqr(self.U[0:2, k]) for k in range(self.N))
        if self.include_heading:
            J += self.w_omega * sum(ca.sumsqr(self.U[2, k]) for k in range(self.N))
        for k in range(1, self.N + 1):
            J += self.w_progress * ca.sumsqr(self.X[0:2, k] - ca.DM(self.defense_center))
        self.opti.minimize(J)
        self.opti.solver("osqp", {"verbose": False})

    def plan(self, red_state, blue_state_plans=None):
        """Returns (u0, X_plan, solved_ok).

        blue_state_plans is accepted for interface parity with EvadeController
        but unused -- ATTACK's nominal cost has no blue-avoidance term.
        """
        del blue_state_plans
        red_state = np.asarray(red_state, dtype=float).reshape(self.nx)
        self.opti.set_value(self.x0_param, red_state)

        try:
            sol = self.opti.solve()
        except RuntimeError:
            return None, None, False

        return sol.value(self.U)[:, 0], sol.value(self.X), True


class RedReconController:
    """Red's RECON-role local MPC: holds current position, recomputed fresh
    every solve. Standalone (not BaseBlueRoleController-derived) to mirror
    AttackController's structure, since red controllers aren't part of that
    blue-only base class hierarchy. Same placeholder caveat as
    ReconController -- a stub for a future scouting controller, with no
    sensing difference from any other role (see sensing.py).
    """

    def __init__(self, horizon, dt, a_max, v_max, w_ue=0.5, w_hold=1.0,
                 include_heading=False, omega_max=2.0 * np.pi, w_omega=0.1):
        self.N = int(horizon)
        self.dt = float(dt)
        self.a_max = float(a_max)
        self.v_max = float(v_max)
        self.w_ue = float(w_ue)
        self.w_hold = float(w_hold)
        self.include_heading = bool(include_heading)
        self.omega_max = float(omega_max)
        self.w_omega = float(w_omega)
        self.nx = 5 if self.include_heading else 4
        self.nu = 3 if self.include_heading else 2

        self.opti = ca.Opti("conic")
        self.X = self.opti.variable(self.nx, self.N + 1)
        self.U = self.opti.variable(self.nu, self.N)
        self.x0_param = self.opti.parameter(self.nx)
        self.target_param = self.opti.parameter(2)

        self.opti.subject_to(self.X[:, 0] == self.x0_param)
        for k in range(self.N):
            px, py = self.X[0, k], self.X[1, k]
            vx, vy = self.X[2, k], self.X[3, k]
            ax, ay = self.U[0, k], self.U[1, k]

            vx_next = vx + self.dt * ax
            vy_next = vy + self.dt * ay
            px_next = px + self.dt * vx_next
            py_next = py + self.dt * vy_next

            if self.include_heading:
                theta_next = self.X[4, k] + self.dt * self.U[2, k]
                self.opti.subject_to(
                    self.X[:, k + 1] == ca.vertcat(px_next, py_next, vx_next, vy_next, theta_next)
                )
                self.opti.subject_to(self.opti.bounded(-self.omega_max, self.U[2, k], self.omega_max))
            else:
                self.opti.subject_to(
                    self.X[:, k + 1] == ca.vertcat(px_next, py_next, vx_next, vy_next)
                )
            self.opti.subject_to(self.opti.bounded(-self.a_max, ax, self.a_max))
            self.opti.subject_to(self.opti.bounded(-self.a_max, ay, self.a_max))
            self.opti.subject_to(self.opti.bounded(-self.v_max, vx_next, self.v_max))
            self.opti.subject_to(self.opti.bounded(-self.v_max, vy_next, self.v_max))

        J = self.w_ue * sum(ca.sumsqr(self.U[0:2, k]) for k in range(self.N))
        if self.include_heading:
            J += self.w_omega * sum(ca.sumsqr(self.U[2, k]) for k in range(self.N))
        for k in range(1, self.N + 1):
            J += self.w_hold * ca.sumsqr(self.X[0:2, k] - self.target_param)
        self.opti.minimize(J)
        self.opti.solver("osqp", {"verbose": False})

    def plan(self, red_state, blue_state_plans=None):
        """Returns (u0, X_plan, solved_ok).

        blue_state_plans is accepted for interface parity with
        EvadeController/AttackController but unused.
        """
        del blue_state_plans
        red_state = np.asarray(red_state, dtype=float).reshape(self.nx)
        self.opti.set_value(self.x0_param, red_state)
        self.opti.set_value(self.target_param, red_state[0:2].copy())

        try:
            sol = self.opti.solve()
        except RuntimeError:
            return None, None, False

        return sol.value(self.U)[:, 0], sol.value(self.X), True
