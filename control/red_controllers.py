"""Red-team decentralized MPC role controllers: EVADE, ATTACK, and RECON.
Each implements BaseRedRoleController's hooks; dynamics, box constraints,
control-effort regularization, and the shared plan() are inherited unchanged
(see control.base.BaseRedRoleController for the shared machinery and why
EVADE alone needs a non-default solver backend).
"""

import casadi as ca
import numpy as np

from control.base import BaseRedRoleController


class EvadeController(BaseRedRoleController):
    """Flees the decentralized blues' intended plans.

    Only reacts to blues red has actually detected (see plan()'s
    detected_blue_indices) -- each blue's contribution to the avoidance term
    is gated by a per-blue weight parameter (1 if detected, 0 otherwise),
    set fresh every solve, rather than the avoidance term always seeing
    every blue's true plan regardless of fog of war.

    The avoidance term is a concave quadratic, not convex-QP-safe -- kept on
    IPOPT/nonlinear Opti (via _make_opti/_configure_solver overrides) unlike
    every other controller (blue or red), which stays on conic/OSQP.
    """

    def __init__(self, horizon, dt, a_max, v_max, n_blues, w_ue=0.5, w_avoid=5.0,
                 include_heading=False, blue_nx=4, omega_max=2.0 * np.pi, w_omega=0.1):
        # blue_nx: stride of the incoming blue_state_plans, which may differ
        # in width from this (red) agent's own state -- only p_i=[0:2] is
        # ever read from it, so its own heading/omega (if any) is irrelevant.
        self.n_blues = int(n_blues)
        self.blue_nx = int(blue_nx)
        self.w_avoid = float(w_avoid)
        super().__init__(
            horizon=horizon, dt=dt, a_max=a_max, v_max=v_max, w_ue=w_ue,
            include_heading=include_heading, omega_max=omega_max, w_omega=w_omega,
        )

    def _make_opti(self):
        return ca.Opti()

    def _configure_solver(self, opti):
        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 500, "print_level": 0, "sb": "yes"}
        opti.solver("ipopt", p_opts, s_opts)

    def _declare_extra_params(self):
        self.blue_traj_param = self.opti.parameter(self.blue_nx * self.n_blues, self.N + 1)
        # 1.0 for a detected blue, 0.0 for an undetected one -- set fresh
        # every solve in _set_extra_values, so the avoidance term below only
        # ever reacts to blues red has actually seen.
        self.detected_weight_param = self.opti.parameter(self.n_blues)

    def _build_role_cost(self):
        J = 0
        for k in range(1, self.N + 1):
            p_e = self.X[0:2, k]
            for i in range(self.n_blues):
                p_i = self.blue_traj_param[i * self.blue_nx : i * self.blue_nx + 2, k]
                J -= self.detected_weight_param[i] * self.w_avoid * ca.sumsqr(p_e - p_i)
        return J

    def _set_extra_values(self, red_state, blue_state_plans, detected_blue_indices=None,
                           agent_id=None):
        del red_state, agent_id
        blue_state_plans = np.asarray(blue_state_plans, dtype=float).reshape(
            self.blue_nx * self.n_blues, self.N + 1
        )
        self.opti.set_value(self.blue_traj_param, blue_state_plans)
        if detected_blue_indices is None:
            weights = np.ones(self.n_blues)
        else:
            detected = set(detected_blue_indices)
            weights = np.array([1.0 if i in detected else 0.0 for i in range(self.n_blues)])
        self.opti.set_value(self.detected_weight_param, weights)

    def _before_solve(self, opti, red_state):
        x_guess = np.tile(red_state[:, None], (1, self.N + 1))
        opti.set_initial(self.X, x_guess)
        opti.set_initial(self.U, np.zeros((self.nu, self.N)))


class AttackController(BaseRedRoleController):
    """Homes in on the defended region.

    No blue-avoidance term in the nominal cost -- pursuer-pursuer/pursuer-evader
    safety coordination is handled downstream by the CBF filter, same as
    DefendController/NeutralizeController on the blue side. Pure convex
    quadratic cost, so (unlike EvadeController) this stays on the default
    conic/OSQP backend.

    defense_center is fixed once at construction (via _set_static_extras),
    not recomputed per solve -- unlike blue's DefendController, which
    recomputes its analogous target from live red state every solve.
    """

    def __init__(self, horizon, dt, a_max, v_max, defense_center=None, arena_size=100.0,
                 w_ue=0.5, w_progress=1.0, include_heading=False, omega_max=2.0 * np.pi,
                 w_omega=0.1):
        self.w_progress = float(w_progress)
        super().__init__(
            horizon=horizon, dt=dt, a_max=a_max, v_max=v_max, w_ue=w_ue,
            include_heading=include_heading, omega_max=omega_max, w_omega=w_omega,
            defense_center=defense_center, arena_size=arena_size,
        )

    def _declare_extra_params(self):
        pass

    def _set_static_extras(self, defense_center=None, arena_size=100.0, **kwargs):
        del kwargs
        self.defense_center = (
            np.array([0.5 * arena_size, 0.5 * arena_size], dtype=float)
            if defense_center is None
            else np.asarray(defense_center, dtype=float).reshape(2)
        )

    def _build_role_cost(self):
        J = 0
        for k in range(1, self.N + 1):
            J += self.w_progress * ca.sumsqr(self.X[0:2, k] - ca.DM(self.defense_center))
        return J

    def _set_extra_values(self, red_state, blue_state_plans, detected_blue_indices=None,
                           agent_id=None):
        # defense_center is fixed at construction (see _set_static_extras) --
        # nothing to refresh per solve. blue_state_plans is accepted for
        # interface parity with EvadeController but unused.
        del red_state, blue_state_plans, detected_blue_indices, agent_id


class RedReconController(BaseRedRoleController):
    """Holds current position, recomputed fresh every solve. Stub for a
    future scouting controller, with no sensing difference from any other
    role (see sensing.py) -- same placeholder caveat as blue's
    ReconController, but with no patrol-waypoint/sweep behavior of its own.
    """

    def __init__(self, horizon, dt, a_max, v_max, w_ue=0.5, w_hold=1.0,
                 include_heading=False, omega_max=2.0 * np.pi, w_omega=0.1):
        self.w_hold = float(w_hold)
        super().__init__(
            horizon=horizon, dt=dt, a_max=a_max, v_max=v_max, w_ue=w_ue,
            include_heading=include_heading, omega_max=omega_max, w_omega=w_omega,
        )

    def _declare_extra_params(self):
        self.target_param = self.opti.parameter(2)

    def _build_role_cost(self):
        J = 0
        for k in range(1, self.N + 1):
            J += self.w_hold * ca.sumsqr(self.X[0:2, k] - self.target_param)
        return J

    def _set_extra_values(self, red_state, blue_state_plans, detected_blue_indices=None,
                           agent_id=None):
        del blue_state_plans, detected_blue_indices, agent_id
        self.opti.set_value(self.target_param, red_state[0:2].copy())
