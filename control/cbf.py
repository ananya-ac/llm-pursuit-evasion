"""Shared discrete-time CBF-QP safety-filter machinery, independent of which
planning strategy (joint-minimax NLP or decentralized per-agent MPC) produced
the nominal control it filters. Works directly off ground-truth state.

Split out of solver.py's BaseMinimaxSolver: this mixin holds only the
CBF-QP filter (blue-blue, blue-red, convex-hull containment, and the red
one-step filter), plus the handful of shared, planning-strategy-agnostic
helpers (`step_blue_dynamics`, `set_fixed_adjacency_from_state`,
`reset_warm_starts`). The joint-minimax NLP boilerplate that used to live in
the same base class (abstract cost/solver-builder hooks, best-response
iteration) now lives in control/joint_minimax.py's JointMinimaxBase, which
inherits from this mixin instead of duplicating it.
"""

import casadi as ca
import numpy as np
import osqp
import scipy.sparse as sparse


class CBFFilterMixin:
    """Shared CBF-QP safety filter for both blue-blue/blue-red collision
    avoidance and the red one-step filter. See module docstring."""

    def __init__(
        self,
        blue_agent,
        red_agent,
        horizon,
        capture_radius,
        v_max=None,
        a_max=None,
        enable_blue_blue_cbf=True,
        enable_blue_red_cbf=True,
        enable_convex_hull_containment=False,
        D_safe_blue=3.0,
        D_safe_blue_red=3.0,
        D_safe_red=3.0,
        gamma_blue_cbf=1.,
        gamma_red_cbf=0.8,
        blue_blue_cbf_slack_weight=1e5,
        blue_red_cbf_slack_weight=1e5,
        convex_hull_slack_weight=1e5,
        red_cbf_slack_weight=1e5,
        cutoff_mode="geometric",
        **_ignored_kwargs,
    ):
        if abs(float(blue_agent.dt) - float(red_agent.dt)) > 1e-12:
            raise ValueError(
                "Minimax solvers expect blue and red to share the same dt."
            )

        self.blue = blue_agent
        self.red = red_agent
        self.dt = float(blue_agent.dt)
        self.N = int(horizon)
        self.N_p = int(blue_agent.count)
        self.N_red = int(red_agent.count)
        # Per-agent state width: 4 normally, 5 when the Agent carries an
        # independent bearing (Agent(include_heading=True)). The CBF-QP
        # machinery below never touches heading/omega (yaw has no bearing on
        # collision safety) -- only state slicing needs this; CBF decision
        # variables always stay a literal 2 (ax, ay) per agent regardless.
        self.blue_nx = self.blue.nx_single
        self.red_nx = self.red.nx_single

        self.v_max = float(blue_agent.v_max if v_max is None else v_max)
        self.a_max = float(blue_agent.a_max if a_max is None else a_max)
        self.r_cap = float(capture_radius)
        self.enable_blue_blue_cbf = bool(enable_blue_blue_cbf)
        self.enable_blue_red_cbf = bool(enable_blue_red_cbf)
        self.enable_convex_hull_containment = bool(enable_convex_hull_containment)
        self.D_safe_blue = float(D_safe_blue)
        self.D_safe_blue_red = float(D_safe_blue_red)
        self.D_safe_red = float(D_safe_red)
        self.gamma_blue_cbf = float(gamma_blue_cbf)
        self.gamma_red_cbf = float(gamma_red_cbf)
        self.blue_blue_cbf_slack_weight = float(blue_blue_cbf_slack_weight)
        self.blue_red_cbf_slack_weight = float(blue_red_cbf_slack_weight)
        self.convex_hull_slack_weight = float(convex_hull_slack_weight)
        self.red_cbf_slack_weight = float(red_cbf_slack_weight)
        self.red_v_max = float(red_agent.v_max)
        self.red_a_max = float(red_agent.a_max)
        self.prev_X_p = None
        self.prev_U_p = None
        self.prev_X_e = None
        self.prev_U_e = None
        self.fixed_hull_order = None

    def reset_warm_starts(self):
        self.prev_X_p = None
        self.prev_U_p = None
        self.prev_X_e = None
        self.prev_U_e = None

    def set_fixed_adjacency_from_state(self, x_state, red_state=None):
        """Freeze a polygon ordering around the red team's centroid at
        simulation start."""
        if self.N_p < 3:
            self.fixed_hull_order = None
            return

        x_state = np.asarray(x_state).reshape(self.blue_nx * self.N_p)
        if red_state is None:
            center = np.mean(
                [x_state[i * self.blue_nx : i * self.blue_nx + 2] for i in range(self.N_p)], axis=0
            )
        elif self.N_red == 0:
            # Defensive: Simulation guarantees N_red >= 1 whenever contact
            # substitution is enabled (a redraw forces >=1 real drone per
            # episode), but guard anyway rather than let an empty-array
            # .mean() silently produce nan.
            center = np.zeros(2)
        else:
            red_state = np.asarray(red_state).reshape(self.red_nx * self.N_red)
            red_positions = np.array(
                [red_state[j * self.red_nx : j * self.red_nx + 2] for j in range(self.N_red)]
            )
            center = red_positions.mean(axis=0)

        blue_positions = np.array(
            [x_state[i * self.blue_nx : i * self.blue_nx + 2] for i in range(self.N_p)]
        )
        angles = np.arctan2(
            blue_positions[:, 1] - center[1], blue_positions[:, 0] - center[0]
        )
        self.fixed_hull_order = list(np.argsort(angles))

    def step_blue_dynamics(self, x_current, u_apply):
        return self.blue.Ad.dot(x_current) + self.blue.Bd.dot(u_apply)

    def _build_shared_cost(self, X_p, U_p, X_e, U_e):
        del X_p, U_p, X_e, U_e
        raise NotImplementedError

    def _build_blue_solver(self):
        raise NotImplementedError

    def _build_red_solver(self):
        raise NotImplementedError

    def _prepare_shared_aux_data(self, X_p_guess, X_e_guess):
        del X_p_guess, X_e_guess
        return None

    def _set_blue_solver_params(
        self, x_current, X_e_guess, U_e_guess, aux_data
    ):
        del aux_data
        self.blue_opti.set_value(self.p_params["x0"], x_current)
        self.blue_opti.set_value(self.p_params["X_e"], X_e_guess)
        self.blue_opti.set_value(self.p_params["U_e"], U_e_guess)

    def _set_red_solver_params(
        self, red_state, X_p_guess, U_p_guess, aux_data
    ):
        del aux_data
        self.red_opti.set_value(self.e_params["x0"], red_state)
        self.red_opti.set_value(self.e_params["X_p"], X_p_guess)
        self.red_opti.set_value(self.e_params["U_p"], U_p_guess)

    def _build_initial_red_guess(self, red_state):
        X_e_guess = np.zeros((4, self.N + 1))
        for k in range(self.N + 1):
            X_e_guess[0, k] = red_state[0] + k * self.dt * red_state[2]
            X_e_guess[1, k] = red_state[1] + k * self.dt * red_state[3]
        X_e_guess[2:4, :] = np.array(red_state[2:4])[:, None]
        U_e_guess = np.zeros((2, self.N))
        return X_e_guess, U_e_guess

    def _build_initial_blue_guess(self, x_current):
        X_p_guess = np.tile(x_current[:, None], (1, self.N + 1))
        U_p_guess = np.zeros((2 * self.N_p, self.N))
        return X_p_guess, U_p_guess

    def _initialize_cbf_qp(self, u_des):
        u_des = np.asarray(u_des).flatten()
        slack_specs = []
        if self.enable_blue_red_cbf:
            slack_specs.append(("blue_red", self.blue_red_cbf_slack_weight))
        if self.enable_blue_blue_cbf:
            slack_specs.append(("blue_blue", self.blue_blue_cbf_slack_weight))
        if self.enable_convex_hull_containment:
            slack_specs.append(("convex_hull", self.convex_hull_slack_weight))

        # CBF decision variables are always exactly [ax, ay] per blue (a
        # literal 2*N_p), regardless of self.blue.nu -- yaw rate never
        # participates in collision-safety filtering, so a widened Agent
        # (nu_single=3, carrying omega) doesn't change this sizing at all.
        blue_nu_cbf = 2 * self.N_p
        slack_indices = {}
        n_dec = blue_nu_cbf + len(slack_specs)
        p_diag = np.ones(n_dec)
        next_idx = blue_nu_cbf
        for slack_name, slack_weight in slack_specs:
            slack_indices[slack_name] = next_idx
            p_diag[next_idx] = float(slack_weight)
            next_idx += 1
        P = sparse.diags(p_diag).tocsc()
        q = np.hstack([-u_des, np.zeros(n_dec - blue_nu_cbf)])
        return u_des, n_dec, slack_indices, P, q

    def _build_cbf_box_constraints(self, n_dec):
        a_box = sparse.eye(n_dec).tolil()
        num_slacks = n_dec - 2 * self.N_p
        l_box = np.hstack([np.tile([-self.a_max, -self.a_max], self.N_p), np.zeros(num_slacks)])
        u_box = np.hstack([np.tile([self.a_max, self.a_max], self.N_p), np.full(num_slacks, np.inf)])
        return a_box.tocsc(), l_box, u_box

    def _build_cbf_velocity_constraints(self, x_current, n_dec):
        blue_nu_cbf = 2 * self.N_p
        a_vel = sparse.lil_matrix((blue_nu_cbf, n_dec))
        l_vel = np.zeros(blue_nu_cbf)
        u_vel = np.zeros(blue_nu_cbf)
        for i in range(self.N_p):
            ctrl_idx = i * 2
            v_i = x_current[i * self.blue_nx + 2 : i * self.blue_nx + 4]
            a_vel[ctrl_idx, ctrl_idx] = self.dt
            a_vel[ctrl_idx + 1, ctrl_idx + 1] = self.dt
            l_vel[ctrl_idx : ctrl_idx + 2] = -self.v_max - v_i
            u_vel[ctrl_idx : ctrl_idx + 2] = self.v_max - v_i
        return a_vel.tocsc(), l_vel, u_vel

    def _manual_barrier_row_terms(self, dp, dv, a_max, d_safe, gamma):
        """Hand-derived primal barrier approximation used in one-step OSQP filters."""
        eps = 1e-5
        pd = np.asarray(dp, dtype=float).reshape(2)
        vd = np.asarray(dv, dtype=float).reshape(2)
        c = pd + vd * self.dt
        c_norm = max(np.linalg.norm(c), eps)
        pd_norm = max(np.linalg.norm(pd), eps)

        h_next = (
            np.dot(c, vd) / c_norm
            + np.sqrt(abs(a_max) * max(c_norm - d_safe, 0.0))
        )
        h_now = (
            np.dot(pd, vd) / pd_norm
            + np.sqrt(abs(a_max) * max(pd_norm - d_safe, 0.0))
        )
        h_const = h_next - (1.0 - gamma) * h_now
        h_u = c * self.dt / c_norm
        return float(h_const), np.asarray(h_u, dtype=float)

    def _build_blue_red_cbf_constraints(
        self, x_current, red_state, n_dec, slack_idx, exempt_blue_indices=None
    ):
        """Builds one row per (active blue, red) pair. `exempt_blue_indices`
        (e.g. NEUTRALIZE-role blues, which are allowed to make contact) are
        skipped entirely -- no blue-red avoidance constraint is emitted for
        them. Returns None if no rows apply (e.g. every blue is exempt)."""
        exempt_blue_indices = exempt_blue_indices or set()
        red_state = np.asarray(red_state).reshape(self.red_nx * self.N_red)
        active_blue_indices = [i for i in range(self.N_p) if i not in exempt_blue_indices]
        num_rows = len(active_blue_indices) * self.N_red
        if num_rows <= 0:
            return None

        a_pe = sparse.lil_matrix((num_rows, n_dec))
        l_pe = np.full(num_rows, -np.inf)
        u_pe = np.zeros(num_rows)

        row = 0
        for j in range(self.N_red):
            p_e = red_state[j * self.red_nx : j * self.red_nx + 2]
            v_e = red_state[j * self.red_nx + 2 : j * self.red_nx + 4]
            for i in active_blue_indices:
                ctrl_idx = i * 2
                p_i = x_current[i * self.blue_nx : i * self.blue_nx + 2]
                v_i = x_current[i * self.blue_nx + 2 : i * self.blue_nx + 4]

                dp = p_i - p_e
                dv = v_i - v_e
                h_const, h_u = self._manual_barrier_row_terms(
                    dp,
                    dv,
                    self.a_max,
                    self.D_safe_blue_red,
                    self.gamma_red_cbf,
                )

                a_pe[row, ctrl_idx : ctrl_idx + 2] = -h_u
                a_pe[row, slack_idx] = -1.0
                u_pe[row] = h_const
                row += 1

        return a_pe.tocsc(), l_pe, u_pe

    def _build_blue_blue_cbf_constraints(self, x_current, n_dec, slack_idx):
        num_pairs = self.N_p * (self.N_p - 1) // 2
        if num_pairs <= 0:
            return None

        a_cbf = sparse.lil_matrix((num_pairs, n_dec))
        l_cbf = np.full(num_pairs, -np.inf)
        u_cbf = np.zeros(num_pairs)

        pair_idx = 0
        for i in range(self.N_p):
            p_i = x_current[i * self.blue_nx : i * self.blue_nx + 2]
            v_i = x_current[i * self.blue_nx + 2 : i * self.blue_nx + 4]
            for j in range(i + 1, self.N_p):
                p_j = x_current[j * self.blue_nx : j * self.blue_nx + 2]
                v_j = x_current[j * self.blue_nx + 2 : j * self.blue_nx + 4]

                dp = p_i - p_j
                dv = v_i - v_j
                h_const, h_u = self._manual_barrier_row_terms(
                    dp,
                    dv,
                    self.a_max,
                    self.D_safe_blue,
                    self.gamma_blue_cbf,
                )

                a_cbf[pair_idx, i * 2 : i * 2 + 2] = -h_u
                a_cbf[pair_idx, j * 2 : j * 2 + 2] = h_u
                a_cbf[pair_idx, slack_idx] = -1.0
                u_cbf[pair_idx] = h_const
                pair_idx += 1

        return a_cbf.tocsc(), l_cbf, u_cbf

    def _build_convex_hull_containment_constraints(
        self, x_current, red_state, n_dec, hull_slack_idx
    ):
        if not self.enable_convex_hull_containment or self.N_p < 3:
            return None
        if not self.fixed_hull_order or len(self.fixed_hull_order) < 3:
            return None

        x_current = np.asarray(x_current).reshape(self.blue_nx * self.N_p)
        red_state = np.asarray(red_state).reshape(self.red_nx * self.N_red)
        red_positions = np.array(
            [red_state[j * self.red_nx : j * self.red_nx + 2] for j in range(self.N_red)]
        )
        p_e = red_positions.mean(axis=0)

        num_edges = len(self.fixed_hull_order)
        a_hull = sparse.lil_matrix((num_edges, n_dec))
        l_hull = np.full(num_edges, -np.inf)
        u_hull = np.zeros(num_edges)

        # Same as the other CBF blocks: the containment constraint's decision
        # variables are always [ax, ay] per blue (2*N_p), never omega.
        blue_nu_cbf = 2 * self.N_p
        u_sym = ca.SX.sym("u", blue_nu_cbf)
        gamma_hull = self.gamma_blue_cbf

        for edge_idx, i in enumerate(self.fixed_hull_order):
            j = self.fixed_hull_order[(edge_idx + 1) % num_edges]

            p_i = x_current[i * self.blue_nx : i * self.blue_nx + 2]
            v_i = x_current[i * self.blue_nx + 2 : i * self.blue_nx + 4]
            p_j = x_current[j * self.blue_nx : j * self.blue_nx + 2]
            v_j = x_current[j * self.blue_nx + 2 : j * self.blue_nx + 4]

            u_i = u_sym[i * 2 : i * 2 + 2]
            u_j = u_sym[j * 2 : j * 2 + 2]

            p_i_next = ca.DM(p_i + self.dt * v_i) + (self.dt ** 2) * u_i
            p_j_next = ca.DM(p_j + self.dt * v_j) + (self.dt ** 2) * u_j
            p_e_dm = ca.DM(p_e)

            edge_vec = p_j_next - p_i_next
            rel_vec = p_e_dm - p_i_next
            h_next_expr = edge_vec[0] * rel_vec[1] - edge_vec[1] * rel_vec[0]
            h_fun = ca.Function(
                f"h_edge_{edge_idx}",
                [u_sym],
                [h_next_expr, ca.jacobian(h_next_expr, u_sym)],
            )

            h_now_vec = p_j - p_i
            rel_now_vec = p_e - p_i
            h_now = h_now_vec[0] * rel_now_vec[1] - h_now_vec[1] * rel_now_vec[0]

            h_next_nom, grad_nom = h_fun(np.zeros(blue_nu_cbf))
            h_next_nom = float(h_next_nom)
            grad_nom = np.asarray(grad_nom).reshape(-1)

            h_const = h_next_nom - (1.0 - gamma_hull) * h_now
            a_hull[edge_idx, :blue_nu_cbf] = -grad_nom
            a_hull[edge_idx, hull_slack_idx] = -1.0
            u_hull[edge_idx] = h_const

        return a_hull.tocsc(), l_hull, u_hull

    def _solve_cbf_qp(self, P, q, a_rows, l_rows, u_rows):
        A = sparse.vstack(a_rows).tocsc()
        l = np.hstack(l_rows)
        u = np.hstack(u_rows)

        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, warm_start=True, verbose=False, adaptive_rho=True)
        return prob.solve()

    def _initialize_red_cbf_qp(self, u_des):
        # Same convention as the blue CBF-QP: decision variables are always
        # exactly [ax, ay] (+1 slack), never omega, regardless of
        # self.red.nu_single.
        u_des = np.asarray(u_des).flatten()
        n_dec = 2 + 1
        slack_idx = 2

        p_diag = np.ones(n_dec)
        p_diag[slack_idx] = self.red_cbf_slack_weight
        P = sparse.diags(p_diag).tocsc()
        q = np.hstack([-u_des, 0.0])
        return u_des, n_dec, slack_idx, P, q

    def one_step_red_cbf_filter(self, x_current, red_state, u_des):
        """One-step red safety filter applied after the nominal red solve."""

        x_current = np.asarray(x_current).reshape(self.blue_nx * self.N_p)
        red_state = np.asarray(red_state).flatten()
        u_des, n_dec, slack_idx, P, q = self._initialize_red_cbf_qp(u_des)

        a_rows = []
        l_rows = []
        u_rows = []

        a_box = sparse.eye(n_dec).tolil()
        l_box = np.array([-self.red_a_max, -self.red_a_max, 0.0])
        u_box = np.array([self.red_a_max, self.red_a_max, np.inf])
        a_rows.append(a_box.tocsc())
        l_rows.append(l_box)
        u_rows.append(u_box)

        a_vel = sparse.lil_matrix((2, n_dec))
        a_vel[0, 0] = self.dt
        a_vel[1, 1] = self.dt
        l_vel = np.array(
            [-self.red_v_max - red_state[2], -self.red_v_max - red_state[3]]
        )
        u_vel = np.array(
            [self.red_v_max - red_state[2], self.red_v_max - red_state[3]]
        )
        a_rows.append(a_vel.tocsc())
        l_rows.append(l_vel)
        u_rows.append(u_vel)

        a_cbf = sparse.lil_matrix((self.N_p, n_dec))
        l_cbf = np.full(self.N_p, -np.inf)
        u_cbf = np.zeros(self.N_p)

        p_e = red_state[0:2]
        v_e = red_state[2:4]
        for i in range(self.N_p):
            p_i = x_current[i * self.blue_nx : i * self.blue_nx + 2]
            v_i = x_current[i * self.blue_nx + 2 : i * self.blue_nx + 4]
            dp_now = p_e - p_i
            dv_now = v_e - v_i
            h_const, h_u = self._manual_barrier_row_terms(
                dp_now,
                dv_now,
                self.red_a_max,
                self.D_safe_red,
                self.gamma_red_cbf,
            )

            a_cbf[i, 0:2] = -h_u
            a_cbf[i, slack_idx] = -1.0
            u_cbf[i] = h_const

        a_rows.append(a_cbf.tocsc())
        l_rows.append(l_cbf)
        u_rows.append(u_cbf)

        res = self._solve_cbf_qp(P, q, a_rows, l_rows, u_rows)
        if res.info.status not in ("solved", "solved inaccurate") or res.x is None:
            return np.clip(u_des, -self.red_a_max, self.red_a_max), np.nan

        # Translational-only result (2-wide), regardless of self.red.nu_single.
        return res.x[:2], res.x[slack_idx]

    def one_step_cbf_filter(self, x_current, red_state, u_des, blue_red_cbf_exempt=None):
        """Outer blue safety filter with optional blue-blue and blue-red CBFs.

        `blue_red_cbf_exempt` is an optional set of blue indices to exclude
        from blue-red avoidance (e.g. NEUTRALIZE-role blues, which are meant
        to make contact rather than avoid it). Defaults to no exemptions,
        matching prior behavior.
        """

        if not (
            self.enable_blue_blue_cbf
            or self.enable_blue_red_cbf
            or self.enable_convex_hull_containment
        ):
            # Clip only [ax, ay] per agent -- if u_des also carries omega
            # (widened Agent), it has no business being clipped to a_max, so
            # pass it through unchanged.
            u_clip = np.asarray(u_des, dtype=float).copy()
            trans_width = 2 * self.N_p
            u_clip[:trans_width] = np.clip(u_clip[:trans_width], -self.a_max, self.a_max)
            return u_clip, 0.0

        red_state = np.asarray(red_state).reshape(self.red_nx * self.N_red)
        u_des, n_dec, slack_indices, P, q = self._initialize_cbf_qp(u_des)

        a_rows = []
        l_rows = []
        u_rows = []

        a_box, l_box, u_box = self._build_cbf_box_constraints(n_dec)
        a_rows.append(a_box.tocsc())
        l_rows.append(l_box)
        u_rows.append(u_box)

        a_vel, l_vel, u_vel = self._build_cbf_velocity_constraints(x_current, n_dec)
        a_rows.append(a_vel)
        l_rows.append(l_vel)
        u_rows.append(u_vel)

        if self.enable_blue_red_cbf:
            blue_red_block = self._build_blue_red_cbf_constraints(
                x_current, red_state, n_dec, slack_indices["blue_red"],
                exempt_blue_indices=blue_red_cbf_exempt,
            )
            if blue_red_block is not None:
                a_pe, l_pe, u_pe = blue_red_block
                a_rows.append(a_pe)
                l_rows.append(l_pe)
                u_rows.append(u_pe)

        if self.enable_blue_blue_cbf:
            blue_cbf_block = self._build_blue_blue_cbf_constraints(
                x_current, n_dec, slack_indices["blue_blue"]
            )
            if blue_cbf_block is not None:
                a_cbf, l_cbf, u_cbf = blue_cbf_block
                a_rows.append(a_cbf)
                l_rows.append(l_cbf)
                u_rows.append(u_cbf)

        if self.enable_convex_hull_containment:
            hull_block = self._build_convex_hull_containment_constraints(
                x_current, red_state, n_dec, slack_indices["convex_hull"]
            )
            if hull_block is not None:
                a_hull, l_hull, u_hull = hull_block
                a_rows.append(a_hull)
                l_rows.append(l_hull)
                u_rows.append(u_hull)

        res = self._solve_cbf_qp(P, q, a_rows, l_rows, u_rows)

        if res.info.status not in ("solved", "solved inaccurate") or res.x is None:
            return np.clip(u_des, -self.a_max, self.a_max), np.nan

        # Translational-only result (2*N_p-wide), regardless of self.blue.nu.
        slack_values = [res.x[idx] for idx in slack_indices.values()]
        if not slack_values:
            return res.x[: 2 * self.N_p], 0.0
        return res.x[: 2 * self.N_p], float(np.max(slack_values))

