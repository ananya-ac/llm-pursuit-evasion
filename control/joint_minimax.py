"""Joint-minimax NLP planning strategy: the older centralized approach where
blue and red each solve one shared-cost trajectory optimization problem
against the other team's most recent guess (best-response iteration), as
opposed to control/{blue,red}_controllers.py's decentralized per-agent MPC.

Kept as an alternate planning strategy behind the same CBF-QP safety filter
(control.cbf.CBFFilterMixin) used by the decentralized path.
"""

import casadi as ca
import numpy as np

from control.cbf import CBFFilterMixin


class JointMinimaxBase(CBFFilterMixin):
    """Shared boilerplate for joint-minimax NLP solvers: abstract cost/solver-
    builder hooks plus the best-response (Nash) iteration loop. Inherits the
    CBF-QP safety filter unchanged from CBFFilterMixin."""

    supports_one_step_cbf = False

    def _debug_pairwise_distances(self, x_current, red_state):
        x_current = np.asarray(x_current).reshape(4 * self.N_p)
        red_state = np.asarray(red_state).reshape(4)

        blue_positions = [
            x_current[i * 4 : i * 4 + 2] for i in range(self.N_p)
        ]
        red_position = red_state[0:2]

        min_pp = np.inf
        for i in range(self.N_p):
            for j in range(i + 1, self.N_p):
                dist = np.linalg.norm(blue_positions[i] - blue_positions[j])
                min_pp = min(min_pp, dist)

        min_pe = np.inf
        for i in range(self.N_p):
            dist = np.linalg.norm(blue_positions[i] - red_position)
            min_pe = min(min_pe, dist)

        if not np.isfinite(min_pp):
            min_pp = np.nan
        if not np.isfinite(min_pe):
            min_pe = np.nan

        print(f"  Min blue-blue distance: {min_pp:.4f}")
        print(f"  Min blue-red distance: {min_pe:.4f}")

    def _debug_red_cbf_margins(self, X_p_guess, X_e_guess):
        if X_p_guess is None or X_e_guess is None:
            print("  Red CBF margins unavailable: missing trajectory guess.")
            return

        min_h = np.inf
        for k in range(self.N + 1):
            p_e = X_e_guess[0:2, k]
            for i in range(self.N_p):
                p_i = X_p_guess[i * 4 : i * 4 + 2, k]
                h = np.dot(p_e - p_i, p_e - p_i) - self.D_safe_red ** 2
                min_h = min(min_h, h)

        print(f"  Min red CBF h value over guess: {min_h:.6f}")

    def solve_minimax_turn(self, x_current, red_state, iters=3):
        x_current = np.asarray(x_current).reshape(4 * self.N_p)
        red_state = np.asarray(red_state).reshape(4)

        if self.prev_X_e is not None:
            X_e_guess = np.roll(self.prev_X_e, -1, axis=1)
            X_e_guess[:, -1] = X_e_guess[:, -2]
            U_e_guess = np.roll(self.prev_U_e, -1, axis=1)
        else:
            X_e_guess, U_e_guess = self._build_initial_red_guess(red_state)

        if self.prev_X_p is not None:
            X_p_guess = self.prev_X_p
        else:
            X_p_guess, _ = self._build_initial_blue_guess(x_current)

        if self.prev_U_p is not None:
            U_p_guess = self.prev_U_p
        else:
            _, U_p_guess = self._build_initial_blue_guess(x_current)

        try:
            for iteration in range(iters):
                aux_data = self._prepare_shared_aux_data(X_p_guess, X_e_guess)

                self._set_blue_solver_params(
                    x_current=x_current,
                    X_e_guess=X_e_guess,
                    U_e_guess=U_e_guess,
                    aux_data=aux_data,
                )
                self.blue_opti.set_initial(self.p_vars["X"], X_p_guess)
                self.blue_opti.set_initial(self.p_vars["U"], U_p_guess)

                try:
                    sol_p = self.blue_opti.solve()
                except RuntimeError as exc:
                    print(f"Blue solve failed at best-response iteration {iteration}.")
                    print(f"  Exception: {exc}")
                    self._debug_pairwise_distances(x_current, red_state)
                    print(
                        "  Current blue guess first state:",
                        np.array2string(X_p_guess[:, 0], precision=3),
                    )
                    print(
                        "  Current red guess first state:",
                        np.array2string(X_e_guess[:, 0], precision=3),
                    )
                    try:
                        x_debug = self.blue_opti.debug.value(self.p_vars["X"])
                        u_debug = self.blue_opti.debug.value(self.p_vars["U"])
                        print(
                            "  Blue debug X first column:",
                            np.array2string(x_debug[:, 0], precision=3),
                        )
                        print(
                            "  Blue debug U first column:",
                            np.array2string(u_debug[:, 0], precision=3),
                        )
                    except Exception as debug_exc:
                        print(f"  Unable to read blue debug values: {debug_exc}")
                    raise
                X_p_guess = sol_p.value(self.p_vars["X"])
                U_p_guess = sol_p.value(self.p_vars["U"])

                self._set_red_solver_params(
                    red_state=red_state,
                    X_p_guess=X_p_guess,
                    U_p_guess=U_p_guess,
                    aux_data=aux_data,
                )
                self.red_opti.set_initial(self.e_vars["X"], X_e_guess)
                self.red_opti.set_initial(self.e_vars["U"], U_e_guess)

                try:
                    sol_e = self.red_opti.solve()
                except RuntimeError as exc:
                    print(f"Red solve failed at best-response iteration {iteration}.")
                    print(f"  Exception: {exc}")
                    self._debug_pairwise_distances(x_current, red_state)
                    self._debug_red_cbf_margins(X_p_guess, X_e_guess)
                    print(
                        "  Current red guess first state:",
                        np.array2string(X_e_guess[:, 0], precision=3),
                    )
                    print(
                        "  Current blue plan first state:",
                        np.array2string(X_p_guess[:, 0], precision=3),
                    )
                    try:
                        x_debug = self.red_opti.debug.value(self.e_vars["X"])
                        u_debug = self.red_opti.debug.value(self.e_vars["U"])
                        print(
                            "  Red debug X first column:",
                            np.array2string(x_debug[:, 0], precision=3),
                        )
                        print(
                            "  Red debug U first column:",
                            np.array2string(u_debug[:, 0], precision=3),
                        )
                    except Exception as debug_exc:
                        print(f"  Unable to read red debug values: {debug_exc}")
                    raise
                X_e_guess = sol_e.value(self.e_vars["X"])
                U_e_guess = sol_e.value(self.e_vars["U"])
        except RuntimeError as exc:
            print(f"Solver failed to converge in Nash Iteration: {exc}")
            return None, None, None, None

        self.prev_X_p, self.prev_U_p = X_p_guess, U_p_guess
        self.prev_X_e, self.prev_U_e = X_e_guess, U_e_guess

        return X_p_guess, U_p_guess, X_e_guess, U_e_guess

    def solve_minimax(self, x_current, red_state, num_best_response_iters=3):
        X_p_guess, U_p_guess, X_e_guess, U_e_guess = self.solve_minimax_turn(
            x_current=x_current,
            red_state=red_state,
            iters=num_best_response_iters,
        )
        if X_p_guess is None:
            return None, None, None, None

        return X_p_guess.T, U_p_guess.T, X_e_guess.T, U_e_guess.T




class PursuitEvasionMinimaxSolver(JointMinimaxBase):
    """Pure zero-sum shared-cost minimax solver for pursuit-evasion."""

    supports_one_step_cbf = True

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
        gamma_blue_cbf=1.0,
        gamma_red_cbf=0.8,
        blue_blue_cbf_slack_weight=1e5,
        blue_red_cbf_slack_weight=1e5,
        red_cbf_slack_weight=1e5,
        cutoff_mode="geometric",
        **kwargs,
    ):
        super().__init__(
            blue_agent=blue_agent,
            red_agent=red_agent,
            horizon=horizon,
            capture_radius=capture_radius,
            v_max=v_max,
            a_max=a_max,
            enable_blue_blue_cbf=enable_blue_blue_cbf,
            enable_blue_red_cbf=enable_blue_red_cbf,
            enable_convex_hull_containment=enable_convex_hull_containment,
            D_safe_blue=D_safe_blue,
            D_safe_blue_red=D_safe_blue_red,
            D_safe_red=D_safe_red,
            gamma_blue_cbf=gamma_blue_cbf,
            gamma_red_cbf=gamma_red_cbf,
            blue_blue_cbf_slack_weight=blue_blue_cbf_slack_weight,
            blue_red_cbf_slack_weight=blue_red_cbf_slack_weight,
            red_cbf_slack_weight=red_cbf_slack_weight,
            **kwargs,
        )
        self.cutoff_mode = str(cutoff_mode)
        if self.cutoff_mode not in ("linear", "geometric"):
            raise ValueError(
                f"Unsupported cutoff_mode '{self.cutoff_mode}'. "
                "Expected 'linear' or 'geometric'."
            )
        self.cutoff_eps = 1e-6
        self.blue_qp_solver = "osqp"

        # Shared zero-sum payoff weights.
        self.w_e = 5.0
        self.w_c = 10.0
        self.k_p = 1.0
        self.w_up = 0.5
        self.w_ue = 0.5

        self.blue_opti, self.p_vars, self.p_params = self._build_blue_solver()
        self.red_opti, self.e_vars, self.e_params = self._build_red_solver()

    def _projector_matrix_from_param(self, P_perp_param, agent_idx, k):
        """Read a frozen 2x2 transverse projector for one blue/stage."""
        base = 4 * agent_idx
        return ca.vertcat(
            ca.horzcat(P_perp_param[base + 0, k], P_perp_param[base + 1, k]),
            ca.horzcat(P_perp_param[base + 2, k], P_perp_param[base + 3, k]),
        )

    def _compute_frozen_projectors(self, X_p_bar, X_e_bar):
        """Freeze line-of-sight projectors from the previous trajectory guess."""
        X_p_bar = np.asarray(X_p_bar)
        X_e_bar = np.asarray(X_e_bar)
        P_perp = np.zeros((4 * self.N_p, self.N + 1))

        for k in range(self.N + 1):
            p_e = X_e_bar[0:2, k]
            for i in range(self.N_p):
                p_i = X_p_bar[i * 4 : i * 4 + 2, k]
                los = p_e - p_i
                los_norm = max(np.linalg.norm(los), self.cutoff_eps)
                r_hat = los / los_norm
                proj = np.eye(2) - np.outer(r_hat, r_hat)

                base = 4 * i
                P_perp[base + 0, k] = proj[0, 0]
                P_perp[base + 1, k] = proj[0, 1]
                P_perp[base + 2, k] = proj[1, 0]
                P_perp[base + 3, k] = proj[1, 1]

        return P_perp

    def _shared_cutoff_target(self, p_i, p_e, v_e):
        return v_e + self.k_p * (p_e - p_i)

    def _prepare_shared_aux_data(self, X_p_guess, X_e_guess):
        return self._compute_frozen_projectors(X_p_guess, X_e_guess)

    def _set_blue_solver_params(
        self, x_current, X_e_guess, U_e_guess, aux_data
    ):
        super()._set_blue_solver_params(x_current, X_e_guess, U_e_guess, aux_data)
        self.blue_opti.set_value(self.p_params["P_perp"], aux_data)

    def _set_red_solver_params(
        self, red_state, X_p_guess, U_p_guess, aux_data
    ):
        super()._set_red_solver_params(red_state, X_p_guess, U_p_guess, aux_data)
        self.red_opti.set_value(self.e_params["P_perp"], aux_data)

    def _build_shared_cost(self, X_p, U_p, X_e, U_e, P_perp_param=None):
        """Strictly shared minimax cost: blue minimizes, red maximizes."""
        J = 0
        for k in range(self.N):
            J += self.w_up * ca.sumsqr(U_p[:, k])
            J -= self.w_ue * ca.sumsqr(U_e[:, k])

        for k in range(1, self.N + 1):
            p_e = X_e[0:2, k]
            v_e = X_e[2:4, k]

            for i in range(self.N_p):
                p_i = X_p[i * 4 : i * 4 + 2, k]
                v_i = X_p[i * 4 + 2 : i * 4 + 4, k]

                J += self.w_e * ca.sumsqr(p_i - p_e)

                if self.cutoff_mode == "geometric":
                    if P_perp_param is None:
                        raise ValueError(
                            "Geometric cutoff mode requires frozen projector parameters."
                        )
                    P_perp = self._projector_matrix_from_param(P_perp_param, i, k)
                    v_error = ca.mtimes(P_perp, (v_i - v_e))
                    J += self.w_c * ca.sumsqr(v_error)
                else:
                    v_cutoff = self._shared_cutoff_target(p_i, p_e, v_e)
                    J += self.w_c * ca.sumsqr(v_i - v_cutoff)

        return J

    def _build_blue_solver(self):
        opti = ca.Opti("conic")

        X_p = opti.variable(4 * self.N_p, self.N + 1)
        U_p = opti.variable(2 * self.N_p, self.N)

        X_e_param = opti.parameter(4, self.N + 1)
        U_e_param = opti.parameter(2, self.N)
        P_perp_param = opti.parameter(4 * self.N_p, self.N + 1)
        x0_p_param = opti.parameter(4 * self.N_p)

        opti.subject_to(X_p[:, 0] == x0_p_param)

        for k in range(self.N):
            for i in range(self.N_p):
                idx_x = i * 4
                idx_u = i * 2

                px, py = X_p[idx_x, k], X_p[idx_x + 1, k]
                vx, vy = X_p[idx_x + 2, k], X_p[idx_x + 3, k]
                ax, ay = U_p[idx_u, k], U_p[idx_u + 1, k]

                vx_next = vx + self.dt * ax
                vy_next = vy + self.dt * ay
                px_next = px + self.dt * vx_next
                py_next = py + self.dt * vy_next

                opti.subject_to(
                    X_p[idx_x : idx_x + 4, k + 1]
                    == ca.vertcat(px_next, py_next, vx_next, vy_next)
                )
                opti.subject_to(opti.bounded(-self.a_max, ax, self.a_max))
                opti.subject_to(opti.bounded(-self.a_max, ay, self.a_max))
                opti.subject_to(opti.bounded(-self.v_max, vx_next, self.v_max))
                opti.subject_to(opti.bounded(-self.v_max, vy_next, self.v_max))

        J = self._build_shared_cost(X_p, U_p, X_e_param, U_e_param, P_perp_param)
        opti.minimize(J)

        qpsol_opts = {"verbose": False}
        opti.solver(self.blue_qp_solver, qpsol_opts)

        return (
            opti,
            {"X": X_p, "U": U_p},
            {
                "X_e": X_e_param,
                "U_e": U_e_param,
                "P_perp": P_perp_param,
                "x0": x0_p_param,
            },
        )

    def _build_red_solver(self):
        opti = ca.Opti()

        X_e = opti.variable(4, self.N + 1)
        U_e = opti.variable(2, self.N)

        X_p_param = opti.parameter(4 * self.N_p, self.N + 1)
        U_p_param = opti.parameter(2 * self.N_p, self.N)
        P_perp_param = opti.parameter(4 * self.N_p, self.N + 1)
        x0_e_param = opti.parameter(4)

        opti.subject_to(X_e[:, 0] == x0_e_param)

        for k in range(self.N):
            px, py = X_e[0, k], X_e[1, k]
            vx, vy = X_e[2, k], X_e[3, k]
            ax, ay = U_e[0, k], U_e[1, k]

            vx_next = vx + self.dt * ax
            vy_next = vy + self.dt * ay
            px_next = px + self.dt * vx_next
            py_next = py + self.dt * vy_next

            opti.subject_to(
                X_e[:, k + 1] == ca.vertcat(px_next, py_next, vx_next, vy_next)
            )
            opti.subject_to(opti.bounded(-self.red_a_max, ax, self.red_a_max))
            opti.subject_to(opti.bounded(-self.red_a_max, ay, self.red_a_max))
            opti.subject_to(opti.bounded(-self.red_v_max, vx_next, self.red_v_max))
            opti.subject_to(opti.bounded(-self.red_v_max, vy_next, self.red_v_max))

        J = self._build_shared_cost(X_p_param, U_p_param, X_e, U_e, P_perp_param)
        opti.minimize(-J)

        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 500, "print_level": 0, "sb": "yes"}
        opti.solver("ipopt", p_opts, s_opts)

        return (
            opti,
            {"X": X_e, "U": U_e},
            {
                "X_p": X_p_param,
                "U_p": U_p_param,
                "P_perp": P_perp_param,
                "x0": x0_e_param,
            },
        )

class PerimeterDefenseMinimaxSolver(JointMinimaxBase):
    """Zero-sum minimax solver for a simple perimeter-defense touchdown game."""

    supports_one_step_cbf = True

    def __init__(
        self,
        blue_agent,
        red_agent,
        horizon,
        capture_radius,
        defense_center=None,
        defense_polygon=None,
        defended_shape_points=None,
        enable_convex_hull_containment=False,
        **kwargs,
    ):
        self.defense_center = (
            None if defense_center is None else np.asarray(defense_center, dtype=float).reshape(2)
        )
        self.defense_polygon = (
            None
            if defense_polygon is None
            else np.asarray(defense_polygon, dtype=float).reshape(-1, 2)
        )
        self.defended_shape_points = (
            None
            if defended_shape_points is None
            else np.asarray(defended_shape_points, dtype=float).reshape(-1, 2)
        )

        # Shared zero-sum payoff weights for touchdown denial.
        self.w_progress = 5.0
        self.w_terminal_progress = 40.0
        self.w_line = 6.0
        self.w_terminal_line = 12.0
        self.w_block = 10.0
        self.w_terminal_block = 20.0

        super().__init__(
            blue_agent=blue_agent,
            red_agent=red_agent,
            horizon=horizon,
            capture_radius=capture_radius,
            cutoff_mode="linear",
            enable_convex_hull_containment=enable_convex_hull_containment,
            **kwargs,
        )
        self.w_up = 0.5
        self.w_ue = 0.5

        self.blue_opti, self.p_vars, self.p_params = self._build_blue_solver()
        self.red_opti, self.e_vars, self.e_params = self._build_red_solver()

    @staticmethod
    def _closest_point_on_segment(point, start, end):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            return start.copy()
        alpha = np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0)
        return start + alpha * segment

    def _polygon_perimeter_data(self):
        polygon = np.asarray(self.defense_polygon, dtype=float)
        closed_polygon = np.vstack([polygon, polygon[0]])
        edge_vectors = np.diff(closed_polygon, axis=0)
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        cumulative_lengths = np.concatenate([[0.0], np.cumsum(edge_lengths)])
        perimeter = float(cumulative_lengths[-1])
        return polygon, closed_polygon, edge_vectors, edge_lengths, cumulative_lengths, perimeter

    def _point_at_polygon_arclength(
        self, distance_along, polygon, edge_vectors, edge_lengths, cumulative_lengths, perimeter
    ):
        wrapped_distance = np.mod(distance_along, perimeter)
        edge_idx = np.searchsorted(cumulative_lengths[1:], wrapped_distance, side="right")
        edge_start = polygon[edge_idx]
        edge_length = max(edge_lengths[edge_idx], 1e-12)
        offset = wrapped_distance - cumulative_lengths[edge_idx]
        return edge_start + (offset / edge_length) * edge_vectors[edge_idx]

    def _nearest_points_on_curve(self, X_e_guess, curve_points):
        curve = np.asarray(curve_points, dtype=float)
        closed_curve = np.vstack([curve, curve[0]])
        nearest_points = np.zeros((2, self.N + 1), dtype=float)
        for k in range(self.N + 1):
            point = np.asarray(X_e_guess[0:2, k], dtype=float)
            best_point = curve[0]
            best_dist_sq = np.inf
            for edge_idx in range(len(curve)):
                candidate = self._closest_point_on_segment(
                    point,
                    closed_curve[edge_idx],
                    closed_curve[edge_idx + 1],
                )
                dist_sq = float(np.sum((candidate - point) ** 2))
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_point = candidate
            nearest_points[:, k] = best_point
        return nearest_points

    def _nearest_boundary_target_sets(self, X_e_guess):
        polygon = np.asarray(self.defense_polygon, dtype=float)
        (
            _,
            closed_polygon,
            edge_vectors,
            edge_lengths,
            cumulative_lengths,
            perimeter,
        ) = self._polygon_perimeter_data()
        boundary_targets = np.zeros((2 * self.N_p, self.N + 1), dtype=float)
        spacing = perimeter / max(16 * self.N_p, 1)
        centered_offsets = spacing * (
            np.arange(self.N_p, dtype=float) - 0.5 * (self.N_p - 1)
        )

        for k in range(self.N + 1):
            point = np.asarray(X_e_guess[0:2, k], dtype=float)
            best_point = polygon[0]
            best_dist_sq = np.inf
            best_distance_along = 0.0
            for edge_idx in range(len(polygon)):
                candidate = self._closest_point_on_segment(
                    point,
                    closed_polygon[edge_idx],
                    closed_polygon[edge_idx + 1],
                )
                dist_sq = float(np.sum((candidate - point) ** 2))
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_point = candidate
                    edge_offset = float(
                        np.linalg.norm(candidate - closed_polygon[edge_idx])
                    )
                    best_distance_along = cumulative_lengths[edge_idx] + edge_offset

            for i, offset in enumerate(centered_offsets):
                target = self._point_at_polygon_arclength(
                    best_distance_along + offset,
                    polygon,
                    edge_vectors,
                    edge_lengths,
                    cumulative_lengths,
                    perimeter,
                )
                boundary_targets[2 * i : 2 * i + 2, k] = target
        return boundary_targets

    def _prepare_shared_aux_data(self, X_p_guess, X_e_guess):
        del X_p_guess
        return {
            "boundary_targets": self._nearest_boundary_target_sets(X_e_guess),
            "progress_targets": self._nearest_points_on_curve(
                X_e_guess,
                self.defended_shape_points,
            ),
        }

    def _set_blue_solver_params(
        self, x_current, X_e_guess, U_e_guess, aux_data
    ):
        super()._set_blue_solver_params(x_current, X_e_guess, U_e_guess, aux_data)
        self.blue_opti.set_value(
            self.p_params["boundary_targets"],
            aux_data["boundary_targets"],
        )
        self.blue_opti.set_value(
            self.p_params["progress_targets"],
            aux_data["progress_targets"],
        )

    def _set_red_solver_params(
        self, red_state, X_p_guess, U_p_guess, aux_data
    ):
        super()._set_red_solver_params(red_state, X_p_guess, U_p_guess, aux_data)
        self.red_opti.set_value(
            self.e_params["boundary_targets"],
            aux_data["boundary_targets"],
        )
        self.red_opti.set_value(
            self.e_params["progress_targets"],
            aux_data["progress_targets"],
        )

    def _build_shared_cost(
        self,
        X_p,
        U_p,
        X_e,
        U_e,
        boundary_targets,
        progress_targets,
        P_perp_param=None,
    ):
        del P_perp_param
        J = 0

        for k in range(self.N):
            J += self.w_up * ca.sumsqr(U_p[:, k])
            J -= self.w_ue * ca.sumsqr(U_e[:, k])
            p_e = X_e[0:2, k]
            J -= self.w_progress * ca.sumsqr(p_e - progress_targets[:, k])

            for i in range(self.N_p):
                p_i = X_p[i * 4 : i * 4 + 2, k]
                target = boundary_targets[2 * i : 2 * i + 2, k]
                J += self.w_line * ca.sumsqr(p_i - target)

        p_e_terminal = X_e[0:2, self.N]
        J -= self.w_terminal_progress * ca.sumsqr(
            p_e_terminal - progress_targets[:, self.N]
        )
        for i in range(self.N_p):
            p_i_terminal = X_p[i * 4 : i * 4 + 2, self.N]
            target = boundary_targets[2 * i : 2 * i + 2, self.N]
            J += self.w_terminal_line * ca.sumsqr(p_i_terminal - target)

        return J

    def _build_blue_solver(self):
        opti = ca.Opti("conic")

        X_p = opti.variable(4 * self.N_p, self.N + 1)
        U_p = opti.variable(2 * self.N_p, self.N)

        X_e_param = opti.parameter(4, self.N + 1)
        U_e_param = opti.parameter(2, self.N)
        boundary_targets_param = opti.parameter(2 * self.N_p, self.N + 1)
        progress_targets_param = opti.parameter(2, self.N + 1)
        x0_p_param = opti.parameter(4 * self.N_p)

        opti.subject_to(X_p[:, 0] == x0_p_param)

        for k in range(self.N):
            for i in range(self.N_p):
                idx_x = i * 4
                idx_u = i * 2

                px, py = X_p[idx_x, k], X_p[idx_x + 1, k]
                vx, vy = X_p[idx_x + 2, k], X_p[idx_x + 3, k]
                ax, ay = U_p[idx_u, k], U_p[idx_u + 1, k]

                vx_next = vx + self.dt * ax
                vy_next = vy + self.dt * ay
                px_next = px + self.dt * vx_next
                py_next = py + self.dt * vy_next

                opti.subject_to(
                    X_p[idx_x : idx_x + 4, k + 1]
                    == ca.vertcat(px_next, py_next, vx_next, vy_next)
                )
                opti.subject_to(opti.bounded(-self.a_max, ax, self.a_max))
                opti.subject_to(opti.bounded(-self.a_max, ay, self.a_max))
                opti.subject_to(opti.bounded(-self.v_max, vx_next, self.v_max))
                opti.subject_to(opti.bounded(-self.v_max, vy_next, self.v_max))

        J = self._build_shared_cost(
            X_p,
            U_p,
            X_e_param,
            U_e_param,
            boundary_targets_param,
            progress_targets_param,
        )
        opti.minimize(J)

        qpsol_opts = {"verbose": False}
        opti.solver("osqp", qpsol_opts)

        return (
            opti,
            {"X": X_p, "U": U_p},
            {
                "X_e": X_e_param,
                "U_e": U_e_param,
                "boundary_targets": boundary_targets_param,
                "progress_targets": progress_targets_param,
                "x0": x0_p_param,
            },
        )

    def _build_red_solver(self):
        opti = ca.Opti()

        X_e = opti.variable(4, self.N + 1)
        U_e = opti.variable(2, self.N)

        X_p_param = opti.parameter(4 * self.N_p, self.N + 1)
        U_p_param = opti.parameter(2 * self.N_p, self.N)
        boundary_targets_param = opti.parameter(2 * self.N_p, self.N + 1)
        progress_targets_param = opti.parameter(2, self.N + 1)
        x0_e_param = opti.parameter(4)

        opti.subject_to(X_e[:, 0] == x0_e_param)

        for k in range(self.N):
            px, py = X_e[0, k], X_e[1, k]
            vx, vy = X_e[2, k], X_e[3, k]
            ax, ay = U_e[0, k], U_e[1, k]

            vx_next = vx + self.dt * ax
            vy_next = vy + self.dt * ay
            px_next = px + self.dt * vx_next
            py_next = py + self.dt * vy_next

            opti.subject_to(
                X_e[:, k + 1] == ca.vertcat(px_next, py_next, vx_next, vy_next)
            )
            opti.subject_to(opti.bounded(-self.a_max, ax, self.a_max))
            opti.subject_to(opti.bounded(-self.a_max, ay, self.a_max))
            opti.subject_to(opti.bounded(-self.v_max, vx_next, self.v_max))
            opti.subject_to(opti.bounded(-self.v_max, vy_next, self.v_max))

        J = self._build_shared_cost(
            X_p_param,
            U_p_param,
            X_e,
            U_e,
            boundary_targets_param,
            progress_targets_param,
        )
        opti.minimize(-J)

        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 500, "print_level": 0, "sb": "yes"}
        opti.solver("ipopt", p_opts, s_opts)

        return (
            opti,
            {"X": X_e, "U": U_e},
            {
                "X_p": X_p_param,
                "U_p": U_p_param,
                "boundary_targets": boundary_targets_param,
                "progress_targets": progress_targets_param,
                "x0": x0_e_param,
            },
        )


