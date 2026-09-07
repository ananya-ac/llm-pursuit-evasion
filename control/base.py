"""Base classes for per-agent decentralized MPC role controllers, one per
team. Each builds its small single-agent QP once at construction (mirroring
control.joint_minimax's pattern) and re-solves it every timestep via
set_value/opti.solve() -- the CasADi graph is never rebuilt per call. Blue-
blue/blue-red safety coordination is handled downstream by control.cbf's
CBF-QP filter, not here; these controllers only produce each agent's nominal
(unfiltered) plan.
"""

import casadi as ca
import numpy as np


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



class BaseRedRoleController:
    """Single-red decentralized MPC controller for one tactical RedRole.

    Structurally parallel to BaseBlueRoleController, with three additions
    the blue base never needed:

    - A pluggable optimizer/solver backend via _make_opti()/_configure_solver
      hooks (default: conic Opti + OSQP, same as blue). EVADE's avoidance
      cost is a concave quadratic -- not conic/OSQP-safe -- so it overrides
      both to build a plain Opti() and use IPOPT instead; every other role
      (blue or red) keeps the default.
    - A split between per-solve extras (_declare_extra_params/
      _set_extra_values, e.g. EVADE's live blue-trajectory/detection-mask
      params, refreshed every plan() call) and static extras fixed once at
      construction (_set_static_extras, e.g. ATTACK's defense_center) --
      makes it explicit when a role has nothing to refresh per solve, rather
      than a silently-empty per-solve hook.
    - A _before_solve(opti, red_state) hook, called immediately before
      opti.solve(), default no-op. EVADE overrides it to warm-start X/U via
      set_initial -- warm-starting matters far more for its IPOPT backend
      than for the OSQP backend every other role uses, so it stays an
      EVADE-specific override rather than a base-level default.

    plan()'s signature is the union of what any red role needs
    (blue_state_plans, detected_blue_indices, agent_id) -- a role that
    doesn't use one of these simply ignores it in its own _set_extra_values,
    the same way blue's DefendController/NeutralizeController already ignore
    other_blue_states/directed_target in BaseBlueRoleController.plan().
    """

    def __init__(self, horizon, dt, a_max, v_max, w_ue=0.5, include_heading=False,
                 omega_max=2.0 * np.pi, w_omega=0.1, **static_extra_kwargs):
        self.N = int(horizon)
        self.dt = float(dt)
        self.a_max = float(a_max)
        self.v_max = float(v_max)
        self.w_ue = float(w_ue)
        self.include_heading = bool(include_heading)
        self.omega_max = float(omega_max)
        self.w_omega = float(w_omega)
        self.nx = 5 if self.include_heading else 4
        self.nu = 3 if self.include_heading else 2

        self.opti = self._make_opti()
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
        self._set_static_extras(**static_extra_kwargs)

        J = self.w_ue * sum(ca.sumsqr(self.U[0:2, k]) for k in range(self.N))
        if self.include_heading:
            J += self.w_omega * sum(ca.sumsqr(self.U[2, k]) for k in range(self.N))
        J += self._build_role_cost()
        self.opti.minimize(J)
        self._configure_solver(self.opti)

    def _make_opti(self):
        return ca.Opti("conic")

    def _configure_solver(self, opti):
        opti.solver("osqp", {"verbose": False})

    def _declare_extra_params(self):
        raise NotImplementedError

    def _set_static_extras(self, **kwargs):
        """Hook for values fixed once at construction (e.g. AttackController's
        defense_center). Default no-op, accepting and discarding any kwargs
        so callers don't need to special-case which role they're
        constructing -- most roles have nothing static to set here."""
        del kwargs

    def _build_role_cost(self):
        raise NotImplementedError

    def _set_extra_values(self, red_state, blue_state_plans, detected_blue_indices=None,
                           agent_id=None):
        raise NotImplementedError

    def _before_solve(self, opti, red_state):
        """Hook called immediately before opti.solve(). Default no-op -- see
        class docstring."""
        del opti, red_state

    def plan(self, red_state, blue_state_plans, detected_blue_indices=None, agent_id=None):
        """Returns (u0, X_plan, U_plan, solved_ok).

        blue_state_plans/detected_blue_indices/agent_id form the union of
        what any red role needs -- see class docstring for how roles that
        don't need one of these simply ignore it.
        """
        red_state = np.asarray(red_state, dtype=float).reshape(self.nx)

        self.opti.set_value(self.x0_param, red_state)
        self._set_extra_values(
            red_state, blue_state_plans, detected_blue_indices=detected_blue_indices,
            agent_id=agent_id,
        )
        self._before_solve(self.opti, red_state)

        try:
            sol = self.opti.solve()
        except RuntimeError:
            return None, None, None, False

        X_plan = sol.value(self.X)
        U_plan = sol.value(self.U)
        return U_plan[:, 0], X_plan, U_plan, True
