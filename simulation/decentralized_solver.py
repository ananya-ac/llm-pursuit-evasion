"""Per-step orchestration for the decentralized planning strategy: each blue
runs its own local controller for its currently assigned BlueRole, each red runs
its own local controller for its currently assigned RedRole, and the
resulting nominal plans are passed through control.cbf's CBF-QP safety
filter (inherited unchanged). Which controller *class* realizes a given role
is a planning-layer decision -- see planning.dispatch -- this module only
constructs those classes' instances for the current run's config and
dispatches to them each step.

Lives in simulation/, not control/: it has no cost function or Opti of its
own, just role->controller dispatch and CBF-filter invocation -- i.e. it's
the piece that runs the simulation bed's per-step control cycle, not a
controller itself.
"""

import numpy as np

from control.blue_controllers import DefendController, NeutralizeController, ReconController
from control.cbf import CBFFilterMixin
from control.red_controllers import AttackController, EvadeController, RedReconController
from planning.dispatch import BLUE_ROLE_CONTROLLER_CLASSES, RED_ROLE_CONTROLLER_CLASSES
from planning.roles import RedRole, BlueRole


class DecentralizedPursuitEvasionSolver(CBFFilterMixin):
    """Per-blue decentralized MPC: each blue runs its own local controller
    for its currently assigned BlueRole (DEFEND or NEUTRALIZE) instead of a joint
    QP across all blues. CBF safety filtering across blues/red is
    still applied centrally (inherited unchanged from CBFFilterMixin) --
    only the nominal per-agent planning is decentralized.
    """

    supports_one_step_cbf = True

    def __init__(
        self,
        blue_agent,
        red_agent,
        horizon,
        arena_size=100.0,
        defense_center=None,
        defense_polygon=None,
        defended_shape_points=None,
        enable_blue_blue_cbf=True,
        enable_blue_red_cbf=True,
        enable_convex_hull_containment=False,
        **kwargs,
    ):
        del defended_shape_points
        super().__init__(
            blue_agent=blue_agent,
            red_agent=red_agent,
            horizon=horizon,
            enable_blue_blue_cbf=enable_blue_blue_cbf,
            enable_blue_red_cbf=enable_blue_red_cbf,
            enable_convex_hull_containment=enable_convex_hull_containment,
            **kwargs,
        )
        self.defense_center = (
            None
            if defense_center is None
            else np.asarray(defense_center, dtype=float).reshape(2)
        )
        self.defense_polygon = (
            None
            if defense_polygon is None
            else np.asarray(defense_polygon, dtype=float).reshape(-1, 2)
        )

        # Threaded to every controller so each one's own state/control width
        # (and, when present, its yaw-rate bound) matches the Agent it steers
        # -- see CBFFilterMixin.__init__ for why CBF machinery is unaffected.
        blue_heading_kwargs = (
            {"include_heading": True, "omega_max": self.blue.omega_max}
            if self.blue.include_heading else {}
        )
        red_heading_kwargs = (
            {"include_heading": True, "omega_max": self.red.omega_max}
            if self.red.include_heading else {}
        )

        self.defend_controller = DefendController(
            horizon=self.N,
            dt=self.dt,
            a_max=self.a_max,
            v_max=self.v_max,
            defense_polygon=self.defense_polygon,
            arena_size=arena_size,
            **blue_heading_kwargs,
        )
        self.neutralize_controller = NeutralizeController(
            horizon=self.N,
            dt=self.dt,
            a_max=self.a_max,
            v_max=self.v_max,
            **blue_heading_kwargs,
        )
        self.recon_controller = ReconController(
            horizon=self.N,
            dt=self.dt,
            a_max=self.a_max,
            v_max=self.v_max,
            arena_size=arena_size,
            **blue_heading_kwargs,
        )
        self.evade_controller = EvadeController(
            horizon=self.N,
            dt=self.dt,
            a_max=self.red_a_max,
            v_max=self.red_v_max,
            n_blues=self.N_p,
            blue_nx=self.blue_nx,
            **red_heading_kwargs,
        )
        self.attack_controller = AttackController(
            horizon=self.N,
            dt=self.dt,
            a_max=self.red_a_max,
            v_max=self.red_v_max,
            defense_center=self.defense_center,
            arena_size=arena_size,
            **red_heading_kwargs,
        )
        self.red_recon_controller = RedReconController(
            horizon=self.N,
            dt=self.dt,
            a_max=self.red_a_max,
            v_max=self.red_v_max,
            **red_heading_kwargs,
        )

        # BlueRole -> instance dispatch, built from planning.dispatch's role ->
        # controller-*class* registry plus this run's already-constructed
        # instances -- keeps "which class realizes a role" a planning-layer
        # fact (single source of truth in planning/dispatch.py) while this
        # solver only owns instance construction/config.
        blue_instances_by_class = {
            DefendController: self.defend_controller,
            NeutralizeController: self.neutralize_controller,
            ReconController: self.recon_controller,
        }
        red_instances_by_class = {
            AttackController: self.attack_controller,
            RedReconController: self.red_recon_controller,
            EvadeController: self.evade_controller,
        }
        self._blue_role_controllers = {
            role: blue_instances_by_class[cls]
            for role, cls in BLUE_ROLE_CONTROLLER_CLASSES.items()
        }
        self._red_role_controllers = {
            role: red_instances_by_class[cls]
            for role, cls in RED_ROLE_CONTROLLER_CLASSES.items()
        }

    def solve_decentralized(
        self, x_current, red_states, role_assignment, red_role_assignment=None,
        target_override=None, recon_target_override=None, detected_blue_indices=None,
        bearing_target_override=None,
    ):
        """Dispatches each blue to its assigned role's controller, then
        dispatches each red to its assigned RedRole controller against the
        resulting blue plans. Returns (u_p_first, X_p_plans, u_e_first,
        X_e_plans), or all-None on any sub-solve failure (matching
        solve_minimax's failure-signaling convention).

        NEUTRALIZE blues are exempt from the blue-red CBF filter
        (CBFFilterMixin.one_step_cbf_filter), allowing them to make contact
        with their target.

        `target_override`: optional {blue_idx: 4-vector state} for NEUTRALIZE
        agents, e.g. from a vision planner's explicit target_contact_id
        (which may resolve to a red OR a decoy bird's state -- this method has
        no notion of "contact"/"bird", it just steers toward whatever state
        it's given). Falls back to the nearest-red heuristic for any
        NEUTRALIZE agent with no override entry (logged, since a
        planner that's supposed to always supply one -- e.g. LLMVisionRolePlanner
        -- silently missing an entry usually signals a bug upstream). DEFEND
        always uses its own nearest-red heuristic regardless of overrides,
        since guarding a perimeter point isn't a single-target action.

        `recon_target_override`: optional {blue_idx: 2-vector waypoint} for
        RECON agents, e.g. a coverage region's center chosen by a role planner
        (see perception.coverage.CoverageTracker / planning.planner.RuleBasedBlueRolePlanner).
        Unlike target_override this is a plain waypoint, not an opposing-team
        state -- ReconController has no nearest-red fallback, so a RECON
        agent with no entry here just keeps its existing random-patrol
        behavior.

        `bearing_target_override`: optional {blue_idx: 2-vector position} --
        a red currently detected by that specific blue's own sensor (see
        Simulation._resolve_bearing_targets), passed straight through to
        every role's controller.plan(bearing_target=...) regardless of role,
        since bearing-tracking is handled uniformly in
        BaseBlueRoleController, not per-role. A blue with no entry here
        detected nothing this step and keeps its role's own default heading
        behavior.
        """
        red_role_assignment = red_role_assignment or {}
        target_override = target_override or {}
        recon_target_override = recon_target_override or {}
        bearing_target_override = bearing_target_override or {}
        blue_nx, blue_nu = self.blue_nx, self.blue.nu_single
        red_nx, red_nu = self.red_nx, self.red.nu_single
        x_current = np.asarray(x_current, dtype=float).reshape(blue_nx * self.N_p)
        red_states = np.asarray(red_states, dtype=float).reshape(red_nx * self.N_red)

        u_p_first = np.zeros(blue_nu * self.N_p)
        X_p_plans = np.zeros((blue_nx * self.N_p, self.N + 1))

        for i in range(self.N_p):
            own_state = x_current[i * blue_nx : i * blue_nx + blue_nx]
            other_states = (
                np.concatenate(
                    [x_current[j * blue_nx : j * blue_nx + blue_nx] for j in range(self.N_p) if j != i]
                )
                if self.N_p > 1
                else np.zeros(0)
            )

            own_pos = own_state[0:2]
            nearest_red_idx = int(np.argmin(
                [np.linalg.norm(own_pos - red_states[j * red_nx : j * red_nx + 2]) for j in range(self.N_red)]
            ))
            nearest_red_state = red_states[nearest_red_idx * red_nx : nearest_red_idx * red_nx + red_nx]

            role = role_assignment.get(i, BlueRole.DEFEND)
            controller = self._blue_role_controllers[role]
            if role == BlueRole.NEUTRALIZE:
                if i in target_override:
                    target_state = target_override[i]
                else:
                    print(
                        f"[solve_decentralized] blue {i} has role {role.value} but no "
                        "target_override entry -- falling back to nearest-red heuristic."
                    )
                    target_state = nearest_red_state
            else:
                # DEFEND and RECON both use the nearest-red heuristic as
                # target_state -- DEFEND actually consumes it (guard point
                # nearest the threat), RECON ignores it (its cost only
                # depends on own_state/directed_target below), kept so
                # nearest_red_state stays ground-truth regardless of role.
                target_state = nearest_red_state
            u0, X_plan, _, ok = controller.plan(
                own_state, target_state, other_states, agent_id=i,
                directed_target=recon_target_override.get(i),
                bearing_target=bearing_target_override.get(i),
            )
            if not ok:
                return None, None, None, None

            u_p_first[i * blue_nu : i * blue_nu + blue_nu] = u0
            X_p_plans[i * blue_nx : i * blue_nx + blue_nx, :] = X_plan

        u_e_first = np.zeros(red_nu * self.N_red)
        X_e_plans = np.zeros((red_nx * self.N_red, self.N + 1))

        for j in range(self.N_red):
            red_state_j = red_states[j * red_nx : j * red_nx + red_nx]
            red_role = red_role_assignment.get(j, RedRole.EVADE)
            controller = self._red_role_controllers[red_role]
            u_e0, X_e_plan, _, ok_e = controller.plan(
                red_state_j, X_p_plans, detected_blue_indices=detected_blue_indices,
            )
            if not ok_e:
                return None, None, None, None

            u_e_first[j * red_nu : j * red_nu + red_nu] = u_e0
            X_e_plans[j * red_nx : j * red_nx + red_nx, :] = X_e_plan

        return u_p_first, X_p_plans, u_e_first, X_e_plans
