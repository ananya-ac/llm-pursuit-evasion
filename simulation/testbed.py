import glob
import inspect
import os
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from agents.dynamics import Agent
from simulation.decentralized_solver import DecentralizedPursuitEvasionSolver
from planning.planner import (
    LLMRedRolePlanner,
    LLMRolePlanner,
    PlannerObservation,
    RuleBasedRedRolePlanner,
    RuleBasedBlueRolePlanner,
)
from planning.roles import RedRole, BlueRole
import perception.sensing as sensing
from perception.coverage import CoverageTracker

_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)
_IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "JPG", "JPEG", "PNG")


def _glob_images(subdir, prefix):
    """All prefix_*.<ext> files under assets/<subdir>/, any of
    _IMAGE_EXTENSIONS -- real sourced photos are as valid a contact image as
    the PIL-drawn placeholders; planner._encode_image_data_uri already
    handles both png and jpg/jpeg for MIME typing."""
    paths = []
    for ext in _IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(_ASSETS_DIR, subdir, f"{prefix}_*.{ext}")))
    return sorted(paths)


_RED_DRONE_IMAGES = _glob_images("drones", "red")
_BIRD_IMAGES = _glob_images("birds", "bird")
_CIVILIAN_IMAGES = _glob_images("civilian", "civilian")


class Simulation:
    def __init__(
        self,
        num_blues=4,
        num_reds=1,
        dt=0.1,
        horizon=60,
        arena_size=100.0,
        D_safe_blue=4.1e-1,
        D_safe_blue_red=3.0,
        D_safe_red=3.0,
        solver_mode="decentralized",
        sim_steps=600,
        seed=130,
        blue_v_max=2.0,
        blue_a_max=2.0,
        red_v_max=2.0,
        red_a_max=2.0,
        red_random_policy=False,
        enable_blue_blue_cbf=False,
        enable_blue_red_cbf=False,
        enable_red_cbf=False,
        enable_convex_hull_containment=False,
        blue_blue_cbf_slack_weight=1e7,
        blue_red_cbf_slack_weight=1e-1,
        red_cbf_slack_weight=1e7,
        convex_hull_slack_weight=1e7,
        contact_tolerance=4e-1,
        planning_interval_seconds=10.0,
        role_planner=None,
        neutralize_fraction=0.5,
        red_role_planner=None,
        neutralize_points=1.0,
        escape_points=1.0,
        num_bird_slots=0,
        bird_spawn_prob=0.2,
        bird_v_max=1.5,
        bird_a_max=1.0,
        num_contact_slots=0,
        p_bird=0.0,
        p_civilian=0.0,
        civilian_v_max=1.5,
        civilian_a_max=1.0,
        fov_half_angle_deg=sensing.DEFAULT_FOV_HALF_ANGLE_DEG,
        fov_range=sensing.DEFAULT_FOV_RANGE,
        fov_bearing_speed_threshold=sensing.DEFAULT_BEARING_SPEED_THRESHOLD,
        omega_max=2.0 * np.pi,
        coverage_grid_dim=4,
    ):
        if solver_mode != "decentralized":
            raise ValueError(
                f"Unsupported solver_mode '{solver_mode}'. Only 'decentralized' is "
                "supported."
            )
        if int(num_contact_slots) > 0:
            if int(num_reds) != 1:
                raise ValueError(
                    "num_reds and num_contact_slots are mutually exclusive -- "
                    "num_contact_slots > 0 derives the real-red count itself from "
                    "the per-slot draw, so passing a non-default num_reds alongside "
                    "it is ambiguous. Leave num_reds at its default."
                )
            if not (0.0 <= p_bird):
                raise ValueError(f"p_bird must be >= 0.0, got {p_bird}")
            if not (0.0 <= p_civilian):
                raise ValueError(f"p_civilian must be >= 0.0, got {p_civilian}")
            if p_bird + p_civilian > 1.0:
                raise ValueError(
                    f"p_bird + p_civilian must be <= 1.0, got {p_bird + p_civilian}"
                )

        self.seed = seed
        np.random.seed(self.seed)

        self.num_blues = num_blues
        self.num_reds = int(num_reds)
        self.num_contact_slots = int(num_contact_slots)
        self.p_bird = float(p_bird)
        self.p_civilian = float(p_civilian)
        self.contact_substitution_enabled = self.num_contact_slots > 0
        self.slot_kinds = []
        self.num_civilians = 0
        self.num_substitution_birds = 0
        if self.contact_substitution_enabled:
            # Local RNG (never np.random), offset from _build_contact_roster's
            # own offset (+4242) so the two draws never collide -- this keeps
            # the whole substitution mechanic a structural no-op with respect
            # to every other seeded draw in the episode (spawn positions,
            # velocities, contact-roster shuffling) whether or not it's used.
            slot_rng = random.Random(self.seed + 9137)

            def _draw_slot_kinds():
                kinds = []
                for _ in range(self.num_contact_slots):
                    r = slot_rng.random()
                    if r < self.p_bird:
                        kinds.append("bird")
                    elif r < self.p_bird + self.p_civilian:
                        kinds.append("civilian")
                    else:
                        kinds.append("drone")
                return kinds

            slot_kinds = _draw_slot_kinds()
            # Guarantee at least one real drone per episode -- an all-decoy
            # episode would otherwise resolve as an instant, degenerate
            # "defenders won" before the simulation even starts (num_reds==0
            # makes num_neutralized+num_escaped>=num_reds trivially true at
            # step 0). This conditions the realized distribution on >=1
            # success rather than the raw Binomial(num_contact_slots,
            # 1-p_bird-p_civilian) -- an explicit, describable adjustment,
            # not a silent bias.
            while "drone" not in slot_kinds:
                slot_kinds = _draw_slot_kinds()
            self.slot_kinds = slot_kinds
            self.num_reds = slot_kinds.count("drone")
            self.num_substitution_birds = slot_kinds.count("bird")
            self.num_civilians = slot_kinds.count("civilian")
        self.dt = dt
        self.horizon = horizon
        self.arena_size = float(arena_size)
        self.arena_min = 0.0
        self.arena_max = self.arena_size
        self.D_safe_blue = D_safe_blue
        self.D_safe_blue_red = D_safe_blue_red
        self.D_safe_red = D_safe_red
        self.solver_mode = solver_mode
        self.sim_steps = sim_steps
        self.red_random_policy = red_random_policy
        self.enable_blue_blue_cbf = enable_blue_blue_cbf
        self.enable_blue_red_cbf = enable_blue_red_cbf
        self.enable_red_cbf = enable_red_cbf
        self.enable_convex_hull_containment = enable_convex_hull_containment
        self.blue_blue_cbf_slack_weight = float(blue_blue_cbf_slack_weight)
        self.blue_red_cbf_slack_weight = float(blue_red_cbf_slack_weight)
        self.red_cbf_slack_weight = float(red_cbf_slack_weight)
        self.convex_hull_slack_weight = convex_hull_slack_weight
        self.contact_tolerance = float(contact_tolerance)
        self.fov_half_angle_rad = np.deg2rad(float(fov_half_angle_deg))
        self.fov_range = float(fov_range)
        self.fov_bearing_speed_threshold = float(fov_bearing_speed_threshold)
        self.omega_max = float(omega_max)
        self.coverage_tracker = CoverageTracker(self.arena_size, grid_dim=coverage_grid_dim)
        self.defended_shape_points = None
        self.defense_polygon = None
        self.defense_center = None
        self.defender_spawn_points = None
        self.planning_interval_seconds = float(planning_interval_seconds)
        self.planning_interval_steps = max(
            1, round(self.planning_interval_seconds / self.dt)
        )
        self.role_planner = (
            role_planner
            if role_planner is not None
            else RuleBasedBlueRolePlanner(
                neutralize_fraction=neutralize_fraction,
            )
        )
        self.role_assignment = {}
        self.red_role_planner = (
            red_role_planner
            if red_role_planner is not None
            else RuleBasedRedRolePlanner()
        )
        self.red_role_assignment = {}
        self.role_assignment_history = []
        self.red_role_assignment_history = []
        self.neutralize_points = float(neutralize_points)
        self.escape_points = float(escape_points)
        self.blue_disabled = [False] * self.num_blues
        self.blue_disabled_history = []

        self.num_bird_slots = int(num_bird_slots)
        self.bird_spawn_prob = float(bird_spawn_prob)
        self.bird_v_max = float(bird_v_max)
        self.bird_a_max = float(bird_a_max)
        self.num_birds = 0
        self.bird_state = np.zeros(0)
        self.history_birds = []
        self.civilian_v_max = float(civilian_v_max)
        self.civilian_a_max = float(civilian_a_max)
        self.civilian_state = np.zeros(0)
        self.history_civilians = []
        self.contact_roster = []  # list of ("red"|"bird"|"civilian", index), position = contact_id
        self.contact_images = {}  # contact_id -> image path
        self.num_bird_engagements = 0
        self.num_civilian_engagements = 0

        self.blue_v_max = float(blue_v_max)
        self.blue_a_max = float(blue_a_max)
        self.red_v_max = float(red_v_max)
        self.red_a_max = float(red_a_max)

        heading_kwargs = {"include_heading": True, "omega_max": self.omega_max}

        self.blue = Agent(
            agent_type=Agent.BLUE,
            dt=self.dt,
            count=self.num_blues,
            v_max=self.blue_v_max,
            a_max=self.blue_a_max,
            **heading_kwargs,
        )
        self.red = Agent(
            agent_type=Agent.RED,
            dt=self.dt,
            count=self.num_reds,
            v_max=self.red_v_max,
            a_max=self.red_a_max,
            use_random_policy=self.red_random_policy,
            **heading_kwargs,
        )
        self.blue_nx = self.blue.nx_single
        self.red_nx = self.red.nx_single
        self._build_perimeter_geometry()
        self.solver = self._build_solver()
        self.solver_supports_cbf = bool(
            getattr(self.solver, "supports_one_step_cbf", False)
        )

        self.x_current = None
        self.red_state = None
        self.history_blues = []
        self.history_red = []
        self.history_cbf_slack = []
        self.neutralized = False
        self.neutralize_step = None
        self.neutralize_agent = None
        self.touchdown = False
        self.touchdown_step = None
        self.touchdown_red_index = None
        self.escaped = False
        self.escape_step = None
        self.collided = False
        self.collision_step = None
        self.collision_agents = None
        self.collision_kind = None
        self.defenders_won = False
        self.defenders_win_step = None
        self.defenders_win_agents = None
        self.red_outcomes = ["active"] * self.num_reds
        self.num_neutralized = 0
        self.num_escaped = 0
        self.blue_score = 0.0
        self.solver_failed = False
        self.solver_failure_step = None

    def _build_solver(self):
        solver_cls = self._select_solver_class()
        candidate_kwargs = {
            "blue_agent": self.blue,
            "red_agent": self.red,
            "horizon": self.horizon,
            "arena_size": self.arena_size,
            "D_safe_blue": self.D_safe_blue,
            "D_safe_blue_red": self.D_safe_blue_red,
            "D_safe_red": self.D_safe_red,
            "enable_blue_blue_cbf": self.enable_blue_blue_cbf,
            "enable_blue_red_cbf": self.enable_blue_red_cbf,
            "enable_convex_hull_containment": self.enable_convex_hull_containment,
            "blue_blue_cbf_slack_weight": self.blue_blue_cbf_slack_weight,
            "blue_red_cbf_slack_weight": self.blue_red_cbf_slack_weight,
            "red_cbf_slack_weight": self.red_cbf_slack_weight,
            "convex_hull_slack_weight": self.convex_hull_slack_weight,
            "defense_center": self.defense_center,
            "defense_polygon": self.defense_polygon,
            "defended_shape_points": self.defended_shape_points,
            "fov_range": self.fov_range,
        }
        solver_signature = inspect.signature(solver_cls.__init__)
        supported_kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in solver_signature.parameters
        }
        return solver_cls(**supported_kwargs)

    def _select_solver_class(self):
        return DecentralizedPursuitEvasionSolver

    def _build_perimeter_geometry(self):
        center = np.array([0.82 * self.arena_size, 0.82 * self.arena_size], dtype=float)
        angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        base_radius = 0.08 * self.arena_size
        radial_scale = 1.0 + 0.18 * np.cos(3.0 * angles) + 0.08 * np.sin(5.0 * angles)
        radii = base_radius * radial_scale
        shape = center + np.column_stack(
            [radii * np.cos(angles), radii * np.sin(angles)]
        )
        polygon_stride = max(1, len(shape) // 12)
        polygon = center + 1.15 * (shape[::polygon_stride] - center)
        self.defended_shape_points = shape
        self.defense_polygon = polygon
        self.defense_center = center
        self.defender_spawn_points = self._sample_polygon_perimeter(
            polygon,
            self.num_blues,
        )

    @staticmethod
    def _sample_polygon_perimeter(polygon, n_points):
        vertices = np.asarray(polygon, dtype=float)
        if n_points <= 0:
            return np.zeros((0, 2), dtype=float)
        closed_vertices = np.vstack([vertices, vertices[0]])
        edge_vectors = np.diff(closed_vertices, axis=0)
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        perimeter = float(np.sum(edge_lengths))
        distances = np.linspace(0.0, perimeter, n_points, endpoint=False)
        anchors = []
        cumulative = np.concatenate([[0.0], np.cumsum(edge_lengths)])
        for distance_along in distances:
            edge_idx = np.searchsorted(cumulative[1:], distance_along, side="right")
            edge_start = vertices[edge_idx]
            edge_vector = edge_vectors[edge_idx]
            edge_length = max(edge_lengths[edge_idx], 1e-9)
            offset = distance_along - cumulative[edge_idx]
            anchors.append(edge_start + (offset / edge_length) * edge_vector)
        return np.asarray(anchors, dtype=float)

    @staticmethod
    def _point_in_polygon(point, polygon):
        x, y = point
        vertices = np.asarray(polygon, dtype=float)
        inside = False
        x_prev, y_prev = vertices[-1]
        for x_curr, y_curr in vertices:
            intersects = ((y_curr > y) != (y_prev > y)) and (
                x < (x_prev - x_curr) * (y - y_curr) / max(y_prev - y_curr, 1e-9) + x_curr
            )
            if intersects:
                inside = not inside
            x_prev, y_prev = x_curr, y_curr
        return inside

    def setup_agents(self):
        arena_margin = 0.5

        min_red_spawn_dist = 4.0
        # Reflecting defense_center through the arena's center puts red's
        # spawn box diametrically opposite the defended region, rather
        # than near the arena's middle -- same half-width (0.10*arena_size)
        # as the old fixed [0.45, 0.65] window, just recentered.
        red_spawn_center = self.arena_size - self.defense_center
        red_spawn_half_width = 0.10 * self.arena_size
        red_positions = []
        for _ in range(self.num_reds):
            red_pos = None
            while red_pos is None or any(
                np.linalg.norm(red_pos - prev) < min_red_spawn_dist
                for prev in red_positions
            ):
                red_pos = np.array(
                    [
                        np.random.uniform(
                            red_spawn_center[0] - red_spawn_half_width,
                            red_spawn_center[0] + red_spawn_half_width,
                        ),
                        np.random.uniform(
                            red_spawn_center[1] - red_spawn_half_width,
                            red_spawn_center[1] + red_spawn_half_width,
                        ),
                    ],
                    dtype=float,
                )
            red_positions.append(red_pos)

        red_state_list = []
        for red_pos in red_positions:
            angle_e = np.random.rand() * 2 * np.pi
            vx_e = self.red.v_max * np.cos(angle_e)
            vy_e = self.red.v_max * np.sin(angle_e)
            row = [red_pos[0], red_pos[1], vx_e, vy_e]
            if self.red.include_heading:
                # Spawn heading always faces the target directly,
                # independent of the (separately randomized) initial
                # velocity direction above.
                to_target = self.defense_center - red_pos
                theta_0 = (
                    float(np.arctan2(to_target[1], to_target[0]))
                    if np.linalg.norm(to_target) > 1e-9
                    else 0.0
                )
                row.append(theta_0)
            red_state_list.append(row)
        self.red_state = np.array(red_state_list, dtype=float).reshape(-1)

        x0_list = []
        for spawn_point in self.defender_spawn_points:
            outward_dir = np.asarray(spawn_point, dtype=float) - self.defense_center
            outward_norm = float(np.linalg.norm(outward_dir))
            if outward_norm > 1e-9:
                outward_dir = outward_dir / outward_norm
            else:
                outward_dir = np.array([1.0, 0.0], dtype=float)
            spawn_base = np.asarray(spawn_point, dtype=float) + 8.0 * outward_dir
            px = np.clip(
                spawn_base[0] + np.random.uniform(-0.75, 0.75),
                arena_margin,
                self.arena_max - arena_margin,
            )
            py = np.clip(
                spawn_base[1] + np.random.uniform(-0.75, 0.75),
                arena_margin,
                self.arena_max - arena_margin,
            )
            row = [px, py, 0.0, 0.0]
            if self.blue.include_heading:
                # Face the reflection of defense_center through the
                # arena's center -- the same point red's spawn box is
                # centered on -- so blue starts out looking toward where
                # the threat actually spawns, not just radially outward
                # from its own guard post.
                row.append(
                    sensing.compute_bearing(
                        [px, py], [0.0, 0.0], self.arena_size - self.defense_center, True,
                        self.fov_bearing_speed_threshold,
                    )
                )
            x0_list.append(row)
        self.x_current = np.asarray(x0_list, dtype=float).reshape(-1)
        self._spawn_birds()
        self._spawn_civilians()
        self._build_contact_roster()

    def _random_inert_spawn_state(self, v_max):
        """One [px, py, vx, vy] draw for a harmless inert entity (bird or
        civilian drone): uniform position in the inner 80% of the arena,
        random heading, speed uniform in [0.3, 1.0] * v_max. Uses np.random
        (the global RNG), matching the pre-existing bird-spawn draw exactly."""
        pos = np.random.uniform(0.1 * self.arena_size, 0.9 * self.arena_size, size=2)
        angle = np.random.rand() * 2 * np.pi
        speed = np.random.uniform(0.3, 1.0) * v_max
        vel = speed * np.array([np.cos(angle), np.sin(angle)])
        return [pos[0], pos[1], vel[0], vel[1]]

    def _spawn_birds(self):
        """Spawns 0..num_bird_slots harmless decoy agents (independent Bernoulli
        draw per slot -- the original additive mechanic, unchanged) plus
        self.num_substitution_birds more (from the three-way per-slot draw in
        __init__, zero unless num_contact_slots was set). Both are motion-model
        identical, so they're merged into one bird_state array/kind label.
        Birds are tactically inert -- pure random-walk motion, never
        neutralized/escaped/counted toward any win condition."""
        bird_states = []
        for _ in range(self.num_bird_slots):
            if np.random.rand() >= self.bird_spawn_prob:
                continue
            bird_states.append(self._random_inert_spawn_state(self.bird_v_max))
        for _ in range(self.num_substitution_birds):
            bird_states.append(self._random_inert_spawn_state(self.bird_v_max))
        self.num_birds = len(bird_states)
        self.bird_state = (
            np.array(bird_states, dtype=float).reshape(-1)
            if bird_states
            else np.zeros(0)
        )

    def _spawn_civilians(self):
        """Spawns self.num_civilians harmless civilian-drone decoys (from
        the three-way per-slot draw in __init__, zero unless num_contact_slots
        was set) -- same inert motion model as birds, but a deliberately
        harder decoy kind visually (a civilian drone is still a drone, unlike
        a bird's plainly different silhouette): the two are meant to be
        distinguishable in the same image pool by something other than
        shape -- see assets/civilian/'s real sourced photos."""
        civilian_states = [
            self._random_inert_spawn_state(self.civilian_v_max)
            for _ in range(self.num_civilians)
        ]
        self.civilian_state = (
            np.array(civilian_states, dtype=float).reshape(-1)
            if civilian_states
            else np.zeros(0)
        )

    @staticmethod
    def _step_random_walk(state, count, v_max, a_max, dt, arena_min, arena_max):
        """Bounded random-walk update shared by birds and civilians: small
        random acceleration, velocity/position integration, and reflection
        off the arena boundary."""
        state = state.copy()
        for b in range(count):
            idx = b * 4
            px, py, vx, vy = state[idx : idx + 4]
            ax, ay = np.clip(
                np.random.normal(0.0, 0.5 * a_max, size=2), -a_max, a_max
            )
            vx = float(np.clip(vx + ax * dt, -v_max, v_max))
            vy = float(np.clip(vy + ay * dt, -v_max, v_max))
            px, py = px + vx * dt, py + vy * dt
            if px < arena_min or px > arena_max:
                vx = -vx
                px = np.clip(px, arena_min, arena_max)
            if py < arena_min or py > arena_max:
                vy = -vy
                py = np.clip(py, arena_min, arena_max)
            state[idx : idx + 4] = [px, py, vx, vy]
        return state

    def _step_birds(self):
        self.bird_state = self._step_random_walk(
            self.bird_state, self.num_birds, self.bird_v_max, self.bird_a_max,
            self.dt, self.arena_min, self.arena_max,
        )

    def _step_civilians(self):
        self.civilian_state = self._step_random_walk(
            self.civilian_state, self.num_civilians, self.civilian_v_max,
            self.civilian_a_max, self.dt, self.arena_min, self.arena_max,
        )

    def _build_contact_roster(self):
        """Builds the fixed, per-episode (kind, index) -> contact_id mapping
        used by vision planners: reds, birds, and civilian drones shuffled
        together so a contact's id carries no positional hint about its true
        identity, plus a stable image assignment per contact.

        Uses a local random.Random instance (seeded from self.seed, offset to
        avoid correlating with anything else), never numpy's global RNG, so
        this is a strict no-op with respect to every other seeded draw in the
        episode -- Experiment I's spawn/motion sequence is unaffected whether
        or not this method runs.
        """
        local_rng = random.Random(self.seed + 4242)

        roster = (
            [("red", j) for j in range(self.num_reds)]
            + [("bird", b) for b in range(self.num_birds)]
            + [("civilian", c) for c in range(self.num_civilians)]
        )
        local_rng.shuffle(roster)
        self.contact_roster = roster

        image_pools = {"bird": _BIRD_IMAGES, "civilian": _CIVILIAN_IMAGES}
        self.contact_images = {}
        for contact_id, (kind, _idx) in enumerate(roster):
            pool = image_pools.get(kind, _RED_DRONE_IMAGES)
            if pool:
                self.contact_images[contact_id] = local_rng.choice(pool)

    def _active_contact_states(self):
        """contact_id -> current 4-state (position+velocity only -- Experiment
        II's vision planner never needs/expects bearing), excluding reds that
        have already been resolved (neutralized/escaped) -- birds/civilians are
        always shown, since they never resolve. NOT detection-filtered --
        callers that build a PlannerObservation must additionally filter this
        down to detected_contact_indices (see _compute_detected_indices)."""
        states = {}
        for contact_id, (kind, idx) in enumerate(self.contact_roster):
            if kind == "red":
                if self.red_outcomes[idx] != "active":
                    continue
                states[contact_id] = self.red_state[idx * self.red_nx : idx * self.red_nx + 4]
            elif kind == "civilian":
                states[contact_id] = self.civilian_state[idx * 4 : idx * 4 + 4]
            else:
                states[contact_id] = self.bird_state[idx * 4 : idx * 4 + 4]
        return states

    def _active_contact_images(self, active_contact_ids):
        return {
            contact_id: path
            for contact_id, path in self.contact_images.items()
            if contact_id in active_contact_ids
        }

    def _resolve_target_override(self, target_contact_ids):
        """Maps a role planner's {blue_id: contact_id} (e.g. LLMVisionRolePlanner
        .last_targets) to {blue_id: 4-state} for solve_decentralized, and
        counts how many resolve to a bird or civilian drone (a wasted
        NEUTRALIZE commitment -- neither is ever in red_outcomes, so
        engaging one can never earn neutralize credit)."""
        target_override = {}
        bird_engagements_this_cycle = 0
        civilian_engagements_this_cycle = 0
        for blue_id, contact_id in target_contact_ids.items():
            if contact_id is None or not (0 <= contact_id < len(self.contact_roster)):
                continue
            kind, idx = self.contact_roster[contact_id]
            if kind == "red":
                target_override[blue_id] = self.red_state[idx * self.red_nx : idx * self.red_nx + 4]
            elif kind == "civilian":
                target_override[blue_id] = self.civilian_state[idx * 4 : idx * 4 + 4]
                civilian_engagements_this_cycle += 1
            else:
                target_override[blue_id] = self.bird_state[idx * 4 : idx * 4 + 4]
                bird_engagements_this_cycle += 1
        self.num_bird_engagements += bird_engagements_this_cycle
        self.num_civilian_engagements += civilian_engagements_this_cycle
        return target_override

    def _compute_bearings(self, states, count, nx):
        """Direct read of each agent's persisted bearing (state[...+4]) --
        theta is now genuine state, evolved by omega via the agent's own
        dynamics (Agent.include_heading), not recomputed from velocity each
        cycle. Every controller currently regularizes omega toward 0, so
        theta stays at its spawn-time value until a real recon controller
        starts spending omega -- see sensing.compute_bearing for that
        spawn-time convention."""
        return {idx: float(states[idx * nx + 4]) for idx in range(count)}

    def _compute_detected_indices(self, sim_step):
        """Returns (detected_red_indices, detected_blue_indices,
        detected_contact_indices): which opposing agents (and, for the
        vision-augmented Experiment II contact roster, which contact_ids)
        each team's combined field of view can see this cycle. Disabled
        blues and non-active reds are excluded from being both observers
        (their sensors are presumably down/gone) and detection targets
        (already-resolved reds aren't real threats). Own-team visibility is
        unaffected by this -- it's handled entirely in
        planner._format_observation, not here.

        detected_contact_indices reuses the same blue_observers/blue_bearings
        against _active_contact_states()'s positions (reds+birds+civilians,
        keyed by contact_id) instead of red_targets -- the same FOV primitive,
        just fed a different target set. This is always computed (cheap, a
        handful of contacts at most); it's simply unused/empty whenever
        contact_roster is red-only (Experiment I, or Experiment II with no
        detection filtering applied by the caller).

        Also updates self.coverage_tracker from the same blue observer/
        bearing data, since "is this cell within some blue's FOV" is the
        same primitive as "is this opposing agent within some blue's FOV" --
        just tested against grid cell centers instead of red positions."""
        active_blues = [i for i in range(self.num_blues) if not self.blue_disabled[i]]
        active_reds = [j for j in range(self.num_reds) if self.red_outcomes[j] == "active"]

        blue_bearings = self._compute_bearings(self.x_current, self.num_blues, self.blue_nx)
        red_bearings = self._compute_bearings(self.red_state, self.num_reds, self.red_nx)

        blue_observers = [(i, self.x_current[i * self.blue_nx : i * self.blue_nx + 2]) for i in active_blues]
        red_targets = [(j, self.red_state[j * self.red_nx : j * self.red_nx + 2]) for j in active_reds]
        detected_red_indices = sensing.team_detected_indices(
            blue_observers, blue_bearings, red_targets, self.fov_half_angle_rad, self.fov_range,
        )

        red_observers = [(j, self.red_state[j * self.red_nx : j * self.red_nx + 2]) for j in active_reds]
        blue_targets = [(i, self.x_current[i * self.blue_nx : i * self.blue_nx + 2]) for i in active_blues]
        detected_blue_indices = sensing.team_detected_indices(
            red_observers, red_bearings, blue_targets, self.fov_half_angle_rad, self.fov_range,
        )

        self.coverage_tracker.update(
            sim_step, blue_observers, blue_bearings, self.fov_half_angle_rad, self.fov_range,
        )

        contact_targets = [
            (contact_id, state[0:2]) for contact_id, state in self._active_contact_states().items()
        ]
        detected_contact_indices = sensing.team_detected_indices(
            blue_observers, blue_bearings, contact_targets, self.fov_half_angle_rad, self.fov_range,
        )

        return detected_red_indices, detected_blue_indices, detected_contact_indices

    def _blue_blue_collision_status(self):
        """Blue-blue hard-contact check only. Blue-red contact is handled
        separately by _resolve_active_reds, since for NEUTRALIZE-role
        contact with red is a resolution event, not a failure -- unlike
        blue-blue contact, which is always a genuine loss.
        """
        for i in range(self.num_blues):
            p_i = self.x_current[i * self.blue_nx : i * self.blue_nx + 2]
            for j in range(i + 1, self.num_blues):
                p_j = self.x_current[j * self.blue_nx : j * self.blue_nx + 2]
                dist_pp = float(np.linalg.norm(p_i - p_j))
                if dist_pp <= self.contact_tolerance:
                    return True, (i, j), dist_pp
        return False, None, None

    def _resolve_red_neutralized(self, step, red_idx, blue_idx, distance):
        self.red_outcomes[red_idx] = "neutralized"
        self.num_neutralized += 1
        self.blue_disabled[blue_idx] = True
        self.blue_score += self.neutralize_points
        self.neutralized = True
        if self.neutralize_step is None:
            self.neutralize_step = step
            self.neutralize_agent = blue_idx
        print(
            f"Red {red_idx + 1} neutralized (contact) at step {step} by blue {blue_idx + 1} "
            f"(distance={distance:.3f}). Blue score += {self.neutralize_points} "
            f"(total={self.blue_score:.1f}) -- now permanently disabled: {blue_idx + 1}."
        )

    def _resolve_active_reds(self, step):
        """Per-red resolution: a single (non-disabled) blue making hard
        contact resolves a red as neutralized immediately -- that blue is
        permanently disabled afterward and the kill is worth
        neutralize_points. Leaving the arena resolves a red as escaped;
        entering the defended region is an immediate breach that ends the
        whole episode. Returns True if a breach occurred this call (caller
        should stop the simulation).
        """
        for j in range(self.num_reds):
            if self.red_outcomes[j] != "active":
                continue
            p_e = self.red_state[j * self.red_nx : j * self.red_nx + 2]

            active_blues = [i for i in range(self.num_blues) if not self.blue_disabled[i]]
            if active_blues:
                blue_distances = {
                    i: float(np.linalg.norm(self.x_current[i * self.blue_nx : i * self.blue_nx + 2] - p_e))
                    for i in active_blues
                }
                closest_agent = min(blue_distances, key=blue_distances.get)
                closest_dist = blue_distances[closest_agent]

                if closest_dist <= self.contact_tolerance:
                    self._resolve_red_neutralized(step, j, closest_agent, closest_dist)
                    continue

            if self._point_in_polygon(p_e, self.defended_shape_points):
                self.touchdown = True
                self.touchdown_step = step
                self.touchdown_red_index = j
                print(
                    f"Perimeter breached at step {step} by red {j + 1} at "
                    f"({p_e[0]:.3f}, {p_e[1]:.3f}). Blue loses."
                )
                return True

            red_x, red_y = float(p_e[0]), float(p_e[1])
            if not (
                self.arena_min <= red_x <= self.arena_max
                and self.arena_min <= red_y <= self.arena_max
            ):
                self.red_outcomes[j] = "escaped"
                self.num_escaped += 1
                self.blue_score += self.escape_points
                self.escaped = True
                if self.escape_step is None:
                    self.escape_step = step
                print(
                    f"Red {j + 1} escaped the arena at step {step} "
                    f"(position=({red_x:.3f}, {red_y:.3f})). Blue score += "
                    f"{self.escape_points} (total={self.blue_score:.1f})."
                )

        return False

    def run(self):
        self.setup_agents()
        self.solver.reset_warm_starts()
        self.solver.set_fixed_adjacency_from_state(self.x_current, self.red_state)
        self.history_blues = [self.x_current.copy()]
        self.history_red = [self.red_state.copy()]
        self.history_birds = [self.bird_state.copy()]
        self.history_civilians = [self.civilian_state.copy()]
        self.history_cbf_slack = []
        self.role_assignment = {}
        self.red_role_assignment = {}
        self.role_assignment_history = [dict(self.role_assignment)]
        self.red_role_assignment_history = [dict(self.red_role_assignment)]
        self.neutralized = False
        self.neutralize_step = None
        self.neutralize_agent = None
        self.touchdown = False
        self.touchdown_step = None
        self.touchdown_red_index = None
        self.escaped = False
        self.escape_step = None
        self.collided = False
        self.collision_step = None
        self.collision_agents = None
        self.collision_kind = None
        self.defenders_won = False
        self.defenders_win_step = None
        self.defenders_win_agents = None
        self.red_outcomes = ["active"] * self.num_reds
        self.num_neutralized = 0
        self.num_escaped = 0
        self.blue_score = 0.0
        self.blue_disabled = [False] * self.num_blues
        self.blue_disabled_history = [list(self.blue_disabled)]
        self.solver_failed = False
        self.solver_failure_step = None
        self.num_bird_engagements = 0
        self.num_civilian_engagements = 0
        self.target_override = {}
        self.recon_target_override = {}

        print("Running decentralized simulation...")
        for step in range(self.sim_steps):
            blue_collided, blue_collision_agents, blue_collision_dist = (
                self._blue_blue_collision_status()
            )
            if blue_collided:
                self.collided = True
                self.collision_step = step
                self.collision_agents = blue_collision_agents
                self.collision_kind = "blue_blue"
                print(
                    f"Swarm lost before solve at step {step} "
                    f"(blue-blue collision, agents={blue_collision_agents}, "
                    f"distance={blue_collision_dist:.3f})."
                )
                break
            if self._resolve_active_reds(step):
                break
            if self.num_neutralized + self.num_escaped >= self.num_reds:
                self.defenders_won = True
                self.defenders_win_step = step
                print(
                    f"All reds resolved before solve at step {step}: "
                    f"neutralized={self.num_neutralized}, escaped={self.num_escaped}, "
                    f"score={self.blue_score:.1f}. Blue wins."
                )
                break

            # Computed fresh every step (cheap) since EvadeController's
            # reactive avoidance below should use the freshest possible
            # sensing, not a stale once-per-planning-cycle snapshot --
            # mirroring how CBF safety filtering is also always
            # every-step/full-ground-truth, distinct from the slower
            # strategic replanning cadence.
            detected_red_indices, detected_blue_indices, detected_contact_indices = (
                self._compute_detected_indices(step)
            )
            if step % self.planning_interval_steps == 0:
                # Detection-gated: a contact (and its image) only enters
                # the observation once some blue's FOV has actually
                # detected it -- Experiment II's vision planner must not
                # see the full, unfiltered contact roster any more than
                # Experiment I's text planner sees every red unconditionally.
                visible_contact_states = {
                    cid: s
                    for cid, s in self._active_contact_states().items()
                    if cid in detected_contact_indices
                }
                obs = PlannerObservation(
                    x_current=self.x_current,
                    red_states=self.red_state,
                    num_blues=self.num_blues,
                    num_reds=self.num_reds,
                    sim_step=step,
                    arena_size=self.arena_size,
                    defense_center=self.defense_center,
                    disabled_blues=[i for i in range(self.num_blues) if self.blue_disabled[i]],
                    detected_red_indices=sorted(detected_red_indices),
                    detected_blue_indices=sorted(detected_blue_indices),
                    detected_contact_ids=sorted(visible_contact_states.keys()),
                    coverage_regions=self.coverage_tracker.summarize(step),
                    contact_states=visible_contact_states,
                    contact_image_paths=self._active_contact_images(visible_contact_states.keys()),
                )
                self.role_assignment = self.role_planner.plan(obs)
                self.red_role_assignment = self.red_role_planner.plan(obs)
                # Two independent target-assignment conventions can coexist:
                # last_targets (Experiment II, contact_id space -- reds+birds
                # merged/shuffled) and last_target_red_ids (Experiment I, raw
                # red index space -- Exp I never obfuscates red identity, only
                # gates visibility via FOV). A given role_planner only ever
                # populates one of the two, so there's no collision.
                target_contact_ids = getattr(self.role_planner, "last_targets", {})
                target_red_ids = getattr(self.role_planner, "last_target_red_ids", {})
                self.target_override = self._resolve_target_override(target_contact_ids)
                for blue_id, red_id in target_red_ids.items():
                    if (
                        red_id is not None
                        and 0 <= red_id < self.num_reds
                        and self.red_outcomes[red_id] == "active"
                    ):
                        self.target_override[blue_id] = self.red_state[
                            red_id * self.red_nx : red_id * self.red_nx + 4
                        ]
                # RECON targeting: separate from target_override above
                # (which feeds NEUTRALIZE's intercept controller with a
                # 4-vector red/contact state) -- ReconController only ever
                # wants a 2D waypoint, resolved from a region id via
                # self.coverage_tracker rather than any agent's state.
                target_region_ids = getattr(self.role_planner, "last_recon_target_regions", {})
                self.recon_target_override = {
                    blue_id: self.coverage_tracker.region_center(region_id)
                    for blue_id, region_id in target_region_ids.items()
                }
                roles_log = {i: r.value for i, r in self.role_assignment.items()}
                red_roles_log = {j: r.value for j, r in self.red_role_assignment.items()}
                print(
                    f"[planner] step {step}: roles = {roles_log}, "
                    f"red_roles = {red_roles_log}, targets = {target_contact_ids}, "
                    f"target_red_ids = {target_red_ids}, "
                    f"recon_target_regions = {target_region_ids}"
                )

            u_p_first, X_p_plans, u_e_first, _ = self.solver.solve_decentralized(
                x_current=self.x_current,
                red_states=self.red_state,
                role_assignment=self.role_assignment,
                red_role_assignment=self.red_role_assignment,
                target_override=self.target_override,
                recon_target_override=self.recon_target_override,
                detected_blue_indices=detected_blue_indices,
            )
            solve_failed = u_p_first is None
            x_p_next_nominal = None if solve_failed else X_p_plans[:, 1]

            if solve_failed:
                self.solver_failed = True
                self.solver_failure_step = step
                print(f"Minimax solve failed at step {step}")
                break

            x_current_before_step = self.x_current.copy()
            if self.solver_supports_cbf and (
                self.enable_blue_blue_cbf
                or self.enable_blue_red_cbf
                or self.enable_convex_hull_containment
            ):
                blue_red_cbf_exempt = {
                    i for i, r in self.role_assignment.items() if r == BlueRole.NEUTRALIZE
                }
                if self.blue.include_heading:
                    # The CBF filter only ever adjusts [ax, ay] -- yaw rate
                    # has no bearing on collision safety, so it's extracted
                    # here, passed through unfiltered, and stitched back in
                    # below rather than threaded through the CBF-QP at all.
                    blue_nu = self.blue.nu_single
                    u_trans_des = np.concatenate(
                        [u_p_first[i * blue_nu : i * blue_nu + 2] for i in range(self.num_blues)]
                    )
                else:
                    u_trans_des = u_p_first
                u_safe, eps_used = self.solver.one_step_cbf_filter(
                    x_current=self.x_current,
                    red_state=self.red_state,
                    u_des=u_trans_des,
                    blue_red_cbf_exempt=blue_red_cbf_exempt,
                )
                if self.blue.include_heading:
                    u_apply = u_p_first.copy()
                    for i in range(self.num_blues):
                        u_apply[i * blue_nu : i * blue_nu + 2] = u_safe[i * 2 : i * 2 + 2]
                        # u_apply[i*blue_nu+2] (omega) stays the nominal
                        # MPC's own value -- unfiltered pass-through.
                else:
                    u_apply = u_safe
                self.x_current = self.solver.step_blue_dynamics(self.x_current, u_apply)
                self.history_cbf_slack.append(eps_used)
            else:
                self.x_current = x_p_next_nominal
            for i in range(self.num_blues):
                if self.blue_disabled[i]:
                    self.x_current[i * self.blue_nx : i * self.blue_nx + self.blue_nx] = (
                        x_current_before_step[i * self.blue_nx : i * self.blue_nx + self.blue_nx]
                    )
            self.history_blues.append(self.x_current.copy())
            self.role_assignment_history.append(dict(self.role_assignment))
            self.blue_disabled_history.append(list(self.blue_disabled))

            red_nu = self.red.nu_single
            for j in range(self.num_reds):
                if self.red_outcomes[j] != "active":
                    continue
                u_e_j_full = u_e_first[j * red_nu : j * red_nu + red_nu]
                u_e_j_trans = u_e_j_full[0:2]
                red_state_j = self.red_state[j * self.red_nx : j * self.red_nx + self.red_nx]
                if self.solver_supports_cbf and self.enable_red_cbf:
                    u_e_safe_trans, _ = self.solver.one_step_red_cbf_filter(
                        x_current=self.x_current,
                        red_state=red_state_j,
                        u_des=u_e_j_trans,
                    )
                else:
                    u_e_safe_trans = u_e_j_trans
                if self.red.include_heading:
                    # Omega (u_e_j_full[2]) bypasses the CBF filter
                    # entirely, same convention as blue.
                    u_e_safe = np.concatenate([u_e_safe_trans, u_e_j_full[2:3]])
                else:
                    u_e_safe = u_e_safe_trans
                self.red_state[j * self.red_nx : j * self.red_nx + self.red_nx] = self.red.step_single(
                    red_state_j, u_e_safe
                )
            self.history_red.append(self.red_state.copy())
            self.red_role_assignment_history.append(dict(self.red_role_assignment))

            self._step_birds()
            self._step_civilians()
            self.history_birds.append(self.bird_state.copy())
            self.history_civilians.append(self.civilian_state.copy())

            blue_collided, blue_collision_agents, blue_collision_dist = (
                self._blue_blue_collision_status()
            )
            if blue_collided:
                self.collided = True
                self.collision_step = step
                self.collision_agents = blue_collision_agents
                self.collision_kind = "blue_blue"
                print(
                    f"Swarm lost at step {step} "
                    f"(blue-blue collision, agents={blue_collision_agents}, "
                    f"distance={blue_collision_dist:.3f})."
                )
                break
            if self._resolve_active_reds(step):
                break
            if self.num_neutralized + self.num_escaped >= self.num_reds:
                self.defenders_won = True
                self.defenders_win_step = step
                print(
                    f"All reds resolved at step {step}: "
                    f"neutralized={self.num_neutralized}, escaped={self.num_escaped}, "
                    f"score={self.blue_score:.1f}. Blue wins."
                )
                break

        self.history_blues = np.array(self.history_blues)
        self.history_red = np.array(self.history_red)
        self.history_birds = np.array(self.history_birds)
        self.history_civilians = np.array(self.history_civilians)
        print("Simulation complete.")

    def generate_animation(self, video_filename=None):
        if len(self.history_blues) <= 1:
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        label_fontsize = 12
        tick_fontsize = 10
        legend_fontsize = 10
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlabel("x", fontsize=label_fontsize)
        ax.set_ylabel("y", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        plot_max = max(self.arena_max, 105.0)
        ax.set_xlim(self.arena_min, plot_max)
        ax.set_ylim(self.arena_min, plot_max)
        arena_border = plt.Rectangle(
            (self.arena_min, self.arena_min),
            self.arena_size,
            self.arena_size,
            fill=False,
            edgecolor="k",
            linewidth=2,
            linestyle="-",
            alpha=0.8,
        )
        ax.add_patch(arena_border)

        red_lines = []
        red_points = []
        red_role_texts = []
        for j in range(self.num_reds):
            line, = ax.plot([], [], "r--", linewidth=1.5, alpha=0.6)
            label = "Red" if self.num_reds == 1 else f"Red {j + 1}"
            point, = ax.plot([], [], "rX", markersize=12, label=label)
            red_lines.append(line)
            red_points.append(point)
            red_role_texts.append(
                ax.text(0, 0, "", fontsize=8, color="darkred", fontweight="bold")
            )

        bird_points = []
        for _ in range(self.num_birds):
            point, = ax.plot(
                [], [], marker="^", color="0.5", linestyle="None", markersize=7,
                label="Bird" if not bird_points else None,
            )
            bird_points.append(point)

        polygon_closed = np.vstack([self.defense_polygon, self.defense_polygon[0]])
        target_line, = ax.plot(
            polygon_closed[:, 0],
            polygon_closed[:, 1],
            color="k",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Defense Polygon",
        )
        shape_closed = np.vstack([self.defended_shape_points, self.defended_shape_points[0]])
        true_shape_line, = ax.plot(
            shape_closed[:, 0],
            shape_closed[:, 1],
            color="0.35",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
            label="Defended Shape",
        )

        blue_lines = []
        blue_points = []
        blue_role_texts = []
        colors = ["b", "g", "c", "m"]

        for i in range(self.num_blues):
            color = colors[i % len(colors)]
            line, = ax.plot([], [], f"{color}-", linewidth=2, alpha=0.7)
            point, = ax.plot([], [], f"{color}o", markersize=8, label=f"Blue {i + 1}")
            blue_lines.append(line)
            blue_points.append(point)
            blue_role_texts.append(
                ax.text(0, 0, "", fontsize=8, color=color, fontweight="bold")
            )

        ax.legend(loc="upper right", fontsize=legend_fontsize)

        label_offset = 0.015 * self.arena_size

        def update(frame):
            red_roles_now = self.red_role_assignment_history[
                min(frame, len(self.red_role_assignment_history) - 1)
            ]
            for j in range(self.num_reds):
                idx = j * self.red_nx
                x, y = self.history_red[frame, idx], self.history_red[frame, idx + 1]
                red_lines[j].set_data(
                    self.history_red[:frame + 1, idx],
                    self.history_red[:frame + 1, idx + 1],
                )
                red_points[j].set_data([x], [y])
                role = red_roles_now.get(j)
                red_role_texts[j].set_position((x + label_offset, y + label_offset))
                red_role_texts[j].set_text(role.value.upper() if role is not None else "")

            blue_roles_now = self.role_assignment_history[
                min(frame, len(self.role_assignment_history) - 1)
            ]
            blue_disabled_now = self.blue_disabled_history[
                min(frame, len(self.blue_disabled_history) - 1)
            ]
            for i in range(self.num_blues):
                idx = i * self.blue_nx
                x, y = self.history_blues[frame, idx], self.history_blues[frame, idx + 1]
                blue_lines[i].set_data(
                    self.history_blues[:frame + 1, idx],
                    self.history_blues[:frame + 1, idx + 1],
                )
                blue_points[i].set_data([x], [y])
                blue_role_texts[i].set_position((x + label_offset, y + label_offset))
                if blue_disabled_now[i]:
                    blue_role_texts[i].set_text("DISABLED")
                    blue_points[i].set_alpha(0.3)
                    blue_lines[i].set_alpha(0.15)
                else:
                    role = blue_roles_now.get(i)
                    blue_role_texts[i].set_text(role.value.upper() if role is not None else "")
                    blue_points[i].set_alpha(1.0)
                    blue_lines[i].set_alpha(0.7)

            bird_frame = min(frame, len(self.history_birds) - 1)
            for b in range(self.num_birds):
                idx = b * 4
                bx, by = self.history_birds[bird_frame, idx], self.history_birds[bird_frame, idx + 1]
                bird_points[b].set_data([bx], [by])

            artists = (
                red_lines + red_points + red_role_texts
                + blue_lines + blue_points + blue_role_texts + bird_points
                + [arena_border, target_line, true_shape_line]
            )
            return artists

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.history_red),
            interval=50,
            blit=True,
        )
        if video_filename is None:
            video_filename = f"decentralized_pursuit_evasion_{self.num_blues}.gif"
        print(f"Saving video to {video_filename}...")
        anim.save(video_filename, writer="pillow", fps=20)
        print(f"Video saved successfully: {video_filename}")

        base_filename = video_filename.rsplit(".", 1)[0]
        final_frame_idx = len(self.history_red) - 1
        update(final_frame_idx)
        png_filename = base_filename + ".png"
        fig.savefig(png_filename, dpi=300, bbox_inches="tight")
        print(f"Final frame saved successfully: {png_filename}")

        # Close the figure to free up memory
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a single simulation and save its animation."
    )
    parser.add_argument("--sim-steps", type=int, default=200, help="Number of simulation steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-reds", type=int, default=1, help="Number of red agents.")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM-backed role planners (via OpenRouter; requires "
        "OPENROUTER_API_KEY) instead of the rule-based stubs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-chat",
        help="OpenRouter model slug to use with --llm.",
    )
    args = parser.parse_args()

    role_planner = LLMRolePlanner(model=args.model) if args.llm else None
    red_role_planner = LLMRedRolePlanner(model=args.model) if args.llm else None
    simulation = Simulation(
        num_blues=4,
        num_reds=args.num_reds,
        sim_steps=args.sim_steps,
        horizon=20,
        enable_blue_blue_cbf=True,
        enable_blue_red_cbf=True,
        role_planner=role_planner,
        red_role_planner=red_role_planner,
        seed=args.seed,
    )

    simulation.run()
    simulation.generate_animation()
