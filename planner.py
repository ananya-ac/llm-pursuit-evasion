import base64
import functools
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from roles import RedRole, Role


def _infer_stride(flat_values, count):
    """Infers the per-agent state width (4 normally, 5 when the agent
    carries an independent bearing -- Simulation opts every decentralized-
    mode Agent into this) from a flattened array's total size, so planners
    don't need PlannerObservation to carry an explicit width field."""
    if count <= 0:
        return 4
    return int(np.asarray(flat_values, dtype=float).size // count)


@dataclass
class PlannerObservation:
    """Everything a RolePlanner (rule-based or LLM-backed) needs to assign roles.

    Bundled as a dataclass so RolePlanner.plan()'s signature stays stable when a
    future LLM planner needs more context to build its prompt.

    The contact_* / own_image_paths fields are only populated when a vision
    planner (Experiment II) is in use; text-only planners (Experiment I)
    ignore them and Simulation leaves them at their None defaults otherwise.

    detected_red_indices / detected_blue_indices: which of the OPPOSING
    team's agents are visible this cycle, per the observing team's combined
    field of view (fog of war -- see sensing.py). A team's own agents are
    always fully visible regardless of these fields. None means "no
    restriction" (back-compat default -- e.g. for hand-built observations in
    tests, or any future caller that doesn't populate them), not "nothing
    detected".
    """

    x_current: np.ndarray
    red_states: np.ndarray
    num_blues: int
    num_reds: int
    sim_step: int
    arena_size: float
    capture_radius: float
    defense_center: Optional[np.ndarray] = None
    disabled_blues: Optional[List[int]] = None
    detected_red_indices: Optional[List[int]] = None
    detected_blue_indices: Optional[List[int]] = None
    coverage_regions: Optional[List[Dict]] = None
    contact_states: Optional[Dict[int, np.ndarray]] = None
    contact_image_paths: Optional[Dict[int, str]] = None
    own_image_paths: Optional[Dict[int, str]] = None


class RolePlanner(ABC):
    """Interface for anything that assigns each blue a Role each planning cycle."""

    @abstractmethod
    def plan(self, obs: PlannerObservation) -> Dict[int, Role]:
        ...


class RuleBasedRolePlanner(RolePlanner):
    """Stub planner: the blues closest to their nearest red are assigned an
    intercept role, split between CAPTURE (closer-in, safe engagement -- CBF
    collision avoidance keeps it outside contact range) and NEUTRALIZE
    (further down the ranking, allowed to make contact); the rest DEFEND,
    except that up to recon_fraction of the DEFEND-ranked agents are
    redirected to RECON, targeting the stalest cells of obs.coverage_regions
    (see coverage.CoverageTracker) -- a non-LLM baseline for the same
    staleness-driven scouting an LLM planner could reason about from the
    same field. recon_fraction defaults to 0.0 (RECON never offered), so
    existing callers are unaffected unless they opt in.
    Stands in for a future LLM-backed planner behind the same RolePlanner
    interface.
    """

    def __init__(
        self,
        neutralize_fraction: float = 0.5,
        min_neutralizers: int = 1,
        capture_fraction: float = 0.5,
        recon_fraction: float = 0.0,
        min_recon: int = 0,
    ):
        self.neutralize_fraction = float(neutralize_fraction)
        self.min_neutralizers = int(min_neutralizers)
        self.capture_fraction = float(capture_fraction)
        self.recon_fraction = float(recon_fraction)
        self.min_recon = int(min_recon)
        self.last_target_red_ids: Dict[int, int] = {}
        self.last_recon_target_regions: Dict[int, int] = {}

    def _select_recon_assignments(
        self, candidates: List[int], obs: PlannerObservation
    ) -> Dict[int, int]:
        """candidates: blue ids currently slated for DEFEND. Redirects up to
        n_recon of them to RECON, each targeting a distinct region -- the
        stalest first (never-swept regions, steps_since_seen == -1, rank
        above any finite value), since staleness is the proxy for how long a
        red could have been sitting undetected there. Returns
        {blue_id: region_id}; empty if recon isn't configured or no coverage
        data was provided."""
        if not candidates or not obs.coverage_regions:
            return {}
        n_recon = max(self.min_recon, int(round(len(candidates) * self.recon_fraction)))
        n_recon = min(n_recon, len(candidates), len(obs.coverage_regions))
        if n_recon <= 0:
            return {}
        ranked_regions = sorted(
            obs.coverage_regions,
            key=lambda r: r["steps_since_seen"] if r["steps_since_seen"] >= 0 else float("inf"),
            reverse=True,
        )
        return {candidates[k]: ranked_regions[k]["region_id"] for k in range(n_recon)}

    def plan(self, obs: PlannerObservation) -> Dict[int, Role]:
        red_nx = _infer_stride(obs.red_states, obs.num_reds)
        blue_nx = _infer_stride(obs.x_current, obs.num_blues)
        red_states = np.asarray(obs.red_states, dtype=float).reshape(red_nx * obs.num_reds)
        x_current = np.asarray(obs.x_current, dtype=float).reshape(blue_nx * obs.num_blues)
        disabled = set(obs.disabled_blues or [])
        active_blues = [i for i in range(obs.num_blues) if i not in disabled]

        detected_reds = (
            list(range(obs.num_reds))
            if obs.detected_red_indices is None
            else list(obs.detected_red_indices)
        )
        if not detected_reds:
            # Fog of war: nothing detected this cycle -- hold the perimeter
            # (modulo recon) rather than ranking against reds we can't
            # currently see.
            self.last_target_red_ids = {}
            recon_targets = self._select_recon_assignments(active_blues, obs)
            self.last_recon_target_regions = recon_targets
            return {
                i: (Role.RECON if i in recon_targets else Role.DEFEND) for i in active_blues
            }

        def nearest_red(blue_pos):
            dists = {
                j: np.linalg.norm(blue_pos - red_states[j * red_nx : j * red_nx + 2]) for j in detected_reds
            }
            best_j = min(dists, key=dists.get)
            return best_j, dists[best_j]

        nearest = {i: nearest_red(x_current[i * blue_nx : i * blue_nx + 2]) for i in active_blues}
        distances = {i: nearest[i][1] for i in active_blues}
        order = sorted(active_blues, key=lambda i: distances[i])
        n_intercept = max(
            self.min_neutralizers,
            int(np.ceil(len(active_blues) * self.neutralize_fraction)),
        )
        n_capture = int(round(n_intercept * self.capture_fraction))

        roles: Dict[int, Role] = {}
        targets: Dict[int, int] = {}
        defend_candidates: List[int] = []
        for rank, idx in enumerate(order):
            if rank < n_capture:
                roles[idx] = Role.CAPTURE
                targets[idx] = nearest[idx][0]
            elif rank < n_intercept:
                roles[idx] = Role.NEUTRALIZE
                targets[idx] = nearest[idx][0]
            else:
                roles[idx] = Role.DEFEND
                defend_candidates.append(idx)

        recon_targets = self._select_recon_assignments(defend_candidates, obs)
        for idx in recon_targets:
            roles[idx] = Role.RECON

        self.last_target_red_ids = targets
        self.last_recon_target_regions = recon_targets
        return roles


class RedRolePlanner(ABC):
    """Interface for anything that assigns each red a RedRole each planning
    cycle."""

    @abstractmethod
    def plan(self, obs: PlannerObservation) -> Dict[int, RedRole]:
        ...


class RuleBasedRedRolePlanner(RedRolePlanner):
    """Stub planner: EVADE is only ever chosen once at least one blue has
    actually been spotted (i.e. obs.detected_blue_indices is non-empty --
    red can't react to a threat it hasn't detected); otherwise every red
    ATTACKs the defended region. RECON is not offered by this planner.
    Stands in for a future LLM-backed planner behind the same RedRolePlanner
    interface.
    """

    def plan(self, obs: PlannerObservation) -> Dict[int, RedRole]:
        disabled = set(obs.disabled_blues or [])
        active_blues = set(i for i in range(obs.num_blues) if i not in disabled)
        detected_blues = (
            active_blues
            if obs.detected_blue_indices is None
            else active_blues & set(obs.detected_blue_indices)
        )
        role = RedRole.EVADE if detected_blues else RedRole.ATTACK
        return {j: role for j in range(obs.num_reds)}


# --- LLM-backed planners -----------------------------------------------
#
# Routed through OpenRouter (OpenAI-compatible wire format), not the native
# Anthropic API -- OPENROUTER_API_KEY, not ANTHROPIC_API_KEY.

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "deepseek/deepseek-chat"
_DEFAULT_VISION_MODEL = "google/gemini-2.5-flash"

_IMAGE_MIME_BY_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@functools.lru_cache(maxsize=256)
def _encode_image_data_uri(path: str) -> str:
    """Base64-encodes a local image file as a data: URI, cached per path since
    the same handful of asset files are reused across every planning call in
    an episode (and across episodes in a sweep)."""
    ext = os.path.splitext(path)[1].lower()
    mime = _IMAGE_MIME_BY_EXT.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_openrouter_client(api_key: Optional[str]) -> OpenAI:
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_key:
        raise ValueError(
            "An OpenRouter API key is required -- pass api_key=, or set the "
            "OPENROUTER_API_KEY environment variable."
        )
    return OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=resolved_key)


def _format_observation(obs: PlannerObservation, perspective: str) -> str:
    """Renders a PlannerObservation as plain text for the user message.

    perspective="blue": blue_agents is the planner's own team (always shown
    in full); red_agents is filtered to obs.detected_red_indices (fog of
    war) -- None means "no restriction" (all shown), not "nothing detected".
    perspective="red": mirrored -- red_agents is own team in full,
    blue_agents filtered to obs.detected_blue_indices (also excluding
    disabled blues, which are never valid detection targets or
    role-assignable regardless of what detected_blue_indices says).
    """
    if perspective not in ("blue", "red"):
        raise ValueError(f"Unsupported perspective '{perspective}'. Expected 'blue' or 'red'.")

    red_nx = _infer_stride(obs.red_states, obs.num_reds)
    blue_nx = _infer_stride(obs.x_current, obs.num_blues)
    red_states = np.asarray(obs.red_states, dtype=float).reshape(red_nx * obs.num_reds)
    x_current = np.asarray(obs.x_current, dtype=float).reshape(blue_nx * obs.num_blues)
    disabled = set(obs.disabled_blues or [])

    lines = [
        f"sim_step: {obs.sim_step}",
        f"arena_size: {obs.arena_size}",
        f"capture_radius: {obs.capture_radius}",
    ]
    if obs.defense_center is not None:
        center = np.asarray(obs.defense_center, dtype=float).reshape(2)
        lines.append(f"defense_center: [{center[0]:.2f}, {center[1]:.2f}]")

    if perspective == "blue":
        blue_ids = range(obs.num_blues)
        red_ids = (
            range(obs.num_reds) if obs.detected_red_indices is None else sorted(obs.detected_red_indices)
        )
    else:
        detected_blues = (
            set(range(obs.num_blues)) if obs.detected_blue_indices is None else set(obs.detected_blue_indices)
        ) - disabled
        blue_ids = sorted(detected_blues)
        red_ids = range(obs.num_reds)

    num_active_blues = obs.num_blues - len(disabled)
    blue_header = f"blue_agents (active={num_active_blues}/{obs.num_blues}):"
    if perspective == "red":
        blue_header = (
            "blue_agents (detected only -- bounded by your team's combined field "
            "of view; undetected blues are NOT listed here but may still exist):"
        )
    lines.append(blue_header)
    for i in blue_ids:
        p = x_current[i * blue_nx : i * blue_nx + 2]
        v = x_current[i * blue_nx + 2 : i * blue_nx + 4]
        status = "disabled -- unavailable, do not assign" if i in disabled else "active"
        lines.append(
            f"  agent_id={i} pos=[{p[0]:.2f}, {p[1]:.2f}] vel=[{v[0]:.2f}, {v[1]:.2f}] status={status}"
        )

    red_header = "red_agents:"
    if perspective == "blue":
        red_header = (
            "red_agents (detected only -- bounded by your team's combined field "
            "of view; undetected reds are NOT listed here but may still exist):"
        )
    lines.append(red_header)
    for j in red_ids:
        p = red_states[j * red_nx : j * red_nx + 2]
        v = red_states[j * red_nx + 2 : j * red_nx + 4]
        lines.append(
            f"  agent_id={j} pos=[{p[0]:.2f}, {p[1]:.2f}] vel=[{v[0]:.2f}, {v[1]:.2f}]"
        )

    if obs.coverage_regions:
        lines.append(
            "coverage (steps since a blue's sensor last swept each region; "
            "-1 = never swept):"
        )
        for region in obs.coverage_regions:
            c = region["center"]
            lines.append(
                f"  region_id={region['region_id']} center=[{c[0]:.2f}, {c[1]:.2f}] "
                f"steps_since_seen={region['steps_since_seen']}"
            )

    return "\n".join(lines)


def _create_completion_with_backoff(
    client, max_rate_limit_retries=5, base_delay=1.0, **kwargs
):
    """Wraps client.chat.completions.create with exponential backoff on
    HTTP 429s specifically -- distinct from, and outside of, the
    schema-validation retry budget in _call_structured_assignment, so a rate
    limit doesn't eat into the retries meant for a non-compliant model
    response. Matters more once games run concurrently (see
    run_sweep_parallel.py): a handful of workers hitting OpenRouter at once
    makes 429s routine rather than exceptional."""
    for attempt in range(max_rate_limit_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == max_rate_limit_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(
                f"[LLM] rate-limited (attempt {attempt + 1}/{max_rate_limit_retries + 1}) "
                f"-- backing off {delay:.1f}s"
            )
            time.sleep(delay)


def _call_structured_assignment(
    client: OpenAI,
    model: str,
    system_prompt: str,
    obs: PlannerObservation,
    schema_name: str,
    response_model: type,
    valid_roles: List[str],
    perspective: str,
    include_target: bool = False,
    include_region_target: bool = False,
    max_retries: int = 2,
):
    """Shared request/parse plumbing for both LLM role planners. Retries up
    to max_retries times on a schema-validation failure -- not every model
    routed through OpenRouter reliably honors the strict json_schema
    response_format below, and a retry often just gets a compliant response
    on the next roll. If every attempt fails, the last exception propagates;
    callers (LLMRolePlanner/LLMRedRolePlanner.plan()) catch it and fall back
    to a safe default role for the cycle rather than crashing the whole run.

    include_target=True (blue only) adds a required target_red_id field to
    the schema, letting the planner pick which specific red each engagement
    role should go after instead of solve_decentralized's nearest-red
    fallback. include_region_target=True (blue only) similarly adds a
    required target_region_id field, letting the planner pick which
    coverage region a RECON-assigned agent should patrol toward (see
    coverage.CoverageTracker) instead of ReconController's random-waypoint
    fallback. Red's schema is unaffected by either flag."""
    properties = {
        "agent_id": {"type": "integer"},
        "role": {"type": "string", "enum": valid_roles},
    }
    required = ["agent_id", "role"]
    if include_target:
        properties["target_red_id"] = {
            "type": "integer",
            "description": (
                "Red agent id (must be one of the ids listed under red_agents) this "
                "agent should engage -- required (>= 0) when role is neutralize or "
                "capture; use -1 for defend/recon."
            ),
        }
        required.append("target_red_id")
    if include_region_target:
        properties["target_region_id"] = {
            "type": "integer",
            "description": (
                "Coverage region id (must be one of the ids listed under coverage) this "
                "agent should patrol toward -- required (>= 0) when role is recon; use "
                "-1 for defend/neutralize/capture."
            ),
        }
        required.append("target_region_id")
    properties["reasoning"] = {
        "type": "string",
        "description": "One sentence explaining why this agent got this role (and target, if applicable).",
    }
    required.append("reasoning")

    schema = {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["assignments"],
        "additionalProperties": False,
    }

    last_exc = None
    for attempt in range(max_retries + 1):
        response = _create_completion_with_backoff(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _format_observation(obs, perspective)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        )
        content = response.choices[0].message.content
        try:
            return response_model.model_validate_json(content)
        except ValidationError as exc:
            last_exc = exc
            print(
                f"[LLM {perspective}] attempt {attempt + 1}/{max_retries + 1}: response "
                f"failed schema validation ({type(exc).__name__}) -- "
                f"{'retrying...' if attempt < max_retries else 'giving up.'}"
            )
    raise last_exc


class _BlueRoleAssignmentItem(BaseModel):
    agent_id: int
    role: Literal["defend", "neutralize", "capture", "recon"]
    target_red_id: int
    target_region_id: int
    reasoning: str


class _BlueRoleAssignmentResponse(BaseModel):
    assignments: List[_BlueRoleAssignmentItem]


class _RedRoleAssignmentItem(BaseModel):
    agent_id: int
    role: Literal["evade", "attack"]
    reasoning: str


class _RedRoleAssignmentResponse(BaseModel):
    assignments: List[_RedRoleAssignmentItem]


class LLMRolePlanner(RolePlanner):
    """LLM-backed blue role planner (DEFEND / NEUTRALIZE / CAPTURE), called
    via OpenRouter. Drop-in replacement for RuleBasedRolePlanner behind the
    same RolePlanner interface -- see Simulation(role_planner=...).
    """

    _VALID_ROLES = ["defend", "neutralize", "capture", "recon"]

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        client: Optional[OpenAI] = None,
        capture_points_capture: float = 2.0,
        capture_points_neutralize: float = 1.0,
        capture_threshold: int = 2,
    ):
        self.model = model
        self.client = client if client is not None else _build_openrouter_client(api_key)
        # Keep these in sync with the matching Simulation(...) kwargs -- the
        # prompt states the real scoring the sim will actually apply, so a
        # mismatch here would just recreate the incentive gap this exists to fix.
        self.capture_points_capture = float(capture_points_capture)
        self.capture_points_neutralize = float(capture_points_neutralize)
        self.capture_threshold = int(capture_threshold)
        self.last_reasoning: Dict[int, str] = {}
        self.last_target_red_ids: Dict[int, int] = {}
        self.last_recon_target_regions: Dict[int, int] = {}
        # Every time the model targets a red that isn't currently detected --
        # a real, expected schema slip, not a bug -- that one agent falls back
        # to DEFEND for the cycle (see plan()) and gets recorded here so
        # per-model reliability can be compared across a sweep.
        self.fallback_events: List[Dict] = []

    def _system_prompt(self) -> str:
        return f"""You are the tactical commander for the BLUE team defending a region against RED intruders in a 2D pursuit-evasion simulation. Each planning cycle, you assign every active blue agent exactly one role:

- DEFEND: hold a guard station on the perimeter of the defended region.
- NEUTRALIZE: a single blue closes to physical contact with a red agent. No collision-avoidance safety margin is applied against red for this role -- contact is allowed and intended. Resolves fast, with just one agent. Consequence: that blue is permanently disabled afterward (it takes no further part in the engagement) and the kill is worth only {self.capture_points_neutralize} points.
- CAPTURE: {self.capture_threshold} or more blues converge on the same red agent simultaneously and hold within capture_radius while maintaining a collision-avoidance safety margin -- red is captured by proximity without any contact. Slower and costlier up front ({self.capture_threshold} committed agents instead of 1), but every participating blue remains fully operational afterward, and the kill is worth {self.capture_points_capture} points -- more than NEUTRALIZE.
- RECON: actively patrols toward a coverage region you choose (see the coverage list below) instead of holding a perimeter station or engaging -- it walks there while sweeping its sensor back and forth, then holds and keeps sweeping once it arrives. Use it to re-establish eyes on ground your team hasn't swept recently, not as a way to park an agent idle.

This is a real tradeoff, not a preference: NEUTRALIZE trades a permanently lost asset and lower points for speed and certainty with a single agent. CAPTURE preserves your whole force and scores higher, but ties up {self.capture_threshold} agents at once and only resolves once they arrive together, so a fast or already-close red may breach or escape before CAPTURE converges. RECON commits an agent away from engagement or perimeter duty in exchange for reducing the chance an undetected red is sitting in a region you haven't looked at recently. Weigh this against how many active (non-disabled) blues you have left, how many red agents are still active, and how urgent this particular red is (close to breaching vs. far away). Disabled blues are listed in the observation and are unavailable -- do not assign them a role.

Your team does not have full battlefield awareness: the red_agents list below only includes reds detected by your team's combined field of view (each blue senses a conic region ahead of its current heading) -- there may be additional reds beyond what's listed that your team simply cannot currently see. A short or empty red_agents list does not mean the defended region is safe, only that nothing is currently detected. The coverage list shows, for a fixed grid of regions spanning the arena, how many steps it's been since any blue's sensor last swept each one (-1 means never swept) -- higher numbers mean a red could have been sitting there undetected for longer, and are the main signal for where to send RECON.

For every agent you assign NEUTRALIZE or CAPTURE, you must also set target_red_id to the id of the specific red agent (from the red_agents list above) it should engage -- you decide which threat each agent goes after, not just what role it plays. target_red_id must be one of the ids currently listed under red_agents (you can only target a red your team has actually detected); for DEFEND and RECON, set target_red_id to -1. When assigning CAPTURE, coordinate: multiple agents assigned CAPTURE against the same target_red_id converge on that one red together, since capture requires several blues on the same red at once.

For every agent you assign RECON, you must also set target_region_id to the id of the specific coverage region (from the coverage list above) it should patrol toward -- prefer regions with a high (or -1, i.e. never-swept) steps_since_seen value. target_region_id must be one of the ids currently listed under coverage; for DEFEND, NEUTRALIZE, and CAPTURE, set target_region_id to -1. If multiple agents are assigned RECON, prefer spreading them across different regions rather than sending them all to the same one.

Assign roles based on the full tactical picture: how many red agents you can currently see, how close each is to blues and to the defended region, and how many blues you can commit to intercepting without leaving the perimeter undefended or over-spending your force. You do not need to use every role every cycle. Every active blue agent id provided must receive exactly one role, and for each one you must give a one-sentence reasoning explaining specifically why that agent got that role and target (referencing the actual tactical picture -- distances, force levels, urgency, coverage staleness -- not a generic restatement of the role's definition).

Respond only with a valid JSON object matching the required schema."""

    def plan(self, obs: PlannerObservation) -> Dict[int, Role]:
        try:
            parsed = _call_structured_assignment(
                self.client,
                self.model,
                self._system_prompt(),
                obs,
                "blue_role_assignment",
                _BlueRoleAssignmentResponse,
                self._VALID_ROLES,
                perspective="blue",
                include_target=True,
                include_region_target=True,
            )
        except Exception as exc:  # noqa: BLE001 -- a non-compliant model response
            # (or a network/API failure) must not crash the whole run: fall
            # back to DEFEND for every active blue this cycle and keep going.
            disabled = set(obs.disabled_blues or [])
            active_blues = [i for i in range(obs.num_blues) if i not in disabled]
            print(
                f"[LLM blue] giving up on this cycle's response ({type(exc).__name__}: "
                f"{exc}) -- falling back to DEFEND for all active agents."
            )
            self.fallback_events.append(
                {
                    "sim_step": obs.sim_step,
                    "agent_id": None,
                    "kind": "schema_failure",
                    "error_type": type(exc).__name__,
                }
            )
            self.last_reasoning = {
                i: f"(fallback to DEFEND: {type(exc).__name__} from LLM response)"
                for i in active_blues
            }
            self.last_target_red_ids = {}
            self.last_recon_target_regions = {}
            return {i: Role.DEFEND for i in active_blues}

        detected_reds = (
            set(range(obs.num_reds)) if obs.detected_red_indices is None else set(obs.detected_red_indices)
        )
        valid_regions = {region["region_id"] for region in (obs.coverage_regions or [])}

        roles: Dict[int, Role] = {}
        targets: Dict[int, int] = {}
        recon_targets: Dict[int, int] = {}
        reasoning: Dict[int, str] = {}
        for item in parsed.assignments:
            role = Role(item.role)
            if role in (Role.NEUTRALIZE, Role.CAPTURE):
                if item.target_red_id not in detected_reds:
                    # The LLM occasionally targets a red that has since dropped
                    # out of FOV (e.g. it was detected a cycle or two ago) --
                    # a real, expected schema slip rather than a bug to crash
                    # the whole run over. Fall back to DEFEND for just this
                    # agent this cycle, matching RuleBasedRolePlanner's
                    # existing "nothing detected -> DEFEND" convention.
                    print(
                        f"[LLM blue] agent {item.agent_id} was assigned {role.value} "
                        f"but target_red_id={item.target_red_id} is not a currently "
                        "detected red -- falling back to DEFEND for this agent this cycle."
                    )
                    self.fallback_events.append(
                        {
                            "sim_step": obs.sim_step,
                            "agent_id": item.agent_id,
                            "kind": "invalid_target_red_id",
                            "attempted_role": role.value,
                            "invalid_target_red_id": item.target_red_id,
                        }
                    )
                    roles[item.agent_id] = Role.DEFEND
                    reasoning[item.agent_id] = (
                        f"(fallback to DEFEND: invalid target_red_id={item.target_red_id}) "
                        f"{item.reasoning}"
                    )
                    continue
                targets[item.agent_id] = item.target_red_id
            elif role == Role.RECON:
                if item.target_region_id not in valid_regions:
                    # Same schema-slip tolerance as the target_red_id case
                    # above -- fall back to DEFEND for just this agent.
                    print(
                        f"[LLM blue] agent {item.agent_id} was assigned RECON but "
                        f"target_region_id={item.target_region_id} is not a valid "
                        "coverage region -- falling back to DEFEND for this agent this cycle."
                    )
                    self.fallback_events.append(
                        {
                            "sim_step": obs.sim_step,
                            "agent_id": item.agent_id,
                            "kind": "invalid_target_region_id",
                            "attempted_role": role.value,
                            "invalid_target_region_id": item.target_region_id,
                        }
                    )
                    roles[item.agent_id] = Role.DEFEND
                    reasoning[item.agent_id] = (
                        f"(fallback to DEFEND: invalid target_region_id={item.target_region_id}) "
                        f"{item.reasoning}"
                    )
                    continue
                recon_targets[item.agent_id] = item.target_region_id
            roles[item.agent_id] = role
            reasoning[item.agent_id] = item.reasoning
            target_str = item.target_red_id if role in (Role.NEUTRALIZE, Role.CAPTURE) else "n/a"
            region_str = item.target_region_id if role == Role.RECON else "n/a"
            print(
                f"[LLM blue] agent {item.agent_id} -> {item.role} "
                f"(target_red={target_str}, target_region={region_str}): {item.reasoning}"
            )

        self.last_reasoning = reasoning
        self.last_recon_target_regions = recon_targets
        self.last_target_red_ids = targets
        return roles


class LLMRedRolePlanner(RedRolePlanner):
    """LLM-backed red role planner (EVADE / ATTACK), called via OpenRouter.
    Drop-in replacement for RuleBasedRedRolePlanner behind the same
    RedRolePlanner interface -- see Simulation(red_role_planner=...).
    """

    _SYSTEM_PROMPT = """You are the tactical commander for the RED team attempting to breach a defended region in a 2D pursuit-evasion simulation, while avoiding capture by BLUE agents. Each planning cycle, you assign every red agent exactly one role:

- ATTACK: advance toward the defended region, attempting to breach it.
- EVADE: flee from nearby threatening blue agents instead of advancing.

EVADE only ever makes sense once you have actually detected a blue agent -- you cannot react to a threat you haven't seen. If the blue_agents list below is empty, no threat is currently detected and every red agent should ATTACK. Blue agents marked status=disabled in the observation are permanently out of action (they were expended neutralizing a red earlier) and pose no threat -- ignore them entirely when judging danger.

Your team does not have full battlefield awareness: the blue_agents list below only includes blues detected by your team's combined field of view (each red senses a conic region ahead of its current heading) -- there may be additional blues beyond what's listed that your team simply cannot currently see.

Every red agent id provided must receive exactly one role, and for each one you must give a one-sentence reasoning explaining specifically why that agent got that role (referencing the actual tactical picture -- distances, threats, opportunity -- not a generic restatement of the role's definition).

Respond only with a valid JSON object matching the required schema."""

    _VALID_ROLES = ["evade", "attack"]

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ):
        self.model = model
        self.client = client if client is not None else _build_openrouter_client(api_key)
        self.last_reasoning: Dict[int, str] = {}
        self.fallback_events: List[Dict] = []

    def plan(self, obs: PlannerObservation) -> Dict[int, RedRole]:
        try:
            parsed = _call_structured_assignment(
                self.client,
                self.model,
                self._SYSTEM_PROMPT,
                obs,
                "red_role_assignment",
                _RedRoleAssignmentResponse,
                self._VALID_ROLES,
                perspective="red",
            )
        except Exception as exc:  # noqa: BLE001 -- same schema-slip tolerance as LLMRolePlanner
            print(
                f"[LLM red] giving up on this cycle's response ({type(exc).__name__}: "
                f"{exc}) -- falling back to ATTACK for all agents."
            )
            self.fallback_events.append(
                {
                    "sim_step": obs.sim_step,
                    "agent_id": None,
                    "kind": "schema_failure",
                    "error_type": type(exc).__name__,
                }
            )
            self.last_reasoning = {
                j: f"(fallback to ATTACK: {type(exc).__name__} from LLM response)"
                for j in range(obs.num_reds)
            }
            return {j: RedRole.ATTACK for j in range(obs.num_reds)}

        self.last_reasoning = {item.agent_id: item.reasoning for item in parsed.assignments}
        for item in parsed.assignments:
            print(f"[LLM red] agent {item.agent_id} -> {item.role}: {item.reasoning}")
        return {item.agent_id: RedRole(item.role) for item in parsed.assignments}


# --- Vision-augmented blue planner (Experiment II) ----------------------
#
# Blue-only: reds and decoy birds are merged into an unlabeled "contacts"
# list (position/velocity + an attached image, no ground-truth tag), so the
# model must use the image -- not the state channel -- to judge whether a
# contact is a real threat. Adds target_contact_id so role assignment also
# controls which contact gets engaged, rather than solve_decentralized's
# blind nearest-red heuristic. Red's planner is unchanged from Experiment I.


def _format_vision_text(obs: PlannerObservation) -> str:
    blue_nx = _infer_stride(obs.x_current, obs.num_blues)
    x_current = np.asarray(obs.x_current, dtype=float).reshape(blue_nx * obs.num_blues)
    lines = [
        f"sim_step: {obs.sim_step}",
        f"arena_size: {obs.arena_size}",
        f"capture_radius: {obs.capture_radius}",
    ]
    if obs.defense_center is not None:
        center = np.asarray(obs.defense_center, dtype=float).reshape(2)
        lines.append(f"defense_center: [{center[0]:.2f}, {center[1]:.2f}]")

    disabled = set(obs.disabled_blues or [])
    num_active_blues = obs.num_blues - len(disabled)
    lines.append(f"blue_agents (active={num_active_blues}/{obs.num_blues}):")
    for i in range(obs.num_blues):
        p = x_current[i * blue_nx : i * blue_nx + 2]
        v = x_current[i * blue_nx + 2 : i * blue_nx + 4]
        status = "disabled -- unavailable, do not assign" if i in disabled else "active"
        lines.append(
            f"  agent_id={i} pos=[{p[0]:.2f}, {p[1]:.2f}] vel=[{v[0]:.2f}, {v[1]:.2f}] status={status}"
        )

    lines.append(
        "contacts (UNIDENTIFIED -- each may be a real red intruder or a harmless bird; "
        "position and velocity alone may not tell them apart, use each contact's attached "
        "image below to judge):"
    )
    contact_states = obs.contact_states or {}
    for contact_id in sorted(contact_states):
        s = np.asarray(contact_states[contact_id], dtype=float).reshape(4)
        lines.append(
            f"  contact_id={contact_id} pos=[{s[0]:.2f}, {s[1]:.2f}] vel=[{s[2]:.2f}, {s[3]:.2f}]"
        )

    return "\n".join(lines)


class _BlueVisionAssignmentItem(BaseModel):
    agent_id: int
    role: Literal["defend", "neutralize", "capture"]
    target_contact_id: int
    reasoning: str


class _BlueVisionAssignmentResponse(BaseModel):
    assignments: List[_BlueVisionAssignmentItem]


class LLMVisionRolePlanner(RolePlanner):
    """Vision-augmented blue role planner (Experiment II). Same role set as
    LLMRolePlanner, plus a required target_contact_id per assignment (-1 for
    DEFEND). Contacts (reds + decoy birds) are shown unlabeled, each with an
    attached image, via PlannerObservation.contact_states/contact_image_paths;
    ground truth (which contact is actually a red vs. a bird) is never
    revealed in the prompt.
    """

    _VALID_ROLES = ["defend", "neutralize", "capture"]

    def __init__(
        self,
        model: str = _DEFAULT_VISION_MODEL,
        api_key: Optional[str] = None,
        client: Optional[OpenAI] = None,
        capture_points_capture: float = 2.0,
        capture_points_neutralize: float = 1.0,
        capture_threshold: int = 2,
        image_detail: str = "low",
    ):
        self.model = model
        self.client = client if client is not None else _build_openrouter_client(api_key)
        self.capture_points_capture = float(capture_points_capture)
        self.capture_points_neutralize = float(capture_points_neutralize)
        self.capture_threshold = int(capture_threshold)
        self.image_detail = image_detail
        self.last_reasoning: Dict[int, str] = {}
        self.last_targets: Dict[int, int] = {}

    def _system_prompt(self) -> str:
        return f"""You are the tactical commander for the BLUE team defending a region against intrusions in a 2D pursuit-evasion simulation. Each planning cycle, you assign every active blue agent exactly one role:

- DEFEND: hold a guard station on the perimeter of the defended region.
- NEUTRALIZE: a single blue closes to physical contact with its assigned target contact. No collision-avoidance safety margin is applied against that contact for this role -- contact is allowed and intended. Consequence: that blue is permanently disabled afterward and the kill is worth only {self.capture_points_neutralize} points.
- CAPTURE: {self.capture_threshold} or more blues converge on the same target contact simultaneously and hold within capture_radius while maintaining a collision-avoidance safety margin -- captured by proximity without contact. Every participating blue remains fully operational afterward, and the kill is worth {self.capture_points_capture} points -- more than NEUTRALIZE.

Critically: every object under "contacts" is UNIDENTIFIED. Some are real intruders; some are harmless birds that pose no threat and are not part of the simulation's win condition. Position and velocity alone will often not tell them apart -- a bird can be moving just as fast, in just as plausible a direction, as a real intruder. Each contact has an image attached below the text observation; use it to judge whether the contact's silhouette looks like a mechanical drone (a real threat worth engaging) or a bird (harmless -- engaging it wastes that agent's role for this cycle while real threats go unaddressed). Weigh this against the tactical picture: how many active blues you have, how close each contact is to the defended region, and how confident the image makes you.

For every agent you assign NEUTRALIZE or CAPTURE, you must also set target_contact_id to the id of the specific contact it should engage -- this is how you decide which threat each agent goes after, not just what role it plays. For DEFEND, set target_contact_id to -1. Disabled blues are listed in the observation and are unavailable -- do not assign them. Every active blue agent id provided must receive exactly one role, and for each one you must give a one-sentence reasoning explaining specifically why that agent got that role and (if applicable) that target -- referencing what you actually see in its image and the tactical picture, not a generic restatement of the role's definition.

Respond only with a valid JSON object matching the required schema."""

    def _build_user_content(self, obs: PlannerObservation) -> List[dict]:
        content = [{"type": "text", "text": _format_vision_text(obs)}]

        own_images = obs.own_image_paths or {}
        for blue_id in sorted(own_images):
            content.append({"type": "text", "text": f"Own team blue agent {blue_id} image:"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _encode_image_data_uri(own_images[blue_id]),
                        "detail": self.image_detail,
                    },
                }
            )

        contact_images = obs.contact_image_paths or {}
        for contact_id in sorted(contact_images):
            content.append({"type": "text", "text": f"Contact {contact_id} image:"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _encode_image_data_uri(contact_images[contact_id]),
                        "detail": self.image_detail,
                    },
                }
            )
        return content

    def _build_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "integer"},
                            "role": {"type": "string", "enum": self._VALID_ROLES},
                            "target_contact_id": {
                                "type": "integer",
                                "description": (
                                    "Contact id this agent should engage, required for "
                                    "NEUTRALIZE/CAPTURE; use -1 for DEFEND."
                                ),
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "One sentence explaining the role and target choice.",
                            },
                        },
                        "required": ["agent_id", "role", "target_contact_id", "reasoning"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["assignments"],
            "additionalProperties": False,
        }

    def plan(self, obs: PlannerObservation) -> Dict[int, Role]:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._build_user_content(obs)},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "blue_vision_role_assignment",
                    "strict": True,
                    "schema": self._build_schema(),
                },
            },
        )
        parsed = _BlueVisionAssignmentResponse.model_validate_json(
            response.choices[0].message.content
        )

        roles: Dict[int, Role] = {}
        targets: Dict[int, int] = {}
        reasoning: Dict[int, str] = {}
        for item in parsed.assignments:
            role = Role(item.role)
            if role != Role.DEFEND and item.target_contact_id < 0:
                raise ValueError(
                    f"LLMVisionRolePlanner: agent {item.agent_id} was assigned "
                    f"{role.value} but no target_contact_id was given -- the schema "
                    "requires a non-negative target for any non-DEFEND role."
                )
            roles[item.agent_id] = role
            if role != Role.DEFEND:
                targets[item.agent_id] = item.target_contact_id
            reasoning[item.agent_id] = item.reasoning
            target_str = item.target_contact_id if role != Role.DEFEND else "n/a"
            print(
                f"[LLM blue-vision] agent {item.agent_id} -> {item.role} "
                f"(target_contact={target_str}): {item.reasoning}"
            )

        self.last_reasoning = reasoning
        self.last_targets = targets
        return roles


class RuleBasedVisionRolePlanner(RolePlanner):
    """Offline test/debug stub for the vision pipeline -- NOT a fair baseline,
    since (unlike a real vision planner) it reads obs.red_states directly
    (ground truth) rather than judging contacts from their images. Exists
    only so solver/simulation wiring for target_contact_id can be smoke
    tested without any API calls.
    """

    def __init__(self, neutralize_fraction: float = 0.5, min_neutralizers: int = 1,
                 capture_fraction: float = 0.5):
        self.neutralize_fraction = float(neutralize_fraction)
        self.min_neutralizers = int(min_neutralizers)
        self.capture_fraction = float(capture_fraction)
        self.last_reasoning: Dict[int, str] = {}
        self.last_targets: Dict[int, int] = {}

    def plan(self, obs: PlannerObservation) -> Dict[int, Role]:
        red_nx = _infer_stride(obs.red_states, obs.num_reds)
        blue_nx = _infer_stride(obs.x_current, obs.num_blues)
        red_states = np.asarray(obs.red_states, dtype=float).reshape(red_nx * obs.num_reds)
        x_current = np.asarray(obs.x_current, dtype=float).reshape(blue_nx * obs.num_blues)
        disabled = set(obs.disabled_blues or [])
        active_blues = [i for i in range(obs.num_blues) if i not in disabled]
        contact_states = obs.contact_states or {}

        def nearest_red_contact(blue_pos):
            best_contact_id, best_dist = None, float("inf")
            for contact_id, state in contact_states.items():
                # contact_states values are always a plain 4-vector
                # (position+velocity only) -- see Simulation._active_contact_states.
                p = np.asarray(state, dtype=float).reshape(4)[0:2]
                for j in range(obs.num_reds):
                    if np.allclose(p, red_states[j * red_nx : j * red_nx + 2]):
                        dist = float(np.linalg.norm(blue_pos - p))
                        if dist < best_dist:
                            best_dist, best_contact_id = dist, contact_id
            return best_contact_id, best_dist

        distances, contacts = {}, {}
        for i in active_blues:
            contact_id, dist = nearest_red_contact(x_current[i * blue_nx : i * blue_nx + 2])
            distances[i] = dist
            contacts[i] = contact_id

        order = sorted(active_blues, key=lambda i: distances[i])
        n_intercept = max(self.min_neutralizers, int(np.ceil(len(active_blues) * self.neutralize_fraction)))
        n_capture = int(round(n_intercept * self.capture_fraction))

        roles: Dict[int, Role] = {}
        targets: Dict[int, int] = {}
        reasoning: Dict[int, str] = {}
        for rank, idx in enumerate(order):
            if rank < n_capture:
                roles[idx] = Role.CAPTURE
            elif rank < n_intercept:
                roles[idx] = Role.NEUTRALIZE
            else:
                roles[idx] = Role.DEFEND
            if roles[idx] != Role.DEFEND and contacts[idx] is not None:
                targets[idx] = contacts[idx]
            reasoning[idx] = "rule-based stub (oracle target, ground-truth-informed)"

        self.last_reasoning = reasoning
        self.last_targets = targets
        return roles
