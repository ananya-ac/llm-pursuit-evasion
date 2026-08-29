"""Fixed, role-independent field-of-view / detection geometry shared by both
teams. FOV is a property of an agent's sensor, not its tactical role -- a
RECON agent gets no different sensing than a DEFEND agent (by design).

Not a sophisticated sensor model: detection is perfect (no noise, no missed
detection) within a conic field of view, and absent entirely outside it.
"""

import numpy as np

DEFAULT_FOV_HALF_ANGLE_DEG = 30.0
DEFAULT_FOV_RANGE = 35.0
DEFAULT_BEARING_SPEED_THRESHOLD = 0.2


def compute_bearing(position, velocity, defense_center, faces_toward_center, speed_threshold):
    """Returns a heading in radians (atan2 convention) for one agent.

    If speed exceeds speed_threshold, bearing is the direction of travel
    (atan2(vy, vx)). Otherwise (stationary/near-stationary -- e.g. at spawn,
    or a DEFEND/RECON agent holding station) falls back to a fixed radial
    convention relative to defense_center: faces_toward_center=False points
    AWAY from it (blue, watching the approach to what it guards);
    faces_toward_center=True points TOWARD it (red, oriented on its
    objective). Degenerates to bearing 0.0 if the agent is exactly at
    defense_center in the fallback case.
    """
    velocity = np.asarray(velocity, dtype=float).reshape(2)
    speed = float(np.linalg.norm(velocity))
    if speed > speed_threshold:
        return float(np.arctan2(velocity[1], velocity[0]))

    position = np.asarray(position, dtype=float).reshape(2)
    defense_center = np.asarray(defense_center, dtype=float).reshape(2)
    radial = position - defense_center
    if faces_toward_center:
        radial = -radial
    norm = float(np.linalg.norm(radial))
    if norm <= 1e-9:
        return 0.0
    return float(np.arctan2(radial[1], radial[0]))


def is_detected(observer_pos, observer_bearing, target_pos, half_angle_rad, max_range):
    """True iff target_pos is within max_range of observer_pos AND within
    half_angle_rad of observer_bearing. Angle difference is computed via
    atan2(sin, cos) to stay well-defined across the +-pi wraparound."""
    delta = np.asarray(target_pos, dtype=float).reshape(2) - np.asarray(
        observer_pos, dtype=float
    ).reshape(2)
    distance = float(np.linalg.norm(delta))
    if distance > max_range:
        return False
    if distance <= 1e-9:
        return True  # coincident points; bearing undefined, treat as detected

    target_angle = float(np.arctan2(delta[1], delta[0]))
    angle_diff = float(
        np.arctan2(
            np.sin(target_angle - observer_bearing), np.cos(target_angle - observer_bearing)
        )
    )
    return abs(angle_diff) <= half_angle_rad


def team_detected_indices(observer_states, observer_bearings, target_states, half_angle_rad, max_range):
    """observer_states/target_states: iterables of (idx, pos_2) pairs, already
    filtered by the caller to active/non-disabled agents. observer_bearings:
    {idx: radians}. Returns the set of target indices detected by at least
    one observer (union-of-sensors semantics, not all-must-see-it)."""
    detected = set()
    for target_idx, target_pos in target_states:
        if target_idx in detected:
            continue
        for observer_idx, observer_pos in observer_states:
            if is_detected(
                observer_pos, observer_bearings[observer_idx], target_pos, half_angle_rad, max_range
            ):
                detected.add(target_idx)
                break
    return detected
