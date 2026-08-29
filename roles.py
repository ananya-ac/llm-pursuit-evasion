from enum import Enum


class Role(Enum):
    """Tactical role assigned to a single blue by a RolePlanner.

    NEUTRALIZE and CAPTURE both run the same intercept controller; the
    distinction is downstream, in the blue-red CBF filter: NEUTRALIZE is
    exempt from blue-red collision avoidance (allowed to make contact),
    CAPTURE is not (converges within capture_radius but stays outside
    D_safe_blue_red).

    RECON (controllers.ReconController) patrols toward a waypoint -- either
    uniformly random, or a specific coverage region chosen by the role
    planner (see coverage.CoverageTracker) -- while independently sweeping
    its bearing. It carries no sensing difference from any other role; field
    of view is fixed per agent, independent of role.
    """

    DEFEND = "defend"
    NEUTRALIZE = "neutralize"
    CAPTURE = "capture"
    RECON = "recon"


class RedRole(Enum):
    """Tactical role assigned to a single red by a RedRolePlanner.

    RECON currently just holds position (controllers.RedReconController) --
    same stub/no-sensing-difference caveat as Role.RECON.
    """

    EVADE = "evade"
    ATTACK = "attack"
    RECON = "recon"
