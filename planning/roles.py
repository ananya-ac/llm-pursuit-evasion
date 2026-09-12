from enum import Enum


class BlueRole(Enum):
    """Tactical role assigned to a single blue by a BlueRolePlanner.

    NEUTRALIZE (controllers via dispatch.NeutralizeController) is exempt
    from blue-red collision avoidance in the CBF filter, allowing it to
    make physical contact with its target red.

    RECON (controllers.ReconController) patrols toward a waypoint -- either
    uniformly random, or a specific coverage region chosen by the role
    planner (see coverage.CoverageTracker) -- while independently sweeping
    its bearing. It carries no sensing difference from any other role; field
    of view is fixed per agent, independent of role.
    """

    DEFEND = "defend"
    NEUTRALIZE = "neutralize"
    RECON = "recon"


class RedRole(Enum):
    """Tactical role assigned to a single red by a RedRolePlanner.

    RECON currently just holds position (controllers.RedReconController) --
    same stub/no-sensing-difference caveat as BlueRole.RECON.
    """

    EVADE = "evade"
    ATTACK = "attack"
    RECON = "recon"
