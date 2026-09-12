"""BlueRole -> controller-class maps, used by the simulation's decentralized
per-step loop (see control.decentralized_solver.DecentralizedPursuitEvasionSolver)
to pick which controller realizes a given agent's currently assigned role.

Deciding *which controller a role maps to* is a planning-layer concern (it's
the same kind of interpretation as assigning the role itself); actually
constructing controller instances, calling .plan() on them, and applying the
CBF safety filter is execution-layer orchestration and lives in
control/simulation code instead -- see
control.decentralized_solver.DecentralizedPursuitEvasionSolver.
"""

from control.blue_controllers import DefendController, NeutralizeController, ReconController
from control.red_controllers import AttackController, EvadeController, RedReconController
from planning.roles import RedRole, BlueRole

BLUE_ROLE_CONTROLLER_CLASSES = {
    BlueRole.DEFEND: DefendController,
    BlueRole.NEUTRALIZE: NeutralizeController,
    BlueRole.RECON: ReconController,
}

RED_ROLE_CONTROLLER_CLASSES = {
    RedRole.ATTACK: AttackController,
    RedRole.RECON: RedReconController,
    RedRole.EVADE: EvadeController,
}
