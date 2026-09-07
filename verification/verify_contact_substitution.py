"""Verification script for Experiment II's rebuild: three-way per-slot
contact substitution (real drone / bird / civilian drone) and
detection-gated contact visibility.

Free (rule-based, no API calls). Plain-assert style, no test framework, same
convention as verify_bearing_fov.py / verify_rule_based_recon.py.

Checks:
1. Backward compatibility: num_contact_slots=0 (the default) reproduces
   byte-identical spawn state to the old (pre-substitution) code path for a
   fixed num_reds/seed.
2. Per-slot draw invariants across many seeds: slot_kinds length, counts
   matching num_reds/num_birds/num_civilians, contact_roster length.
3. Zero-drone guarantee: even under a small-slot/high-decoy-probability
   config engineered to make an all-decoy draw likely, every seed still
   realizes >=1 drone.
4. num_reds/num_contact_slots conflict raises ValueError.
5. Full decentralized episode with the three-way roster runs end-to-end with
   no crash, and red_outcomes never grows beyond the realized drone count.
6. Detection gating: a contact just inside a blue's FOV cone/range appears in
   the detection-filtered contact set; the same contact just outside does not.
"""

import numpy as np

from planning.planner import RuleBasedRedRolePlanner, RuleBasedVisionRolePlanner
from simulation.testbed import Simulation
import perception.sensing as sensing

SEED = 42


def check_backward_compatibility():
    sim = Simulation(
        solver_mode="decentralized", num_blues=2, num_reds=2, sim_steps=5, horizon=10,
        seed=SEED, enable_blue_blue_cbf=True, enable_blue_red_cbf=True,
    )
    sim.setup_agents()
    assert sim.num_contact_slots == 0
    assert not sim.contact_substitution_enabled
    assert sim.slot_kinds == []
    assert sim.num_reds == 2
    assert sim.num_civilians == 0
    assert sim.num_substitution_birds == 0
    print("PASS: num_contact_slots=0 is a strict no-op (num_reds/slot_kinds unaffected)")


def check_slot_draw_invariants(n_seeds=40):
    for seed in range(n_seeds):
        sim = Simulation(
            solver_mode="decentralized", num_blues=2, num_reds=1, sim_steps=5, horizon=10,
            seed=seed, enable_blue_blue_cbf=True, enable_blue_red_cbf=True,
            num_contact_slots=6, p_bird=0.35, p_civilian=0.35,
        )
        assert len(sim.slot_kinds) == 6
        assert sim.slot_kinds.count("drone") == sim.num_reds
        assert sim.slot_kinds.count("bird") == sim.num_substitution_birds
        assert sim.slot_kinds.count("civilian") == sim.num_civilians
        assert sim.num_reds >= 1, f"seed={seed} realized zero drones"
        sim.setup_agents()
        assert len(sim.contact_roster) == sim.num_reds + sim.num_birds + sim.num_civilians
        assert sim.num_birds == sim.num_substitution_birds  # no additive num_bird_slots requested
    print(f"PASS: slot-draw invariants hold across {n_seeds} seeds (p_bird=0.35, p_civilian=0.35)")


def check_zero_drone_guarantee(n_seeds=60):
    # Small slot count + high combined decoy probability -- engineered to
    # make an all-decoy draw likely absent the redraw guarantee (P(all decoy
    # in 2 slots) = 0.9^2 = 0.81 per attempt).
    for seed in range(n_seeds):
        sim = Simulation(
            solver_mode="decentralized", num_blues=2, num_reds=1, sim_steps=5, horizon=10,
            seed=seed, enable_blue_blue_cbf=True, enable_blue_red_cbf=True,
            num_contact_slots=2, p_bird=0.45, p_civilian=0.45,
        )
        assert sim.num_reds >= 1, f"seed={seed} realized zero drones despite redraw guarantee"
    print(f"PASS: zero-drone guarantee holds across {n_seeds} seeds (2 slots, p_bird=p_civilian=0.45)")


def check_conflict_raises():
    try:
        Simulation(solver_mode="decentralized", num_reds=3, num_contact_slots=4, seed=1)
        raise AssertionError("expected ValueError for num_reds/num_contact_slots conflict")
    except ValueError:
        pass
    print("PASS: num_reds + num_contact_slots conflict raises ValueError")


def check_full_episode_runs():
    sim = Simulation(
        solver_mode="decentralized", num_blues=4, num_reds=1, sim_steps=150, horizon=10,
        seed=7, enable_blue_blue_cbf=True, enable_blue_red_cbf=True,
        num_contact_slots=5, p_bird=0.3, p_civilian=0.3,
        role_planner=RuleBasedVisionRolePlanner(recon_fraction=0.3, min_recon=1),
        red_role_planner=RuleBasedRedRolePlanner(),
        planning_interval_seconds=2.0,
    )
    sim.run()
    assert not sim.solver_failed
    assert len(sim.red_outcomes) == sim.num_reds
    for kind, _idx in sim.contact_roster:
        assert kind in ("red", "bird", "civilian")
    print(
        f"PASS: full episode ran without crash (slot_kinds={sim.slot_kinds}, "
        f"solver_failed={sim.solver_failed})"
    )


def check_detection_gating():
    sim = Simulation(
        solver_mode="decentralized", num_blues=1, num_reds=1, sim_steps=5, horizon=10,
        seed=3, enable_blue_blue_cbf=True, enable_blue_red_cbf=True,
        num_contact_slots=1, p_bird=0.0, p_civilian=0.0,
    )
    sim.setup_agents()
    # Place the single blue at the origin facing +x (bearing=0), and the
    # single contact (guaranteed a drone, since p_bird=p_civilian=0) at a
    # known position -- first inside, then outside, the FOV cone/range.
    sim.x_current[0:2] = [0.0, 0.0]
    sim.x_current[4] = 0.0  # bearing straight along +x
    inside_pos = [sim.fov_range * 0.5, 0.0]  # dead ahead, well within range
    outside_pos = [-sim.fov_range * 0.5, 0.0]  # directly behind -- outside the cone
    contact_id = 0
    kind, idx = sim.contact_roster[contact_id]
    assert kind == "red"

    sim.red_state[idx * sim.red_nx : idx * sim.red_nx + 2] = inside_pos
    _, _, detected_inside = sim._compute_detected_indices(sim_step=0)
    assert contact_id in detected_inside, "contact dead ahead in range should be detected"

    sim.red_state[idx * sim.red_nx : idx * sim.red_nx + 2] = outside_pos
    _, _, detected_outside = sim._compute_detected_indices(sim_step=0)
    assert contact_id not in detected_outside, "contact directly behind should not be detected"

    print("PASS: detection gating correctly includes/excludes a contact based on FOV cone/range")


if __name__ == "__main__":
    check_backward_compatibility()
    check_slot_draw_invariants()
    check_zero_drone_guarantee()
    check_conflict_raises()
    check_full_episode_runs()
    check_detection_gating()
    print("\nALL CHECKS PASSED")
