import argparse
import contextlib
import io
import traceback
from collections import Counter

from planning.planner import LLMRedRolePlanner, LLMVisionRolePlanner
from scripts.run_red_count_grid import save_csv

# Only these three of Experiment I's six models are actually vision-capable
# via OpenRouter -- DeepSeek V3, Qwen3-30B-A3B, and Llama 3.3 70B are
# text-only and will error (or silently drop the image) if sent one.
VISION_CAPABLE_MODELS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5-nano",
    "google/gemini-2.5-flash",
]

DEFAULT_CONTACT_SLOT_COUNTS = [1, 2, 3, 4]
DEFAULT_NUM_BLUES = 4
DEFAULT_N_SEEDS = 10
DEFAULT_SIM_STEPS = 600
DEFAULT_HORIZON = 20
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_PLANNING_INTERVAL_SECONDS = 10.0
DEFAULT_P_BIRD = 0.15
DEFAULT_P_CIVILIAN = 0.15

RESULT_FIELDNAMES = [
    "game_idx",
    "seed",
    "num_contact_slots",
    "num_blues",
    "sim_steps_limit",
    "model",
    "planning_interval_seconds",
    "p_bird",
    "p_civilian",
    "num_drones_realized",
    "num_birds_realized",
    "num_civilians_realized",
    "num_bird_engagements",
    "num_civilian_engagements",
    "blue_planner_fallback_count",
    "red_planner_fallback_count",
    "outcome",
    "steps_completed",
    "num_captured",
    "num_escaped",
    "num_red_active_at_end",
    "red_outcomes",
    "blue_score",
    "final_num_blue_disabled",
    "touchdown",
    "touchdown_step",
    "touchdown_red_index",
    "collided",
    "collision_step",
    "collision_kind",
    "collision_agents",
    "defenders_won",
    "defenders_win_step",
    "solver_failed",
    "solver_failure_step",
    "error_message",
]

REASONING_FIELDNAMES = [
    "game_idx",
    "seed",
    "num_contact_slots",
    "num_blues",
    "model",
    "planning_interval_seconds",
    "sim_step",
    "team",
    "agent_id",
    "role",
    "target_contact_id",
    "target_region_id",
    "target_kind",
    "reasoning",
]


class ReasoningLoggingPlanner:
    """Wraps a BlueRolePlanner and records every (step, agent, role, target,
    reasoning) tuple it produces. target_contact_id/target_region_id/
    last_targets/last_recon_target_regions only exist on vision planners
    (LLMVisionRolePlanner) -- for red's text-only planner they're just
    logged empty, same as reasoning is for rule-based stubs."""

    def __init__(self, inner, team):
        self.inner = inner
        self.team = team
        self.log = []
        self.last_targets = {}

    def plan(self, obs):
        roles = self.inner.plan(obs)
        reasoning = getattr(self.inner, "last_reasoning", {})
        targets = getattr(self.inner, "last_targets", {})
        recon_targets = getattr(self.inner, "last_recon_target_regions", {})
        self.last_targets = targets  # passthrough so Simulation.run() can read it via getattr
        for agent_id, role in roles.items():
            self.log.append(
                {
                    "sim_step": obs.sim_step,
                    "team": self.team,
                    "agent_id": agent_id,
                    "role": role.value,
                    "target_contact_id": targets.get(agent_id, ""),
                    "target_region_id": recon_targets.get(agent_id, ""),
                    "reasoning": reasoning.get(agent_id, ""),
                }
            )
        return roles


def determine_outcome(simulation):
    if simulation.solver_failed:
        return "solver_failure"
    if simulation.collided:
        return "lost_collision"
    if simulation.touchdown:
        return "lost_touchdown"
    if simulation.defenders_won:
        return "blue_win_all_resolved"
    return "incomplete"


def annotate_target_ground_truth(reasoning_rows, simulation):
    """Fills in target_kind (drone/bird/civilian) by cross-referencing the
    now-finished episode's contact_roster -- ground truth the planner itself
    never saw."""
    for row in reasoning_rows:
        contact_id = row["target_contact_id"]
        if contact_id == "" or not (0 <= contact_id < len(simulation.contact_roster)):
            row["target_kind"] = ""
            continue
        kind, _idx = simulation.contact_roster[contact_id]
        row["target_kind"] = kind


def run_single_game(
    game_idx,
    seed,
    num_contact_slots,
    num_blues,
    sim_steps,
    horizon,
    model,
    planning_interval_seconds,
    p_bird,
    p_civilian,
    verbose_games,
):
    # Local import to avoid a hard dependency on Simulation at module import
    # time for callers that only need the CSV-schema/planner-wiring helpers
    # above (mirrors run_red_count_grid.py's structure).
    from simulation import Simulation

    blue_planner = ReasoningLoggingPlanner(LLMVisionRolePlanner(model=model), "blue")
    red_planner = ReasoningLoggingPlanner(LLMRedRolePlanner(model=model), "red")

    row = {
        "game_idx": game_idx,
        "seed": seed,
        "num_contact_slots": num_contact_slots,
        "num_blues": num_blues,
        "sim_steps_limit": sim_steps,
        "model": model,
        "planning_interval_seconds": planning_interval_seconds,
        "p_bird": p_bird,
        "p_civilian": p_civilian,
        "error_message": "",
    }

    try:
        simulation = Simulation(
            solver_mode="decentralized",
            num_blues=num_blues,
            sim_steps=sim_steps,
            horizon=horizon,
            seed=seed,
            enable_blue_blue_cbf=True,
            enable_blue_red_cbf=True,
            role_planner=blue_planner,
            red_role_planner=red_planner,
            planning_interval_seconds=planning_interval_seconds,
            num_contact_slots=num_contact_slots,
            p_bird=p_bird,
            p_civilian=p_civilian,
        )
        if verbose_games:
            simulation.run()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                simulation.run()

        row.update(
            {
                "num_drones_realized": simulation.num_reds,
                "num_birds_realized": simulation.num_birds,
                "num_civilians_realized": simulation.num_civilians,
                "num_bird_engagements": simulation.num_bird_engagements,
                "num_civilian_engagements": simulation.num_civilian_engagements,
                "blue_planner_fallback_count": len(
                    getattr(blue_planner.inner, "fallback_events", [])
                ),
                "red_planner_fallback_count": len(
                    getattr(red_planner.inner, "fallback_events", [])
                ),
                "outcome": determine_outcome(simulation),
                "steps_completed": max(len(simulation.history_red) - 1, 0),
                "num_captured": simulation.num_captured,
                "num_escaped": simulation.num_escaped,
                "num_red_active_at_end": simulation.red_outcomes.count("active"),
                "red_outcomes": ",".join(simulation.red_outcomes),
                "blue_score": simulation.blue_score,
                "final_num_blue_disabled": sum(simulation.blue_disabled),
                "touchdown": int(simulation.touchdown),
                "touchdown_step": simulation.touchdown_step,
                "touchdown_red_index": simulation.touchdown_red_index,
                "collided": int(simulation.collided),
                "collision_step": simulation.collision_step,
                "collision_kind": simulation.collision_kind,
                "collision_agents": simulation.collision_agents,
                "defenders_won": int(simulation.defenders_won),
                "defenders_win_step": simulation.defenders_win_step,
                "solver_failed": int(simulation.solver_failed),
                "solver_failure_step": simulation.solver_failure_step,
            }
        )
        reasoning_rows = []
        for entry in blue_planner.log + red_planner.log:
            reasoning_rows.append(
                {
                    "game_idx": game_idx,
                    "seed": seed,
                    "num_contact_slots": num_contact_slots,
                    "num_blues": num_blues,
                    "model": model,
                    "planning_interval_seconds": planning_interval_seconds,
                    **entry,
                }
            )
        annotate_target_ground_truth(reasoning_rows, simulation)
        reasoning_rows.sort(key=lambda r: (r["sim_step"], r["team"], r["agent_id"]))
        return row, reasoning_rows

    except Exception as exc:  # noqa: BLE001 -- batch sweep must not die on one game
        row.update(
            {
                "num_drones_realized": None,
                "num_birds_realized": None,
                "num_civilians_realized": None,
                "num_bird_engagements": None,
                "num_civilian_engagements": None,
                "blue_planner_fallback_count": len(
                    getattr(blue_planner.inner, "fallback_events", [])
                ),
                "red_planner_fallback_count": len(
                    getattr(red_planner.inner, "fallback_events", [])
                ),
                "outcome": "error",
                "steps_completed": None,
                "num_captured": None,
                "num_escaped": None,
                "num_red_active_at_end": None,
                "red_outcomes": "",
                "blue_score": None,
                "final_num_blue_disabled": None,
                "touchdown": None,
                "touchdown_step": None,
                "touchdown_red_index": None,
                "collided": None,
                "collision_step": None,
                "collision_kind": None,
                "collision_agents": None,
                "defenders_won": None,
                "defenders_win_step": None,
                "solver_failed": None,
                "solver_failure_step": None,
                "error_message": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"[game_idx={game_idx} seed={seed} num_contact_slots={num_contact_slots}] FAILED: {exc}")
        traceback.print_exc()
        reasoning_rows = [
            {
                "game_idx": game_idx,
                "seed": seed,
                "num_contact_slots": num_contact_slots,
                "num_blues": num_blues,
                "model": model,
                "planning_interval_seconds": planning_interval_seconds,
                "target_kind": "",
                **entry,
            }
            for entry in blue_planner.log + red_planner.log
        ]
        return row, reasoning_rows


def parse_int_list(raw_value, default):
    if raw_value is None:
        return default
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("Count list cannot be empty.")
    return values


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Experiment II: vision-augmented blue planner, otherwise identical "
            "to Experiment I (RECON, coverage, retry/fallback). Each contact "
            "slot independently resolves to a real antagonistic drone, a "
            "harmless bird, or a harmless civilian drone; blue sees an "
            "image of every DETECTED contact and must also assign each "
            "engagement role a specific target. Sweeps contact-slot-count x "
            "seeds for one (model, planning-interval) combination per run."
        )
    )
    parser.add_argument(
        "--contact-slot-counts", type=str, default=None, help="Default: 1,2,3,4"
    )
    parser.add_argument("--num-blues", type=int, default=DEFAULT_NUM_BLUES)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--sim-steps", type=int, default=DEFAULT_SIM_STEPS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=VISION_CAPABLE_MODELS,
        help=f"OpenRouter model slug -- restricted to vision-capable models: {VISION_CAPABLE_MODELS}",
    )
    parser.add_argument(
        "--planning-interval-seconds", type=float, default=DEFAULT_PLANNING_INTERVAL_SECONDS
    )
    parser.add_argument("--p-bird", type=float, default=DEFAULT_P_BIRD)
    parser.add_argument("--p-civilian", type=float, default=DEFAULT_P_CIVILIAN)
    parser.add_argument("--output-results-csv", type=str, default=None)
    parser.add_argument("--output-reasoning-csv", type=str, default=None)
    parser.add_argument("--verbose-games", action="store_true")
    args = parser.parse_args()

    contact_slot_counts = parse_int_list(args.contact_slot_counts, DEFAULT_CONTACT_SLOT_COUNTS)
    model_slug = args.model.replace("/", "_").replace(":", "_")
    stem = f"vision_experiment_{args.n_seeds}seeds_{model_slug}_{args.planning_interval_seconds:g}s"
    output_results_csv = args.output_results_csv or f"{stem}_results.csv"
    output_reasoning_csv = args.output_reasoning_csv or f"{stem}_reasoning.csv"

    result_rows = []
    reasoning_rows = []
    game_idx = 0
    seed = args.base_seed
    for num_contact_slots in contact_slot_counts:
        for local_idx in range(args.n_seeds):
            print(
                f"Running game_idx={game_idx} (num_contact_slots={num_contact_slots}, "
                f"num_blues={args.num_blues}, model={args.model}, "
                f"interval={args.planning_interval_seconds}s, p_bird={args.p_bird}, "
                f"p_civilian={args.p_civilian}, seed={seed}, run {local_idx + 1}/{args.n_seeds}) ..."
            )
            row, game_reasoning_rows = run_single_game(
                game_idx=game_idx,
                seed=seed,
                num_contact_slots=num_contact_slots,
                num_blues=args.num_blues,
                sim_steps=args.sim_steps,
                horizon=args.horizon,
                model=args.model,
                planning_interval_seconds=args.planning_interval_seconds,
                p_bird=args.p_bird,
                p_civilian=args.p_civilian,
                verbose_games=args.verbose_games,
            )
            result_rows.append(row)
            reasoning_rows.extend(game_reasoning_rows)
            print(
                f"  -> outcome={row['outcome']} steps={row['steps_completed']} "
                f"drones={row.get('num_drones_realized')} birds={row.get('num_birds_realized')} "
                f"civilians={row.get('num_civilians_realized')} "
                f"blue_fallback={row.get('blue_planner_fallback_count')}"
            )
            game_idx += 1
            seed += 1

    save_csv(result_rows, RESULT_FIELDNAMES, output_results_csv)
    save_csv(reasoning_rows, REASONING_FIELDNAMES, output_reasoning_csv)

    print(f"\nCompleted vision-experiment sweep over {contact_slot_counts} x {args.n_seeds} seeds each.")
    print(f"Per-game results CSV: {output_results_csv}")
    print(f"Planner reasoning CSV: {output_reasoning_csv}")
    print("Outcome counts:")
    for outcome, count in sorted(Counter(row["outcome"] for row in result_rows).items()):
        print(f"  {outcome}: {count}")


if __name__ == "__main__":
    main()
