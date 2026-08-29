import argparse
import contextlib
import csv
import io
import traceback
from collections import Counter

from planner import LLMRedRolePlanner, LLMVisionRolePlanner
from simulation import Simulation


# Only these three of Experiment I's six models are actually vision-capable
# via OpenRouter -- DeepSeek V3, Qwen3-30B-A3B, and Llama 3.3 70B are
# text-only and will error (or silently drop the image) if sent one.
VISION_CAPABLE_MODELS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5-nano",
    "google/gemini-2.5-flash",
]

DEFAULT_RED_COUNTS = [1, 2, 3, 4]
DEFAULT_NUM_BLUES = 4
DEFAULT_N_SEEDS = 10
DEFAULT_SIM_STEPS = 600
DEFAULT_HORIZON = 20
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_PLANNING_INTERVAL_SECONDS = 10.0
DEFAULT_NUM_BIRD_SLOTS = 3
DEFAULT_BIRD_SPAWN_PROB = 0.2

RESULT_FIELDNAMES = [
    "game_idx",
    "seed",
    "num_reds",
    "num_blues",
    "sim_steps_limit",
    "model",
    "planning_interval_seconds",
    "num_bird_slots",
    "bird_spawn_prob",
    "num_birds_spawned",
    "num_bird_engagements",
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
    "num_reds",
    "num_blues",
    "model",
    "planning_interval_seconds",
    "sim_step",
    "team",
    "agent_id",
    "role",
    "target_contact_id",
    "target_was_bird",
    "reasoning",
]


class ReasoningLoggingPlanner:
    """Wraps a RolePlanner and records every (step, agent, role, target,
    reasoning) tuple it produces. target_contact_id/last_targets only exists
    on vision planners (LLMVisionRolePlanner) -- for red's text-only planner
    it's just logged empty, same as reasoning is for rule-based stubs."""

    def __init__(self, inner, team):
        self.inner = inner
        self.team = team
        self.log = []
        self.last_targets = {}

    def plan(self, obs):
        roles = self.inner.plan(obs)
        reasoning = getattr(self.inner, "last_reasoning", {})
        targets = getattr(self.inner, "last_targets", {})
        self.last_targets = targets  # passthrough so Simulation.run() can read it via getattr
        for agent_id, role in roles.items():
            self.log.append(
                {
                    "sim_step": obs.sim_step,
                    "team": self.team,
                    "agent_id": agent_id,
                    "role": role.value,
                    "target_contact_id": targets.get(agent_id, ""),
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
    """Fills in target_was_bird by cross-referencing the now-finished episode's
    contact_roster -- ground truth the planner itself never saw."""
    for row in reasoning_rows:
        contact_id = row["target_contact_id"]
        if contact_id == "" or not (0 <= contact_id < len(simulation.contact_roster)):
            row["target_was_bird"] = ""
            continue
        kind, _idx = simulation.contact_roster[contact_id]
        row["target_was_bird"] = kind == "bird"


def run_single_game(
    game_idx,
    seed,
    num_reds,
    num_blues,
    sim_steps,
    horizon,
    model,
    planning_interval_seconds,
    num_bird_slots,
    bird_spawn_prob,
    verbose_games,
):
    blue_planner = ReasoningLoggingPlanner(LLMVisionRolePlanner(model=model), "blue")
    red_planner = ReasoningLoggingPlanner(LLMRedRolePlanner(model=model), "red")

    row = {
        "game_idx": game_idx,
        "seed": seed,
        "num_reds": num_reds,
        "num_blues": num_blues,
        "sim_steps_limit": sim_steps,
        "model": model,
        "planning_interval_seconds": planning_interval_seconds,
        "num_bird_slots": num_bird_slots,
        "bird_spawn_prob": bird_spawn_prob,
        "error_message": "",
    }

    try:
        simulation = Simulation(
            solver_mode="decentralized",
            num_blues=num_blues,
            num_reds=num_reds,
            sim_steps=sim_steps,
            horizon=horizon,
            seed=seed,
            enable_blue_blue_cbf=True,
            enable_blue_red_cbf=True,
            role_planner=blue_planner,
            red_role_planner=red_planner,
            planning_interval_seconds=planning_interval_seconds,
            num_bird_slots=num_bird_slots,
            bird_spawn_prob=bird_spawn_prob,
        )
        if verbose_games:
            simulation.run()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                simulation.run()

        row.update(
            {
                "num_birds_spawned": simulation.num_birds,
                "num_bird_engagements": simulation.num_bird_engagements,
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
                    "num_reds": num_reds,
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
                "num_birds_spawned": None,
                "num_bird_engagements": None,
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
        print(f"[game_idx={game_idx} seed={seed} num_reds={num_reds}] FAILED: {exc}")
        traceback.print_exc()
        reasoning_rows = [
            {
                "game_idx": game_idx,
                "seed": seed,
                "num_reds": num_reds,
                "num_blues": num_blues,
                "model": model,
                "planning_interval_seconds": planning_interval_seconds,
                "target_was_bird": "",
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


def save_csv(rows, fieldnames, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Experiment II: vision-augmented blue planner. Reds and decoy birds "
            "are shown to blue as unlabeled image-attached contacts; blue must "
            "also assign each engagement role a specific target. Sweeps red-team "
            "size x seeds for one (model, planning-interval) combination per run."
        )
    )
    parser.add_argument("--red-counts", type=str, default=None, help="Default: 1,2,3,4")
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
    parser.add_argument("--num-bird-slots", type=int, default=DEFAULT_NUM_BIRD_SLOTS)
    parser.add_argument("--bird-spawn-prob", type=float, default=DEFAULT_BIRD_SPAWN_PROB)
    parser.add_argument("--output-results-csv", type=str, default=None)
    parser.add_argument("--output-reasoning-csv", type=str, default=None)
    parser.add_argument("--verbose-games", action="store_true")
    args = parser.parse_args()

    red_counts = parse_int_list(args.red_counts, DEFAULT_RED_COUNTS)
    model_slug = args.model.replace("/", "_").replace(":", "_")
    stem = f"vision_experiment_{args.n_seeds}seeds_{model_slug}_{args.planning_interval_seconds:g}s"
    output_results_csv = args.output_results_csv or f"{stem}_results.csv"
    output_reasoning_csv = args.output_reasoning_csv or f"{stem}_reasoning.csv"

    result_rows = []
    reasoning_rows = []
    game_idx = 0
    seed = args.base_seed
    for num_reds in red_counts:
        for local_idx in range(args.n_seeds):
            print(
                f"Running game_idx={game_idx} (num_reds={num_reds}, num_blues={args.num_blues}, "
                f"model={args.model}, interval={args.planning_interval_seconds}s, "
                f"seed={seed}, run {local_idx + 1}/{args.n_seeds}) ..."
            )
            row, game_reasoning_rows = run_single_game(
                game_idx=game_idx,
                seed=seed,
                num_reds=num_reds,
                num_blues=args.num_blues,
                sim_steps=args.sim_steps,
                horizon=args.horizon,
                model=args.model,
                planning_interval_seconds=args.planning_interval_seconds,
                num_bird_slots=args.num_bird_slots,
                bird_spawn_prob=args.bird_spawn_prob,
                verbose_games=args.verbose_games,
            )
            result_rows.append(row)
            reasoning_rows.extend(game_reasoning_rows)
            print(
                f"  -> outcome={row['outcome']} steps={row['steps_completed']} "
                f"bird_engagements={row.get('num_bird_engagements')}"
            )
            game_idx += 1
            seed += 1

    save_csv(result_rows, RESULT_FIELDNAMES, output_results_csv)
    save_csv(reasoning_rows, REASONING_FIELDNAMES, output_reasoning_csv)

    print(f"\nCompleted vision-experiment sweep over {red_counts} x {args.n_seeds} seeds each.")
    print(f"Per-game results CSV: {output_results_csv}")
    print(f"Planner reasoning CSV: {output_reasoning_csv}")
    print("Outcome counts:")
    for outcome, count in sorted(Counter(row["outcome"] for row in result_rows).items()):
        print(f"  {outcome}: {count}")


if __name__ == "__main__":
    main()
