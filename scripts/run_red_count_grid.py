import argparse
import contextlib
import csv
import io
import os
import traceback
from collections import Counter

from planning.planner import (
    LLMRedRolePlanner,
    LLMRolePlanner,
    RuleBasedRedRolePlanner,
    RuleBasedBlueRolePlanner,
)
from simulation.testbed import Simulation


DEFAULT_RED_COUNTS = [1, 2, 3, 4]
DEFAULT_NUM_BLUES = 4
DEFAULT_N_SEEDS = 10
DEFAULT_SIM_STEPS = 600
DEFAULT_HORIZON = 20
DEFAULT_MODEL = "deepseek/deepseek-chat"
DEFAULT_PLANNING_INTERVAL_SECONDS = 10.0

RESULT_FIELDNAMES = [
    "game_idx",
    "seed",
    "num_reds",
    "num_blues",
    "sim_steps_limit",
    "planner_type",
    "model",
    "planning_interval_seconds",
    "blue_v_max",
    "blue_a_max",
    "red_v_max",
    "red_a_max",
    "outcome",
    "steps_completed",
    "num_neutralized",
    "num_escaped",
    "num_red_active_at_end",
    "red_outcomes",
    "blue_score",
    "blue_planner_fallback_count",
    "red_planner_fallback_count",
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
    "reasoning",
]


class ReasoningLoggingPlanner:
    """Wraps a BlueRolePlanner/RedRolePlanner and records every (step, agent,
    role, reasoning) tuple it produces, without changing its behavior. Works
    with both LLM-backed planners (which populate last_reasoning) and the
    rule-based stubs (which don't -- reasoning is just logged empty for those).
    """

    def __init__(self, inner, team):
        self.inner = inner
        self.team = team
        self.log = []

    def plan(self, obs):
        roles = self.inner.plan(obs)
        reasoning = getattr(self.inner, "last_reasoning", {})
        for agent_id, role in roles.items():
            self.log.append(
                {
                    "sim_step": obs.sim_step,
                    "team": self.team,
                    "agent_id": agent_id,
                    "role": role.value,
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


def build_planners(planner_type, model):
    if planner_type == "llm":
        blue_planner = LLMRolePlanner(model=model)
        red_planner = LLMRedRolePlanner(model=model)
    else:
        blue_planner = RuleBasedBlueRolePlanner()
        red_planner = RuleBasedRedRolePlanner()
    return ReasoningLoggingPlanner(blue_planner, "blue"), ReasoningLoggingPlanner(
        red_planner, "red"
    )


def run_single_game(
    game_idx,
    seed,
    num_reds,
    num_blues,
    sim_steps,
    horizon,
    planner_type,
    model,
    planning_interval_seconds,
    verbose_games,
    blue_v_max=2.0,
    blue_a_max=2.0,
    red_v_max=2.0,
    red_a_max=2.0,
):
    role_planner, red_role_planner = build_planners(planner_type, model)

    row = {
        "game_idx": game_idx,
        "seed": seed,
        "num_reds": num_reds,
        "num_blues": num_blues,
        "sim_steps_limit": sim_steps,
        "planner_type": planner_type,
        "model": model if planner_type == "llm" else "",
        "planning_interval_seconds": planning_interval_seconds,
        "blue_v_max": blue_v_max,
        "blue_a_max": blue_a_max,
        "red_v_max": red_v_max,
        "red_a_max": red_a_max,
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
            role_planner=role_planner,
            red_role_planner=red_role_planner,
            planning_interval_seconds=planning_interval_seconds,
            blue_v_max=blue_v_max,
            blue_a_max=blue_a_max,
            red_v_max=red_v_max,
            red_a_max=red_a_max,
        )
        if verbose_games:
            simulation.run()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                simulation.run()

        row.update(
            {
                "outcome": determine_outcome(simulation),
                "steps_completed": max(len(simulation.history_red) - 1, 0),
                "num_neutralized": simulation.num_neutralized,
                "num_escaped": simulation.num_escaped,
                "num_red_active_at_end": simulation.red_outcomes.count("active"),
                "red_outcomes": ",".join(simulation.red_outcomes),
                "blue_score": simulation.blue_score,
                "blue_planner_fallback_count": len(
                    getattr(role_planner.inner, "fallback_events", [])
                ),
                "red_planner_fallback_count": len(
                    getattr(red_role_planner.inner, "fallback_events", [])
                ),
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
        for entry in role_planner.log + red_role_planner.log:
            reasoning_rows.append(
                {
                    "game_idx": game_idx,
                    "seed": seed,
                    "num_reds": num_reds,
                    "num_blues": num_blues,
                    "model": model if planner_type == "llm" else "",
                    "planning_interval_seconds": planning_interval_seconds,
                    **entry,
                }
            )
        reasoning_rows.sort(key=lambda r: (r["sim_step"], r["team"], r["agent_id"]))
        return row, reasoning_rows

    except Exception as exc:  # noqa: BLE001 -- batch sweep must not die on one game
        row.update(
            {
                "outcome": "error",
                "steps_completed": None,
                "num_neutralized": None,
                "num_escaped": None,
                "num_red_active_at_end": None,
                "red_outcomes": "",
                "blue_score": None,
                "blue_planner_fallback_count": len(
                    getattr(role_planner.inner, "fallback_events", [])
                ),
                "red_planner_fallback_count": len(
                    getattr(red_role_planner.inner, "fallback_events", [])
                ),
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
        reasoning_rows = []
        for entry in role_planner.log + red_role_planner.log:
            reasoning_rows.append(
                {
                    "game_idx": game_idx,
                    "seed": seed,
                    "num_reds": num_reds,
                    "num_blues": num_blues,
                    "model": model if planner_type == "llm" else "",
                    "planning_interval_seconds": planning_interval_seconds,
                    **entry,
                }
            )
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
            "Sweep the decentralized pursuit-evasion sim over red-team size "
            "(1-4 reds, fixed blue count) across multiple random seeds, "
            "recording per-game outcomes and every planner reasoning string."
        )
    )
    parser.add_argument(
        "--red-counts",
        type=str,
        default=None,
        help="Comma-separated red-team sizes. Default: 1,2,3,4",
    )
    parser.add_argument("--num-blues", type=int, default=DEFAULT_NUM_BLUES)
    parser.add_argument(
        "--n-seeds", type=int, default=DEFAULT_N_SEEDS, help="Seeds per red-count config."
    )
    parser.add_argument(
        "--base-seed", type=int, default=1000, help="First seed; each game gets a distinct seed."
    )
    parser.add_argument("--sim-steps", type=int, default=DEFAULT_SIM_STEPS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--planner",
        type=str,
        choices=["llm", "rule"],
        default="llm",
        help="'llm' uses OpenRouter-backed planners (requires OPENROUTER_API_KEY) "
        "and produces real reasoning text; 'rule' uses the free rule-based stubs "
        "(reasoning column will be empty).",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenRouter model slug.")
    parser.add_argument(
        "--planning-interval-seconds",
        type=float,
        default=DEFAULT_PLANNING_INTERVAL_SECONDS,
        help="How often (in simulated seconds) the role planners replan. Default: 10.0",
    )
    parser.add_argument("--output-results-csv", type=str, default=None)
    parser.add_argument("--output-reasoning-csv", type=str, default=None)
    parser.add_argument(
        "--verbose-games", action="store_true", help="Print each game's internal simulation logs."
    )
    args = parser.parse_args()

    red_counts = parse_int_list(args.red_counts, DEFAULT_RED_COUNTS)
    stem = f"red_count_grid_{args.n_seeds}seeds_{args.planner}"
    if args.planner == "llm":
        model_slug = args.model.replace("/", "_").replace(":", "_")
        interval_slug = f"{args.planning_interval_seconds:g}s"
        stem += f"_{model_slug}_{interval_slug}"
    output_results_csv = args.output_results_csv or f"{stem}_results.csv"
    output_reasoning_csv = args.output_reasoning_csv or f"{stem}_reasoning.csv"

    result_rows = []
    reasoning_rows = []
    game_idx = 0
    seed = args.base_seed
    for num_reds in red_counts:
        for local_idx in range(args.n_seeds):
            print(
                f"Running game_idx={game_idx} (num_reds={num_reds}, "
                f"num_blues={args.num_blues}, seed={seed}, run {local_idx + 1}/{args.n_seeds}) ..."
            )
            row, game_reasoning_rows = run_single_game(
                game_idx=game_idx,
                seed=seed,
                num_reds=num_reds,
                num_blues=args.num_blues,
                sim_steps=args.sim_steps,
                horizon=args.horizon,
                planner_type=args.planner,
                model=args.model,
                planning_interval_seconds=args.planning_interval_seconds,
                verbose_games=args.verbose_games,
            )
            result_rows.append(row)
            reasoning_rows.extend(game_reasoning_rows)
            print(f"  -> outcome={row['outcome']} steps={row['steps_completed']}")
            game_idx += 1
            seed += 1

    save_csv(result_rows, RESULT_FIELDNAMES, output_results_csv)
    save_csv(reasoning_rows, REASONING_FIELDNAMES, output_reasoning_csv)

    print(f"\nCompleted red-count sweep over {red_counts} x {args.n_seeds} seeds each.")
    print(f"Per-game results CSV: {output_results_csv}")
    print(f"Planner reasoning CSV: {output_reasoning_csv}")
    print("Outcome counts:")
    for outcome, count in sorted(Counter(row["outcome"] for row in result_rows).items()):
        print(f"  {outcome}: {count}")


if __name__ == "__main__":
    main()
