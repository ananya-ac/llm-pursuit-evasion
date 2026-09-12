import argparse
import contextlib
import csv
import io
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from simulation.testbed import Simulation


DEFAULT_BLUE_COUNTS = [2, 4, 6, 8, 10]
DEFAULT_RED_VELOCITIES = [1, 2, 3, 4, 5]
HEATMAP_FIGSIZE = (8, 6)
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
ANNOTATION_FONTSIZE = 15
COLORBAR_FONTSIZE = 16


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


def parse_count_list(raw_value):
    if raw_value is None:
        return DEFAULT_BLUE_COUNTS
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("Blue count list cannot be empty.")
    return values


def parse_velocity_list(raw_value):
    if raw_value is None:
        return DEFAULT_RED_VELOCITIES
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("Red velocity list cannot be empty.")
    return values


def run_single_game(
    game_idx,
    base_seed,
    num_blues,
    red_v_max,
    verbose_games,
    return_simulation=False,
):
    seed = base_seed + game_idx
    simulation = Simulation(
        seed=seed,
        num_blues=int(num_blues),
        blue_v_max=2.0,
        blue_a_max=1.0,
        red_v_max=float(red_v_max),
        red_a_max=1.0,
        enable_blue_blue_cbf=True,
        enable_blue_red_cbf=True,
    )

    if verbose_games:
        simulation.run()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            simulation.run()

    outcome = determine_outcome(simulation)
    steps_completed = max(len(simulation.history_red) - 1, 0)

    row = {
        "game_idx": game_idx,
        "seed": seed,
        "num_blues": num_blues,
        "blue_v_max": 2.0,
        "blue_a_max": 1.0,
        "red_v_max": red_v_max,
        "red_a_max": 1.0,
        "outcome": outcome,
        "steps_completed": steps_completed,
        "num_neutralized": simulation.num_neutralized,
        "num_escaped": simulation.num_escaped,
        "defenders_won": int(simulation.defenders_won),
        "touchdown": int(simulation.touchdown),
        "touchdown_step": simulation.touchdown_step,
        "collided": int(simulation.collided),
        "collision_step": simulation.collision_step,
        "collision_kind": simulation.collision_kind,
        "collision_agents": simulation.collision_agents,
        "solver_failed": int(simulation.solver_failed),
        "solver_failure_step": simulation.solver_failure_step,
        "final_red_x": float(simulation.red_state[0]),
        "final_red_y": float(simulation.red_state[1]),
    }
    if return_simulation:
        return row, simulation
    return row


def build_default_paths(n_games):
    stem = f"blue_count_grid_{n_games}games"
    return (
        f"{stem}.csv",
        f"{stem}_win_count_heatmap.png",
        f"{stem}_win_rate_heatmap.png",
    )


def build_default_gif_dir(n_games):
    return os.path.join("gifs", f"blue_count_grid_{n_games}games")


def save_rows_csv(rows, output_csv):
    fieldnames = [
        "game_idx",
        "seed",
        "num_blues",
        "blue_v_max",
        "blue_a_max",
        "red_v_max",
        "red_a_max",
        "outcome",
        "steps_completed",
        "num_neutralized",
        "num_escaped",
        "defenders_won",
        "touchdown",
        "touchdown_step",
        "collided",
        "collision_step",
        "collision_kind",
        "collision_agents",
        "solver_failed",
        "solver_failure_step",
        "final_red_x",
        "final_red_y",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_win_metrics(rows, blue_counts, red_velocities, n_games):
    win_counts = np.zeros((len(blue_counts), len(red_velocities)), dtype=int)
    win_rates = np.zeros((len(blue_counts), len(red_velocities)), dtype=float)

    for i, count in enumerate(blue_counts):
        for j, e_vel in enumerate(red_velocities):
            wins = sum(
                1
                for row in rows
                if row["num_blues"] == count
                and row["red_v_max"] == e_vel
                and row["outcome"] == "blue_win_all_resolved"
            )
            win_counts[i, j] = wins
            win_rates[i, j] = wins / float(n_games)

    return win_counts, win_rates


def save_heatmap(matrix, blue_counts, red_velocities, title, colorbar_label, fmt, output_png):
    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    image = ax.imshow(matrix, cmap="Reds", origin="upper")

    ax.set_xticks(np.arange(len(red_velocities)))
    ax.set_yticks(np.arange(len(blue_counts)))
    ax.set_xticklabels(red_velocities)
    ax.set_yticklabels(blue_counts)
    ax.set_xlabel("Red Max Velocity", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Number of Blues", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color="black",
                fontsize=ANNOTATION_FONTSIZE,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label, fontsize=COLORBAR_FONTSIZE)
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run a decentralized pursuit-evasion sweep over number of blues and save win-rate heatmaps."
    )
    parser.add_argument("--n-games", type=int, default=3, help="Games per blue count.")
    parser.add_argument(
        "--blue-counts",
        type=str,
        default=None,
        help="Comma-separated blue counts. Default: 4,6,8,10",
    )
    parser.add_argument(
        "--red-velocities",
        type=str,
        default=None,
        help="Comma-separated red max velocities. Default: 1,2,3,4,5",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for the sweep.")
    parser.add_argument("--output-csv", type=str, default=None, help="Per-game CSV output path.")
    parser.add_argument(
        "--output-count-heatmap",
        type=str,
        default=None,
        help="Win-count heatmap output path.",
    )
    parser.add_argument(
        "--output-rate-heatmap",
        type=str,
        default=None,
        help="Win-rate heatmap output path.",
    )
    parser.add_argument(
        "--take-gifs",
        action="store_true",
        help="Save one representative GIF and one solver-failure debug GIF when encountered.",
    )
    parser.add_argument(
        "--gif-dir",
        type=str,
        default=None,
        help="Directory to store GIFs. Default: gifs/blue_count_grid_Ygames",
    )
    parser.add_argument(
        "--verbose-games",
        action="store_true",
        help="Print each game's internal simulation logs.",
    )
    args = parser.parse_args()

    blue_counts = parse_count_list(args.blue_counts)
    red_velocities = parse_velocity_list(args.red_velocities)
    default_csv, default_count_heatmap, default_rate_heatmap = build_default_paths(
        args.n_games
    )
    output_csv = args.output_csv or default_csv
    output_count_heatmap = args.output_count_heatmap or default_count_heatmap
    output_rate_heatmap = args.output_rate_heatmap or default_rate_heatmap
    gif_dir = args.gif_dir or build_default_gif_dir(args.n_games)
    if args.take_gifs:
        os.makedirs(gif_dir, exist_ok=True)

    rows = []
    seed_offset = 0
    saved_solver_failure_gif = False
    saw_solver_failure = False
    for count in blue_counts:
        for e_vel in red_velocities:
            for game_idx in range(args.n_games):
                need_representative_gif = args.take_gifs and game_idx == 0
                need_solver_failure_debug_gif = args.take_gifs and not saved_solver_failure_gif
                need_simulation = need_representative_gif or need_solver_failure_debug_gif
                result = run_single_game(
                    game_idx=game_idx,
                    base_seed=args.seed + seed_offset,
                    num_blues=count,
                    red_v_max=e_vel,
                    verbose_games=args.verbose_games,
                    return_simulation=need_simulation,
                )
                if need_simulation:
                    row, simulation = result
                    if need_representative_gif:
                        gif_name = f"blues_{count}_red_v{e_vel}.gif"
                        gif_path = os.path.join(gif_dir, gif_name)
                        with contextlib.redirect_stdout(io.StringIO()):
                            simulation.generate_animation(video_filename=gif_path)
                    if (
                        need_solver_failure_debug_gif
                        and row["outcome"] == "solver_failure"
                    ):
                        saw_solver_failure = True
                        if len(simulation.history_red) > 1:
                            failure_gif_name = (
                                f"solver_failure_blues_{count}_red_v{e_vel}_game{game_idx}.gif"
                            )
                            failure_gif_path = os.path.join(gif_dir, failure_gif_name)
                            with contextlib.redirect_stdout(io.StringIO()):
                                simulation.generate_animation(video_filename=failure_gif_path)
                            saved_solver_failure_gif = True
                else:
                    row = result
                    if row["outcome"] == "solver_failure":
                        saw_solver_failure = True
                rows.append(row)
            seed_offset += args.n_games

    save_rows_csv(rows, output_csv)
    win_counts, win_rates = build_win_metrics(
        rows, blue_counts, red_velocities, args.n_games
    )
    save_heatmap(
        win_counts,
        blue_counts,
        red_velocities,
        title=f"Win Count by Blue Count and Red Velocity ({args.n_games} Games Each)",
        colorbar_label="Number of Victories",
        fmt="d",
        output_png=output_count_heatmap,
    )
    save_heatmap(
        win_rates,
        blue_counts,
        red_velocities,
        title=f"Win Rate by Blue Count and Red Velocity ({args.n_games} Games Each)",
        colorbar_label="Win Rate",
        fmt=".2f",
        output_png=output_rate_heatmap,
    )

    print("Completed blue-count sweep.")
    print(f"CSV results written to: {output_csv}")
    print(f"Win-count heatmap written to: {output_count_heatmap}")
    print(f"Win-rate heatmap written to: {output_rate_heatmap}")
    if args.take_gifs:
        print(f"Representative GIFs written to: {gif_dir}")
        if saved_solver_failure_gif:
            print("Also wrote one solver-failure debug GIF.")
        elif saw_solver_failure:
            print(
                "Solver-failure games were encountered, but none had enough trajectory history "
                "to render a debug GIF."
            )
        else:
            print("No solver-failure game was encountered, so no solver-failure debug GIF was written.")
    print("Outcome counts:")
    for outcome, count in sorted(Counter(row["outcome"] for row in rows).items()):
        print(f"{outcome}: {count}")


if __name__ == "__main__":
    main()
