"""Coarse recency-based coverage tracking, used to steer RECON assignments
toward genuinely under-observed ground instead of picking a target region
blind (or, in ReconController's default mode, uniformly at random).

Deliberately not a full occupancy grid (no belief/probability update, no
false positives/negatives) -- this sim's sensor model is already perfect
detection within a conic FOV (see sensing.py), so "steps since any blue's
sensor last swept this cell" is the whole story: it's a direct proxy for how
long a red could have been sitting undetected there.
"""

import numpy as np

import sensing


class CoverageTracker:
    """Tracks last-seen sim_step for each cell of a small, fixed grid_dim x
    grid_dim grid over the arena.

    grid_dim is kept small (default 4x4 = 16 cells) so region_id stays a
    stable, bounded identifier -- suitable for a rule-based heuristic now,
    and for an LLM schema field (e.g. a future target_region_id) later,
    rather than a per-pixel raster no planner could reason over.
    """

    def __init__(self, arena_size, grid_dim=4):
        self.arena_size = float(arena_size)
        self.grid_dim = int(grid_dim)
        cell = self.arena_size / self.grid_dim
        self.cell_size = cell
        self.centers = np.array(
            [
                [(i + 0.5) * cell, (j + 0.5) * cell]
                for i in range(self.grid_dim)
                for j in range(self.grid_dim)
            ]
        )
        self.last_seen_step = np.full(len(self.centers), -1, dtype=int)

    def update(self, sim_step, blue_positions, blue_bearings, half_angle_rad, max_range):
        """blue_positions: iterable of (agent_id, pos_2) pairs for active
        blues; blue_bearings: {agent_id: radians}. Mirrors the observer loop
        shape in Simulation._compute_detected_indices, just testing grid
        cell centers instead of opposing-team agent positions."""
        for cell_idx, center in enumerate(self.centers):
            for agent_id, pos in blue_positions:
                if sensing.is_detected(pos, blue_bearings[agent_id], center, half_angle_rad, max_range):
                    self.last_seen_step[cell_idx] = sim_step
                    break

    def summarize(self, sim_step):
        """Returns [{"region_id", "center", "steps_since_seen"}, ...] --
        steps_since_seen is -1 for a cell no blue sensor has ever swept."""
        regions = []
        for cell_idx, center in enumerate(self.centers):
            last_seen = int(self.last_seen_step[cell_idx])
            steps_since_seen = -1 if last_seen < 0 else int(sim_step) - last_seen
            regions.append(
                {
                    "region_id": cell_idx,
                    "center": center,
                    "steps_since_seen": steps_since_seen,
                }
            )
        return regions

    def region_center(self, region_id):
        return self.centers[region_id]

    def stalest_region_id(self, sim_step):
        """Region id with the largest steps_since_seen (ties broken by lowest
        region_id); never-seen cells (-1) outrank any finite value."""
        never_seen = np.where(self.last_seen_step < 0)[0]
        if len(never_seen) > 0:
            return int(never_seen[0])
        return int(np.argmin(self.last_seen_step))
