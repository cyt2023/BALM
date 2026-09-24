"""Small causal and evaluation-window checks for the hard-capacity replay."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import sim
from capacity_sim import simulate_capacity


def trace_from_counts(counts, footprints):
    counts = sparse.csc_matrix(np.asarray(counts, dtype=np.int32))
    n_apps, n_minutes = counts.shape
    return {
        "counts": counts,
        "mem_mb": np.asarray(footprints, dtype=np.float32),
        "mean_ms": np.full(n_apps, 100.0, dtype=np.float32),
        "mean_s": np.full(n_apps, 0.1, dtype=np.float32),
        "q_s": np.full((n_apps, 1), 0.1, dtype=np.float32),
        "q_w": np.array([1.0], dtype=np.float32),
        "n_apps": n_apps,
        "n_minutes": n_minutes,
    }


class WarmFirstTwo(sim.BasePolicy):
    name = "warm-first-two"

    def replan(self, t, active_apps, counts):
        self.warm_until[:] = t - 1
        self.warm_until[:2] = t


class CapacityProtocolTests(unittest.TestCase):
    def test_eviction_cannot_prefer_an_app_known_to_be_inactive_later(self):
        # A and B are warm and equally old. C arrives first, and A arrives
        # later. An online LRU eviction removes A (id 0); a lookahead that
        # knows A will be requested would incorrectly remove B instead.
        trace = trace_from_counts([[1], [0], [1_000_000]], [60, 60, 60])
        result = simulate_capacity(
            trace, WarmFirstTwo(3, trace["mem_mb"]),
            total_capacity_mb=120, seed=0,
        )
        self.assertEqual(result["cold_events"], 2)
        self.assertEqual(result["capacity_evictions"], 2)
        self.assertEqual(result["served_requests"], 1_000_001)
        self.assertEqual(result["capacity_violations"], 0)

    def test_burn_in_rejection_is_not_an_evaluation_rejection(self):
        # The first application cannot fit in the capacity during burn-in;
        # only the second minute's request belongs in evaluation metrics.
        trace = trace_from_counts([[1, 0], [0, 1]], [150, 50])
        result = simulate_capacity(
            trace, sim.BasePolicy(2, trace["mem_mb"]),
            total_capacity_mb=100, seed=0, eval_start=1,
        )
        self.assertEqual(result["served_requests"], 1)
        self.assertEqual(result["rejected_requests"], 0)
        self.assertEqual(result["service_completion_rate"], 1)
        self.assertEqual(result["cold_events"], 1)
        self.assertEqual(result["capacity_violations"], 0)


if __name__ == "__main__":
    unittest.main()
