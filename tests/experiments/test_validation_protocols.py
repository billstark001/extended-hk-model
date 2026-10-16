import math
import unittest

import numpy as np

from common.statistics import bootstrap_weights, mean_interval
from theory.release import batch_half_times, clocks, confirmed_clocks, curve_clock_interval, energy_tail_check

ANALYSIS = {
    "bootstrap_samples": 4000,
    "bootstrap_seed": 2026100701,
    "confidence_level": 0.95,
    "tail_fractions": (0.1, 0.2),
}


class TestScientificChecks(unittest.TestCase):
    def test_plateau_requires_both_channels_and_patience(self):
        stopping = {
            "check_every": 2,
            "minimum_steps": 6,
            "patience_steps": 4,
            "energy_absolute_tolerance": 0.0,
            "energy_relative_tolerance": 0.0,
        }
        values = (1.0, 0.5, 0.25, 0.125, 0.125, 0.125, 0.125, 0.125)
        frames = [{"step": 2 * i, "node_energy": e, "edge_energy": e} for i, e in enumerate(values)]
        self.assertFalse(energy_tail_check(frames[:5], stopping)["energy_confirmed"])
        self.assertTrue(energy_tail_check(frames[:6], stopping)["energy_confirmed"])
        frames[6]["edge_energy"] += 1e-4
        self.assertFalse(energy_tail_check(frames, stopping)["energy_confirmed"])
        with self.assertRaisesRegex(ValueError, "missing energy checkpoint"):
            energy_tail_check(frames[:4] + frames[5:], stopping)

    def test_vector_clocks_match_scalar_first_crossings(self):
        times = np.linspace(0, 20, 101)
        curves = np.asarray(
            [
                np.exp(-times),
                np.exp(-3 * times),
                np.exp(-times) + 0.2 * np.sin(2 * times) ** 2,
                np.ones(len(times)),
                np.linspace(0, 1, len(times)),
            ]
        )
        vector = batch_half_times(times, curves, 0.1)
        for i, curve in enumerate(curves):
            frames = [{"time": float(t), "node_energy": float(e)} for t, e in zip(times, curve)]
            scalar = clocks(frames, False, "node_energy")["T50"]
            if scalar is None:
                self.assertTrue(np.isnan(vector[i]))
            else:
                self.assertAlmostEqual(vector[i], scalar, places=12)
        protocol = ANALYSIS
        weights = bootstrap_weights(2, protocol)
        ci, _, _ = curve_clock_interval(times, curves[:2], weights, protocol)
        self.assertGreater(abs(vector[:2].mean() - ci["T50_of_mean_curve"]), 0.02)
        self.assertLessEqual(ci["lower"], ci["T50_of_mean_curve"])
        self.assertGreaterEqual(ci["upper"], ci["T50_of_mean_curve"])

    def test_no_false_precision_for_one_or_missing_clock(self):
        weights = bootstrap_weights(1, ANALYSIS)
        stat = mean_interval([0.4], weights, 0.95)
        self.assertEqual(stat["mean"], 0.4)
        self.assertIsNone(stat["lower"])
        stat = mean_interval([0.4, np.nan], bootstrap_weights(2, ANALYSIS), 0.95)
        self.assertEqual(stat["valid_processes"], 1)
        self.assertIsNone(stat["upper"])

    def test_roundoff_does_not_create_clock_order(self):
        frames = [
            {
                "step": t,
                "time": float(t),
                "node_energy": math.exp(-t) if t < 10 else 0.0,
                "edge_energy": math.exp(-(1 + 1e-12) * t) if t < 10 else 0.0,
            }
            for t in range(21)
        ]
        stop = {
            "check_every": 1,
            "minimum_steps": 1,
            "patience_steps": 3,
            "energy_absolute_tolerance": 0.0,
            "energy_relative_tolerance": 0.0,
        }
        output = confirmed_clocks({"frames": frames}, stop, 1e-9)
        self.assertTrue(output["energy_confirmed"])
        self.assertEqual([output[f"delta{p}"] for p in (10, 50, 90)], [0.0, 0.0, 0.0])
