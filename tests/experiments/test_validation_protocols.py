import collections
import json
import math
import unittest
from copy import deepcopy

import numpy as np

from common.io.json import fingerprint
from common.statistics import bootstrap_weights, mean_interval
from experiments.paper_theory import protocols as definitions
from experiments.paper_theory.designs import generate
from experiments.paper_theory.protocols import ANALYSIS
from experiments.paper_theory.runtime.tasks import request, task
from theory.release import batch_half_times, clocks, confirmed_clocks, curve_clock_interval, energy_tail_check
from theory.runtime.store import resolve_frozen_plan


class TestScientificChecks(unittest.TestCase):
    def frozen_plan_fixture(self):
        candidate = {
            "schema": 1,
            "profile": "core",
            "jobs": [generate(preset="ensembles", smoke=True)[0]],
            "analysis_protocol": ANALYSIS,
            "definitions_sha256": {"parameters.py": "original-file-hash"},
        }
        frozen = json.loads(json.dumps({**candidate, "plan_fingerprint": fingerprint(candidate)}))
        return candidate, frozen

    def test_definition_byte_change_keeps_frozen_identity(self):
        candidate, frozen = self.frozen_plan_fixture()
        candidate["definitions_sha256"]["parameters.py"] = "formatted-file-hash"
        snapshot = deepcopy(frozen)
        selected, identity, changed = resolve_frozen_plan(candidate, frozen)
        self.assertTrue(changed)
        self.assertEqual(identity, frozen["plan_fingerprint"])
        self.assertEqual(selected["definitions_sha256"], frozen["definitions_sha256"])
        self.assertEqual(frozen, snapshot)
        self.assertEqual(fingerprint(selected), identity)

    def test_real_design_or_analysis_changes_still_rejected(self):
        for field in ("seed", "request", "profile", "analysis"):
            candidate, frozen = self.frozen_plan_fixture()
            candidate["definitions_sha256"]["parameters.py"] = "new-file-hash"
            if field == "seed":
                candidate["jobs"][0]["payload"]["process_seed"] += 1
            elif field == "request":
                candidate["jobs"][0]["payload"]["request"]["population"] += 1
            elif field == "profile":
                candidate["profile"] = "minimum"
            else:
                candidate["analysis_protocol"] = {**ANALYSIS, "confidence_level": 0.9}
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "plan changed"):
                resolve_frozen_plan(candidate, frozen)

    def test_corrupted_frozen_plan_rejected(self):
        candidate, frozen = self.frozen_plan_fixture()
        frozen["jobs"][0]["payload"]["process_seed"] += 1
        with self.assertRaisesRegex(RuntimeError, "content/hash mismatch"):
            resolve_frozen_plan(candidate, frozen)

    def test_design_and_conditional_repeats(self):
        expected = {"minimum": (94, 62), "core": (236, 110), "extended": (240, 130)}
        for profile, counts in expected.items():
            jobs = generate(preset="ensembles", profile=profile)
            actual = collections.Counter(j["family"] for j in jobs)
            self.assertEqual((actual["micro"], actual["clocks"]), counts)
            self.assertEqual(jobs, generate(preset="ensembles", profile=profile))
            self.assertEqual(len(jobs), len({j["id"] for j in jobs}))
            physical = []
            for job in jobs:
                payload = deepcopy(job["payload"])
                payload.pop("id")
                payload["request"]["request_id"] = "revision"
                physical.append(fingerprint(payload))
            self.assertEqual(len(physical), len(set(physical)))
        jobs = generate(preset="ensembles")
        for initial in definitions.INITIAL_REPLICATES:
            cohort = [
                j
                for j in jobs
                if j["family"] == "micro"
                and j["meta"]["role"] in ("basin_grid", "basin_ensemble")
                and j["meta"]["b"] == 161
                and j["meta"]["quadrature"] == 33
                and j["meta"]["initial_replicate"] == initial
            ]
            self.assertEqual(len(cohort), 16)
            self.assertEqual(len({j["payload"]["init_seed"] for j in cohort}), 1)
            self.assertEqual(len({j["payload"]["process_seed"] for j in cohort}), 16)
            original = task("micro", request(horizon=30), kind="micro", replicate=initial)
            first = next(j for j in cohort if j["meta"]["process_replicate"] == 0)
            self.assertEqual(first["payload"]["process_seed"], original["payload"]["process_seed"])

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
