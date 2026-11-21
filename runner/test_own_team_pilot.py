"""Pilot protocol tests using the pinned tokenizer and synthetic scores and activations."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from runner import own_team_pilot as pilot
from runner.prompts import load_tokenizer
from runner.test_runner import FakeTok, tiny_model


class TestPilotInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = pilot.package_files()
        cls.digest = pilot.sha256(pilot.PACKAGE / "manifest.json")
        cls.groups = pilot.frozen_prompts(cls.files, cls.digest)

    def test_grid_differs_only_in_team_or_labels(self):
        tok = load_tokenizer()
        for prompts in self.groups.values():
            self.assertEqual(len(pilot.encode_rows(tok, prompts)), 96)
            for own_a, neutral_a, own_b, neutral_b in zip(*[iter(prompts)] * 4, strict=True):
                for own, neutral in ((own_a, neutral_a), (own_b, neutral_b)):
                    self.assertEqual(own["marked"].split("\n", 1)[1], neutral["marked"].split("\n", 1)[1])
                    self.assertEqual(own["referee_team"], own["violating_team"])
                    self.assertNotEqual(neutral["referee_team"], own["violating_team"])
                for a, b in ((own_a, own_b), (neutral_a, neutral_b)):
                    swapped = a["marked"].replace("A. Waive the warning\nB. Issue the warning",
                                                  "A. Issue the warning\nB. Waive the warning")
                    self.assertEqual(swapped, b["marked"])

    def test_rejects_hash_change_and_duplicate_rule(self):
        files = dict(self.files)
        files["fresh_cases.json"] += b" "
        with self.assertRaisesRegex(ValueError, "changed"):
            pilot.frozen_prompts(files, self.digest)
        cases = {split: json.loads(files[f"{split}_cases.json"]) for split in pilot.SPLITS}
        cases["fresh"][0]["rule"] = cases["pilot"][0]["rule"]
        with self.assertRaisesRegex(ValueError, "duplicate rule"):
            pilot.render(cases)


class TestPilotAnalysis(unittest.TestCase):
    def test_score_sign_label_swap_and_raw_mass(self):
        expected = [{"case_id": str(i), "waive_label": label, "affiliation": affiliation}
                    for i in range(24) for label in pilot.LABELS for affiliation in pilot.AFFILIATIONS]
        logprobs = np.log(np.tile([[0.15, 0.05], [0.04, 0.16], [0.05, 0.15], [0.16, 0.04]], (24, 1)))
        result, gaps = pilot.behavior(expected, logprobs)
        np.testing.assert_allclose(gaps, 0.55)
        self.assertAlmostEqual(result["waiver_means"]["own"], 0.75)
        self.assertAlmostEqual(result["waiver_means"]["neutral"], 0.20)
        self.assertAlmostEqual(result["mean_label_mass"], 0.20)
        self.assertTrue(result["gate"]["passed"])
        self.assertTrue(pilot.parity_error(logprobs, logprobs + 9e-6, expected)["passed"])
        self.assertFalse(pilot.parity_error(logprobs, logprobs + 2e-5, expected)["passed"])
        shifted = logprobs.copy()
        shifted[0, 0] += 9e-6
        self.assertFalse(pilot.parity_error(logprobs, shifted, expected)["passed"])

    def test_gate_thresholds_inclusive_except_positive_sign(self):
        self.assertTrue(pilot.pilot_gate(np.full((24, 2), 0.05))["passed"])
        observations = np.array([[0.025, 0.125]] * 24)
        self.assertTrue(pilot.pilot_gate(observations)["passed"])
        observations[:, 0] -= 1e-8
        self.assertFalse(pilot.pilot_gate(observations)["checks"]["waive_A_gap"]["passed"])
        observations = np.array([[0.125, 0.125]] * 16 + [[0, 0]] * 8)
        self.assertTrue(pilot.pilot_gate(observations)["passed"])
        observations[15] = 0
        self.assertFalse(pilot.pilot_gate(observations)["checks"]["positive_cases"]["passed"])
        self.assertFalse(pilot.pilot_gate(np.full((24, 2), 0.05 - 1e-8))["checks"]["mean_gap"]["passed"])

    def test_rank_polarity_constants_and_ties(self):
        x = np.arange(1, 25, dtype=np.float32)
        gaps = np.repeat(x[:, None], 2, axis=1)
        grid = np.zeros((24, 2, 2, 2, 5), dtype=np.float32)
        for layer in range(2):
            grid[:, 0, 0, layer, :2] = x[:, None]
            grid[:, 1, 0, layer, :2] = x[:, None]
            grid[:, :, 0, layer, 2] = -x[:, None]
            grid[:, :, 0, layer, 3] = 5
            grid[:, 0, 0, layer, 4] = x
            grid[:, 1, 0, layer, 4] = x[::-1]
        ranked = pilot.rank_sides(grid.reshape(96, 2, 5), gaps)
        self.assertEqual([r[1:4] for r in ranked],
                         [(layer, neuron, side) for layer in range(2)
                          for neuron, side in ((0, "+"), (1, "+"), (2, "-"))])
        self.assertEqual(pilot.rank_sides(grid.reshape(96, 2, 5), np.ones((24, 2))), [])

        # Within each pair of equal gaps, one case has a positive activation and one a negative.
        alternating = np.arange(24) % 2 == 0
        paired = np.tile(np.repeat([2, 4, 6, 8], 2), 3)
        grid = np.zeros((24, 2, 2, 1, 1), dtype=np.float32)
        grid[:, :, 0, 0, 0] = np.where(alternating, paired, -paired)[:, None]
        ranked = pilot.rank_sides(grid.reshape(96, 1, 1), np.repeat(paired[:, None], 2, axis=1))
        self.assertEqual([r[3] for r in ranked], ["+", "-"])
        self.assertEqual(ranked[0][0], ranked[1][0])

    def test_analyze_rejects_missing_rows_and_never_ranks_invalid_or_failed_gate(self):
        expected = [{"case_id": str(i), "waive_label": label}
                    for i in range(24) for label in pilot.LABELS for _ in pilot.AFFILIATIONS]
        rows = [dict(row, logprobs_unhooked=[-2.0, -2.0], logprobs_hooked=[-2.0, -2.0])
                for row in expected]
        with self.assertRaisesRegex(ValueError, "96 score rows"):
            pilot.verify_score_rows(rows[:-1], expected)
        with self.assertRaisesRegex(ValueError, "differs"):
            pilot.verify_score_rows(list(reversed(rows)), expected)
        with tempfile.TemporaryDirectory() as directory, patch.object(pilot, "LAYERS", 2), \
                patch.object(pilot, "NEURONS", 3), patch.object(pilot, "rank_sides") as rank:
            path = Path(directory)
            np.save(path / "activations.npy", np.zeros((96, 2, 3), dtype=np.float32))
            (path / "run.json").write_text("{}")
            (path / "scores.jsonl").write_text(pilot.jsonl_text(rows))
            self.assertEqual(pilot.analyze(path, expected, path)["status"], "gate_failed")
            rows[0]["logprobs_hooked"][0] += 0.1
            (path / "scores.jsonl").write_text(pilot.jsonl_text(rows))
            self.assertEqual(pilot.analyze(path, expected, path)["status"], "invalid_parity")
            rank.assert_not_called()


class TestFixturePath(unittest.TestCase):
    def test_fixture_checks_run_on_tiny_model(self):
        from runner import own_team_pilot_modal as modal
        from runner.model import Marin

        m = Marin(tiny_model(), FakeTok(), stats=None)
        encoded = [{"token_ids": [1, 9, 12, 7], "cue_index": 3},
                   {"token_ids": [1, 3, 5, 9, 12, 15], "cue_index": 5}]
        with patch.object(modal, "LAYERS", 3), patch.object(modal, "NEURONS", 48), \
                patch.object(modal, "ANSWER_IDS", (2, 9)), patch.object(modal, "encode_rows", return_value=encoded):
            report = modal.fixture_checks(m)
        self.assertTrue(report["capture_parity"]["passed"])
        self.assertTrue(report["noop_parity"]["passed"])
        self.assertTrue(report["single_coordinate_zeroing"])
        self.assertTrue(report["hooks_removed"])


if __name__ == "__main__":
    unittest.main()
