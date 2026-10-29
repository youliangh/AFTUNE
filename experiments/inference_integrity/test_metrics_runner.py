"""Focused tests for integrity verdicts and audit sampling."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from experiments.inference_integrity.config import ExperimentConfig
from experiments.inference_integrity.metrics import (
    calibrate_threshold,
    decide_sequence,
    exact_hit_probability,
    sampled_block_ids,
    simulate_sampling,
    summarize_cases,
    summarize_sampling,
)
from experiments.inference_integrity.runner import _condition_kind, _dataset_rows, _injection_spec, _positive_indices


class VerdictTests(unittest.TestCase):
    def test_calibration_uses_sequence_maximum_and_rejects_bad_controls(self):
        threshold = calibrate_threshold([
            [{"overall_relative_l2": 0.01}, {"overall_relative_l2": 0.03}],
            [{"overall_relative_l2": 0.02}],
        ])
        self.assertEqual(threshold, 0.03)
        self.assertEqual(decide_sequence([
            {"block_id": 0, "overall_relative_l2": 0.04},
            {"block_id": 1, "overall_relative_l2": 0.001},
        ], threshold)["verdict"], "DETECTED")
        with self.assertRaises(ValueError):
            calibrate_threshold([[{"error": "missing metadata"}]])

    def test_integrity_rejection_is_distinct_from_setup_error(self):
        output = decide_sequence([
            {"block_id": 0, "error": "layer_0: Computed output hash mismatch"},
            {"block_id": 1, "error": "Cannot load metadata"},
        ], 1.0)
        self.assertEqual(output["verdict"], "DETECTED")
        self.assertEqual(output["rejected_blocks"], ["0"])
        self.assertEqual(len(output["error_blocks"]), 1)
        self.assertEqual(decide_sequence([
            {"block_id": 0, "overall_relative_l2": float("nan")}
        ], 0.0)["verdict"], "ERROR")

    def test_unknown_injection_is_error_not_failed_injection(self):
        cases = [
            {"planned_positive": True, "injection_success": None, "verdict": "ERROR"},
            {"planned_positive": True, "injection_success": False, "verdict": "ERROR"},
            {"planned_positive": True, "injection_success": True, "verdict": "PASSED"},
            {"planned_positive": False, "injection_success": True, "verdict": "PASSED"},
        ]
        summary = summarize_cases(cases)
        self.assertEqual(summary["counts"]["FN"], 1)
        self.assertEqual(summary["counts"]["TN"], 1)
        self.assertEqual(summary["counts"]["INJECTION_FAILED"], 1)
        self.assertEqual(summary["counts"]["ERROR_POSITIVE"], 1)
        self.assertEqual(summary["counts"]["INJECTION_STATUS_UNKNOWN"], 1)
        self.assertEqual(summary["successful_positive"], 1)

    def test_sampling_distinguishes_positive_and_negative(self):
        positive = simulate_sampling(
            "positive", [{"overall_relative_l2": 1.0}, {"overall_relative_l2": 0.0}],
            0.5, (0.5,), 100, 1234,
        )[0]
        negative = simulate_sampling(
            "negative", [{"overall_relative_l2": 0.0}] * 2,
            0.5, (0.5,), 100, 4321,
        )[0]
        merged = summarize_sampling([
            {"planned_positive": True, "injection_success": True, "sampling": [positive]},
            {"planned_positive": False, "injection_success": True, "sampling": [negative]},
        ], (0.5,), 100)["0.5"]
        self.assertEqual(merged["counts"]["FP"], 0)
        self.assertEqual(merged["counts"]["TN"], 100)
        self.assertEqual(merged["counts"]["TP"] + merged["counts"]["FN"], 100)
        self.assertEqual(exact_hit_probability(18, 0, 18), 0.0)
        self.assertEqual(exact_hit_probability(18, 1, 18), 1.0)
        self.assertEqual(
            sampled_block_ids("positive", 18, 0.1, 7, 9876),
            sampled_block_ids("positive", 18, 0.1, 7, 9876),
        )


class PreparationTests(unittest.TestCase):
    def test_pinned_dataset_checksum_gate(self):
        with tempfile.TemporaryDirectory() as location:
            root = Path(location)
            source = root / "inputs.jsonl"
            source.write_text(json.dumps({"input": "one two three"}) + "\n", encoding="utf-8")
            checksum = sha256(source.read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps({
                "derived_sha256": checksum, "revision": "test", "rows": 1,
            }), encoding="utf-8")
            config = ExperimentConfig(dataset_jsonl=str(source), dataset_revision="test")
            rows, metadata = _dataset_rows(config)
            self.assertEqual(rows[0]["input"], "one two three")
            self.assertEqual(metadata["sha256"], checksum)
            source.write_text(source.read_text() + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                _dataset_rows(config)

    def test_model_origin_and_positive_assignment(self):
        config = ExperimentConfig()
        self.assertEqual(_condition_kind("model_mismatch"), "model")
        self.assertEqual(len(_positive_indices(config, 0)), config.n)
        self.assertEqual(_positive_indices(config, 0), _positive_indices(config, 0))

    def test_serving_has_only_r6_and_native_block_configuration(self):
        config = ExperimentConfig()
        from experiments.inference_integrity.runner import ALL_CONDITIONS
        self.assertEqual(config.layers_per_block, 2)
        self.assertEqual(config.activation_interval, 4)
        self.assertEqual(config.verification_device, 'cpu')
        self.assertEqual(config.sample_rates, (0.08,))
        self.assertEqual(config.serving_round_bits_sweep, ())
        self.assertEqual(len(ALL_CONDITIONS), 6)
        for index in (0, 63, 255):
            spec = _injection_spec(config, 2, 'serving_config', index)
            self.assertEqual(spec.rounding_bits, 6)
            self.assertTrue(2 <= spec.step <= 96)


if __name__ == "__main__":
    unittest.main()
