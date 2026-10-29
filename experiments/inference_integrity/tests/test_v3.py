"""Real tiny-model checks for sparse live evidence and shared-control views."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.inference_integrity.config import ExperimentConfig
from experiments.inference_integrity import native, runner


class SparseEvidenceTests(unittest.TestCase):
    def test_two_layer_blocks_preserve_live_fault_and_sparse_commitment(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=16, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64,
        )).float().eval()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / 'reference'
            native.prepare_reference(model, reference, 'tiny-llama',
                                     layers_per_block=2, activation_interval=4)
            def fault(_module, _inputs, output):
                changed = output.clone()
                changed[:, -1, 0] += 1
                return changed
            for label, hooks in [('clean', ()), ('fault', [(model.model.layers[3].mlp, fault)])]:
                record = root / label
                native.record_forward(model, torch.tensor([1, 2, 3, 4]), record,
                                      reference, 'tiny-llama', hooks)
                preparation = json.loads((record / 'activation_preparation.json').read_text())
                self.assertTrue(preparation['complete'])
                self.assertEqual(len(preparation['responses']), 6)
                base_outputs = [name for name in preparation['base_files'] if name.endswith('/output.pt')]
                self.assertEqual(len(base_outputs), 3)
                self.assertEqual({int(name.split('_')[1]) for name in base_outputs}, {0, 4, 8})
                paired = native.verify_record_both_modes(record, reference)
                self.assertEqual(len(paired['numeric']), 9)
                self.assertFalse(any(row.get('error') for row in paired['numeric']))
                if label == 'clean':
                    self.assertLess(max(row['overall_relative_l2'] for row in paired['numeric']), 1e-5)
                else:
                    self.assertGreater(paired['numeric'][2]['overall_relative_l2'], 1e-4)
                    self.assertIn('Computed output hash mismatch', paired['strict_hash'][2]['error'])
            original = next(iter(preparation['base_files']))
            with (record / original).open('ab') as stream:
                stream.write(b'corruption')
            self.assertIn('Disk data hash mismatch', native.verify_record(record, reference)[0]['error'])


class SharedControlsTests(unittest.TestCase):
    def test_shared_paths_and_summary_reuse_identical_negative_sampling(self):
        config = ExperimentConfig(m=4, n=1, calibration_size=1, sample_repeats=3)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            positives = runner._positive_indices(config, 0)
            for condition in ('shared_clean', *runner.CONDITIONS):
                for index in range(config.m):
                    positive = index in positives
                    if positive == (condition == 'shared_clean'):
                        continue
                    case = runner._case_dir(root, 0, condition, index)
                    origins = runner.EVIDENCE_ORIGINS if not positive else (runner._condition_origin(condition),)
                    runner._atomic_json(case / 'result.json', {
                        'origins': list(origins), 'stage_seconds': {'generation': 1.0}, 'proof_bytes': 10,
                    })
                    for origin in origins:
                        blocks = [{'block_id': str(i), 'overall_relative_l2': 0.0} for i in range(9)]
                        runner._save_case_result(case / origin, condition=condition, seed=0,
                            index=index, positive=positive, injection_success=True, origin=origin,
                            blocks={'numeric': blocks, 'strict_hash': blocks},
                            thresholds={o: 0.0 for o in runner.EVIDENCE_ORIGINS}, status_error=None,
                            config=config)
            with patch.object(runner, 'prepare_corpus'), patch.object(runner, '_load_thresholds',
                    return_value={o: 0.0 for o in runner.EVIDENCE_ORIGINS}):
                summary = runner.summarize(config, root)
            self.assertEqual(summary['unique_generation_tasks'], 3 + 6)
            self.assertEqual(summary['unique_sequence_evidence'], 3 * 2 + 6)
            self.assertEqual(summary['condition_sample_references'], 4 * 6)
            self.assertEqual(summary['actual_costs']['proof_bytes'], 90)
            d_groups = [g for key, g in summary['groups'].items() if key.endswith('/generation_live')]
            negative = next(i for i in range(config.m) if i not in positives)
            sources = {g['sources'][str(negative)] for g in d_groups}
            self.assertEqual(len(sources), 1)
            self.assertTrue(all(g['sampling'] == d_groups[0]['sampling'] for g in d_groups))


if __name__ == '__main__':
    unittest.main()
