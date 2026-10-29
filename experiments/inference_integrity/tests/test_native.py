"""Small CPU integration tests for the native AFTUNE experiment adapter."""

import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.inference_integrity.generation import generate_llama
from experiments.inference_integrity.injections import InjectionSpec
from experiments.inference_integrity.native import (
    VerifiedReference,
    prepare_reference,
    record_forward,
    verify_record,
    verify_record_both_modes,
)


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(7)
        self.model = LlamaForCausalLM(LlamaConfig(
            vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
        )).float().eval()
        self.reference = self.root / 'reference'
        prepare_reference(self.model, self.reference, 'test-llama')

    def assert_both_modes_match_native(self, record, reference_cache=None):
        old_numeric = verify_record(record, self.reference,
                                    verified_reference=reference_cache)
        old_strict = verify_record(record, self.reference, strict_hash=True,
                                   verified_reference=reference_cache)
        both = verify_record_both_modes(record, self.reference,
                                        verified_reference=reference_cache)
        for mode, expected in [('numeric', old_numeric), ('strict_hash', old_strict)]:
            self.assertEqual(len(both[mode]), len(expected))
            for projected, original in zip(both[mode], expected):
                self.assertEqual(projected['block_id'], original['block_id'])
                self.assertEqual(projected['layer_name'], original['layer_name'])
                self.assertEqual(projected.get('error'), original.get('error'))
                if 'overall_relative_l2' in original:
                    self.assertAlmostEqual(projected['overall_relative_l2'],
                                           original['overall_relative_l2'], places=8)

    def test_clean_short_long_short_reference_cache_matches_native(self):
        cached_reference = VerifiedReference(self.reference)
        for i, token_ids in enumerate(([1, 2, 3], [1, 2, 3, 4, 5, 6], [1, 2, 3])):
            record = self.root / f'clean_{i}'
            record_forward(self.model, torch.tensor(token_ids), record,
                           self.reference, 'test-llama')
            cold = verify_record(record, self.reference)
            cached = verify_record(record, self.reference,
                                   verified_reference=cached_reference)
            self.assertEqual(len(cold), 4)
            self.assertEqual([x['layer_name'] for x in cold],
                             ['embedding', 'layer_0', 'layer_1', 'norm'])
            for a, b in zip(cold, cached):
                self.assertNotIn('error', a)
                self.assertNotIn('error', b)
                self.assertAlmostEqual(a['overall_relative_l2'],
                                       b['overall_relative_l2'], places=8)
            self.assert_both_modes_match_native(record, cached_reference)

    def test_live_fault_and_disk_corruption_have_distinct_rejections(self):
        def fault(_module, _inputs, output):
            changed = output.clone()
            changed[:, -1, 0] += 0.75
            return changed

        record = self.root / 'fault'
        record_forward(self.model, torch.tensor([1, 2, 3, 4]), record,
                       self.reference, 'test-llama',
                       extra_hooks=[(self.model.model.layers[0].mlp, fault)])
        reference = VerifiedReference(self.reference)
        numerical = verify_record(record, self.reference,
                                  verified_reference=reference)
        strict = verify_record(record, self.reference, strict_hash=True,
                               verified_reference=reference)
        self.assertGreater(numerical[1]['overall_relative_l2'], 0)
        self.assertIn('Computed output hash mismatch', strict[1]['error'])
        self.assert_both_modes_match_native(record, reference)

        output_file = record / 'block_1_0' / 'layer_0' / 'step_0' / 'output.pt'
        tensor = torch.load(output_file, weights_only=False)
        tensor[:, -1, 0] += 1
        torch.save(tensor, output_file)
        corrupted = verify_record(record, self.reference,
                                  verified_reference=reference)
        self.assertIn('Disk data hash mismatch', corrupted[1]['error'])
        self.assert_both_modes_match_native(record, reference)

    def test_corrupt_trusted_parameter_stays_an_error_in_both_modes(self):
        record = self.root / 'clean'
        record_forward(self.model, torch.tensor([1, 2, 3, 4]), record,
                       self.reference, 'test-llama')
        param_file = self.reference / 'final_model' / 'layer_0' / 'parameters' / 'param_0.pt'
        tensor = torch.load(param_file, weights_only=False)
        tensor.flatten()[0] += 1
        torch.save(tensor, param_file)
        both = verify_record_both_modes(record, self.reference)
        self.assertIn('hash mismatch', both['numeric'][1]['error'])
        self.assertEqual(both['numeric'][1]['error'], both['strict_hash'][1]['error'])
        self.assert_both_modes_match_native(record)

    def test_kv_context_simulation_is_seen_by_outer_decoder_recorder(self):
        class Tokenizer:
            eos_token_id = 2
            all_special_ids = [1, 2]

            @staticmethod
            def decode(ids, **_kwargs):
                return ' '.join(map(str, ids))

        original = list(range(1, 33))
        donor = [1] + [((i + 11) % 60) + 3 for i in range(32)]

        def live_record(prefix, step):
            return record_forward(self.model, prefix, self.root / f'kv_step_{step}',
                                  self.reference, 'test-llama')

        generated = generate_llama(
            self.model, Tokenizer(), original, steps=2,
            spec=InjectionSpec(kind='kv_cache', step=2, layer=0,
                               donor_position=3, seed=42),
            donor_inputs=donor, forward=live_record,
        )
        self.assertEqual(len(generated.generated_ids), 2)
        self.assertEqual(len(generated.events), 1)
        self.assertGreater(generated.events[0]['output_delta_l2'], 0)
        cache = VerifiedReference(self.reference)
        clean = verify_record_both_modes(self.root / 'kv_step_1', self.reference,
                                         verified_reference=cache)
        fault = verify_record_both_modes(self.root / 'kv_step_2', self.reference,
                                         verified_reference=cache)
        self.assertEqual(clean['numeric'][1]['overall_relative_l2'], 0.0)
        self.assertGreater(fault['numeric'][1]['overall_relative_l2'], 0.0)
        self.assertIn('Computed output hash mismatch', fault['strict_hash'][1]['error'])


if __name__ == '__main__':
    unittest.main()
