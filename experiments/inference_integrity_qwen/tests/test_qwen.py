"""Qwen mixed-attention live evidence and paired protocol regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5ForCausalLM, LlamaConfig, LlamaForCausalLM

from experiments.inference_integrity_qwen import native, runner
from experiments.inference_integrity_qwen.config import load_config
from experiments.inference_integrity_qwen.generation import (
    _greedy_with_fixed_length, generate_declared, generate_substitute_source, retokenize_response)
from experiments.inference_integrity_qwen.injections import InjectionSpec, kv_context_fault, next_legal_token
from experiments.inference_integrity_qwen.metrics import simulate_sampling
from experiments.inference_integrity_qwen.model_profile import build_profile
from experiments.inference_integrity_qwen.workload import build_paired_corpus, reference_inputs
from experiments.inference_integrity_qwen.audit import summary_agrees
from experiments.inference_integrity_qwen.metrics import summarize_cases
from dataclasses import replace


class Tokenizer:
    bos_token_id = None
    eos_token_id = 2
    pad_token_id = 0
    all_special_ids = [0, 2]

    def get_vocab(self):
        return {str(i): i for i in range(64)}

    def encode(self, text, add_special_tokens=False):
        return [1, 3, 4, 5, 6, 7, 8] if add_special_tokens else [3, 4, 5]

    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))


def tiny_qwen():
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=24,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=4, linear_num_value_heads=4, linear_conv_kernel_dim=4,
        layer_types=(['linear_attention'] * 3 + ['full_attention']) * 6,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                         'partial_rotary_factor': 0.5, 'mrope_section': [1, 1, 0],
                         'mrope_interleaved': True},
    )
    config.layer_types = ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'] * 6
    config._attn_implementation = 'sdpa'
    return Qwen3_5ForCausalLM(config).float().eval()


class QwenEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        cls.model = tiny_qwen()

    def test_nested_checkpoint_dtype_cannot_override_fp32(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            saved = tiny_qwen().bfloat16()
            saved.save_pretrained(checkpoint)
            Qwen3_5Config(text_config=saved.config.to_dict()).save_pretrained(checkpoint)
            config = replace(load_config(Path(__file__).parents[1]/'formal-qwen-seed0-cal32-v4.json'),
                             declared_model_path=str(checkpoint), generation_device='cpu')
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()):
                loaded, _ = runner._load_declared(config)
            self.assertEqual(loaded.config.dtype, torch.float32)
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in loaded.parameters()))
            with torch.no_grad():
                self.assertEqual(loaded(input_ids=torch.tensor([[3,4,5]]),use_cache=False).logits.dtype,
                                 torch.float32)

    def test_sparse_13_block_record_replays_clean_and_live_fault(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native.prepare_reference(self.model, root/'reference', 'tiny-qwen3.5',
                                     layers_per_block=2, activation_interval=4)
            reference = native.VerifiedReference(root/'reference')
            for name, fault_layer in [('clean', None), ('linear_fault', 0), ('full_fault', 3)]:
                def fault(_module, _inputs, output):
                    changed = output.clone()
                    changed[:, -1, 0] += 1
                    return changed
                hooks = [] if fault_layer is None else [(self.model.model.layers[fault_layer].mlp, fault)]
                record = root/name
                native.record_forward(self.model, torch.tensor([3,4,5,6]), record,
                                      root/'reference', 'tiny-qwen3.5', hooks)
                prep = json.loads((record/'activation_preparation.json').read_text())
                self.assertEqual({int(n.split('_')[1]) for n in prep['base_files'] if n.endswith('/output.pt')}, {0,4,8,12})
                self.assertEqual({row['block_id'] for row in prep['responses']}, {1,2,3,5,6,7,9,10,11})
                both = native.verify_record_both_modes(record, root/'reference', verified_reference=reference)
                self.assertEqual(len(both['numeric']), 13)
                self.assertFalse(any(row.get('error') for row in both['numeric']))
                if fault_layer is None:
                    self.assertEqual(max(row['overall_relative_l2'] for row in both['numeric']), 0)
                else:
                    self.assertGreater(both['numeric'][(fault_layer+1)//2]['overall_relative_l2'], 1e-4)
            base = next(iter(prep['base_files']))
            with (record/base).open('ab') as stream:stream.write(b'tampered')
            self.assertIn('Disk data hash mismatch', native.verify_record(record, root/'reference')[0]['error'])

    def test_every_full_layer_kv_identity_and_single_changed_history_slot(self):
        target = [3,4,5,6,7,8]
        donor = [3,14,15,16,17,18]
        with torch.no_grad():
            baseline = self.model(torch.tensor([target]), use_cache=False).logits
            for layer in (3,7,11,15,19,23):
                spec = InjectionSpec('kv_cache', step=2, layer=layer, donor_position=2, seed=1)
                actions = []
                with kv_context_fault(self.model, target, spec, 2, actions, allow_identical_donor=True):
                    identity = self.model(torch.tensor([target]), use_cache=False).logits
                self.assertTrue(torch.equal(identity, baseline))
                self.assertEqual(actions[0]['output_delta_l2'], 0)
                actions = []
                with kv_context_fault(self.model, donor, spec, 2, actions):
                    changed = self.model(torch.tensor([target]), use_cache=False).logits
                self.assertTrue(torch.equal(changed[:, :-1], baseline[:, :-1]))
                self.assertEqual(actions[0]['historical_output_delta_l2'], 0)
                self.assertGreater(actions[0]['output_delta_l2'], 0)
                self.assertNotEqual(actions[0]['before_k_sha256'], actions[0]['after_k_sha256'])
            with self.assertRaises(ValueError):
                with kv_context_fault(self.model, donor, InjectionSpec('kv_cache',step=2,layer=0),2,[]):pass

    def test_generation_and_reverse_bridge_have_distinct_length_contracts(self):
        tokenizer = Tokenizer()
        original = list(range(3,35))
        generated = generate_declared(self.model, tokenizer, original, steps=2,
                                     spec=InjectionSpec('token_selection',step=2))
        event = generated.events[0]
        self.assertNotEqual(event['normal_token'], event['replacement_token'])
        self.assertEqual(generated.generated_ids[1], event['replacement_token'])
        llama = LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
                                           num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2)).float().eval()
        source = generate_substitute_source(llama, tokenizer, 'same request',steps=2,
                                            spec=InjectionSpec('model'))
        self.assertEqual(len(source.input_ids), 7)
        self.assertEqual(len(source.generated_ids), 2)
        bridge = retokenize_response(original, source.response_text, tokenizer)
        self.assertEqual(bridge[:32], original)
        self.assertEqual(len(bridge), 35)


class ProtocolTests(unittest.TestCase):
    def test_persisted_Wilson_intervals_match_but_changed_numbers_are_rejected(self):
        rows=[{'planned_positive':positive,'injection_success':True,'verdict':verdict,
               'strict_hash_verdict':'DETECTED'} for positive,verdict in
              [(True,'DETECTED'),(True,'PASSED'),(False,'PASSED'),(False,'DETECTED')]]
        saved=json.loads(json.dumps(summarize_cases(rows)))
        self.assertTrue(summary_agrees(saved,rows))
        saved['tpr_wilson_95'][0]+=0.001
        self.assertFalse(summary_agrees(saved,rows))
        strict=json.loads(json.dumps(summarize_cases([{**row,'verdict':row['strict_hash_verdict']} for row in rows])))
        self.assertTrue(summary_agrees(strict,rows,'strict_hash'))

    def test_shared_evidence_sampling_and_costs_are_reused_in_every_view(self):
        config=replace(load_config(Path(__file__).parents[1]/'formal-qwen-seed0-cal32-v4.json'),
                       m=4,n=1,calibration_size=1,sample_repeats=3)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for condition in ('shared_clean',*runner.CONDITIONS):
                for index in range(4):
                    positive=index==1
                    if positive==(condition=='shared_clean'):continue
                    case=runner._case_dir(root,0,condition,index)
                    origins=runner.EVIDENCE_ORIGINS if not positive else (runner._condition_origin(condition),)
                    runner._atomic_json(case/'result.json',{'origins':list(origins),'proof_bytes':10,
                                                          'stage_seconds':{'generation':1.0}})
                    for origin in origins:
                        blocks=[{'block_id':str(i),'overall_relative_l2':0.0} for i in range(13)]
                        runner._save_case_result(case/origin,condition=condition,seed=0,index=index,
                            positive=positive,injection_success=True,origin=origin,
                            blocks={'numeric':blocks,'strict_hash':blocks},
                            thresholds={value:0 for value in runner.EVIDENCE_ORIGINS},status_error=None,config=config)
            with patch.object(runner,'prepare_corpus'),patch.object(runner,'_positive_indices',return_value={1}),\
                    patch.object(runner,'_load_thresholds',return_value={origin:0 for origin in runner.EVIDENCE_ORIGINS}):
                summary=runner.summarize(config,root)
            self.assertEqual((summary['unique_generation_tasks'],summary['unique_sequence_evidence'],
                              summary['condition_sample_references']),(9,12,24))
            self.assertEqual(summary['actual_costs']['proof_bytes'],90)
            groups=[group for key,group in summary['groups'].items() if key.endswith('/generation_live')]
            self.assertEqual(len({group['sources']['0'] for group in groups}),1)
            self.assertTrue(all(group['sampling']==groups[0]['sampling'] for group in groups))

    def test_interrupted_attempt_is_archived_and_saved_substitute_source_preserved(self):
        config=load_config(Path(__file__).parents[1]/'formal-qwen-seed0-cal32-v4.json')
        with tempfile.TemporaryDirectory() as temporary:
            case=Path(temporary)/'model_mismatch/0004'
            (case/'records').mkdir(parents=True)
            (case/'records/partial.pt').write_bytes(b'partial proof')
            runner._atomic_json(case/'oracle/model_source.json',{'generated_ids':[3,4]})
            self.assertFalse(runner._case_result_exists(case,config))
            self.assertTrue((case/'oracle/model_source.json').is_file())
            archives=list((case.parent/'_diagnostics').iterdir())
            self.assertEqual((archives[0]/'records/partial.pt').read_bytes(),b'partial proof')
            runner._atomic_json(case/'result.json',{'config_sha256':config.fingerprint(),
                                                   'origins':['model_recompute_live']})
            with self.assertRaises(ValueError):runner._case_result_exists(case,config)
            self.assertFalse(runner._case_result_exists(case,config,retry_errors=True))
            self.assertEqual(len(list((case.parent/'_diagnostics').iterdir())),2)

    def test_unassigned_ids_and_only_tokenizer_eos_are_masked(self):
        logits = torch.zeros(66)
        logits[65]=100;logits[2]=99;logits[0]=10;logits[3]=9
        selected, allowed = _greedy_with_fixed_length(logits, Tokenizer())
        self.assertEqual(selected,0)  # PAD is deliberately not an extra greedy stop token.
        self.assertTrue(torch.isneginf(allowed[65]) and torch.isneginf(allowed[2]))
        self.assertEqual(next_legal_token(allowed,{0,2}), (0,3))

    def test_dynamic_block_counts_and_postcommit_sampling_budget(self):
        profile = build_profile(tiny_qwen().config, Tokenizer(), 'tiny-qwen')
        self.assertEqual(profile['block_count'],13)
        self.assertEqual(profile['kv_eligible_layers'],[3,7,11,15,19,23])
        for count, sample in [(1248,100),(13,2)]:
            blocks=[{'block_id':str(i),'overall_relative_l2':0} for i in range(count)]
            result=simulate_sampling('source',blocks,0,(0.08,),3,42)[0]
            self.assertEqual(result['sample_count'],sample)

    def test_frozen_rows_labels_and_model_independent_schedule_are_reused(self):
        config=load_config(Path(__file__).parents[1]/'formal-qwen-seed0-cal32-v4.json')
        rows,dataset=runner._dataset_rows(config)
        corpus,schedule,provenance=build_paired_corpus(config,rows,dataset)
        old=reference_inputs(config.reference_run)[1]
        self.assertEqual(corpus['positive_indices'],old['positive_indices'])
        self.assertEqual([r['source_row'] for r in corpus['evaluation']['0']],
                         [r['source_row'] for r in old['evaluation']['0']])
        self.assertEqual(len({tuple(r['input_ids']) for r in corpus['calibration']+corpus['evaluation']['0']}),288)
        for condition in runner.CONDITIONS:
            for index in corpus['positive_indices']['0']:
                spec=runner._injection_spec(config,0,condition,index)
                expected=schedule['0'][condition][str(index)]
                self.assertEqual(spec.step,expected.get('step'))
                self.assertEqual(spec.donor_position,expected.get('donor_position'))
                if spec.kind=='kv_cache':self.assertIn(spec.layer,[3,7,11,15,19,23])
                if spec.kind=='computation':self.assertIn(spec.layer,range(24))
                if spec.kind=='input':self.assertEqual(spec.swap_positions,tuple(expected['swap_positions']))


if __name__=='__main__':unittest.main()
