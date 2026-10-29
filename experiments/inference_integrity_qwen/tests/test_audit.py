"""Persist a miniature complete workload and exercise the actual v4 auditor."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.inference_integrity_qwen import audit as a, runner as r
from experiments.inference_integrity_qwen.config import load_config
from experiments.inference_integrity_qwen.injections import InjectionSpec
from experiments.inference_integrity_qwen.model_profile import build_profile
from experiments.inference_integrity_qwen.preflight import REQUIRED_CHECKS
from experiments.inference_integrity_qwen.tests.test_qwen import Tokenizer,tiny_qwen


class CompletionAuditTests(unittest.TestCase):
    def test_complete_persisted_views_and_sampling_pass_and_modified_summary_fails(self):
        torch.set_num_threads(1)
        r.configure_determinism(0)
        config=replace(load_config(Path(__file__).parents[1]/'formal-qwen-seed0-cal32-v4.json'),
                       m=4,n=1,calibration_size=1,generation_device='cpu',
                       retain_record_indices=(0,),sample_repeats=3)
        tokenizer=Tokenizer()
        profile=build_profile(tiny_qwen().config,tokenizer,'tiny-qwen3.5')
        profile['backends']={'full_attention':'sdpa','linear_attention':'torch_fallback'}
        def request(index):
            return {'source_row':index,'input_ids':list(range(3+index,35+index)),
                    'request_text':f'request {index}','request_text_sha256':sha256(str(index).encode()).hexdigest()}
        corpus={'calibration':[request(4)],'evaluation':{'0':[request(i) for i in range(4)]},
                'positive_indices':{'0':[1]}}
        schedule={'0':{condition:{'1':{}} for condition in r.CONDITIONS}}
        dataset={'sha256':'a'*64}
        provenance={'reference_run':config.reference_run}
        def spec(_config,_seed,condition,_index):
            kind=r._condition_kind(condition)
            return InjectionSpec(kind,step=2 if kind not in ('model','input') else None,
                                 layer=3 if kind=='kv_cache' else 0 if kind=='computation' else None,
                                 donor_position=1 if kind=='kv_cache' else None,
                                 swap_positions=(1,5) if kind=='input' else None)
        def blocks(origin):
            steps=range(1,97) if origin=='generation_live' else (0,)
            return [{'block_id':f'{step}/{index}','layer_block_id':index,'step':step,
                     'layer_name':profile['block_to_layers'][str(index)][-1],
                     'overall_relative_l2':0.0} for step in steps for index in range(13)]
        def generation(ids,events):
            return {'input_ids':list(ids),'generated_ids':[3]*96,'full_ids':list(ids)+[3]*96,
                    'response_text':'response','events':events}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            r._atomic_json(root/'corpus.json',corpus)
            r._atomic_json(root/'paired_schedule.json',schedule)
            r._atomic_json(root/'reference/metadata.json',{'layer_order':profile['tracked_layers'],
                'block_to_layers':profile['block_to_layers'],'layers_per_block':2,
                'activation_interval':4,'steps_per_block':1,'model_name':profile['model_name']})
            r._atomic_json(root/'reference/ready.json',{'files':r._directory_hashes(root/'reference')})
            models={str(config.declared_model_path):{'path':config.declared_model_path,'files':{}},
                    str(config.substitute_model_path):{'path':config.substitute_model_path,'files':{}}}
            manifest={'config':a.json_value(config.as_dict()),'config_sha256':config.fingerprint(),
                'experiment_code_sha256':r._code_digest(),'paired_reference':provenance,'dataset':dataset,
                'corpus_sha256':r._hash_file(root/'corpus.json'),'paired_schedule_sha256':r._hash_file(root/'paired_schedule.json'),
                'declared_model':models[config.declared_model_path],'substitute_model':models[config.substitute_model_path],
                'model_profile':profile,'environment':r._environment_manifest(config),
                'shared_controls':{'0':{'indices':[0,2,3],
                    'condition_origins':{condition:r._condition_origin(condition) for condition in r.CONDITIONS}}}}
            r._atomic_json(root/'manifest.json',manifest)
            gate_root=root/'diagnostic'
            r._atomic_json(gate_root/'preflight.json',{'passed':True,'diagnostic_only':True,
                'config_sha256':config.fingerprint(),'experiment_code_sha256':r._code_digest(),
                'checks':{name:True for name in REQUIRED_CHECKS}})
            r._atomic_json(root/'preflight_gate.json',{'path':str(gate_root),
                'preflight_sha256':r._hash_file(gate_root/'preflight.json')})
            thresholds={origin:0.0 for origin in r.EVIDENCE_ORIGINS}
            r._atomic_json(root/'thresholds.json',{'config_sha256':config.fingerprint(),
                'calibration_size':1,'thresholds':thresholds})
            for origin in r.EVIDENCE_ORIGINS:r._atomic_json(root/f'calibration/0000/blocks_{origin}.json',blocks(origin))
            r._atomic_json(root/'calibration/0000/calibration.json',{'request_sha256':corpus['calibration'][0]['request_text_sha256']})
            for condition in ('shared_clean',*r.CONDITIONS):
                for index in ([0,2,3] if condition=='shared_clean' else [1]):
                    positive=condition!='shared_clean'
                    case=r._case_dir(root,0,condition,index)
                    req=corpus['evaluation']['0'][index]
                    ids=req['input_ids'].copy();events=[]
                    injection=spec(config,0,condition,index) if positive else None
                    if condition=='serving_config':
                        events=[{'kind':'serving','event':'route_switch','step':2,'rounding_bits':6}]+[
                            {'kind':'serving','step':step,'layer':layer,'rounding_bits':6,'changed_elements':1}
                            for step in range(2,97) for layer in range(24)]
                    elif condition=='kv_cache':events=[{'kind':'kv_cache','layer':3,'step':2,'requested_donor_position':1,
                        'output_delta_l2':0.1,'historical_output_delta_l2':0,'before_k_sha256':'a','after_k_sha256':'b',
                        'before_v_sha256':'a','after_v_sha256':'b'}]
                    elif condition=='computation':events=[{'kind':'computation','layer':0,'step':2,'changed_elements':1,
                        'before_sha256':'a','after_sha256':'b','delta_l2':0.01,'before_l2':1}]
                    elif condition=='token_selection':events=[{'kind':'token_selection','step':2,'triggered':True,
                        'normal_token':4,'replacement_token':3,'normal_logit':10,'replacement_logit':9}]
                    elif condition=='input_mismatch':
                        ids[1:5],ids[5:9]=ids[5:9],ids[1:5]
                        events=[{'kind':'input','positions':[1,5],'span_width':4,'before':req['input_ids'],'after':ids}]
                    source=generation(ids,events)
                    truth={'planned_positive':positive,'injection_success':True,'source_request':req,
                           'source_model':config.declared_model_path,'generation':source,
                           'spec':vars(injection) if injection else None}
                    if condition=='kv_cache':
                        truth['donor_request']=corpus['evaluation']['0'][(index+1)%4]
                        truth['donor_generation']=generation(truth['donor_request']['input_ids'],[])
                    if condition=='model_mismatch':
                        source=generation(tokenizer.encode(req['request_text'],add_special_tokens=True),[{'kind':'model'}])
                        r._atomic_json(case/'oracle/model_source.json',source)
                        truth['source_generation']=source;truth['source_model']=config.substitute_model_path
                    if condition in ('shared_clean','model_mismatch'):
                        truth['declared_retokenized_ids']=req['input_ids']+tokenizer.encode(source['response_text'],add_special_tokens=False)
                    r._atomic_json(case/'oracle/truth.json',truth)
                    origins=r.EVIDENCE_ORIGINS if not positive else (r._condition_origin(condition),)
                    if index==0:
                        for origin in origins:
                            steps=range(1,97) if origin=='generation_live' else (0,)
                            for step in steps:
                                record=case/'records'/origin/(f'step_{step:04d}' if step else '')
                                base_files={};responses=[]
                                for block in range(13):
                                    layer=profile['block_to_layers'][str(block)][-1]
                                    name=f'block_{block}_0/{layer}/step_0/output.pt'
                                    path=record/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'tensor fixture')
                                    digest=r._hash_file(path)
                                    r._atomic_json(path.with_name('hashes.json'),{'output_hash':digest})
                                    if block%4==0:base_files[name]=digest
                                    else:responses.append({'block_id':block,'path':name,'sha256':digest,'output_hash':digest})
                                r._atomic_json(record/'activation_preparation.json',{'method':'original_live_tensor_response',
                                    'complete':True,'base_files':base_files,'responses':responses})
                        files={str(path.relative_to(case)):r._hash_file(path) for path in (case/'records').rglob('*') if path.is_file()}
                    else:files={'records/purged.pt':'a'*64}
                    records=97 if not positive else 1 if condition=='model_mismatch' else 96
                    r._atomic_json(case/'commitment.json',{'files':files,'frozen_at':r._timestamp(),
                        'activation_preparation':{'method':'original_live_tensor_response','records':records}})
                    for origin in origins:
                        values=blocks(origin)
                        r._save_case_result(case/origin,condition=condition,seed=0,index=index,positive=positive,
                            injection_success=True,origin=origin,blocks={'numeric':values,'strict_hash':values},
                            thresholds=thresholds,status_error=None,config=config)
                    r._atomic_json(case/'result.json',{'origins':list(origins),'proof_bytes':10,'stage_seconds':{'generation':1}})
            with patch.object(r,'prepare_corpus'),patch.object(r,'_positive_indices',return_value={1}),\
                 patch.object(r,'_load_thresholds',return_value=thresholds):r.summarize(config,root)
            with patch.object(a,'load_config',return_value=config),patch.object(r,'_dataset_rows',return_value=([],dataset)),\
                 patch.object(a,'build_paired_corpus',return_value=(corpus,schedule,provenance)),\
                 patch.object(a,'checkpoint_profile',return_value=profile),\
                 patch.object(r,'_model_digest',side_effect=lambda path:models[str(path)]),\
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer),\
                 patch.object(r,'_injection_spec',side_effect=spec):
                result=a.audit(root,full_sampling=True)
                self.assertTrue(result['passed'],result)
                self.assertEqual(result['observed'],{'unique_generation_tasks':9,'unique_sequence_evidence':12,'condition_sample_references':24})
                summary=r._read_json(root/'summary.json')
                summary['groups']['0/kv_cache/generation_live']['numeric']['tpr_wilson_95'][0]+=0.01
                r._atomic_json(root/'summary.json',summary)
                rejected=a.audit(root,full_sampling=True)
                self.assertFalse(rejected['passed'])
                self.assertEqual(rejected['issue_counts'],{'numeric_summary':1})


if __name__=='__main__':unittest.main()
