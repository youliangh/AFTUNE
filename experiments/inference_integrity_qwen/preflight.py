"""Full-length actual-checkpoint GPU-to-CPU gate, separate from formal results."""
import argparse
from dataclasses import replace
from pathlib import Path
import time

import torch

from .config import load_config
from .generation import generate_declared, _greedy_with_fixed_length
from .injections import InjectionSpec, kv_context_fault, next_legal_token, swap_input_spans
from . import runner as r


REQUIRED_CHECKS = ('actual_cuda_generation_cpu_verification', 'all_six_injections',
                   'all_full_attention_KV_layers', 'KV_identity_and_historical_rows',
                   'linear_and_full_computation', 'serving_all_24_layers',
                   'vocab_EOS_and_special_boundaries', 'first_content_token_protected',
                   'reverse_bridge_and_Model_origin_isolation', 'sparse_4_plus_9')


def inspect_case(case, condition, expected_blocks):
    marker = r._read_json(case/'result.json')
    if marker.get('status_error'):
        raise RuntimeError(f'Preflight execution failure: {case}: {marker}')
    outcomes = {}
    for origin in marker['origins']:
        row = r._read_json(case/origin/'result.json')
        expected = expected_blocks * (96 if origin == 'generation_live' else 1)
        if row.get('error_blocks') or row.get('status_error') or row['block_count'] != expected:
            raise RuntimeError(f'Preflight invalid block evidence: {case}/{origin}')
        if condition != 'shared_clean' and row['injection_success'] is not True:
            raise RuntimeError(f'Preflight injection failed: {case}')
        outcomes[origin] = {name: row.get(name) for name in (
            'verdict','score','block_count','injection_success','strict_hash_verdict')}
    return outcomes


@torch.no_grad()
def kv_controls(config, root, model, tokenizer, requests, index, eligible):
    r._check_gpu_budget(config)
    target = generate_declared(model,tokenizer,requests[index]['input_ids'],steps=96)
    donor = generate_declared(model,tokenizer,requests[(index+1)%config.m]['input_ids'],steps=96)
    r._atomic_json(root/'kv_control_sources.json', {'target':r._generation_payload(target),
                                                 'donor':r._generation_payload(donor)})
    observations = []
    for layer in eligible:
        for step in (2,96):
            r._check_gpu_budget(config)
            prefix = target.full_ids[:32+step-1]
            ids = torch.tensor([prefix],device=config.generation_device)
            baseline = model(input_ids=ids,use_cache=False,logits_to_keep=1).logits
            spec = InjectionSpec('kv_cache',step=step,layer=layer,seed=0,donor_position=1)
            identity_actions = []
            with kv_context_fault(model,prefix,spec,step,identity_actions,allow_identical_donor=True):
                identity = model(input_ids=ids,use_cache=False,logits_to_keep=1).logits
            if not torch.equal(baseline,identity) or identity_actions[0]['output_delta_l2'] != 0:
                raise RuntimeError(f'Qwen KV no-change reconstruction differs at layer {layer}, step {step}')
            actions = []
            with kv_context_fault(model,donor.full_ids[:len(prefix)],spec,step,actions):
                model(input_ids=ids,use_cache=False,logits_to_keep=1)
            event = actions[0]
            if len(actions)!=1 or event['historical_output_delta_l2']!=0 or event['output_delta_l2']<=0:
                raise RuntimeError('KV diagnostic did not change only the final row')
            observations.append({'layer':layer,'step':step,'identity_equal':True,**event})
            r._progress(root,'preflight_KV_control',layer=layer,step=step,output_delta_l2=event['output_delta_l2'])
    return observations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('formal-qwen-seed0-cal32-v4.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    if not config.generation_device.startswith('cuda') or config.verification_device!='cpu':
        parser.error('The actual-checkpoint gate requires GPU generation and CPU verification')
    root=args.output.resolve()
    r.configure_determinism(0)
    torch.set_num_threads(config.cpu_threads)
    started=time.monotonic()
    r.prepare_reference(config,root)
    r.calibrate(config,root,limit=1)
    if (root/'thresholds.json').exists():
        raise RuntimeError('Preflight must not freeze official thresholds')
    thresholds={origin:r.calibrate_threshold([r._read_json(root/f'calibration/0000/blocks_{origin}.json')])
                for origin in r.EVIDENCE_ORIGINS}
    corpus=r._load_corpus(config,root)
    manifest=r._read_json(root/'manifest.json')
    profile=manifest['model_profile']
    seed=config.seeds[0]
    requests=corpus['evaluation'][str(seed)]
    positives=r._positive_indices(config,seed)
    index=min(positives)
    r._prepare_model_sources(config,root,seed,requests,positives,[index],False)
    reference=r._verified_reference(config,root)
    model,tokenizer=r._load_declared(config)
    outcomes={}
    try:
        controls=kv_controls(config,root,model,tokenizer,requests,index,profile['kv_eligible_layers'])
        for condition in ('shared_clean',*r.CONDITIONS):
            r._check_disk_budget(root)
            r._check_gpu_budget(config)
            if condition=='model_mismatch':
                r._run_model_case(config,root,seed,index,requests[index],True,model,tokenizer,reference,thresholds)
            else:
                r._run_non_model_case(config,root,seed,condition,index,requests[index],condition!='shared_clean',
                                      model,tokenizer,reference,thresholds,requests[(index+1)%config.m])
            case=r._case_dir(root,seed,condition,index)
            outcomes[condition]=inspect_case(case,condition,profile['block_count'])
            r._progress(root,'preflight_condition_finished',condition=condition,outcomes=outcomes[condition])
        first_spec=r._injection_spec(config,seed,'computation',index)
        first_type=profile['layer_types'][first_spec.layer]
        other_layer=3 if first_type=='linear_attention' else 0
        other_index=sorted(positives)[1]
        other_spec=replace(r._injection_spec(config,seed,'computation',other_index),layer=other_layer)
        r._run_non_model_case(config,root,seed,'computation',other_index,requests[other_index],True,
                              model,tokenizer,reference,thresholds,requests[(other_index+1)%config.m],diagnostic_spec=other_spec)
        extra=inspect_case(r._case_dir(root,seed,'computation',other_index),'computation',profile['block_count'])
        # Explicitly exercise legal-ID, tokenizer-EOS and PAD policy boundaries.
        vocab=set(tokenizer.get_vocab().values())
        special=set(tokenizer.all_special_ids)
        legal=next(token for token in sorted(vocab-special) if token!=tokenizer.eos_token_id)
        unassigned=next(token for token in range(profile['head_vocab_size']) if token not in vocab)
        logits=torch.zeros(profile['head_vocab_size'],device=config.generation_device)
        logits[unassigned]=100;logits[tokenizer.eos_token_id]=99
        logits[tokenizer.pad_token_id]=10;logits[legal]=9
        greedy,allowed=_greedy_with_fixed_length(logits,tokenizer)
        if greedy!=tokenizer.pad_token_id or next_legal_token(allowed,special)!=(greedy,legal):
            raise RuntimeError('Frozen vocabulary/EOS/PAD policy failed its boundary control')
        positions=tuple(corpus['evaluation'][str(seed)][index]['input_ids'])
        paired=r._injection_spec(config,seed,'input_mismatch',index)
        swapped,event=swap_input_spans(list(positions),seed=paired.seed,positions=paired.swap_positions)
        if swapped[0]!=positions[0] or len(swapped)!=32 or not event['triggered']:
            raise RuntimeError('First-content-token input policy failed')
        model_marker=r._read_json(r._case_dir(root,seed,'model_mismatch',index)/'result.json')
        if model_marker['origins']!=['model_recompute_live']:
            raise RuntimeError('Substitute trace leaked into Model evidence origins')
        preparation=r._read_json(root/'calibration/0000/records/generation_live/step_0001/activation_preparation.json')
        base={int(name.split('_')[1]) for name in preparation['base_files'] if name.endswith('/output.pt')}
        if base!={0,4,8,12} or len(preparation['responses'])!=9 or not preparation['complete']:
            raise RuntimeError('Actual-checkpoint sparse path differs from 4+9')
    finally:
        del model,reference
        r._collect_device()
    report={'passed':True,'diagnostic_only':True,'config_sha256':config.fingerprint(),
            'experiment_code_sha256':r._code_digest(),'checks':{name:True for name in REQUIRED_CHECKS},
            'calibration_requests':1,'diagnostic_thresholds':thresholds,'outcomes':outcomes,
            'kv_controls':controls,'computation_layer_types':[first_type,profile['layer_types'][other_layer]],
            'extra_computation':extra,'elapsed_seconds':time.monotonic()-started}
    r._atomic_json(root/'preflight.json',report)
    r._progress(root,'preflight_complete',passed=True)
    print(report,flush=True)


if __name__=='__main__':main()
