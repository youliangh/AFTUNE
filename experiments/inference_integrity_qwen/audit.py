"""Complete Qwen v4 evidence, paired-schedule and post-commit sampling audit."""
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from .config import CONDITIONS, EVIDENCE_ORIGINS, load_config
from .metrics import (INTEGRITY_ERRORS, calibrate_threshold, decide_sequence,
                      simulate_sampling, summarize_cases, summarize_sampling)
from .model_profile import checkpoint_profile
from .workload import build_paired_corpus, reference_inputs
from . import runner as r


def json_value(value):
    """Compare persisted metrics without changing numbers or accepting NaN."""
    return json.loads(json.dumps(value, allow_nan=False))


def summary_agrees(saved, rows, mode='numeric'):
    if mode == 'strict_hash':
        rows = [{**row, 'verdict': row['strict_hash_verdict']} for row in rows]
    return saved == json_value(summarize_cases(rows))


def _verdict(blocks, threshold):
    rejected = invalid = False
    if not blocks:
        return 'ERROR'
    for block in blocks:
        error = str(block.get('error') or '')
        if error:
            if any(marker in error for marker in INTEGRITY_ERRORS):
                rejected = True
            else:
                invalid = True
            continue
        try:
            score = float(block['overall_relative_l2'])
            if not math.isfinite(score) or score < 0:
                invalid = True
            elif score > threshold:
                rejected = True
        except (KeyError, ValueError, TypeError):
            invalid = True
    return 'DETECTED' if rejected else 'ERROR' if invalid else 'PASSED'


def audit(root: Path, *, full_sampling: bool = True) -> dict:
    from transformers import AutoTokenizer
    root = root.resolve()
    manifest = r._read_json(root/'manifest.json')
    config = load_config(Path(__file__).with_name('formal-qwen-seed0-cal32-v4.json'))
    issues, counts = [], Counter()
    def check(ok, code, location, detail=''):
        if not ok:
            counts[code] += 1
            if len(issues) < 60:
                issues.append({'code': code, 'location': str(location), 'detail': str(detail)[:500]})
    def read(path):
        try:
            return r._read_json(path)
        except (OSError, ValueError) as exc:
            check(False, 'missing_or_invalid_json', path, exc)
            return None
    config_hash = config.fingerprint()
    check(manifest['config'] == json_value(config.as_dict()) and manifest['config_sha256'] == config_hash,
          'protocol_config', 'manifest.json')
    check(r._code_digest() == manifest['experiment_code_sha256'], 'code_hash', 'manifest.json')
    environment=r._environment_manifest(replace(config,generation_device='cpu'))
    check({key:value for key,value in environment.items() if key!='gpu_name'} ==
          {key:value for key,value in manifest['environment'].items() if key!='gpu_name'},
          'frozen_environment', root)
    accepted = read(root/'preflight_gate.json')
    if accepted:
        gate_path = Path(accepted['path'])/'preflight.json'
        gate = read(gate_path)
        from .preflight import REQUIRED_CHECKS
        check(bool(gate) and gate.get('passed') and gate.get('diagnostic_only') and
              gate.get('config_sha256') == config_hash and gate.get('experiment_code_sha256') == r._code_digest() and
              all(gate.get('checks',{}).get(name) is True for name in REQUIRED_CHECKS) and
              r._hash_file(gate_path) == accepted['preflight_sha256'], 'preflight_gate', root)
    reference_inputs.cache_clear()
    raw, dataset = r._dataset_rows(config)
    expected_corpus, expected_schedule, provenance = build_paired_corpus(config, raw, dataset)
    corpus, schedule = read(root/'corpus.json'), read(root/'paired_schedule.json')
    check(corpus == expected_corpus, 'paired_corpus', 'corpus.json')
    check(schedule == expected_schedule, 'paired_schedule', 'paired_schedule.json')
    check(provenance == manifest['paired_reference'], 'paired_reference_hashes', 'manifest.json')
    check(r._hash_file(root/'corpus.json') == manifest['corpus_sha256'], 'corpus_hash', root)
    check(r._hash_file(root/'paired_schedule.json') == manifest['paired_schedule_sha256'], 'schedule_hash', root)
    check(dataset['sha256'] == manifest['dataset']['sha256'], 'dataset_hash', root)
    for role in ('declared_model', 'substitute_model'):
        check(r._model_digest(manifest[role]['path']) == manifest[role], 'model_hash', role)
    profile = checkpoint_profile(config.declared_model_path)
    check(profile == manifest['model_profile'], 'model_profile', root)
    block_count = profile['block_count']
    check(block_count == 13 and profile['base_output_blocks'] == [0,4,8,12] and
          profile['kv_eligible_layers'] == [3,7,11,15,19,23], 'Qwen_architecture', root)
    metadata, ready = read(root/'reference/metadata.json'), read(root/'reference/ready.json')
    if metadata and ready:
        check(r._directory_hashes(root/'reference', exclude={'ready.json'}) == ready['files'], 'reference_hash', root)
        check(metadata['layer_order'] == profile['tracked_layers'] and
              metadata['block_to_layers'] == profile['block_to_layers'], 'layer_partition', root)
        check(metadata['layers_per_block'] == 2 and metadata['activation_interval'] == 4 and
              metadata['steps_per_block'] == 1 and metadata['model_name'] == profile['model_name'], 'native_parameters', root)
    thresholds, summary = read(root/'thresholds.json'), read(root/'summary.json')
    if not all(value is not None for value in (corpus, schedule, thresholds, summary)):
        return {'passed': False, 'issue_counts': dict(counts), 'issues': issues}
    check(thresholds['config_sha256'] == config_hash and thresholds['calibration_size'] == config.calibration_size and
          set(thresholds['thresholds']) == set(EVIDENCE_ORIGINS), 'calibration_identity', root)
    def grid(blocks, origin, location):
        steps = range(1, config.generated_tokens+1) if origin == 'generation_live' else (0,)
        expected = {(step, block) for step in steps for block in range(block_count)}
        coordinates = [(block.get('step'), block.get('layer_block_id')) for block in blocks]
        check(len(coordinates) == len(expected) and set(coordinates) == expected, 'block_grid', location)
        for block in blocks:
            index = block.get('layer_block_id')
            if type(index) is int and 0 <= index < block_count:
                check(block.get('block_id') == f"{block['step']}/{index}" and
                      block.get('layer_name') == profile['block_to_layers'][str(index)][-1], 'block_identity', location)
    for origin in EVIDENCE_ORIGINS:
        calibration = []
        for index in range(config.calibration_size):
            blocks = read(root/f'calibration/{index:04d}/blocks_{origin}.json')
            marker = read(root/f'calibration/{index:04d}/calibration.json')
            if blocks is not None and marker is not None:
                check(marker['request_sha256'] == corpus['calibration'][index]['request_text_sha256'], 'calibration_request', index)
                grid(blocks, origin, f'calibration/{index}/{origin}')
                calibration.append(blocks)
        try:
            check(len(calibration) == config.calibration_size and calibrate_threshold(calibration) == thresholds['thresholds'][origin], 'calibration_threshold', origin)
        except ValueError as exc:
            check(False, 'calibration_error', origin, exc)
    tokenizer = AutoTokenizer.from_pretrained(config.declared_model_path, local_files_only=True)
    substitute_tokenizer = AutoTokenizer.from_pretrained(config.substitute_model_path, local_files_only=True)
    valid = {'declared': set(tokenizer.get_vocab().values()), 'substitute': set(substitute_tokenizer.get_vocab().values())}
    special = set(tokenizer.all_special_ids)
    unique, tasks, task_markers, groups = {}, set(), {}, {}
    bridge_lengths = {'negative': Counter(), 'positive': Counter()}
    token_rejections, local_computation = {}, 0
    prep_counts, retained_records = Counter(), 0
    audited_case_events = set()
    audited_retention = set()
    for seed in config.seeds:
        positives = set(expected_corpus['positive_indices'][str(seed)])
        requests = corpus['evaluation'][str(seed)]
        mapping = manifest['shared_controls'][str(seed)]
        check(mapping['indices'] == sorted(set(range(config.m))-positives), 'shared_indices', seed)
        check(not ({tuple(q['input_ids']) for q in requests} & {tuple(q['input_ids']) for q in corpus['calibration']}), 'calibration_overlap', seed)
        for condition in CONDITIONS:
            origin = r._condition_origin(condition)
            check(mapping['condition_origins'][condition] == origin, 'shared_origin', condition)
            key, rows = f'{seed}/{condition}/{origin}', []
            for index in range(config.m):
                positive = index in positives
                source_condition = condition if positive else 'shared_clean'
                case = r._case_dir(root, seed, source_condition, index)
                relative = str((case/origin/'result.json').relative_to(root))
                check(summary['groups'].get(key, {}).get('sources', {}).get(str(index)) == relative, 'source_reference', f'{key}/{index}')
                if relative not in unique:
                    row, marker = read(root/relative), read(case/'result.json')
                    oracle, commitment = read(case/'oracle/truth.json'), read(case/'commitment.json')
                    if any(value is None for value in (row, marker, oracle, commitment)):
                        continue
                    unique[relative] = row
                    tasks.add(case)
                    task_markers[case] = marker
                    check((row.get('seed'), row.get('index'), row.get('config_sha256'), row.get('evidence_origin'),
                           row.get('condition'), row.get('planned_positive'), row.get('case_key')) ==
                          (seed,index,config_hash,origin,source_condition,positive,f'{seed}/{source_condition}/{index:04d}/{origin}'), 'source_identity', relative)
                    check(not marker.get('status_error') and not row.get('status_error') and not row.get('error_blocks'), 'sample_execution_error', relative)
                    check(row.get('injection_success') is True and oracle.get('injection_success') is True,
                          'injection_success', relative)
                    check(oracle.get('source_request') == requests[index] and oracle.get('planned_positive') is positive,
                          'oracle_request', relative)
                    expected_origins = [origin] if positive else list(EVIDENCE_ORIGINS)
                    check(marker['origins'] == expected_origins, 'evidence_origins', case)
                    check(all(re.fullmatch(r'[0-9a-f]{64}', digest) for digest in commitment['files'].values()) and bool(commitment['files']), 'commitment_structure', case)
                    check(datetime.fromisoformat(commitment['frozen_at']) <= datetime.fromisoformat(row['challenge_created_at']) <=
                          datetime.fromisoformat(row['finished_at']), 'postcommit_challenge', relative)
                    prep = commitment.get('activation_preparation', {})
                    expected_records = (1 if condition == 'model_mismatch' else 96) if positive else 97
                    check(prep.get('method') == 'original_live_tensor_response' and prep.get('records') == expected_records,
                          'activation_preparation', case)
                    if case not in audited_case_events:
                        audited_case_events.add(case)
                        prep_counts['records'] += prep.get('records', 0)
                        source = read(case/'oracle/model_source.json') if positive and condition == 'model_mismatch' else oracle.get('generation')
                        if source is not None:
                            role = 'substitute' if positive and condition == 'model_mismatch' else 'declared'
                            source_eos = substitute_tokenizer.eos_token_id if role == 'substitute' else tokenizer.eos_token_id
                            check(len(source['generated_ids']) == 96 and source['full_ids'] == source['input_ids']+source['generated_ids'], 'generation_length', case)
                            check(all(token in valid[role] and token != source_eos for token in source['generated_ids']), 'generation_token_policy', case)
                            events = source['events']
                            if not positive:
                                check(source['input_ids'] == requests[index]['input_ids'] and not events, 'clean_generation', case)
                            elif condition == 'model_mismatch':
                                check(source == oracle.get('source_generation') and len(events) == 1 and events[0].get('kind') == 'model', 'substitute_source', case)
                                check(source['input_ids'] == substitute_tokenizer.encode(requests[index]['request_text'], add_special_tokens=True), 'substitute_prompt', case)
                                check(oracle.get('source_model') == config.substitute_model_path, 'source_model', case)
                            else:
                                spec = r._injection_spec(config, seed, condition, index)
                                check(oracle['spec'] == json_value(vars(spec)), 'paired_injection_spec', case)
                                check(source['input_ids'] == requests[index]['input_ids'] or condition == 'input_mismatch', 'declared_input', case)
                                if condition == 'serving_config':
                                    switches = [event for event in events if event.get('event') == 'route_switch']
                                    rounded = [event for event in events if event.get('event') != 'route_switch']
                                    coordinates = {(event['step'], event['layer']) for event in rounded}
                                    expected = {(step,layer) for step in range(spec.step,97) for layer in range(profile['decoder_count'])}
                                    check(len(switches) == 1 and switches[0]['step'] == spec.step and len(rounded) == len(expected) and coordinates == expected and
                                          all(event.get('rounding_bits') == 6 for event in events) and any(event.get('changed_elements',0)>0 for event in rounded), 'serving_events', case)
                                elif len(events) != 1:
                                    check(False, 'single_injection_event', case)
                                else:
                                    event = events[0]
                                    if condition == 'kv_cache':
                                        check(event['layer'] == spec.layer and event['step'] == spec.step and event['layer'] in profile['kv_eligible_layers'] and
                                              event['requested_donor_position'] == spec.donor_position and event['output_delta_l2']>0 and event['historical_output_delta_l2']==0 and
                                              (event['before_k_sha256'] != event['after_k_sha256'] or event['before_v_sha256'] != event['after_v_sha256']), 'kv_event', case)
                                        donor = oracle.get('donor_generation')
                                        check(bool(donor) and len(donor['generated_ids'])==96 and not donor['events'] and
                                              oracle['donor_request'] == requests[(index+1)%config.m], 'kv_donor', case)
                                    elif condition == 'computation':
                                        check(event['layer']==spec.layer and event['step']==spec.step and event['changed_elements']>0 and
                                              event['before_sha256']!=event['after_sha256'] and math.isclose(event['delta_l2']/event['before_l2'],0.01,rel_tol=1e-4), 'computation_event', case)
                                        local_computation += f'{spec.step}/{(spec.layer+1)//2}' in row['rejected_blocks']
                                    elif condition == 'token_selection':
                                        check(event['step']==spec.step and event['triggered'] is True and event['normal_token']!=event['replacement_token'] and
                                              event['replacement_token'] in valid['declared']-special and source['generated_ids'][spec.step-1]==event['replacement_token'] and
                                              event['normal_logit']>=event['replacement_logit'], 'token_selection_event', case)
                                        if row['verdict']=='DETECTED':
                                            steps=[int(value.split('/')[0]) for value in row['rejected_blocks']]
                                            token_rejections[index]={'selection_step':spec.step,'rejected_block_count':len(steps),'rejected_before_or_at_selection':sum(step<=spec.step for step in steps)}
                                    elif condition == 'input_mismatch':
                                        a,b=spec.swap_positions
                                        original=requests[index]['input_ids'];changed=original.copy()
                                        changed[a:a+4],changed[b:b+4]=original[b:b+4],original[a:a+4]
                                        check(event['before']==original and event['after']==changed==source['input_ids'] and event['positions']==[a,b] and
                                              event['span_width']==4 and 1<=a<b<=28 and b-a>=4 and original[a:a+4]!=original[b:b+4], 'input_swap_event', case)
                            if source_condition == 'shared_clean' or condition == 'model_mismatch':
                                expected_bridge=requests[index]['input_ids']+tokenizer.encode(source['response_text'],add_special_tokens=False)
                                check(oracle.get('declared_retokenized_ids')==expected_bridge, 'text_bridge', case)
                                bridge_lengths['positive' if positive else 'negative'][len(expected_bridge)]+=1
                    if index in config.retain_record_indices:
                        check((case/'records').is_dir(), 'required_tensor_retention', case)
                    if (case/'records').is_dir() and case not in audited_retention:
                        audited_retention.add(case)
                        check(r._directory_hashes(case/'records') == {name.removeprefix('records/'):digest for name,digest in commitment['files'].items()}, 'retained_evidence_hash', case)
                        preparations=list((case/'records').rglob('activation_preparation.json'))
                        for path in preparations:
                            prepared=read(path)
                            if prepared:
                                outputs={int(Path(name).parts[0].split('_')[1]) for name in prepared['base_files'] if name.endswith('/output.pt')}
                                check(outputs==set(profile['base_output_blocks']) and len(prepared['responses'])==9 and
                                      {value['block_id'] for value in prepared['responses']}==set(profile['revealed_output_blocks']) and prepared['complete'], 'retained_sparse_structure', path)
                                for response in prepared['responses']:
                                    hashes=read((path.parent/response['path']).with_name('hashes.json'))
                                    check(bool(hashes) and hashes['output_hash']==response['output_hash'], 'retained_original_response_hash', path)
                                    relative_response=str((path.parent/response['path']).relative_to(case))
                                    check(commitment['files'].get(relative_response)==response['sha256'], 'retained_response_file_hash', path)
                                for name,digest in prepared['base_files'].items():
                                    relative_base=str((path.parent/name).relative_to(case))
                                    check(commitment['files'].get(relative_base)==digest, 'retained_base_hash', path)
                        retained_records+=len(preparations)
                    for mode in ('numeric','strict_hash'):
                        blocks=read(case/origin/f'blocks_{mode}.json')
                        if blocks is None:continue
                        grid(blocks,origin,relative)
                        threshold=thresholds['thresholds'][origin] if mode=='numeric' else math.inf
                        check(row['verdict' if mode=='numeric' else 'strict_hash_verdict']==_verdict(blocks,threshold), 'sequence_verdict', f'{relative}/{mode}')
                        decisions=decide_sequence(blocks,threshold)
                        check(not decisions['error_blocks'], 'invalid_block_scores', relative)
                        if mode=='numeric':
                            check(all(row.get(field)==decisions[field] for field in ('score','verdict','checked_blocks','rejected_blocks','error_blocks')) and row['block_count']==len(blocks), 'numeric_decision_details', relative)
                            if full_sampling:
                                expected=simulate_sampling(row['case_key'],blocks,threshold,config.sample_rates,config.sample_repeats,row['audit_seed'])
                                check(expected==row['sampling'], 'sampling_reconstruction', relative)
                        else:
                            check(row['strict_hash_rejected_blocks']==decisions['rejected_blocks'], 'strict_decision_details', relative)
                if relative in unique:rows.append(unique[relative])
            groups[key]=rows
            saved=summary['groups'].get(key,{})
            check(len(rows)==config.m and saved.get('observed_cases')==config.m, 'group_size', key)
            check(summary_agrees(saved.get('numeric'),rows), 'numeric_summary', key)
            check(summary_agrees(saved.get('strict_hash'),rows,'strict_hash'), 'strict_summary', key)
            check(saved.get('sampling')==summarize_sampling(rows,config.sample_rates,config.sample_repeats), 'sampling_summary', key)
    seed_count=len(config.seeds)
    check(len(tasks)==summary['unique_generation_tasks']==seed_count*(config.m+5*config.n), 'unique_tasks', root)
    check(len(unique)==summary['unique_sequence_evidence']==seed_count*(2*config.m+4*config.n), 'unique_evidence', root)
    check(summary['condition_sample_references']==seed_count*6*config.m and summary['complete'], 'complete_views', root)
    expected_retained=0
    for seed in config.seeds:
        positive=set(corpus['positive_indices'][str(seed)])
        for index in config.retain_record_indices:
            if index<config.m:expected_retained+=(5*96+1 if index in positive else 97)
    check(prep_counts['records']==seed_count*(97*config.m+384*config.n) and retained_records==expected_retained, 'activation_preparation_counts', root)
    check(sum(marker.get('proof_bytes') or 0 for marker in task_markers.values())==summary['actual_costs']['proof_bytes'], 'unique_proof_cost', root)
    stages=defaultdict(list)
    for marker in task_markers.values():
        for stage,seconds in marker.get('stage_seconds',{}).items():stages[stage].append(float(seconds))
    check(set(stages)==set(summary['actual_costs']['stage_seconds']) and all(math.isclose(math.fsum(values),summary['actual_costs']['stage_seconds'][stage],rel_tol=1e-12,abs_tol=1e-6) for stage,values in stages.items()), 'unique_stage_cost', root)
    return {'audited_at':datetime.now(timezone.utc).isoformat(),'passed':not counts,'issue_counts':dict(counts),'issues':issues,
            'full_sampling_reconstruction':full_sampling,
            'observed':{'unique_generation_tasks':len(tasks),'unique_sequence_evidence':len(unique),'condition_sample_references':sum(map(len,groups.values()))},
            'model_bridge_lengths':{label:dict(sorted(values.items())) for label,values in bridge_lengths.items()},
            'token_numeric_rejections':token_rejections,'computation_injected_block_detected':local_computation,
            'activation_preparation_records':prep_counts['records'],'retained_preparation_records':retained_records}


def render(root: Path, audit_report: dict) -> str:
    if not audit_report.get('passed'):
        raise ValueError('Full completion audit must pass before formal reporting')
    summary, manifest = r._read_json(root/'summary.json'), r._read_json(root/'manifest.json')
    previous = r._read_json(Path(manifest['paired_reference']['reference_run'])/'summary.json')
    def pct(value):return '未定义' if value is None else f'{100*value:.2f}%'
    def ci(value):return '未定义' if value is None else f'[{pct(value[0])}, {pct(value[1])}]'
    lines=['# Qwen3.5-0.8B 推理完整性实验结果（v4）','',
           '声明/可信参考模型为 Qwen3.5-0.8B；Model Mismatch 源为 Llama-3.2-1B-Instruct。GPU推理与记录、CPU验证，FP32，32内容token＋96生成token，不加BOS或chat template。','',
           'B_L=2、I_A=4、原生B_S=1；26个跟踪模块构成13块，初始保存0/4/8/12并揭示其余9个原始输出；full attention使用SDPA，linear attention使用冻结的Torch fallback。','',
           '复用v3的288个原始数据行、阳性标签及模型无关注入时序，重新生成全部响应、证据和阈值。224个共享阴性＋六类各32个阳性＝416个正式源任务，640条去重证据、1536次引用。数据源与时序配对不等于严格相同截断文本、KV层覆盖或审计成本；引用和抽样重复不扩大独立样本量。','']
    for mode,title in [('numeric','数值主模式'),('strict_hash','严格输出哈希模式')]:
        lines += [f'## {title}','','| 条件 | TP | FN | FP | TN | ERROR | TPR | FPR | Precision | F1 | TPR 95% Wilson | FPR 95% Wilson | 覆盖率 | deployment recall |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|']
        for key,group in summary['groups'].items():
            m=group[mode];c=m['counts']
            lines.append('| '+' | '.join([key.split('/')[1],*[str(c[name]) for name in ('TP','FN','FP','TN','ERROR')],*[pct(m[name]) for name in ('tpr','fpr','precision','f1')],ci(m['tpr_wilson_95']),ci(m['fpr_wilson_95']),pct(m['decidable_coverage']),pct(m['deployment_recall'])])+' |')
    lines += ['','## 8%抽样模拟','','D为100/1248（8.01%），Model为2/13（15.38%）；每条证据1000次相关重复，全量核对后模拟，不作为真实重复TEE调用性能。','','| 条件 | TP | FN | FP | TN | 错误 | 抽样TPR | 抽样FPR | 理论平均阳性命中率 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,group in summary['groups'].items():
        m=group['sampling']['0.08'];c=m['counts']
        lines.append('| '+' | '.join([key.split('/')[1],*[str(c[name]) for name in ('TP','FN','FP','TN')],str(c['ERROR_POSITIVE']+c['ERROR_NEGATIVE']),pct(m['tpr']),pct(m['fpr']),pct(m['mean_exact_positive_hit_probability'])])+' |')
    lines += ['','## 与v3配对数据源的对照','','| 条件 | v3全量TPR | v4全量TPR | v3抽样TPR | v4抽样TPR |','|---|---:|---:|---:|---:|']
    for key,group in summary['groups'].items():
        old=previous['groups'][key]
        lines.append('| '+' | '.join([key.split('/')[1],pct(old['numeric']['tpr']),pct(group['numeric']['tpr']),pct(old['sampling']['0.08']['tpr']),pct(group['sampling']['0.08']['tpr'])])+' |')
    lines += ['','## 实测长度、真值与成本','',f"Model桥接总长度分布：`{audit_report['model_bridge_lengths']}`；不强制跨tokenizer等长。",'',f"Token数值拒绝时序：`{audit_report['token_numeric_rejections']}`。选择发生在该step的forward之后，事前拒绝不能直接归因于替换。",'',f"Computation拒绝覆盖实际注入块：{audit_report['computation_injected_block_detected']}/32。",'',f"激活准备记录{audit_report['activation_preparation_records']}份，必保留索引0/64/128/192的准备记录{audit_report['retained_preparation_records']}份。",'',f"去重任务累计证据产出：{summary['actual_costs']['proof_bytes']/2**30:.2f} GiB；包含按方案已释放的临时张量。阶段耗时秒数：`{summary['actual_costs']['stage_seconds']}`。校准、参考快照和失败/中断重试另计。",'',
           '各类注入成功率、未知/失败/不支持数以及全部源映射详见summary.json；错误不塞入四格。GPU与CPU的正常FP32差异可造成数值或严格哈希拒绝，严格模式单列。KV只评估6个full-attention层的完整前缀K/V替换，不覆盖linear recurrent state；Serving是舍入代理。Model证据是Qwen重算trace，不单独绑定真实生成模型；Input证据不包含可信客户端原始输入锚；lm_head及返回token关系未新增验证，未实施TEE远程认证。','',
           f"配置SHA-256：`{manifest['config_sha256']}`；语料SHA-256：`{manifest['corpus_sha256']}`；配对时序SHA-256：`{manifest['paired_schedule_sha256']}`。",'',f'结果目录：`{root}`。完整审计及全部抽样重建通过。','']
    return '\n'.join(lines)
