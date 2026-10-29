"""Audit v3 source evidence, shared references, decisions and sampling."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import re

from .config import CONDITIONS, EVIDENCE_ORIGINS, load_config
from .metrics import calibrate_threshold, simulate_sampling, summarize_cases, summarize_sampling
from .stage_audit import _verdict
from . import runner as r


def audit(root: Path, *, full_sampling: bool = True) -> dict:
    root = root.resolve()
    manifest = r._read_json(root / 'manifest.json')
    config = manifest['config']
    issues = []
    counts = Counter()
    def check(ok, code, location, detail=''):
        if not ok:
            counts[code] += 1
            if len(issues) < 50:
                issues.append({'code': code, 'location': str(location), 'detail': str(detail)[:500]})
    def read(path):
        try:
            return r._read_json(path)
        except (OSError, ValueError) as exc:
            check(False, 'missing_or_invalid_json', path, exc)
            return None
    config_hash = sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    check(manifest['config_sha256'] == config_hash, 'config_hash', 'manifest.json')
    expected_config = load_config(Path(__file__).with_name('formal-seed0-cal32-v3.json'))
    # Paths are absolute in manifests; compare all protocol settings directly.
    for key, value in expected_config.as_dict().items():
        if key not in ('declared_model_path', 'substitute_model_path', 'dataset_jsonl'):
            check(config.get(key) == (list(value) if isinstance(value, tuple) else value),
                  'protocol_config', key, config.get(key))
    check(r._code_digest() == manifest['experiment_code_sha256'], 'code_hash', 'manifest.json')
    check(r._hash_file(root / 'corpus.json') == manifest['corpus_sha256'], 'corpus_hash', 'corpus.json')
    check(r._hash_file(Path(manifest['dataset']['path'])) == manifest['dataset']['sha256'],
          'dataset_hash', 'manifest.json')
    for key in ('declared_model', 'substitute_model'):
        for name, digest in manifest[key]['files'].items():
            check(r._hash_file(Path(manifest[key]['path']) / name) == digest, 'model_hash', f'{key}/{name}')
    ready = read(root / 'reference' / 'ready.json')
    if ready:
        check(r._directory_hashes(root / 'reference', exclude={'ready.json'}) == ready['files'],
              'reference_hash', 'reference/ready.json')
    metadata = read(root / 'reference' / 'metadata.json')
    layers = ['embedding'] + [f'layer_{i}' for i in range(16)] + ['norm']
    expected_blocks = {str(i): layers[i * 2:(i + 1) * 2] for i in range(9)}
    if metadata:
        check(metadata['block_to_layers'] == expected_blocks, 'layer_partition', 'reference/metadata.json')
        check(metadata['activation_interval'] == 4 and metadata['steps_per_block'] == 1,
              'native_recording_parameters', 'reference/metadata.json')
    corpus = read(root / 'corpus.json')
    thresholds = read(root / 'thresholds.json')
    summary = read(root / 'summary.json')
    if not corpus or not thresholds or not summary:
        return {'passed': False, 'issue_counts': dict(counts), 'issues': issues}
    check(thresholds['config_sha256'] == config_hash and thresholds['calibration_size'] == 32,
          'threshold_identity', 'thresholds.json')
    check(set(thresholds['thresholds']) == set(EVIDENCE_ORIGINS), 'threshold_origins', 'thresholds.json')
    check(len(corpus['calibration']) == 32, 'calibration_size', 'corpus.json')
    cal_ids = {tuple(row['input_ids']) for row in corpus['calibration']}
    cal_rows = {row['source_row'] for row in corpus['calibration']}
    cal_texts = {row['request_text_sha256'] for row in corpus['calibration']}

    def block_grid(blocks, origin, location):
        expected_steps = range(1, 97) if origin == 'generation_live' else (0,)
        expected = {(step, block) for step in expected_steps for block in range(9)}
        coordinates = [(b.get('step'), b.get('layer_block_id')) for b in blocks]
        check(len(coordinates) == len(expected) and set(coordinates) == expected,
              'block_grid', location)
        for b in blocks:
            block = b.get('layer_block_id')
            if isinstance(block, int) and 0 <= block < 9:
                check(b.get('block_id') == f'{b["step"]}/{block}' and
                      b.get('layer_name') == expected_blocks[str(block)][-1], 'block_identity', location)

    for origin in EVIDENCE_ORIGINS:
        cal = []
        for index in range(32):
            row = read(root / 'calibration' / f'{index:04d}' / f'blocks_{origin}.json')
            marker = read(root / 'calibration' / f'{index:04d}' / 'calibration.json')
            if row and marker:
                check(marker['request_sha256'] == corpus['calibration'][index]['request_text_sha256'],
                      'calibration_request', index)
                block_grid(row, origin, f'calibration/{index}/{origin}')
                cal.append(row)
        try:
            check(len(cal) == 32 and calibrate_threshold(cal) == thresholds['thresholds'][origin],
                  'calibration_threshold', origin)
        except ValueError as exc:
            check(False, 'calibration_error', origin, exc)

    unique = {}
    tasks = set()
    group_rows = {}
    for seed in config['seeds']:
        seed_bytes = json.dumps(['labels', seed, config_hash], ensure_ascii=False, separators=(',', ':')).encode()
        positives = set(random.Random(int.from_bytes(sha256(seed_bytes).digest()[:8], 'big')).sample(range(256), 32))
        check(corpus.get('positive_indices', {}).get(str(seed)) == sorted(positives), 'frozen_labels', seed)
        requests = corpus['evaluation'][str(seed)]
        check(len(requests) == 256 and len({tuple(q['input_ids']) for q in requests}) == 256,
              'evaluation_size', seed)
        check(not ({tuple(q['input_ids']) for q in requests} & cal_ids) and
              not ({q['source_row'] for q in requests} & cal_rows) and
              not ({q['request_text_sha256'] for q in requests} & cal_texts), 'calibration_overlap', seed)
        mapping = manifest['shared_controls'][str(seed)]
        check(mapping['indices'] == sorted(set(range(256)) - positives), 'shared_control_indices', seed)
        for condition in CONDITIONS:
            origin = 'model_recompute_live' if condition == 'model_mismatch' else 'generation_live'
            check(mapping['condition_origins'].get(condition) == origin, 'shared_origin', condition)
            key = f'{seed}/{condition}/{origin}'
            rows = []
            for index in range(256):
                positive = index in positives
                case = (root / 'cases' / str(seed) / condition / f'{index:04d}' if positive else
                        root / 'shared_clean' / str(seed) / f'{index:04d}')
                relative = str((case / origin / 'result.json').relative_to(root))
                check(summary.get('groups', {}).get(key, {}).get('sources', {}).get(str(index)) == relative,
                      'source_reference', f'{key}/{index}')
                if relative not in unique:
                    row = read(root / relative)
                    marker = read(case / 'result.json')
                    oracle = read(case / 'oracle' / 'truth.json')
                    commitment = read(case / 'commitment.json')
                    if not all(x is not None for x in (row, marker, oracle, commitment)):
                        continue
                    tasks.add(str(case.relative_to(root)))
                    unique[relative] = row
                    identity = (row.get('seed'), row.get('index'), row.get('config_sha256'),
                                row.get('evidence_origin'), row.get('planned_positive'))
                    check(identity == (seed, index, config_hash, origin, positive), 'source_identity', relative)
                    source_condition = condition if positive else 'shared_clean'
                    check(row.get('condition') == source_condition and
                          row.get('case_key') == f'{seed}/{source_condition}/{index:04d}/{origin}',
                          'sampling_identity', relative)
                    check(not marker.get('status_error') and not row.get('status_error'), 'execution_error', relative)
                    check(oracle.get('source_request') == requests[index] and
                          oracle.get('planned_positive') == positive, 'oracle_request', relative)
                    check(row.get('injection_success') is True, 'injection_success', relative)
                    check('final_recompute' not in marker['origins'], 'unexpected_origin', case)
                    if positive:
                        check(marker['origins'] == [origin], 'positive_origins', case)
                        if condition == 'model_mismatch':
                            source = read(case / 'oracle' / 'model_source.json')
                            check(bool(source) and len(source['generated_ids']) == 96 and
                                  len(source['events']) == 1 and source['events'][0].get('kind') == 'model',
                                  'model_source', case)
                            check(oracle.get('source_model') == config['substitute_model_path'], 'substitute_model', case)
                        else:
                            events = oracle['generation']['events']
                            check(len(oracle['generation']['generated_ids']) == 96 and bool(events), 'positive_events', case)
                            if condition == 'serving_config':
                                check(oracle['spec']['rounding_bits'] == 6 and
                                      sum(e.get('event') == 'route_switch' for e in events) == 1,
                                      'serving_r6', case)
                            elif condition == 'kv_cache':
                                check(len(events) == 1 and events[0]['output_delta_l2'] > 0 and
                                      (events[0]['before_k_sha256'] != events[0]['after_k_sha256'] or
                                       events[0]['before_v_sha256'] != events[0]['after_v_sha256']), 'kv_change', case)
                            else:
                                check(len(events) == 1, 'single_injection', case)
                    else:
                        check(marker['origins'] == list(EVIDENCE_ORIGINS) and
                              len(oracle['generation']['generated_ids']) == 96 and not oracle['generation']['events'],
                              'shared_clean_generation', case)
                        if origin == 'model_recompute_live':
                            check('declared_retokenized_ids' in oracle, 'clean_text_bridge', case)
                    check(bool(commitment.get('files')) and all(re.fullmatch(r'[0-9a-f]{64}', v)
                          for v in commitment['files'].values()), 'commitment_structure', case)
                    check(commitment.get('activation_preparation', {}).get('method') == 'original_live_tensor_response',
                          'activation_preparation_method', case)
                    if (case / 'records').exists():
                        check(r._directory_hashes(case / 'records') == {
                            name.removeprefix('records/'): digest for name, digest in commitment['files'].items()
                        }, 'retained_evidence_hash', case)
                    for mode in ('numeric', 'strict_hash'):
                        blocks = read(case / origin / f'blocks_{mode}.json')
                        if blocks is None:
                            continue
                        block_grid(blocks, origin, relative)
                        expected_verdict = _verdict(blocks, thresholds['thresholds'][origin] if mode == 'numeric' else math.inf)
                        check(row['verdict' if mode == 'numeric' else 'strict_hash_verdict'] == expected_verdict,
                              'verdict', f'{relative}/{mode}')
                        check(not row.get('error_blocks') and expected_verdict != 'ERROR', 'undecidable_evidence', relative)
                        if mode == 'numeric':
                            check(row['block_count'] == len(blocks), 'block_count', relative)
                            if full_sampling:
                                sampled = simulate_sampling(row['case_key'], blocks, thresholds['thresholds'][origin],
                                                            config['sample_rates'], config['sample_repeats'], row['audit_seed'])
                                check(sampled == row.get('sampling'), 'sampling_reconstruction', relative)
                if relative in unique:
                    rows.append(unique[relative])
            group_rows[key] = rows
            group = summary['groups'].get(key, {})
            check(len(rows) == 256 and group.get('observed_cases') == 256, 'group_size', key)
            # Wilson intervals are tuples in memory and arrays in saved JSON.
            # Compare the persisted representation without changing any values.
            numeric_summary = json.loads(json.dumps(summarize_cases(rows), allow_nan=False))
            strict_summary = json.loads(json.dumps(summarize_cases([
                {**row, 'verdict': row.get('strict_hash_verdict', 'ERROR')} for row in rows]), allow_nan=False))
            check(group.get('numeric') == numeric_summary, 'numeric_summary', key)
            check(group.get('strict_hash') == strict_summary, 'strict_summary', key)
            check(group.get('sampling') == summarize_sampling(rows, config['sample_rates'], config['sample_repeats']),
                  'sampling_summary', key)
    expected_tasks = len(config['seeds']) * 416
    expected_evidence = len(config['seeds']) * 640
    check(len(tasks) == expected_tasks and summary['unique_generation_tasks'] == expected_tasks, 'unique_tasks', root)
    check(len(unique) == expected_evidence and summary['unique_sequence_evidence'] == expected_evidence, 'unique_evidence', root)
    check(summary['condition_sample_references'] == len(config['seeds']) * 1536 and summary['complete'], 'complete_views', root)
    return {'audited_at': datetime.now(timezone.utc).isoformat(), 'passed': not counts,
            'issue_counts': dict(counts), 'issues': issues, 'full_sampling_reconstruction': full_sampling,
            'observed': {'unique_generation_tasks': len(tasks), 'unique_sequence_evidence': len(unique),
                         'condition_sample_references': sum(len(rows) for rows in group_rows.values())}}


def render(root: Path, audit_report: dict) -> str:
    if not audit_report.get('passed'):
        raise ValueError('Completion audit must pass before rendering formal results')
    summary = r._read_json(root / 'summary.json')
    manifest = r._read_json(root / 'manifest.json')
    lines = ['# AFTUNE 推理完整性实验结果（v3）', '',
             'GPU 推理与证据记录，CPU 验证；FP32；B_L=2、I_A=4；Serving 仅 r6；采样率 8%。', '',
             '六类实验共享 224 个阴性生成样本，各有 32 个阳性；共 416 个正式目标生成任务、640 条去重序列证据、1,536 次条件统计引用。共享样本及重复抽样不增加独立样本量。', '',
             '| 条件/证据 | TP | FN | FP | TN | ERROR | TPR | FPR | Precision | F1 | TPR 95% Wilson | FPR 95% Wilson |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    def pct(value):
        return '未定义' if value is None else f'{value * 100:.2f}%'
    def interval(value):
        return '未定义' if value is None else f'[{pct(value[0])}, {pct(value[1])}]'
    for key, group in summary['groups'].items():
        row = group['numeric']
        c = row['counts']
        lines.append('| ' + ' | '.join([key, *[str(c[k]) for k in ('TP', 'FN', 'FP', 'TN', 'ERROR')],
                     *[pct(row[k]) for k in ('tpr', 'fpr', 'precision', 'f1')],
                     interval(row['tpr_wilson_95']), interval(row['fpr_wilson_95'])]) + ' |')
    lines.extend(['', '## 8% 抽样结果', '',
                  'D 每轮抽取 70/864 个单元；Model 抽取 1/9 个块。每份源证据重复 1,000 次，五类 D 实验复用共享阴性的同一套抽样。以下为相关重复的汇总，不据此计算独立请求置信区间。', '',
                  '| 条件/证据 | 采样 TP | FN | FP | TN | 错误 | 采样 TPR | 采样 FPR |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for key, group in summary['groups'].items():
        row = group['sampling']['0.08']
        c = row['counts']
        lines.append('| ' + ' | '.join([key, *[str(c[k]) for k in ('TP', 'FN', 'FP', 'TN')],
                     str(c['ERROR_POSITIVE'] + c['ERROR_NEGATIVE']), pct(row['tpr']), pct(row['fpr'])]) + ' |')
    lines.extend(['', '## 严格输出哈希模式', '',
                  'GPU 与 CPU 的正常 FP32 数值差异也可能产生哈希不一致；该模式单列，不能直接把哈希拒绝解释为违例。', '',
                  '| 条件/证据 | TP | FN | FP | TN | ERROR |', '|---|---:|---:|---:|---:|---:|'])
    for key, group in summary['groups'].items():
        c = group['strict_hash']['counts']
        lines.append('| ' + ' | '.join([key, *[str(c[k]) for k in ('TP', 'FN', 'FP', 'TN', 'ERROR')]]) + ' |')
    lines.extend(['', '## 复现与边界', '',
                  f"D/Model 冻结阈值：`{r._read_json(root / 'thresholds.json')['thresholds']}`。", '',
                  f"配置 SHA-256：`{manifest['config_sha256']}`。", '',
                  f"语料 SHA-256：`{manifest['corpus_sha256']}`。", '',
                  f'运行目录：`{root}`。完整性审计通过，抽样结果已由保存的源证据标识和种子重建核对。', '',
                  '稀疏记录初始仅保存块 0、4、8 的激活；全量审计准备阶段揭示真实前向暂存的其余原始边界张量并核验原承诺，不使用干净重算替换。补齐会增加实际存储及执行开销，相关成本保存在各任务 commitment 与 summary 中。', '',
                  '这是固定 seed、小模型、短输入的机制评估。KV 为完整前缀模拟，Serving 为数值舍入代理；未实施 TEE 远程认证。Model 重算通过只能证明提交的 Llama trace 自洽，不能证明源文本由 Llama 生成。', ''])
    return '\n'.join(lines)
