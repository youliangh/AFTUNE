"""Full-length GPU-to-CPU gate for the v3 protocol, isolated from formal results."""
import argparse
from pathlib import Path
import time

import torch

from .config import load_config
from . import runner as r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.output.resolve()
    r.configure_determinism(0)
    torch.set_num_threads(config.cpu_threads)
    started = time.monotonic()
    r.prepare_reference(config, root)
    r.calibrate(config, root, limit=1)
    thresholds = {origin: r.calibrate_threshold([
        r._read_json(root / 'calibration' / '0000' / f'blocks_{origin}.json')
    ]) for origin in r.EVIDENCE_ORIGINS}
    corpus = r._load_corpus(config, root)
    seed = config.seeds[0]
    requests = corpus['evaluation'][str(seed)]
    positives = r._positive_indices(config, seed)
    index = min(positives)
    r._prepare_model_sources(config, root, seed, requests, positives, [index], False)
    reference = r._verified_reference(config, root)
    model, tokenizer = r._load_llama(config)
    outcomes = {}
    try:
        for condition in ('shared_clean', *r.CONDITIONS):
            if condition == 'model_mismatch':
                r._run_model_case(config, root, seed, index, requests[index], True,
                                  model, tokenizer, reference, thresholds)
            else:
                r._run_non_model_case(config, root, seed, condition, index, requests[index],
                                      condition != 'shared_clean', model, tokenizer, reference,
                                      thresholds, requests[(index + 1) % config.m])
            case = r._case_dir(root, seed, condition, index)
            marker = r._read_json(case / 'result.json')
            rows = {origin: r._read_json(case / origin / 'result.json') for origin in marker['origins']}
            if marker.get('status_error') or any(row.get('error_blocks') for row in rows.values()):
                raise RuntimeError(f'Preflight execution error in {condition}: {marker}')
            if condition != 'shared_clean' and not all(row['injection_success'] for row in rows.values()):
                raise RuntimeError(f'Preflight injection failed: {condition}')
            if condition == 'shared_clean' and any(row['verdict'] == 'ERROR' for row in rows.values()):
                raise RuntimeError('Shared clean preflight was undecidable')
            outcomes[condition] = {origin: {key: row.get(key) for key in (
                'verdict', 'score', 'block_count', 'injection_success', 'strict_hash_verdict')}
                for origin, row in rows.items()}
            r._progress(root, 'preflight_condition_finished', condition=condition,
                        outcomes=outcomes[condition])
    finally:
        del model
        r._collect_device()
    report = {'passed': True, 'diagnostic_only': True, 'config_sha256': config.fingerprint(),
              'calibration_requests': 1, 'diagnostic_thresholds': thresholds, 'outcomes': outcomes,
              'elapsed_seconds': time.monotonic() - started}
    r._atomic_json(root / 'preflight.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
