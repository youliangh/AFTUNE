"""Verify archive hashes, all score decisions/metrics and every saved sample draw.

Uses the standard library and the frozen dependency-free metrics module. This
does not rerun GPU generation, CPU model replay or original tensor commitments.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.inference_integrity.metrics import (calibrate_threshold, decide_sequence,
    simulate_sampling, summarize_cases, summarize_sampling)


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def persisted(value):
    return json.loads(json.dumps(value, allow_nan=False))


def unpack(archive_path, metadata, root):
    assert sha(archive_path.read_bytes()) == metadata['sha256'], archive_path
    observed = set()
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert member.name in metadata['files'] and member.name not in observed
            observed.add(member.name)
            data = archive.extractfile(member).read()
            expected = metadata['files'][member.name]
            assert len(data) == member.size == expected['bytes'] and sha(data) == expected['sha256'], member.name
            output = root/path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
    assert observed == set(metadata['files'])


def verify(root, full_sampling):
    manifest, summary, thresholds, corpus, audit = [read(root/name) for name in
        ('manifest.json','summary.json','thresholds.json','corpus.json','completion_audit.json')]
    assert audit['passed'] and audit['full_sampling_reconstruction'] and summary['complete']
    assert sha((root/'corpus.json').read_bytes()) == manifest['corpus_sha256']
    assert sha(json.dumps(manifest['config'],sort_keys=True,separators=(',',':')).encode()) == manifest['config_sha256']
    assert thresholds['config_sha256'] == summary['config_sha256'] == manifest['config_sha256']
    for name, expected in manifest['experiment_code_sha256'].items():
        package = 'inference_integrity_qwen' if manifest['config']['protocol_version']=='v4' else 'inference_integrity'
        path = REPO/'experiments'/package/name
        if not path.is_file():
            path = REPO/name
        assert sha(path.read_bytes()) == expected, path
    for origin, threshold in thresholds['thresholds'].items():
        blocks = [read(root/f'calibration/{i:04d}/blocks_{origin}.json') for i in range(32)]
        assert calibrate_threshold(blocks) == threshold
    unique, cases = {}, set()
    for key, group in summary['groups'].items():
        rows = []
        for index, relative in group['sources'].items():
            if relative not in unique:
                row = read(root/relative)
                case = (root/relative).parent.parent
                commitment = read(case/'commitment.export.json')
                truth = read(case/'oracle/truth.json')
                assert truth['injection_success'] is True and row['injection_success'] is True
                assert not row.get('status_error') and not row['error_blocks']
                assert truth['source_request'] == corpus['evaluation'][str(row['seed'])][row['index']]
                frozen, finished = datetime.fromisoformat(commitment['frozen_at']), datetime.fromisoformat(row['finished_at'])
                assert datetime.fromisoformat(thresholds['created_at']) <= frozen <= finished
                if manifest['config']['protocol_version']=='v4':
                    assert frozen <= datetime.fromisoformat(row['challenge_created_at']) <= finished
                origin = row['evidence_origin']
                blocks_count = 13 if manifest['config']['protocol_version']=='v4' else 9
                steps = range(1,97) if origin=='generation_live' else (0,)
                for mode in ('numeric','strict_hash'):
                    blocks = read(case/origin/f'blocks_{mode}.json')
                    assert len(blocks)==len(steps)*blocks_count
                    assert {(b['step'],b['layer_block_id']) for b in blocks}=={(s,b) for s in steps for b in range(blocks_count)}
                    threshold = thresholds['thresholds'][origin] if mode=='numeric' else math.inf
                    decision = decide_sequence(blocks, threshold)
                    assert not decision['error_blocks']
                    assert decision['verdict']==row['verdict' if mode=='numeric' else 'strict_hash_verdict']
                    assert decision['rejected_blocks']==row['rejected_blocks' if mode=='numeric' else 'strict_hash_rejected_blocks']
                    if mode=='numeric':
                        assert all(row[k]==decision[k] for k in ('score','checked_blocks','verdict','rejected_blocks','error_blocks'))
                        if full_sampling:
                            assert simulate_sampling(row['case_key'],blocks,threshold,manifest['config']['sample_rates'],1000,row['audit_seed'])==row['sampling']
                unique[relative] = row
                cases.add(case)
                if len(unique)%128==0:
                    print(json.dumps({'evidence_checked':len(unique),'full_sampling':full_sampling}),flush=True)
            row = unique[relative]
            assert row['index']==int(index)
            seed,condition,origin = key.split('/')
            assert row['seed']==int(seed) and row['evidence_origin']==origin
            assert row['config_sha256']==manifest['config_sha256']
            assert row['planned_positive']==(int(index) in corpus['positive_indices'][seed])
            assert row['condition']==(condition if row['planned_positive'] else 'shared_clean')
            rows.append(row)
        assert len(rows)==256 and sum(bool(r['planned_positive']) for r in rows)==32
        assert group['numeric']==persisted(summarize_cases(rows))
        assert group['strict_hash']==persisted(summarize_cases([{**r,'verdict':r['strict_hash_verdict']} for r in rows]))
        assert group['sampling']==summarize_sampling(rows,manifest['config']['sample_rates'],1000)
    assert len(unique)==summary['unique_sequence_evidence']==640
    assert len(cases)==summary['unique_generation_tasks']==416
    assert sum(len(g['sources']) for g in summary['groups'].values())==summary['condition_sample_references']==1536
    assert sum(read(case/'result.json')['proof_bytes'] for case in cases)==summary['actual_costs']['proof_bytes']
    markers = [read(case/'result.json') for case in cases]
    for stage, seconds in summary['actual_costs']['stage_seconds'].items():
        assert math.isclose(math.fsum(marker.get('stage_seconds',{}).get(stage,0) for marker in markers),seconds,rel_tol=1e-12,abs_tol=1e-6)
    return {'sources':len(cases),'evidence':len(unique),'references':1536,'full_sampling_reconstruction':full_sampling}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path(__file__).parent)
    parser.add_argument('--quick',action='store_true',help='Skip rebuilding 1,000 draws; still verify hashes, grids, metrics and stored sampling aggregates.')
    args = parser.parse_args()
    metadata = read(args.bundle/'bundle-manifest.json')
    for label, run in metadata['runs'].items():
        with tempfile.TemporaryDirectory(prefix='aftune-artifact-') as directory:
            root = Path(directory)
            unpack(args.bundle/run['archive'],run,root)
            observed = verify(root,not args.quick)
            assert {k:observed[k] for k in ('sources','evidence','references')}=={'sources':416,'evidence':640,'references':1536}
            print(json.dumps({'run':label,'passed':True,**observed}),flush=True)


if __name__ == '__main__':
    main()
