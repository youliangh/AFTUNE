"""Export the completed v3/v4 score evidence without model/tensor binaries."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

REPO = Path(__file__).resolve().parents[2]
RUNS = {
    'llama-v3': 'aftune-inference-integrity-seed0-cal32-v3',
    'qwen-v4': 'aftune-inference-integrity-qwen-seed0-cal32-v4',
}
ROOT_FILES = (
    'manifest.json', 'corpus.json', 'thresholds.json', 'summary.json',
    'completion_audit.json', 'completion_review.json', 'final_verification.json',
    'runner_state.json', 'paired_schedule.json', 'paired_comparison.json',
    'preflight_gate.json', 'post_audit_review.json', 'pause_checkpoint.json',
    'resume_checkpoint.json', 'resume_launch.json', 'monitor.latest.json',
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def payloads(root):
    for name in ROOT_FILES:
        path = root/name
        if path.is_file():
            yield name, path.read_bytes()
    for path in sorted(root.glob('calibration/*/*.json')):
        yield path.relative_to(root).as_posix(), path.read_bytes()
    cases = sorted(root.glob('shared_clean/*/*/result.json')) + sorted(root.glob('cases/*/*/*/result.json'))
    assert len(cases) == 416
    for marker_path in cases:
        case = marker_path.parent
        marker = json.loads(marker_path.read_bytes())
        assert not marker.get('status_error')
        for path in [marker_path, *sorted((case/'oracle').glob('*.json'))]:
            yield path.relative_to(root).as_posix(), path.read_bytes()
        commitment_path = case/'commitment.json'
        original = commitment_path.read_bytes()
        commitment = json.loads(original)
        files = commitment.pop('files')
        commitment['export'] = {
            'original_commitment_sha256': sha(original),
            'committed_file_count': len(files),
            'omitted': 'Per-tensor file hash map; original tensor evidence remains in the local run.',
        }
        yield (case/'commitment.export.json').relative_to(root).as_posix(), (json.dumps(commitment, ensure_ascii=False, indent=2)+'\n').encode()
        for origin in marker['origins']:
            for name in ('result.json', 'blocks_numeric.json', 'blocks_strict_hash.json'):
                path = case/origin/name
                yield path.relative_to(root).as_posix(), path.read_bytes()
    report = root.with_name(root.name+'-实验结果.md')
    yield 'report.md', report.read_bytes()
    for directory in (root.with_name(root.name+'-preflight'), root.with_name(root.name+'-preflight-2')):
        for name in ('preflight.json', 'failure_diagnosis.json'):
            path = directory/name
            if path.is_file():
                yield f'diagnostics/{directory.name}/{name}', path.read_bytes()
    analysis = root/'reporting/post_audit_report.py'
    if analysis.is_file():
        yield 'reporting/post_audit_report.py', analysis.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=REPO.parent/'.cache/experiment-runs')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = {'format_version': 1, 'scope': 'Original JSON score/truth evidence; no model or activation tensors.',
              'commitments': 'commitment.export.json preserves header/count/original SHA-256, omitting the tensor file map.',
              'runs': {}}
    for label, name in RUNS.items():
        root = args.runs/name
        audit = json.loads((root/'completion_audit.json').read_bytes())
        assert audit['passed'] and audit['full_sampling_reconstruction']
        target = args.output/(label+'-evidence.tar.gz')
        entries = {}
        with target.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                    for name, data in payloads(root):
                        assert name not in entries, name
                        member = tarfile.TarInfo(name)
                        member.size = len(data)
                        member.mode = 0o644
                        member.mtime = 0
                        archive.addfile(member, io.BytesIO(data))
                        entries[name] = {'sha256':sha(data), 'bytes':len(data)}
        bundle['runs'][label] = {'source_run':root.name, 'archive':target.name,
                                 'sha256':sha(target.read_bytes()), 'bytes':target.stat().st_size,
                                 'files':entries, 'observed':audit['observed']}
        print(json.dumps({'run':label,'files':len(entries),'archive_bytes':target.stat().st_size}), flush=True)
    (args.output/'bundle-manifest.json').write_text(json.dumps(bundle, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
