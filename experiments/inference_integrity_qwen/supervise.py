"""Run and monitor the complete Qwen v4 protocol with an exclusive lock."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import traceback

import torch

from .config import load_config
from .monitor import read_status
from . import runner as r
from .audit import audit, render


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('formal-qwen-seed0-cal32-v4.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--preflight', type=Path, help='Passed actual-checkpoint diagnostic directory')
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error('--interval must be positive')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'run.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('An experiment process already holds this run directory')
    config = load_config(args.config)
    torch.set_num_threads(config.cpu_threads)
    r.configure_determinism(0)
    from .preflight import REQUIRED_CHECKS
    preflight = (args.preflight or root.with_name(root.name + '-preflight')).resolve()
    gate = r._read_json(preflight / 'preflight.json')
    if (not gate.get('passed') or not gate.get('diagnostic_only') or
            gate.get('config_sha256') != config.fingerprint() or
            gate.get('experiment_code_sha256') != r._code_digest() or
            not all(gate.get('checks', {}).get(name) is True for name in REQUIRED_CHECKS)):
        raise SystemExit('A passed full-length GPU-to-CPU preflight for the exact frozen code is required')
    r._atomic_json(root / 'preflight_gate.json', {
        'path': str(preflight), 'preflight_sha256': r._hash_file(preflight / 'preflight.json'),
        'accepted_at': datetime.now(timezone.utc).isoformat(), 'passed': True,
    })
    stopped = threading.Event()
    state_lock = threading.Lock()
    stage = {'name': 'prepare'}
    started = time.monotonic()
    identity = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ', 1)[-1].split()[19]
    def state(status, **extra):
        with state_lock:
            r._atomic_json(root / 'runner_state.json', {
                'pid': os.getpid(), 'process_start_ticks': identity, 'status': status,
                'stage': stage['name'], 'updated_at': datetime.now(timezone.utc).isoformat(),
                'config_sha256': config.fingerprint(), 'elapsed_seconds': time.monotonic() - started,
                **extra,
            })
    def snapshot():
        row = read_status(root, os.getpid())
        row['stage'] = stage['name']
        r._atomic_json(root / 'monitor.latest.json', row)
        with (root / 'monitor.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'monitor': row}, ensure_ascii=False), flush=True)
        state('running')
    def monitor():
        while not stopped.wait(args.interval):
            try:
                snapshot()
            except Exception as exc:
                print(f'Monitor snapshot failed: {exc!r}', flush=True)
    state('running')
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        r.prepare_reference(config, root)
        stage['name'] = 'calibrate'
        state('running')
        r.calibrate(config, root)
        stage['name'] = 'run'
        state('running')
        r.run_cases(config, root, retry_errors=args.retry_errors)
        stage['name'] = 'summarize'
        state('running')
        r.summarize(config, root)
        stage['name'] = 'audit'
        state('running')
        audit_report = audit(root, full_sampling=True)
        r._atomic_json(root / 'completion_audit.json', audit_report)
        if not audit_report['passed']:
            raise RuntimeError(f'Completion audit failed: {audit_report["issue_counts"]}')
        report = root.parent / f'{root.name}-实验结果.md'
        report.write_text(render(root, audit_report), encoding='utf-8')
        stopped.set()
        thread.join(timeout=10)
        snapshot()
        state('complete', report=str(report), audit_passed=True)
        print(json.dumps({'complete': True, 'output': str(root), 'report': str(report)}), flush=True)
    except BaseException as exc:
        stopped.set()
        thread.join(timeout=10)
        state('failed', error=repr(exc), traceback=traceback.format_exc())
        r._progress(root, 'runner_failed', stage=stage['name'], error=repr(exc))
        raise
    finally:
        stopped.set()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
