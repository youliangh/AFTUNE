"""Read-only progress and resource monitor for a running experiment.

Example: python -m experiments.inference_integrity.monitor --watch --pid 12345
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / ".cache/experiment-runs/aftune-inference-integrity-seed0-cal32-v3"


def _gpu_status() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            try:
                rows.append({
                    "index": int(parts[0]), "used_mib": int(parts[1]),
                    "free_mib": int(parts[2]), "utilization_percent": int(parts[3]),
                })
            except ValueError:
                continue
    return {"devices": rows} if rows else None


def _pid_alive(pid: int) -> bool:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[-1].split()
        if fields[0] in ('Z', 'X'):
            return False
    except FileNotFoundError:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_status(root: str | Path, pid: int | None = None) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    events: list[dict[str, Any]] = []
    progress = root / "progress.jsonl"
    if progress.is_file():
        with progress.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    events.append(json.loads(line))
    counts = Counter(event["phase"] for event in events)
    latest_case_state = {}
    for event in events:
        if event.get("phase") in {"case_finished", "case_error"}:
            key = (event.get("seed"), event.get("condition"), event.get("index"))
            latest_case_state[key] = event["phase"]
    latest = events[-1] if events else None
    now = datetime.now(timezone.utc)
    errors = [e for e in events if e.get("phase") in {"case_error", "calibration_error"}]
    disk_path = root
    while not disk_path.exists():
        disk_path = disk_path.parent
    disk = shutil.disk_usage(disk_path)
    manifest_path = root / "manifest.json"
    config: dict[str, Any] = {}
    if manifest_path.is_file():
        config = json.loads(manifest_path.read_text(encoding="utf-8")).get("config", {})
    if config.get('protocol_version') == 'v3':
        expected = len(config.get('seeds', [])) * (int(config.get('m', 0)) + 5 * int(config.get('n', 0)))
    else:
        expected = (len(config.get("seeds", [])) * int(config.get("m", 0)) *
                    (6 + len(config.get("serving_round_bits_sweep", []))))
    finished = sum(phase == "case_finished" for phase in latest_case_state.values())
    first_case = next((e for e in events if e.get("phase") == "case_started"), None)
    rate = None
    if first_case and finished:
        elapsed_hours = (now - datetime.fromisoformat(first_case["at"])).total_seconds() / 3600
        rate = finished / elapsed_hours if elapsed_hours > 0 else None
    return {
        "observed_at_utc": now.isoformat(),
        "output": str(root),
        "pid": pid,
        "process_alive": _pid_alive(pid) if pid is not None else None,
        "last_event": latest,
        "seconds_since_last_event": round((now - datetime.fromisoformat(latest["at"])).total_seconds(), 1) if latest else None,
        "calibration_finished": counts["calibration_finished"],
        "calibration_expected": config.get("calibration_size"),
        "cases_finished": finished,
        "cases_failed": sum(phase == "case_error" for phase in latest_case_state.values()),
        "cases_expected": expected or None,
        'shared_clean_finished': sum(phase == 'case_finished' and key[1] == 'shared_clean'
                                     for key, phase in latest_case_state.items()),
        'positive_tasks_finished': sum(phase == 'case_finished' and key[1] != 'shared_clean'
                                      for key, phase in latest_case_state.items()),
        "observed_cases_per_hour": round(rate, 2) if rate is not None else None,
        "eta_hours_at_observed_rate": round((expected - finished) / rate, 2) if rate and expected else None,
        "last_error": errors[-1] if errors else None,
        "formal_run_complete": counts["formal_run_complete"] > 0,
        "disk_free_gib": round(disk.free / 2**30, 2),
        "gpu": _gpu_status(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error("--interval must be positive")
    while True:
        status = read_status(args.output, args.pid)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if not args.watch or status["formal_run_complete"] or status["process_alive"] is False:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
