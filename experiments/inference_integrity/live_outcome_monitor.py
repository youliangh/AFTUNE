"""Read-only JSONL snapshots of completed inference-integrity outcomes.

Run once for a snapshot, or pass ``--watch PID`` to snapshot every 60 seconds
until that process terminates. Existing experiment artifacts are never edited.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import time
from typing import Any


CONDITIONS = (
    "kv_cache", "serving_config", "model_mismatch", "token_selection",
    "computation", "input_mismatch", "serving_config_r2", "serving_config_r10",
)


def _origins(condition: str) -> tuple[str, ...]:
    return ("model_recompute_live",) if condition == "model_mismatch" else (
        "generation_live", "final_recompute"
    )


def _positive_indices(seed: int, config_hash: str, m: int, n: int) -> set[int]:
    payload = json.dumps(["labels", seed, config_hash], ensure_ascii=False, separators=(",", ":"))
    random_seed = int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "big")
    return set(random.Random(random_seed).sample(range(m), n))


def _pid_identity(pid: int) -> str | None:
    """Use Linux start time to avoid following a reused PID; zombies are terminal."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    fields = stat.rsplit(") ", 1)[-1].split()
    if len(fields) < 20 or fields[0] in ("Z", "X"):
        return None
    return fields[19]


class LiveOutcomeMonitor:
    """Cache only small result fields so periodic scans avoid rereading old JSON."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.config = manifest["config"]
        self.config_hash = manifest["config_sha256"]
        self.seeds = tuple(int(seed) for seed in self.config["seeds"])
        self.m = int(self.config["m"])
        self.n = int(self.config["n"])
        self.labels = {
            seed: _positive_indices(seed, self.config_hash, self.m, self.n)
            for seed in self.seeds
        }
        self._cache: dict[Path, tuple[int, int, dict[str, Any]]] = {}

    def _result(self, path: Path) -> dict[str, Any] | None:
        try:
            metadata = path.stat()
        except FileNotFoundError:
            self._cache.pop(path, None)
            return None
        cached = self._cache.get(path)
        if cached is not None and cached[:2] == (metadata.st_mtime_ns, metadata.st_size):
            return cached[2]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("result is not a JSON object")
            keep = (
                "case_key", "condition", "seed", "index", "config_sha256", "origins",
                "planned_positive", "injection_success", "evidence_origin", "verdict",
                "strict_hash_verdict", "status_error",
            )
            small = {key: payload.get(key) for key in keep}
        except (OSError, ValueError) as exc:
            small = {"read_error": str(exc)}
        self._cache[path] = (metadata.st_mtime_ns, metadata.st_size, small)
        return small

    def snapshot(self, pid: int | None = None) -> dict[str, Any]:
        if self.config.get('protocol_version') == 'v3':
            return self._snapshot_v3(pid)
        groups: dict[tuple[str, str, bool], dict[str, Any]] = defaultdict(lambda: {
            "verdict_counts": Counter(),
            "strict_hash_verdict_counts": Counter(),
            "injection_success_counts": Counter(),
            "status_error_count": 0,
        })
        by_condition: dict[str, dict[str, int]] = {}
        anomalies: Counter[str] = Counter()
        error_examples: list[dict[str, str]] = []
        observed_cases = 0
        observed_origins = 0
        expected_origins = 0

        def anomaly(code: str, path: Path, detail: Any = None) -> None:
            anomalies[code] += 1
            if len(error_examples) < 10:
                error_examples.append({
                    "kind": code, "path": str(path.relative_to(self.root)),
                    "detail": str(detail)[:300] if detail is not None else "",
                })

        for seed in self.seeds:
            for condition in CONDITIONS:
                origins = _origins(condition)
                expected_origins += self.m * len(origins)
                condition_key = f"{seed}/{condition}"
                status = {"expected_cases": self.m, "case_results": 0,
                          "expected_origin_results": self.m * len(origins),
                          "origin_results": 0, "case_markers_missing_origins": 0}
                for index in range(self.m):
                    case_dir = self.root / "cases" / str(seed) / condition / f"{index:04d}"
                    case_path = case_dir / "result.json"
                    case = self._result(case_path)
                    if case is not None:
                        observed_cases += 1
                        status["case_results"] += 1
                        if case.get("read_error"):
                            anomaly("invalid_case_result", case_path, case["read_error"])
                        elif (case.get("condition"), case.get("seed"), case.get("index"),
                              case.get("config_sha256"), case.get("origins")) != (
                              condition, seed, index, self.config_hash, list(origins)):
                            anomaly("case_identity_mismatch", case_path)
                        if case.get("status_error"):
                            anomaly("case_status_error", case_path, case["status_error"])
                        if not (case_dir / "commitment.json").is_file():
                            anomaly("case_commitment_missing", case_dir)
                    for origin in origins:
                        path = case_dir / origin / "result.json"
                        row = self._result(path)
                        if row is None:
                            if case is not None:
                                status["case_markers_missing_origins"] += 1
                                anomaly("completed_case_missing_origin", path)
                            continue
                        observed_origins += 1
                        status["origin_results"] += 1
                        if row.get("read_error"):
                            anomaly("invalid_origin_result", path, row["read_error"])
                            continue
                        expected_positive = index in self.labels[seed]
                        if row.get("planned_positive") is not expected_positive:
                            anomaly("label_mismatch", path, row.get("planned_positive"))
                        expected_key = f"{seed}/{condition}/{index:04d}/{origin}"
                        if (row.get("case_key"), row.get("condition"), row.get("seed"),
                            row.get("index"), row.get("evidence_origin"), row.get("config_sha256")) != (
                            expected_key, condition, seed, index, origin, self.config_hash):
                            anomaly("origin_identity_mismatch", path)
                        label = bool(row.get("planned_positive"))
                        group = groups[(condition, origin, label)]
                        verdict = str(row.get("verdict") or "MISSING")
                        group["verdict_counts"][verdict] += 1
                        group["strict_hash_verdict_counts"][str(row.get("strict_hash_verdict") or "MISSING")] += 1
                        injected = row.get("injection_success")
                        group["injection_success_counts"]["true" if injected is True else
                                                          "false" if injected is False else "unknown"] += 1
                        if injected is False or (label and injected is not True):
                            anomaly("injection_failed_or_unknown", path, injected)
                        if verdict in ("ERROR", "UNSUPPORTED", "MISSING"):
                            anomaly("undecidable_verdict", path, verdict)
                        if row.get("status_error"):
                            group["status_error_count"] += 1
                            anomaly("origin_status_error", path, row["status_error"])
                status["missing_case_results"] = self.m - status["case_results"]
                status["missing_origin_results"] = status["expected_origin_results"] - status["origin_results"]
                by_condition[condition_key] = status

        compact_groups = []
        for (condition, origin, positive), group in sorted(groups.items()):
            compact_groups.append({
                "condition": condition, "origin": origin, "positive": positive,
                "completed": sum(group["verdict_counts"].values()),
                "verdict_counts": dict(group["verdict_counts"]),
                "strict_hash_verdict_counts": dict(group["strict_hash_verdict_counts"]),
                "injection_success_counts": dict(group["injection_success_counts"]),
                "status_error_count": group["status_error_count"],
            })
        expected_cases = len(self.seeds) * len(CONDITIONS) * self.m
        return {
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(self.root), "pid": pid,
            "process_alive": _pid_identity(pid) is not None if pid is not None else None,
            "config_sha256": self.config_hash,
            "totals": {
                "expected_cases": expected_cases,
                "case_results": observed_cases,
                "missing_case_results": expected_cases - observed_cases,
                "expected_origin_results": expected_origins,
                "origin_results": observed_origins,
                "missing_origin_results": expected_origins - observed_origins,
            },
            "by_condition": by_condition,
            "outcomes": compact_groups,
            "anomaly_counts": dict(anomalies),
            "first_10_anomalies": error_examples,
        }

    def _snapshot_v3(self, pid: int | None) -> dict[str, Any]:
        from .config import CONDITIONS as conditions
        unique_cases, unique_origins = set(), set()
        by_condition = {}
        outcomes = []
        anomalies = Counter()
        for seed in self.seeds:
            for condition in conditions:
                origin = 'model_recompute_live' if condition == 'model_mismatch' else 'generation_live'
                status = {'expected_cases': self.m, 'case_results': 0, 'origin_results': 0}
                groups = {positive: {'verdict_counts': Counter(), 'strict_hash_verdict_counts': Counter(),
                                      'injection_success_counts': Counter(), 'status_error_count': 0}
                          for positive in (False, True)}
                for index in range(self.m):
                    positive = index in self.labels[seed]
                    case = (self.root / 'cases' / str(seed) / condition / f'{index:04d}' if positive else
                            self.root / 'shared_clean' / str(seed) / f'{index:04d}')
                    marker = self._result(case / 'result.json')
                    row = self._result(case / origin / 'result.json')
                    if marker:
                        unique_cases.add(case)
                        status['case_results'] += 1
                    if row:
                        unique_origins.add(case / origin)
                        status['origin_results'] += 1
                        group = groups[positive]
                        group['verdict_counts'][row.get('verdict', 'ERROR')] += 1
                        group['strict_hash_verdict_counts'][row.get('strict_hash_verdict', 'ERROR')] += 1
                        group['injection_success_counts'][str(row.get('injection_success'))] += 1
                        group['status_error_count'] += bool(row.get('status_error'))
                        if row.get('read_error') or row.get('config_sha256') != self.config_hash:
                            anomalies['source_result_identity'] += 1
                by_condition[f'{seed}/{condition}'] = status
                for positive, group in groups.items():
                    outcomes.append({'seed': seed, 'condition': condition, 'origin': origin,
                                     'planned_positive': positive, **{key: dict(value) if isinstance(value, Counter) else value
                                                                      for key, value in group.items()}})
        return {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'root': str(self.root),
                'pid': pid, 'process_alive': _pid_identity(pid) is not None if pid else None,
                'config_sha256': self.config_hash, 'protocol_version': 'v3',
                'totals': {'expected_unique_tasks': len(self.seeds) * (self.m + 5 * self.n),
                           'unique_tasks': len(unique_cases),
                           'expected_unique_evidence': len(self.seeds) * (2 * (self.m - self.n) + 6 * self.n),
                           'unique_evidence': len(unique_origins),
                           'condition_sample_references': sum(row['origin_results'] for row in by_condition.values())},
                'by_condition': by_condition, 'outcomes': outcomes,
                'anomaly_counts': dict(anomalies), 'first_10_anomalies': []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Formal experiment output directory")
    parser.add_argument("--jsonl", type=Path, help="Optional append-only snapshot file outside experiment artifacts")
    parser.add_argument("--watch", type=int, metavar="PID", help="Repeat until this PID terminates")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between snapshots (default: 60)")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be positive")
    monitor = LiveOutcomeMonitor(args.root)
    process_identity = _pid_identity(args.watch) if args.watch is not None else None
    while True:
        row = monitor.snapshot(args.watch)
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        print(line, flush=True)
        if args.jsonl is not None:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.jsonl.open("a", encoding="utf-8") as output:
                output.write(line + "\n")
        if args.watch is None or process_identity is None or _pid_identity(args.watch) != process_identity:
            break
        time.sleep(min(args.interval, 60.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
