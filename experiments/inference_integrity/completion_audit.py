"""Read-only consistency audit for a completed inference-integrity run.

Proof tensors are never loaded. The audit reads manifests, calibration scores,
case verdicts, saved block results, and sampling summaries. It independently
checks each formal condition's block grid and recomputes sequence verdicts.
This utility is outside the frozen experiment-code digest.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

from .metrics import calibrate_threshold, summarize_cases, summarize_sampling


CONDITIONS = (
    "kv_cache", "serving_config", "model_mismatch", "token_selection",
    "computation", "input_mismatch", "serving_config_r2", "serving_config_r10",
)
ORIGINS = ("generation_live", "final_recompute", "model_recompute_live")
CORE_LOCAL = ("config.py", "metrics.py", "runner.py", "native.py", "injections.py", "generation.py")
CORE_PARENT = (
    "verify_inference.py", "recorder.py", "replay_utils.py", "verify_block.py",
    "finetune.py", "load_model.py",
)


@dataclass(frozen=True)
class Expectations:
    m: int = 256
    n: int = 32
    calibration_size: int = 32
    seeds: tuple[int, ...] = (0,)
    prefill_tokens: int = 32
    max_length: int = 128
    sample_repeats: int = 1000
    generation_device: str = "cuda:0"
    verification_device: str = "cuda:0"


class Findings:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: list[dict[str, str]] = []

    def add(self, code: str, location: str, detail: str = "") -> None:
        self.counts[code] += 1
        if len(self.examples) < 50:
            self.examples.append({"code": code, "location": location, "detail": detail})

    def check(self, condition: bool, code: str, location: str, detail: str = "") -> None:
        if not condition:
            self.add(code, location, detail)


def _read(path: Path, findings: Findings) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        findings.add("missing_or_invalid_json", str(path), str(exc))
        return None


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_digest() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    aftune = directory.parents[1]
    return {
        **{name: _sha_file(directory / name) for name in CORE_LOCAL},
        **{name: _sha_file(aftune / name) for name in CORE_PARENT},
    }


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _positive_indices(seed: int, config_hash: str, m: int, n: int) -> set[int]:
    payload = json.dumps(["labels", seed, config_hash], ensure_ascii=False, separators=(",", ":"))
    random_seed = int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "big")
    return set(random.Random(random_seed).sample(range(m), n))


def _origins(condition: str) -> tuple[str, ...]:
    return ("model_recompute_live",) if condition == "model_mismatch" else (
        "generation_live", "final_recompute"
    )


def _same_number(actual: Any, expected: Any) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _check_sampling(row: dict[str, Any], config: dict[str, Any], location: str,
                    findings: Findings) -> None:
    rates = tuple(config.get("sample_rates", ()))
    repeats = config.get("sample_repeats")
    samples = row.get("sampling")
    if not isinstance(samples, list) or len(samples) != len(rates):
        findings.add("sampling_structure", location, "missing or wrong number of rates")
        return
    for sample, rate in zip(samples, rates):
        if not isinstance(sample, dict):
            findings.add("sampling_structure", location, "sample entry is not an object")
            continue
        block_count = row.get("block_count")
        expected_count = min(block_count, max(1, math.ceil(rate * block_count))) if isinstance(block_count, int) and block_count > 0 else None
        valid = (
            sample.get("rate") == rate
            and sample.get("repeats") == repeats
            and sample.get("sample_count") == expected_count
            and sample.get("root_seed") == row.get("audit_seed")
            and isinstance(sample.get("first_sampled_indices"), list)
            and len(sample["first_sampled_indices"]) == expected_count
        )
        for key in ("hits", "errors"):
            valid = valid and isinstance(sample.get(key), int) and 0 <= sample[key] <= repeats
        if valid:
            valid = sample["hits"] + sample["errors"] <= repeats
            valid = valid and all(isinstance(i, int) and 0 <= i < block_count
                                  for i in sample["first_sampled_indices"])
            valid = valid and len(set(sample["first_sampled_indices"])) == expected_count
        findings.check(valid, "sampling_structure", location, f"rate={rate}")


def _check_calibration(root: Path, config: dict[str, Any], corpus: dict[str, Any],
                       config_hash: str, findings: Findings) -> dict[str, float]:
    frozen = _read(root / "thresholds.json", findings)
    if not isinstance(frozen, dict):
        return {}
    findings.check(frozen.get("config_sha256") == config_hash, "threshold_config", "thresholds.json")
    findings.check(frozen.get("calibration_size") == config["calibration_size"], "threshold_size", "thresholds.json")
    thresholds = frozen.get("thresholds")
    if not isinstance(thresholds, dict):
        findings.add("threshold_structure", "thresholds.json")
        return {}
    findings.check(set(thresholds) == set(ORIGINS), "threshold_origins", "thresholds.json")
    block_sets: dict[str, list[list[dict[str, Any]]]] = {origin: [] for origin in ORIGINS}
    calibration = corpus.get("calibration", [])
    for index in range(config["calibration_size"]):
        case_dir = root / "calibration" / f"{index:04d}"
        marker = _read(case_dir / "calibration.json", findings)
        if not isinstance(marker, dict):
            continue
        findings.check(marker.get("index") == index, "calibration_index", str(case_dir))
        if index < len(calibration):
            findings.check(marker.get("request_sha256") == calibration[index].get("request_text_sha256"),
                           "calibration_request", str(case_dir))
        commitment = case_dir / "commitment.json"
        findings.check(commitment.is_file() and commitment.stat().st_size > 0,
                       "missing_commitment", str(case_dir))
        block_files = marker.get("block_files", {})
        findings.check(set(block_files) == set(ORIGINS), "calibration_origins", str(case_dir))
        for origin in ORIGINS:
            expected_path = f"calibration/{index:04d}/blocks_{origin}.json"
            findings.check(block_files.get(origin) == expected_path, "calibration_block_path", str(case_dir), origin)
            blocks = _read(root / expected_path, findings)
            if isinstance(blocks, list):
                expected_blocks = 18 * (config["max_length"] - config["prefill_tokens"]) if origin == "generation_live" else 18
                findings.check(len(blocks) == expected_blocks, "calibration_block_count", expected_path)
                block_sets[origin].append(blocks)
    recomputed: dict[str, float] = {}
    for origin, sets in block_sets.items():
        if len(sets) != config["calibration_size"]:
            findings.add("calibration_incomplete", origin, f"{len(sets)} of {config['calibration_size']}")
            continue
        try:
            recomputed[origin] = calibrate_threshold(sets)
        except (ValueError, TypeError) as exc:
            findings.add("calibration_invalid", origin, str(exc))
            continue
        findings.check(_same_number(thresholds.get(origin), recomputed[origin]),
                       "threshold_mismatch", origin, f"frozen={thresholds.get(origin)}, recomputed={recomputed[origin]}")
    return recomputed


def audit_run(root: Path, expected: Expectations = Expectations()) -> dict[str, Any]:
    """Audit a terminal run without reading or changing proof tensor files."""
    root = root.expanduser().resolve()
    findings = Findings()
    manifest = _read(root / "manifest.json", findings)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
        findings.add("missing_manifest_config", str(root / "manifest.json"))
        return _report(root, expected, findings, {})
    config = manifest["config"]
    for field in ("m", "n", "calibration_size", "prefill_tokens", "max_length",
                  "sample_repeats", "generation_device", "verification_device"):
        findings.check(config.get(field) == getattr(expected, field), "unexpected_config", field,
                       f"observed={config.get(field)!r}, expected={getattr(expected, field)!r}")
    findings.check(tuple(config.get("seeds", ())) == expected.seeds, "unexpected_config", "seeds")
    findings.check(config.get("dtype") == "float32" and config.get("attention_backend") == "sdpa",
                   "unexpected_numeric_backend", "manifest.json")
    findings.check(config.get("sample_rates") == [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
                   "unexpected_sample_rates", "manifest.json")
    config_hash = _config_hash(config)
    findings.check(manifest.get("config_sha256") == config_hash, "config_digest", "manifest.json")
    code_digest = manifest.get("experiment_code_sha256")
    findings.check(isinstance(code_digest, dict) and set(code_digest) == set(CORE_LOCAL + CORE_PARENT),
                   "code_digest_structure", "manifest.json")
    try:
        current_digest = _code_digest()
        for name, digest in current_digest.items():
            findings.check(isinstance(code_digest, dict) and code_digest.get(name) == digest,
                           "code_digest_mismatch", name)
    except OSError as exc:
        findings.add("code_digest_unreadable", "source tree", str(exc))

    corpus = _read(root / "corpus.json", findings)
    if not isinstance(corpus, dict):
        findings.add("corpus_structure", "corpus.json")
        return _report(root, expected, findings, {"config_sha256": config_hash})
    findings.check(_sha_file(root / "corpus.json") == manifest.get("corpus_sha256"),
                   "corpus_digest", "corpus.json")
    calibration = corpus.get("calibration", [])
    evaluation = corpus.get("evaluation", {})
    findings.check(isinstance(calibration, list) and len(calibration) == config["calibration_size"],
                   "calibration_corpus_size", "corpus.json")
    findings.check(isinstance(evaluation, dict) and set(evaluation) == {str(s) for s in expected.seeds},
                   "evaluation_seeds", "corpus.json")
    identities: list[tuple[int, ...]] = []
    for location, rows in [("calibration", calibration)] + [
        (f"evaluation/{seed}", evaluation.get(str(seed), [])) for seed in expected.seeds
    ]:
        if not isinstance(rows, list):
            findings.add("corpus_rows", location)
            continue
        if location != "calibration":
            findings.check(len(rows) == config["m"], "evaluation_corpus_size", location)
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("input_ids"), list):
                findings.add("corpus_row", location)
                continue
            ids = tuple(row["input_ids"])
            identities.append(ids)
            findings.check(len(ids) == config["prefill_tokens"], "request_token_length", location)
            request_text = row.get("request_text", "")
            findings.check(isinstance(request_text, str) and
                           sha256(request_text.encode("utf-8")).hexdigest() == row.get("request_text_sha256"),
                           "request_text_digest", location)
    findings.check(len(identities) == len(set(identities)), "calibration_evaluation_overlap", "corpus.json")

    thresholds = _check_calibration(root, config, corpus, config_hash, findings)
    summary = _read(root / "summary.json", findings)
    if not isinstance(summary, dict):
        summary = {}
    findings.check(summary.get("complete") is True and summary.get("missing_count") == 0,
                   "summary_incomplete", "summary.json")
    findings.check(summary.get("config_sha256") == config_hash, "summary_config", "summary.json")
    groups = summary.get("groups", {})
    if not isinstance(groups, dict):
        findings.add("summary_groups", "summary.json")
        groups = {}
    expected_groups = {
        f"{seed}/{condition}/{origin}"
        for seed in expected.seeds for condition in CONDITIONS for origin in _origins(condition)
    }
    findings.check(set(groups) == expected_groups, "summary_group_count", "summary.json",
                   f"observed={len(groups)}, expected={len(expected_groups)}")

    observed_cases = 0
    observed_origins = 0
    positives_by_group: dict[str, int] = {}
    for seed in expected.seeds:
        labels = _positive_indices(seed, config_hash, config["m"], config["n"])
        for condition in CONDITIONS:
            case_rows: dict[str, list[dict[str, Any]]] = {origin: [] for origin in _origins(condition)}
            paired_specs: dict[int, dict[str, Any]] = {}
            for index in range(config["m"]):
                case_dir = root / "cases" / str(seed) / condition / f"{index:04d}"
                case = _read(case_dir / "result.json", findings)
                oracle = _read(case_dir / "oracle" / "truth.json", findings)
                if not isinstance(case, dict) or not isinstance(oracle, dict):
                    continue
                observed_cases += 1
                positive = index in labels
                findings.check(case.get("condition") == condition and case.get("seed") == seed
                               and case.get("index") == index and case.get("config_sha256") == config_hash,
                               "case_identity", str(case_dir))
                findings.check(case.get("origins") == list(_origins(condition)),
                               "case_origins", str(case_dir))
                findings.check(oracle.get("planned_positive") is positive,
                               "oracle_label", str(case_dir))
                expected_request = evaluation.get(str(seed), [])[index] if index < len(evaluation.get(str(seed), [])) else None
                if expected_request is not None:
                    findings.check(oracle.get("source_request", {}).get("input_ids") == expected_request.get("input_ids"),
                                   "oracle_request", str(case_dir))
                if positive:
                    findings.check(oracle.get("injection_success") is True,
                                   "failed_injection", str(case_dir))
                commitment = case_dir / "commitment.json"
                findings.check(commitment.is_file() and commitment.stat().st_size > 0,
                               "missing_commitment", str(case_dir))
                if condition.startswith("serving_config") and positive:
                    spec = oracle.get("spec")
                    if isinstance(spec, dict):
                        paired_specs[index] = spec
                for origin in _origins(condition):
                    path = case_dir / origin / "result.json"
                    row = _read(path, findings)
                    if not isinstance(row, dict):
                        continue
                    observed_origins += 1
                    case_rows[origin].append(row)
                    expected_key = f"{seed}/{condition}/{index:04d}/{origin}"
                    findings.check(row.get("case_key") == expected_key and row.get("condition") == condition
                                   and row.get("seed") == seed and row.get("index") == index
                                   and row.get("evidence_origin") == origin
                                   and row.get("config_sha256") == config_hash,
                                   "origin_identity", str(path))
                    findings.check(row.get("planned_positive") is positive
                                   and row.get("injection_success") == oracle.get("injection_success"),
                                   "origin_truth", str(path))
                    findings.check(row.get("status_error") is None
                                   and row.get("verdict") in ("PASSED", "DETECTED")
                                   and row.get("strict_hash_verdict") in ("PASSED", "DETECTED"),
                                   "case_error", str(path), str(row.get("status_error")))
                    expected_blocks = 18 * (config["max_length"] - config["prefill_tokens"]) if origin == "generation_live" else 18
                    findings.check(row.get("block_count") == expected_blocks,
                                   "origin_block_count", str(path))
                    findings.check(_same_number(row.get("threshold"), thresholds.get(origin)),
                                   "origin_threshold", str(path))
                    findings.check((case_dir / origin / "blocks_numeric.json").is_file()
                                   and (case_dir / origin / "blocks_strict_hash.json").is_file(),
                                   "missing_block_verdicts", str(path))
                    if row.get("verdict") in ("PASSED", "DETECTED"):
                        _check_sampling(row, config, str(path), findings)
                if condition == "model_mismatch":
                    findings.check(not (case_dir / "generation_live" / "result.json").exists()
                                   and not (case_dir / "final_recompute" / "result.json").exists(),
                                   "model_duplicate_evidence", str(case_dir))
            if condition.startswith("serving_config"):
                # Pair the strength arms by request, label, injection seed, and switch step.
                for index, spec in paired_specs.items():
                    canonical = root / "cases" / str(seed) / "serving_config" / f"{index:04d}" / "oracle" / "truth.json"
                    base = _read(canonical, findings)
                    base_spec = base.get("spec") if isinstance(base, dict) else None
                    if isinstance(base_spec, dict):
                        comparable = {k: v for k, v in spec.items() if k != "rounding_bits"}
                        expected_spec = {k: v for k, v in base_spec.items() if k != "rounding_bits"}
                        findings.check(comparable == expected_spec, "serving_pairing", f"{seed}/{condition}/{index:04d}")
                        bits = 2 if condition == "serving_config_r2" else 10 if condition == "serving_config_r10" else 6
                        findings.check(spec.get("rounding_bits") == bits, "serving_strength", f"{seed}/{condition}/{index:04d}")
            for origin, rows in case_rows.items():
                group_key = f"{seed}/{condition}/{origin}"
                positives_by_group[group_key] = sum(row.get("planned_positive") is True for row in rows)
                findings.check(len(rows) == config["m"], "group_case_count", group_key,
                               f"observed={len(rows)}, expected={config['m']}")
                findings.check(positives_by_group[group_key] == config["n"], "group_positive_count", group_key)
                group = groups.get(group_key)
                if not isinstance(group, dict):
                    findings.add("missing_summary_group", group_key)
                    continue
                findings.check(group.get("observed_cases") == config["m"] and
                               group.get("expected_cases") == config["m"],
                               "summary_case_count", group_key)
                findings.check(_same_number(group.get("threshold"), thresholds.get(origin)),
                               "summary_threshold", group_key)
                actual_numeric = summarize_cases(rows)
                strict_rows = [{**row, "verdict": row.get("strict_hash_verdict", "ERROR")} for row in rows]
                actual_strict = summarize_cases(strict_rows)
                for name, actual in (("numeric", actual_numeric), ("strict_hash", actual_strict)):
                    reported = group.get(name, {})
                    findings.check(reported.get("counts") == actual["counts"],
                                   "summary_counts", f"{group_key}/{name}")
                    findings.check(reported.get("planned_sequences") == len(rows),
                                   "summary_planned", f"{group_key}/{name}")
                    # Include TPR/FPR, Wilson intervals, coverage, precision,
                    # and deployment recall in the raw-result cross-check.
                    canonical_actual = json.loads(json.dumps(actual))
                    findings.check(reported == canonical_actual,
                                   "summary_derived_metrics", f"{group_key}/{name}")
                    counts = actual["counts"]
                    findings.check(not any(counts.get(key, 0) for key in
                                           ("ERROR", "UNSUPPORTED", "INJECTION_FAILED", "INJECTION_STATUS_UNKNOWN")),
                                   "unexplained_failure", f"{group_key}/{name}", str(counts))
                    findings.check(counts.get("TP", 0) + counts.get("FN", 0) == config["n"] and
                                   counts.get("FP", 0) + counts.get("TN", 0) == config["m"] - config["n"],
                                   "confusion_denominators", f"{group_key}/{name}")
                try:
                    actual_sampling = summarize_sampling(rows, config["sample_rates"], config["sample_repeats"])
                except (KeyError, TypeError, ValueError) as exc:
                    findings.add("sampling_summary_invalid", group_key, str(exc))
                    continue
                reported_sampling = group.get("sampling", {})
                findings.check(set(reported_sampling) == set(actual_sampling),
                               "summary_sampling_rates", group_key)
                findings.check(reported_sampling == actual_sampling,
                               "summary_sampling_metrics", group_key)
                for rate, actual in actual_sampling.items():
                    reported = reported_sampling.get(rate, {})
                    findings.check(reported.get("counts") == actual["counts"] and
                                   reported.get("repeat_count_per_sequence") == config["sample_repeats"],
                                   "summary_sampling_counts", f"{group_key}/{rate}")
                    total = sum(actual["counts"].get(k, 0) for k in
                                ("TP", "FN", "FP", "TN", "ERROR_POSITIVE", "ERROR_NEGATIVE",
                                 "INJECTION_FAILED", "INJECTION_STATUS_UNKNOWN"))
                    findings.check(total == config["m"] * config["sample_repeats"],
                                   "sampling_denominator", f"{group_key}/{rate}")
    observed = {
        "config_sha256": config_hash,
        "calibration_controls": len(calibration) if isinstance(calibration, list) else None,
        "expected_cases": len(expected.seeds) * len(CONDITIONS) * config["m"],
        "observed_cases": observed_cases,
        "expected_evidence_groups": len(expected_groups),
        "observed_summary_groups": len(groups),
        "expected_origin_results": len(expected.seeds) * 15 * config["m"],
        "observed_origin_results": observed_origins,
        "positives_by_group": positives_by_group,
        "thresholds_recomputed": thresholds,
    }
    findings.check(observed_cases == observed["expected_cases"], "total_case_count", "cases")
    findings.check(observed_origins == observed["expected_origin_results"], "total_origin_count", "cases")
    # Re-derive every sequence verdict from its saved blocks.  The summary
    # cross-check above only re-derives metrics from stored sequence verdicts.
    stage_audits: dict[str, dict[str, Any]] = {}
    if config.get("m") == 256 and config.get("n") == 32:
        from .stage_audit import audit_stage

        for seed in expected.seeds:
            for condition in CONDITIONS:
                key = f"{seed}/{condition}"
                try:
                    stage = audit_stage(root, seed, condition)
                except Exception as exc:
                    findings.add("stage_audit_exception", key, repr(exc))
                    continue
                stage_audits[key] = {
                    "passed": stage["passed"],
                    "root_results": stage["root_results"],
                    "origin_results": stage["origin_results"],
                    "block_result_files": stage["block_result_files"],
                    "groups": stage["groups"],
                }
                findings.check(stage["passed"], "stage_audit_failed", key,
                               str(stage["first_30_issues"][:3]))
        findings.check(len(stage_audits) == len(expected.seeds) * len(CONDITIONS),
                       "stage_audit_count", "cases")
    observed["stage_audits"] = stage_audits
    audited_files: dict[str, str] = {}
    for name in ("manifest.json", "corpus.json", "thresholds.json", "summary.json"):
        try:
            audited_files[name] = _sha_file(root / name)
        except OSError as exc:
            findings.add("audited_file_unreadable", name, str(exc))
    return _report(root, expected, findings, observed, audited_files)


def _report(root: Path, expected: Expectations, findings: Findings,
            observed: dict[str, Any],
            audited_files: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root),
        "passed": not findings.counts,
        "expected": {
            "m": expected.m, "n": expected.n, "calibration_size": expected.calibration_size,
            "seeds": list(expected.seeds), "prefill_tokens": expected.prefill_tokens,
            "max_length": expected.max_length, "sample_repeats": expected.sample_repeats,
            "generation_device": expected.generation_device,
            "verification_device": expected.verification_device,
        },
        "observed": observed,
        "audited_file_sha256": audited_files or {},
        "issue_counts": dict(findings.counts),
        "first_50_issues": findings.examples,
    }


def _pid_identity(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    fields = stat.rsplit(") ", 1)[-1].split()
    if len(fields) < 20 or fields[0] in ("Z", "X"):
        return None
    return fields[19]  # Linux stat field 22: process start time.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Experiment output directory")
    parser.add_argument("--output", type=Path, help="Audit JSON path; defaults to ROOT/completion_audit.json")
    parser.add_argument("--expect-m", type=int, default=256)
    parser.add_argument("--expect-n", type=int, default=32)
    parser.add_argument("--expect-calibration", type=int, default=32)
    parser.add_argument("--expect-seeds", default="0", help="Comma-separated seed IDs")
    parser.add_argument("--expect-prefill", type=int, default=32)
    parser.add_argument("--expect-max-length", type=int, default=128)
    parser.add_argument("--expect-repeats", type=int, default=1000)
    parser.add_argument("--expect-generation-device", default="cuda:0")
    parser.add_argument("--expect-verification-device", default="cuda:0")
    parser.add_argument("--watch", type=int, metavar="PID", help="Wait for this process to terminate before auditing")
    parser.add_argument("--interval", type=float, default=30.0, help="Watch polling interval in seconds")
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.expect_m <= args.expect_n or args.expect_n < 1:
        parser.error("Require interval > 0 and m > n >= 1")
    seeds = tuple(int(part) for part in args.expect_seeds.split(",") if part)
    if not seeds:
        parser.error("At least one expected seed is required")
    expected = Expectations(
        m=args.expect_m, n=args.expect_n, calibration_size=args.expect_calibration,
        seeds=seeds, prefill_tokens=args.expect_prefill, max_length=args.expect_max_length,
        sample_repeats=args.expect_repeats,
        generation_device=args.expect_generation_device,
        verification_device=args.expect_verification_device,
    )
    if args.watch is not None:
        identity = _pid_identity(args.watch)
        while identity is not None and _pid_identity(args.watch) == identity:
            time.sleep(min(args.interval, 60.0))
    report = audit_run(args.root, expected)
    output = args.output or args.root / "completion_audit.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "passed": report["passed"], "issue_counts": report["issue_counts"],
        "observed_cases": report["observed"].get("observed_cases"),
        "observed_origin_results": report["observed"].get("observed_origin_results"),
        "output": str(output),
    }, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
