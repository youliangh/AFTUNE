"""Read-only audit for one or more completed 256-case experiment conditions.

This checks saved block verdicts and commitments without loading tensor proofs.
The formal run may continue with other conditions while this script runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from .completion_audit import CONDITIONS, _code_digest, _config_hash, _origins, _positive_indices, _sha_file


INTEGRITY_ERRORS = ("Computed output hash mismatch", "Disk data hash mismatch", "Input hash mismatch")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _read(path: Path, issues: list[dict[str, str]], code: str = "missing_or_invalid_json") -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if len(issues) < 30:
            issues.append({"code": code, "path": str(path), "detail": str(exc)[:300]})
        return None


def _issue(issues: list[dict[str, str]], code: str, path: Path, detail: Any = "") -> None:
    if len(issues) < 30:
        issues.append({"code": code, "path": str(path), "detail": str(detail)[:300]})


def _verdict(blocks: list[dict[str, Any]], threshold: float) -> str:
    """Independent small implementation of the frozen sequence decision rule."""
    if not blocks:
        return "ERROR"
    rejected = False
    invalid = False
    for block in blocks:
        error = str(block.get("error") or "")
        if error:
            if any(marker in error for marker in INTEGRITY_ERRORS):
                rejected = True
            else:
                invalid = True
            continue
        try:
            value = float(block["overall_relative_l2"])
        except (KeyError, TypeError, ValueError):
            invalid = True
            continue
        if not math.isfinite(value) or value < 0:
            invalid = True
        elif value > threshold:
            rejected = True
    return "DETECTED" if rejected else "ERROR" if invalid else "PASSED"


def _counts(rows: list[tuple[bool, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for positive, verdict in rows:
        if verdict == "DETECTED":
            counts["TP" if positive else "FP"] += 1
        elif verdict == "PASSED":
            counts["FN" if positive else "TN"] += 1
        else:
            counts["ERROR"] += 1
    return {name: counts[name] for name in ("TP", "FN", "FP", "TN", "ERROR")}


def _check_block_grid(blocks: list[dict[str, Any]], origin: str,
                      generation_steps: int, path: Path,
                      issues: list[dict[str, str]]) -> None:
    """Require each expected step/layer exactly once, not just the right count."""
    steps = range(1, generation_steps + 1) if origin == "generation_live" else (0,)
    expected = {(step, block_id) for step in steps for block_id in range(18)}
    actual: set[tuple[int, int]] = set()
    valid = True
    for block in blocks:
        if not isinstance(block, dict):
            valid = False
            continue
        step, block_id = block.get("step"), block.get("block_id")
        if type(step) is not int or type(block_id) is not int:
            valid = False
            continue
        coordinate = (step, block_id)
        if coordinate in actual:
            valid = False
        actual.add(coordinate)
        expected_name = ("embedding" if block_id == 0 else
                         "norm" if block_id == 17 else f"layer_{block_id - 1}")
        if block.get("layer_name") != expected_name:
            valid = False
    if not valid or actual != expected:
        _issue(issues, "block_grid_mismatch", path,
               f"observed={len(actual)}, expected={len(expected)}, valid={valid}")


def audit_stage(root: Path, seed: int, condition: str) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    issues: list[dict[str, str]] = []
    manifest_path = root / "manifest.json"
    manifest = _read(manifest_path, issues)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
        return {"condition": condition, "seed": seed, "passed": False, "issues": issues}
    config = manifest["config"]
    config_hash = _config_hash(config)
    if manifest.get("config_sha256") != config_hash:
        _issue(issues, "config_hash_mismatch", manifest_path)
    if manifest.get("experiment_code_sha256") != _code_digest():
        _issue(issues, "core_code_mismatch", manifest_path)
    if seed not in config.get("seeds", []):
        _issue(issues, "seed_not_in_manifest", manifest_path, seed)
    m = int(config["m"])
    n = int(config["n"])
    if (m, n) != (256, 32):
        _issue(issues, "unexpected_formal_size", manifest_path, f"m={m}, n={n}")
    origins = _origins(condition)
    expected_blocks = {
        "generation_live": 18 * (int(config["max_length"]) - int(config["prefill_tokens"])),
        "final_recompute": 18,
        "model_recompute_live": 18,
    }
    thresholds_path = root / "thresholds.json"
    threshold_data = _read(thresholds_path, issues)
    thresholds = threshold_data.get("thresholds", {}) if isinstance(threshold_data, dict) else {}
    if not isinstance(threshold_data, dict) or threshold_data.get("config_sha256") != config_hash:
        _issue(issues, "threshold_config_mismatch", thresholds_path)
    corpus_path = root / "corpus.json"
    corpus = _read(corpus_path, issues)
    evaluation = corpus.get("evaluation", {}).get(str(seed), []) if isinstance(corpus, dict) else []
    if not isinstance(corpus, dict) or _sha_file(corpus_path) != manifest.get("corpus_sha256"):
        _issue(issues, "corpus_hash_mismatch", corpus_path)
    if not isinstance(evaluation, list) or len(evaluation) != m:
        _issue(issues, "evaluation_size", corpus_path, len(evaluation) if isinstance(evaluation, list) else "invalid")
        evaluation = []
    labels = _positive_indices(seed, config_hash, m, n)
    result_rows: dict[str, dict[str, list[tuple[bool, str]]]] = {
        origin: {"numeric": [], "strict_hash": []} for origin in origins
    }
    root_results = 0
    origin_results = 0
    oracle_count = 0
    commitment_count = 0
    block_results = 0
    positive_count = 0
    negative_count = 0
    for index in range(m):
        case_dir = root / "cases" / str(seed) / condition / f"{index:04d}"
        positive = index in labels
        positive_count += int(positive)
        negative_count += int(not positive)
        marker_path = case_dir / "result.json"
        marker = _read(marker_path, issues)
        if isinstance(marker, dict):
            root_results += 1
            if (marker.get("condition"), marker.get("seed"), marker.get("index"),
                marker.get("config_sha256"), marker.get("origins")) != (
                condition, seed, index, config_hash, list(origins)):
                _issue(issues, "case_marker_mismatch", marker_path)
            if marker.get("status_error"):
                _issue(issues, "case_status_error", marker_path, marker["status_error"])
        oracle_path = case_dir / "oracle" / "truth.json"
        oracle = _read(oracle_path, issues)
        if isinstance(oracle, dict):
            oracle_count += 1
            if oracle.get("planned_positive") is not positive:
                _issue(issues, "oracle_label_mismatch", oracle_path)
            if oracle.get("injection_success") is not True:
                _issue(issues, "injection_not_successful", oracle_path, oracle.get("injection_success"))
            if index < len(evaluation):
                request = oracle.get("source_request", {})
                if not isinstance(request, dict) or request.get("input_ids") != evaluation[index].get("input_ids"):
                    _issue(issues, "oracle_request_mismatch", oracle_path)
            expected_model = config["substitute_model_path"] if condition == "model_mismatch" and positive else config["declared_model_path"]
            if oracle.get("source_model") != expected_model:
                _issue(issues, "oracle_source_model", oracle_path, oracle.get("source_model"))
            if condition == "model_mismatch":
                if oracle.get("kind") != "model" or not isinstance(oracle.get("declared_retokenized_ids"), list):
                    _issue(issues, "model_bridge_missing", oracle_path)
            elif positive and not isinstance(oracle.get("spec"), dict):
                _issue(issues, "injection_spec_missing", oracle_path)
        commitment_path = case_dir / "commitment.json"
        commitment = _read(commitment_path, issues)
        if isinstance(commitment, dict):
            commitment_count += 1
            files = commitment.get("files")
            if commitment.get("algorithm") != "sha256" or not isinstance(files, dict) or not files:
                _issue(issues, "commitment_structure", commitment_path)
            else:
                expected_prefixes = tuple(f"records/{origin}/" for origin in origins)
                actual_prefixes = {origin for origin in origins
                                   if any(path.startswith(f"records/{origin}/") for path in files)}
                if actual_prefixes != set(origins) or any(not path.startswith(expected_prefixes) for path in files):
                    _issue(issues, "commitment_origins", commitment_path)
                if any(not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
                       for value in files.values()):
                    _issue(issues, "commitment_hash_format", commitment_path)
        for origin in origins:
            path = case_dir / origin / "result.json"
            row = _read(path, issues)
            if not isinstance(row, dict):
                continue
            origin_results += 1
            expected_key = f"{seed}/{condition}/{index:04d}/{origin}"
            if (row.get("case_key"), row.get("condition"), row.get("seed"),
                row.get("index"), row.get("evidence_origin"), row.get("config_sha256")) != (
                expected_key, condition, seed, index, origin, config_hash):
                _issue(issues, "origin_result_identity", path)
            if row.get("planned_positive") is not positive or row.get("injection_success") is not True:
                _issue(issues, "origin_truth_mismatch", path)
            if row.get("status_error"):
                _issue(issues, "origin_status_error", path, row["status_error"])
            if row.get("block_count") != expected_blocks[origin]:
                _issue(issues, "reported_block_count", path, row.get("block_count"))
            if row.get("threshold") != thresholds.get(origin):
                _issue(issues, "threshold_mismatch", path)
            if isinstance(commitment, dict) and commitment.get("frozen_at") and row.get("finished_at"):
                if commitment["frozen_at"] > row["finished_at"]:
                    _issue(issues, "commitment_after_result", path)
            for mode in ("numeric", "strict_hash"):
                blocks_path = case_dir / origin / f"blocks_{mode}.json"
                blocks = _read(blocks_path, issues)
                if not isinstance(blocks, list):
                    continue
                block_results += 1
                if len(blocks) != expected_blocks[origin]:
                    _issue(issues, "actual_block_count", blocks_path, len(blocks))
                _check_block_grid(blocks, origin,
                                  int(config["max_length"]) - int(config["prefill_tokens"]),
                                  blocks_path, issues)
                verdict = _verdict(blocks, float(thresholds.get(origin, 0.0)) if mode == "numeric" else float("inf"))
                saved_verdict = row.get("verdict" if mode == "numeric" else "strict_hash_verdict")
                if verdict != saved_verdict:
                    _issue(issues, "saved_verdict_mismatch", path, f"{mode}: stored={saved_verdict}, derived={verdict}")
                if verdict not in ("PASSED", "DETECTED"):
                    _issue(issues, "undecidable_verdict", path, mode)
                result_rows[origin][mode].append((positive, verdict))
        if condition == "model_mismatch" and (
            (case_dir / "generation_live" / "result.json").exists()
            or (case_dir / "final_recompute" / "result.json").exists()
        ):
            _issue(issues, "model_duplicate_origin", case_dir)
    group_results: dict[str, Any] = {}
    for origin, modes in result_rows.items():
        group_results[origin] = {mode: _counts(rows) for mode, rows in modes.items()}
        for mode, rows in modes.items():
            counts = group_results[origin][mode]
            if len(rows) != m or counts["TP"] + counts["FN"] != n or counts["FP"] + counts["TN"] != m - n or counts["ERROR"]:
                _issue(issues, "four_cell_denominator", root / "cases" / str(seed) / condition,
                       f"{origin}/{mode}: {counts}")
    if root_results != m or oracle_count != m or commitment_count != m or origin_results != m * len(origins) or block_results != 2 * m * len(origins):
        _issue(issues, "condition_incomplete", root / "cases" / str(seed) / condition,
               f"case={root_results}, oracle={oracle_count}, commit={commitment_count}, origin={origin_results}, blocks={block_results}")
    return {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "seed": seed, "condition": condition,
        "passed": not issues,
        "expected_cases": m, "root_results": root_results,
        "positive_cases": positive_count, "negative_cases": negative_count,
        "oracle_results": oracle_count, "commitments": commitment_count,
        "expected_origin_results": m * len(origins), "origin_results": origin_results,
        "expected_block_result_files": 2 * m * len(origins), "block_result_files": block_results,
        "groups": group_results, "first_30_issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", action="append", choices=CONDITIONS, required=True,
                        help="Repeat to audit several completed conditions")
    parser.add_argument("--jsonl", type=Path, help="Optional report file outside the run directory")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.jsonl is not None:
        target = args.jsonl.expanduser().resolve()
        if target == root or root in target.parents:
            parser.error("--jsonl must be outside the experiment run directory")
        if not target.parent.is_dir():
            parser.error("--jsonl parent directory must already exist")
    results = [audit_stage(root, args.seed, condition) for condition in dict.fromkeys(args.condition)]
    for result in results:
        line = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        print(line)
        if args.jsonl is not None:
            with args.jsonl.open("a", encoding="utf-8") as output:
                output.write(line + "\n")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
