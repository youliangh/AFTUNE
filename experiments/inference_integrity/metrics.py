"""Sequence decisions, confusion matrices, and post-commit block sampling."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
import random
from typing import Any, Iterable


INTEGRITY_ERRORS = (
    "Computed output hash mismatch",
    "Disk data hash mismatch",
    "Input hash mismatch",
)


def block_rejection(block: dict[str, Any], threshold: float) -> bool:
    """Only a tested trace violation counts as rejection; setup faults do not."""
    error = str(block.get("error") or "")
    if error:
        return any(marker in error for marker in INTEGRITY_ERRORS)
    value = block.get("overall_relative_l2")
    return value is not None and math.isfinite(float(value)) and float(value) > threshold


def decide_sequence(blocks: Iterable[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Apply one frozen threshold to every supported block in a sequence."""
    block_list = list(blocks)
    if not block_list:
        return {"verdict": "ERROR", "score": None, "rejected_blocks": [], "error_blocks": []}
    rejected: list[str] = []
    errors: list[dict[str, str]] = []
    values: list[float] = []
    for i, block in enumerate(block_list):
        block_id = str(block.get("block_id", i))
        if block.get("error"):
            if block_rejection(block, threshold):
                rejected.append(block_id)
            else:
                errors.append({"block_id": block_id, "error": str(block["error"])})
            continue
        try:
            value = float(block["overall_relative_l2"])
        except (KeyError, TypeError, ValueError):
            errors.append({"block_id": block_id, "error": "missing_or_invalid_score"})
            continue
        if not math.isfinite(value) or value < 0:
            errors.append({"block_id": block_id, "error": "nonfinite_or_negative_score"})
            continue
        values.append(value)
        if value > threshold:
            rejected.append(block_id)
    return {
        "verdict": "DETECTED" if rejected else "ERROR" if errors else "PASSED",
        "score": max(values, default=None),
        "rejected_blocks": rejected,
        "error_blocks": errors,
        "checked_blocks": len(block_list),
    }


def calibrate_threshold(clean_block_sets: Iterable[Iterable[dict[str, Any]]]) -> float:
    """Use the sequence maximum; calibration failures are a failed gate."""
    maxima: list[float] = []
    for blocks in clean_block_sets:
        blocks = list(blocks)
        if not blocks:
            raise ValueError("Calibration sequence has no verifiable blocks")
        values: list[float] = []
        for block in blocks:
            if block.get("error"):
                raise ValueError(f"Calibration block failed: {block['error']}")
            try:
                value = float(block["overall_relative_l2"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Calibration block has no valid score") from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError("Calibration block score is nonfinite or negative")
            values.append(value)
        maxima.append(max(values))
    if not maxima:
        raise ValueError("No clean calibration sequences")
    return max(maxima)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if trials == 0:
        return None
    p = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    radius = (z / denominator) * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_cases(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Keep failed injections and undecidable runs outside the four cells."""
    rows = list(cases)
    counts: Counter[str] = Counter()
    for case in rows:
        positive = bool(case["planned_positive"])
        injected = case.get("injection_success", None if positive else True)
        if positive and injected is False:
            counts["INJECTION_FAILED"] += 1
            continue
        verdict = case.get("verdict", "ERROR")
        if positive and injected is None:
            counts["INJECTION_STATUS_UNKNOWN"] += 1
            verdict = "ERROR"
        if verdict not in ("ERROR", "UNSUPPORTED", "DETECTED", "PASSED"):
            raise ValueError(f"Unknown verdict: {verdict}")
        if verdict in ("ERROR", "UNSUPPORTED"):
            counts[verdict] += 1
            counts[f"{verdict}_POSITIVE" if positive else f"{verdict}_NEGATIVE"] += 1
        elif positive:
            counts["TP" if verdict == "DETECTED" else "FN"] += 1
        else:
            counts["FP" if verdict == "DETECTED" else "TN"] += 1
    for key in ("TP", "FP", "TN", "FN", "ERROR", "UNSUPPORTED", "INJECTION_FAILED"):
        counts.setdefault(key, 0)
    successful_positive = counts["TP"] + counts["FN"] + counts["ERROR_POSITIVE"] + counts["UNSUPPORTED_POSITIVE"] - counts["INJECTION_STATUS_UNKNOWN"]
    negative = counts["TN"] + counts["FP"] + counts["ERROR_NEGATIVE"] + counts["UNSUPPORTED_NEGATIVE"]
    detectable_positive = counts["TP"] + counts["FN"]
    detectable_negative = counts["FP"] + counts["TN"]

    def safe_ratio(num: int, den: int) -> float | None:
        return num / den if den else None

    precision_den = counts["TP"] + counts["FP"]
    precision = safe_ratio(counts["TP"], precision_den)
    recall = safe_ratio(counts["TP"], detectable_positive)
    return {
        "counts": dict(counts),
        "planned_sequences": len(rows),
        "successful_positive": successful_positive,
        "negative": negative,
        "injection_status_unknown": counts["INJECTION_STATUS_UNKNOWN"],
        "injection_success_rate": safe_ratio(successful_positive, successful_positive + counts["INJECTION_FAILED"] + counts["INJECTION_STATUS_UNKNOWN"]),
        "decidable_coverage": safe_ratio(detectable_positive + detectable_negative, successful_positive + negative + counts["INJECTION_STATUS_UNKNOWN"]),
        "tpr": recall,
        "fpr": safe_ratio(counts["FP"], detectable_negative),
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None,
        "deployment_recall": safe_ratio(counts["TP"], successful_positive),
        "tpr_wilson_95": wilson_interval(counts["TP"], detectable_positive),
        "fpr_wilson_95": wilson_interval(counts["FP"], detectable_negative),
    }


def exact_hit_probability(block_count: int, rejected_count: int, sample_count: int) -> float:
    if not 0 <= rejected_count <= block_count:
        raise ValueError("Invalid rejected block count")
    if not 0 <= sample_count <= block_count:
        raise ValueError("Invalid sample count")
    if sample_count == 0 or rejected_count == 0:
        return 0.0
    if sample_count > block_count - rejected_count:
        return 1.0
    probability_of_miss = 1.0
    for index in range(sample_count):
        probability_of_miss *= (block_count - rejected_count - index) / (block_count - index)
    return 1.0 - probability_of_miss


def _sample_seed(case_key: str, rate: float, repeat: int, root_seed: int) -> int:
    payload = json.dumps([case_key, rate, repeat, root_seed], separators=(",", ":"))
    return int.from_bytes(sha256(payload.encode()).digest()[:8], "big")


def sampled_block_ids(case_key: str, block_count: int, rate: float, repeat: int, root_seed: int) -> list[int]:
    """The saved case key and seeds reproduce every sampled index exactly."""
    if not 0 < rate <= 1:
        raise ValueError("rate must be in (0, 1]")
    sample_count = min(block_count, max(1, math.ceil(rate * block_count)))
    rng = random.Random(_sample_seed(case_key, rate, repeat, root_seed))
    return sorted(rng.sample(range(block_count), sample_count))


def simulate_sampling(
    case_key: str,
    blocks: list[dict[str, Any]],
    threshold: float,
    rates: Iterable[float],
    repeats: int,
    root_seed: int,
) -> list[dict[str, Any]]:
    """Replay sample decisions after the complete evidence is frozen."""
    if not blocks:
        raise ValueError("Cannot sample an empty evidence set")
    rejected_indices = {i for i, block in enumerate(blocks) if block_rejection(block, threshold)}
    output: list[dict[str, Any]] = []
    for rate in rates:
        sample_count = min(len(blocks), max(1, math.ceil(rate * len(blocks))))
        hits = 0
        errors = 0
        first_sample: list[int] = []
        for repeat in range(repeats):
            sampled = sampled_block_ids(case_key, len(blocks), rate, repeat, root_seed)
            if repeat == 0:
                first_sample = sampled
            if any(index in rejected_indices for index in sampled):
                hits += 1
            elif any(blocks[index].get("error") for index in sampled):
                errors += 1
        output.append({
            "rate": rate,
            "sample_count": sample_count,
            "repeats": repeats,
            "hits": hits,
            "errors": errors,
            "estimated_hit_rate": hits / repeats,
            "exact_hit_probability": exact_hit_probability(len(blocks), len(rejected_indices), sample_count),
            "rejected_block_count": len(rejected_indices),
            "first_sampled_indices": first_sample,
            "sample_seed_scheme": "sha256([case_key,rate,repeat,root_seed])[:8] + Python Random.sample",
            "root_seed": root_seed,
        })
    return output


def summarize_sampling(
    cases: Iterable[dict[str, Any]], rates: Iterable[float], repeats: int,
) -> dict[str, dict[str, Any]]:
    """Report sampled TP/FN/FP/TN with explicit error and failed-injection counts.

    Repeats from one request are correlated, so this reports empirical rates
    without presenting a per-trial confidence interval as independent data.
    """
    rows = list(cases)
    output: dict[str, dict[str, Any]] = {}
    for rate in rates:
        counts: Counter[str] = Counter()
        positive_exact: list[float] = []
        negative_exact: list[float] = []
        for case in rows:
            positive = bool(case["planned_positive"])
            injected = case.get("injection_success", None if positive else True)
            if positive and injected is False:
                counts["INJECTION_FAILED"] += repeats
                continue
            if positive and injected is None:
                counts["INJECTION_STATUS_UNKNOWN"] += repeats
                continue
            sample = next((s for s in case.get("sampling", []) if s["rate"] == rate), None)
            if sample is None:
                counts["ERROR_POSITIVE" if positive else "ERROR_NEGATIVE"] += repeats
                continue
            if sample["repeats"] != repeats:
                raise ValueError("Sampling repeat count differs from frozen configuration")
            hits = int(sample["hits"])
            errors = int(sample["errors"])
            passed = repeats - hits - errors
            if min(hits, errors, passed) < 0:
                raise ValueError("Invalid sampled outcome counts")
            if positive:
                counts["TP"] += hits
                counts["FN"] += passed
                counts["ERROR_POSITIVE"] += errors
                positive_exact.append(float(sample["exact_hit_probability"]))
            else:
                counts["FP"] += hits
                counts["TN"] += passed
                counts["ERROR_NEGATIVE"] += errors
                negative_exact.append(float(sample["exact_hit_probability"]))
        for key in ("TP", "FN", "FP", "TN", "ERROR_POSITIVE", "ERROR_NEGATIVE",
                    "INJECTION_FAILED", "INJECTION_STATUS_UNKNOWN"):
            counts.setdefault(key, 0)
        decidable_positive = counts["TP"] + counts["FN"]
        decidable_negative = counts["FP"] + counts["TN"]
        eligible = (
            decidable_positive + decidable_negative
            + counts["ERROR_POSITIVE"] + counts["ERROR_NEGATIVE"]
            + counts["INJECTION_STATUS_UNKNOWN"]
        )
        output[str(rate)] = {
            "counts": dict(counts),
            "tpr": counts["TP"] / decidable_positive if decidable_positive else None,
            "fpr": counts["FP"] / decidable_negative if decidable_negative else None,
            "decidable_coverage": (decidable_positive + decidable_negative) / eligible if eligible else None,
            "mean_exact_positive_hit_probability": (
                sum(positive_exact) / len(positive_exact) if positive_exact else None
            ),
            "mean_exact_negative_hit_probability": (
                sum(negative_exact) / len(negative_exact) if negative_exact else None
            ),
            "repeat_count_per_sequence": repeats,
        }
    return output
