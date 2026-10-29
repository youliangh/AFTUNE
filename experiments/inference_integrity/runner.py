"""Resumable, evidence-preserving orchestration of the integrity experiment.

Ground truth is written under ``oracle/`` and never supplied to the native
verifier. A model-substitution case exposes only the Llama recomputation trace.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import platform
from pathlib import Path
import random
import secrets
import shutil
import sys
import time
from typing import Any, Iterator

from .config import CONDITIONS, EVIDENCE_ORIGINS, ExperimentConfig
from .metrics import calibrate_threshold, decide_sequence, simulate_sampling, summarize_cases, summarize_sampling


ALL_CONDITIONS = CONDITIONS


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derive_seed(*parts: Any) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "big")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _progress(root: Path, phase: str, **details: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    row = {"at": _timestamp(), "phase": phase, **details}
    with (root / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def _dataset_rows(config: ExperimentConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only a pinned local source and validate its frozen derivation hash."""
    if not config.dataset_jsonl:
        raise FileNotFoundError("dataset_jsonl is required; prepare a local frozen MentalChat16K JSONL")
    path = Path(config.dataset_jsonl).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen dataset missing: {path}")
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset checksum manifest missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    actual_hash = _hash_file(path)
    if manifest.get("derived_sha256") != actual_hash:
        raise ValueError("Frozen dataset checksum differs from its manifest")
    if config.dataset_revision and manifest.get("revision") != config.dataset_revision:
        raise ValueError("Dataset revision differs from experiment configuration")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("input"), str):
                raise ValueError(f"Invalid dataset row {line_number}")
            if row["input"].strip():
                rows.append(row)
    if len(rows) != manifest.get("rows"):
        raise ValueError("Frozen dataset row count differs from its manifest")
    return rows, {"path": str(path), "sha256": actual_hash, "manifest": manifest}


def _encode_prompt(text: str, tokenizer: Any, length: int) -> list[int] | None:
    if tokenizer.bos_token_id is None:
        raise ValueError("Llama tokenizer has no BOS token")
    # The declared input is exactly one BOS plus 31 content tokens. The
    # default chat template alone would exceed the blueprint's P=32 budget.
    content = tokenizer.encode(text, add_special_tokens=False)
    if len(content) < length - 1:
        return None
    return [int(tokenizer.bos_token_id)] + [int(token) for token in content[: length - 1]]


def _model_digest(model_path: str) -> dict[str, Any]:
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory missing: {root}")
    important = sorted(root.glob("*.safetensors")) + sorted(root.glob("*.safetensors.index.json"))
    important += sorted(root.glob("config.json")) + sorted(root.glob("tokenizer.json"))
    if not important:
        raise FileNotFoundError(f"No model checkpoint/config files in {root}")
    return {"path": str(root), "files": {p.name: _hash_file(p) for p in important}}


def _code_digest() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    core = (
        "config.py", "metrics.py", "runner.py", "native.py", "injections.py", "generation.py",
    )
    result = {name: _hash_file(directory / name) for name in core}
    parent = directory.parents[1]
    for name in (
        "verify_inference.py", "recorder.py", "replay_utils.py", "verify_block.py",
        "finetune.py", "load_model.py",
    ):
        result[name] = _hash_file(parent / name)
    return result


def _environment_manifest(config: ExperimentConfig) -> dict[str, Any]:
    from importlib.metadata import version
    import torch

    gpu_name = None
    if config.generation_device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(config.generation_device).index or 0
        gpu_name = torch.cuda.get_device_name(index)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "datasets": version("datasets"),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
        "attention_backend": config.attention_backend,
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def configure_determinism(seed: int) -> None:
    """Freeze Python/NumPy/Torch randomness and FP32 backend behavior."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def prepare_corpus(config: ExperimentConfig, root: Path) -> dict[str, Any]:
    """Freeze calibration/evaluation requests and labels before any inference."""
    import transformers

    existing = root / "manifest.json"
    if existing.exists():
        payload = _read_json(existing)
        if payload.get("config_sha256") != config.fingerprint():
            raise ValueError("Output directory already belongs to a different configuration")
        path = root / "corpus.json"
        if not path.is_file() or _hash_file(path) != payload.get("corpus_sha256"):
            raise ValueError("Frozen corpus is missing or altered")
        _, dataset = _dataset_rows(config)
        if dataset["sha256"] != payload.get("dataset", {}).get("sha256"):
            raise ValueError("Dataset changed since the experiment was prepared")
        if _code_digest() != payload.get("experiment_code_sha256"):
            raise ValueError("Experiment code changed since this output was prepared; use a separate output directory")
        for key, model_path in (
            ("declared_model", config.declared_model_path),
            ("substitute_model", config.substitute_model_path),
        ):
            if _model_digest(model_path) != payload.get(key):
                raise ValueError(f"{key} changed since the experiment was prepared")
        return payload

    rows, dataset = _dataset_rows(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.declared_model_path, local_files_only=True
    )
    seen_ids: set[tuple[int, ...]] = set()
    eligible: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows):
        ids = _encode_prompt(row["input"], tokenizer, config.prefill_tokens)
        if ids is None or tuple(ids) in seen_ids:
            continue
        seen_ids.add(tuple(ids))
        eligible.append({
            "source_row": row_number,
            "source_file": row.get("source_file"),
            "source_text_sha256": sha256(row["input"].encode("utf-8")).hexdigest(),
            "input_ids": ids,
            "request_text": tokenizer.decode(ids[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False),
        })
        eligible[-1]["request_text_sha256"] = sha256(
            eligible[-1]["request_text"].encode("utf-8")
        ).hexdigest()
    needed = config.calibration_size + config.m * len(config.seeds)
    if len(eligible) < needed:
        raise ValueError(f"Need {needed} unique >=32-token requests; found {len(eligible)}")
    random.Random(_derive_seed("corpus", config.fingerprint())).shuffle(eligible)
    calibration = eligible[: config.calibration_size]
    eval_rows = eligible[config.calibration_size : needed]
    evaluation = {
        str(seed): eval_rows[i * config.m : (i + 1) * config.m]
        for i, seed in enumerate(config.seeds)
    }
    corpus = {"calibration": calibration, "evaluation": evaluation}
    corpus['positive_indices'] = {str(seed): sorted(_positive_indices(config, seed))
                                  for seed in config.seeds}
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "corpus.json", corpus)
    payload = {
        "created_at": _timestamp(),
        "config": config.as_dict(),
        "config_sha256": config.fingerprint(),
        "corpus_sha256": _hash_file(root / "corpus.json"),
        "dataset": dataset,
        "declared_model": _model_digest(config.declared_model_path),
        "substitute_model": _model_digest(config.substitute_model_path),
        "environment": _environment_manifest(config),
        "experiment_code_sha256": _code_digest(),
        "label_seed_scheme": "sha256(['labels', seed, config_sha256])[:8] + Python Random.sample",
        "injection_seed_scheme": "sha256(['inject', seed, condition, row, config_sha256])[:8]",
        "shared_controls": {
            str(seed): {
                'indices': [i for i in range(config.m) if i not in _positive_indices(config, seed)],
                'source_directory': f'shared_clean/{seed}',
                'condition_origins': {c: _condition_origin(c) for c in CONDITIONS},
            } for seed in config.seeds
        },
    }
    _atomic_json(existing, payload)
    _progress(root, "prepared", eligible_rows=len(eligible), selected_requests=needed)
    return payload


def _load_corpus(config: ExperimentConfig, root: Path) -> dict[str, Any]:
    prepare_corpus(config, root)
    return _read_json(root / "corpus.json")


def _positive_indices(config: ExperimentConfig, seed: int) -> set[int]:
    rng = random.Random(_derive_seed("labels", seed, config.fingerprint()))
    return set(rng.sample(range(config.m), config.n))


def _load_llama(config: ExperimentConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config.generation_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA generation device requested but CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(config.declared_model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.declared_model_path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation=config.attention_backend,
    ).to(config.generation_device)
    model.eval()
    return model, tokenizer


def _load_qwen(config: ExperimentConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config.generation_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA generation device requested but CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(config.substitute_model_path, local_files_only=True)
    model, loading = AutoModelForCausalLM.from_pretrained(
        config.substitute_model_path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation=config.attention_backend,
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Qwen checkpoint did not load exactly: {loading}")
    model = model.to(config.generation_device)
    model.eval()
    return model, tokenizer


def _collect_device() -> None:
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reference_dir(root: Path) -> Path:
    return root / "reference"


def prepare_reference(config: ExperimentConfig, root: Path) -> None:
    from . import native

    configure_determinism(0)
    prepare_corpus(config, root)
    reference_dir = _reference_dir(root)
    marker = reference_dir / "ready.json"
    if marker.exists():
        data = _read_json(marker)
        if data.get("config_sha256") != config.fingerprint():
            raise ValueError("Reference snapshot belongs to a different configuration")
        if _directory_hashes(reference_dir, exclude={"ready.json"}) != data.get("files"):
            raise ValueError("Reference snapshot files are missing or altered")
        return
    temporary = root / "reference.building"
    if temporary.exists():
        shutil.rmtree(temporary)
    if reference_dir.exists():
        # The directory has no ready marker and belongs to this output root.
        shutil.rmtree(reference_dir)
    model, _ = _load_llama(config)
    try:
        native.prepare_reference(model, temporary, "Llama-3.2-1B-Instruct",
                                 layers_per_block=config.layers_per_block,
                                 activation_interval=config.activation_interval)
        # Recorder persists its module structure directory as an absolute
        # path. Rewrite that pointer to the final atomic destination before
        # hashing or renaming the snapshot.
        metadata_path = temporary / "metadata.json"
        metadata = _read_json(metadata_path)
        metadata["module_structure_dir"] = str(reference_dir.resolve())
        _atomic_json(metadata_path, metadata)
        files = _directory_hashes(temporary)
        _atomic_json(temporary / "ready.json", {
            "config_sha256": config.fingerprint(), "created_at": _timestamp(), "files": files,
        })
        os.replace(temporary, reference_dir)
    finally:
        del model
        _collect_device()
    _progress(root, "reference_prepared")


def _directory_hashes(root: Path, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    return {
        str(path.relative_to(root)): _hash_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in excluded
    }


def _case_dir(root: Path, seed: int, condition: str, index: int) -> Path:
    if condition == 'shared_clean':
        return root / 'shared_clean' / str(seed) / f'{index:04d}'
    return root / "cases" / str(seed) / condition / f"{index:04d}"


def _condition_origin(condition: str) -> str:
    return 'model_recompute_live' if condition == 'model_mismatch' else 'generation_live'


def _source_case_dir(root: Path, seed: int, condition: str, index: int,
                     positives: set[int]) -> Path:
    return _case_dir(root, seed, condition if index in positives else 'shared_clean', index)


def _condition_kind(condition: str) -> str:
    if condition.startswith("serving_config"):
        return "serving"
    return {
        "kv_cache": "kv_cache",
        "model_mismatch": "model",
        "token_selection": "token_selection",
        "computation": "computation",
        "input_mismatch": "input",
    }[condition]


def _injection_spec(config: ExperimentConfig, seed: int, condition: str, index: int):
    from .injections import InjectionSpec

    kind = _condition_kind(condition)
    if kind == "model":
        return InjectionSpec(kind="model", seed=_derive_seed("inject", seed, condition, index, config.fingerprint()))
    seed_condition = condition
    injection_seed = _derive_seed("inject", seed, seed_condition, index, config.fingerprint())
    rng = random.Random(injection_seed)
    step = rng.randint(2, config.generated_tokens) if kind not in ("input",) else None
    layer = rng.randrange(16) if kind in ("kv_cache", "computation") else None
    bits = config.serving_round_bits
    donor_position = rng.randint(1, config.prefill_tokens + step - 3) if kind == "kv_cache" else None
    return InjectionSpec(
        kind=kind,
        step=step,
        layer=layer,
        seed=injection_seed,
        epsilon=config.computation_relative_l2,
        rounding_bits=bits,
        donor_position=donor_position,
    )


def _recorded_forward(model: Any, reference_dir: Path, case_dir: Path):
    from . import native

    def callback(prefix, step):
        logits = native.record_forward(
            model,
            prefix,
            case_dir / "records" / "generation_live" / f"step_{step:04d}",
            reference_dir,
            "Llama-3.2-1B-Instruct",
        )
        if step == 1 or step % 16 == 0:
            _case_progress(case_dir, 'recording_progress', step=step)
        return logits

    return callback


def _case_progress(case_dir: Path, phase: str, **details: Any) -> None:
    for parent in case_dir.parents:
        if (parent / 'manifest.json').is_file():
            _progress(parent, phase, case=str(case_dir.relative_to(parent)), **details)
            return


def _record_recompute(model: Any, full_ids: list[int], reference_dir: Path, case_dir: Path, origin: str) -> None:
    import torch
    from . import native

    if len(full_ids) < 2:
        raise ValueError("Cannot recompute a sequence of fewer than two tokens")
    prefix = torch.tensor(full_ids[:-1], dtype=torch.long, device=next(model.parameters()).device)
    native.record_forward(
        model,
        prefix,
        case_dir / "records" / origin,
        reference_dir,
        "Llama-3.2-1B-Instruct",
    )


def _verify_evidence(
    case_dir: Path, reference_dir: Path, config: ExperimentConfig, origin: str,
    verified_reference: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    from . import native

    origin_dir = case_dir / "records" / origin
    if origin == "generation_live":
        record_dirs = sorted(origin_dir.glob("step_*"))
        if len(record_dirs) != config.generated_tokens:
            raise RuntimeError(f"Expected {config.generated_tokens} live step records, found {len(record_dirs)}")
    else:
        record_dirs = [origin_dir]
    if not all(path.is_dir() for path in record_dirs):
        raise FileNotFoundError(f"Missing evidence under {origin_dir}")
    outputs: dict[str, list[dict[str, Any]]] = {"numeric": [], "strict_hash": []}
    for step, path in enumerate(record_dirs, 1):
        paired = native.verify_record_both_modes(
            path,
            reference_dir,
            device=config.verification_device,
            verified_reference=verified_reference,
        )
        for mode, blocks in paired.items():
            expected_blocks = (18 + config.layers_per_block - 1) // config.layers_per_block
            if len(blocks) != expected_blocks:
                raise RuntimeError(f"Expected {expected_blocks} supported blocks in {path}, found {len(blocks)}")
            for block in blocks:
                step_id = step if origin == 'generation_live' else 0
                outputs[mode].append({**block, 'layer_block_id': block['block_id'],
                                      'block_id': f'{step_id}/{block["block_id"]}', 'step': step_id})
        if step == 1 or step % 16 == 0 or step == len(record_dirs):
            _case_progress(case_dir, 'verification_progress', origin=origin,
                           step=step, total_steps=len(record_dirs))
    return outputs


def _generation_payload(result: Any) -> dict[str, Any]:
    return {
        "input_ids": list(result.input_ids),
        "generated_ids": list(result.generated_ids),
        "full_ids": list(result.full_ids),
        "response_text": result.response_text,
        "events": result.events,
    }


def _generation_success(kind: str, events: list[dict[str, Any]]) -> bool:
    if kind == "serving":
        switches = [e for e in events if e.get("kind") == "serving" and e.get("event") == "route_switch"]
        changes = [e for e in events if e.get("kind") == "serving" and e.get("changed_elements", 0) > 0]
        return len(switches) == 1 and bool(changes)
    named = [e for e in events if e.get("kind") == kind]
    return len(named) == 1 and bool(named[0].get("triggered", named[0].get("changed_elements", 1) > 0))


def _write_oracle(case_dir: Path, data: dict[str, Any]) -> None:
    _atomic_json(case_dir / "oracle" / "truth.json", data)


def _freeze_evidence(case_dir: Path) -> dict[str, str]:
    """Commit all trace files before selecting any audit blocks."""
    paths = sorted(path for path in (case_dir / "records").rglob("*") if path.is_file())
    if not paths:
        raise RuntimeError("No recorded evidence to freeze")
    digests = {str(path.relative_to(case_dir)): _hash_file(path) for path in paths}
    _atomic_json(case_dir / "commitment.json", {
        "frozen_at": _timestamp(),
        "algorithm": "sha256",
        "files": digests,
        'activation_preparation': _activation_costs(case_dir),
    })
    return digests


def _activation_costs(case_dir: Path) -> dict[str, Any]:
    rows = [_read_json(p) for p in (case_dir / 'records').rglob('activation_preparation.json')]
    return {'method': 'original_live_tensor_response', 'records': len(rows),
            'elapsed_seconds': sum(row['elapsed_seconds'] for row in rows),
            'base_bytes': sum(row['base_bytes'] for row in rows),
            'response_bytes': sum(row['response_bytes'] for row in rows)}


def _record_bytes(case_dir: Path) -> int:
    return sum(path.stat().st_size for path in (case_dir / "records").rglob("*") if path.is_file())


def _save_case_result(
    case_dir: Path, *, condition: str, seed: int, index: int, positive: bool,
    injection_success: bool | None, origin: str, blocks: dict[str, list[dict[str, Any]]] | None,
    thresholds: dict[str, float] | None, status_error: str | None, config: ExperimentConfig,
    stage_seconds: dict[str, float] | None = None, proof_bytes: int | None = None,
) -> dict[str, Any]:
    key = f"{seed}/{condition}/{index:04d}/{origin}"
    result: dict[str, Any] = {
        "case_key": key,
        "condition": condition,
        "seed": seed,
        "index": index,
        "planned_positive": positive,
        "injection_success": injection_success,
        "evidence_origin": origin,
        "config_sha256": config.fingerprint(),
        "status_error": status_error,
        "finished_at": _timestamp(),
        "stage_seconds": stage_seconds or {},
        "proof_bytes": proof_bytes,
    }
    if blocks is None:
        result.update({"verdict": "ERROR", "strict_hash_verdict": "ERROR", "score": None})
    else:
        _atomic_json(case_dir / "blocks_numeric.json", blocks["numeric"])
        _atomic_json(case_dir / "blocks_strict_hash.json", blocks["strict_hash"])
        if thresholds is None or origin not in thresholds:
            raise ValueError(f"Missing frozen calibration threshold for {origin}")
        numeric = decide_sequence(blocks["numeric"], thresholds[origin])
        strict = decide_sequence(blocks["strict_hash"], float("inf"))
        result.update(numeric)
        result["strict_hash_verdict"] = strict["verdict"]
        result["strict_hash_rejected_blocks"] = strict["rejected_blocks"]
        result["threshold"] = thresholds[origin]
        result["block_count"] = len(blocks["numeric"])
        # Challenge randomness is drawn only after the trace commitment was
        # saved. It is private to the evaluator until the audit is complete.
        audit_seed = secrets.randbits(128)
        result["audit_seed"] = audit_seed
        result["sampling"] = simulate_sampling(
            key,
            blocks["numeric"],
            thresholds[origin],
            config.sample_rates,
            config.sample_repeats,
            audit_seed,
        )
    _atomic_json(case_dir / "result.json", result)
    return result


def _purge_unretained_records(case_dir: Path, config: ExperimentConfig, index: int) -> None:
    if index in config.retain_record_indices:
        return
    records = case_dir / "records"
    if records.exists():
        shutil.rmtree(records)


def _verified_reference(config: ExperimentConfig, root: Path):
    from . import native

    return native.VerifiedReference(
        _reference_dir(root), device=config.verification_device, use_float32=True
    )


def _park_generation_model(model: Any, config: ExperimentConfig) -> bool:
    """Free GPU space while the independent reference is replayed on GPU."""
    if config.generation_device.startswith("cuda") and config.verification_device.startswith("cuda"):
        model.to("cpu")
        _collect_device()
        return True
    return False


def _restore_generation_model(model: Any, config: ExperimentConfig, parked: bool) -> None:
    if parked:
        model.to(config.generation_device)


def _calibration_dir(root: Path, index: int) -> Path:
    return root / "calibration" / f"{index:04d}"


def calibrate(config: ExperimentConfig, root: Path, limit: int | None = None) -> dict[str, float]:
    """Freeze origin-specific maxima from disjoint clean requests."""
    from . import native
    from .generation import generate_llama, llama_retokenize_response

    configure_determinism(_derive_seed("calibration", config.fingerprint()))
    corpus = _load_corpus(config, root)
    prepare_reference(config, root)
    marker = root / "thresholds.json"
    if marker.exists():
        data = _read_json(marker)
        if data.get("config_sha256") != config.fingerprint():
            raise ValueError("Thresholds belong to another configuration")
        return data["thresholds"]
    target = min(config.calibration_size, limit) if limit is not None else config.calibration_size
    if target < 1:
        raise ValueError("Calibration limit must be positive")
    model, tokenizer = _load_llama(config)
    cached_reference = _verified_reference(config, root) if config.verification_device == 'cpu' else None
    try:
        for index in range(target):
            request = corpus["calibration"][index]
            case_dir = _calibration_dir(root, index)
            done = case_dir / "calibration.json"
            if done.exists():
                continue
            records = case_dir / "records"
            if records.exists():
                shutil.rmtree(records)
            _progress(root, "calibration_started", index=index)
            started = time.monotonic()
            result = generate_llama(
                model, tokenizer, request["input_ids"], steps=config.generated_tokens,
                forward=_recorded_forward(model, _reference_dir(root), case_dir),
            )
            bridge = llama_retokenize_response(request["input_ids"], result.response_text, tokenizer)
            _record_recompute(
                model, bridge, _reference_dir(root), case_dir, "model_recompute_live"
            )
            _freeze_evidence(case_dir)
            parked = _park_generation_model(model, config)
            reference = None
            try:
                reference = cached_reference or _verified_reference(config, root)
                by_origin: dict[str, str] = {}
                for origin in EVIDENCE_ORIGINS:
                    verified = _verify_evidence(
                        case_dir, _reference_dir(root), config, origin, reference
                    )
                    output = case_dir / f"blocks_{origin}.json"
                    _atomic_json(output, verified["numeric"])
                    by_origin[origin] = str(output.relative_to(root))
            finally:
                del reference
                _collect_device()
                _restore_generation_model(model, config, parked)
            _atomic_json(done, {
                "index": index,
                "request_sha256": request["request_text_sha256"],
                "block_files": by_origin,
                "elapsed_seconds": time.monotonic() - started,
            })
            _purge_unretained_records(case_dir, config, index)
            _progress(root, "calibration_finished", index=index)
    finally:
        del model
        _collect_device()
    if target < config.calibration_size:
        _progress(root, "calibration_partial", processed=target, required=config.calibration_size)
        return {}
    thresholds = {
        origin: calibrate_threshold(
            _read_json(_calibration_dir(root, index) / f"blocks_{origin}.json")
            for index in range(config.calibration_size)
        )
        for origin in EVIDENCE_ORIGINS
    }
    _atomic_json(marker, {
        "created_at": _timestamp(),
        "config_sha256": config.fingerprint(),
        "calibration_size": config.calibration_size,
        "thresholds": thresholds,
    })
    _progress(root, "calibration_complete", thresholds=thresholds)
    return thresholds


def _load_thresholds(config: ExperimentConfig, root: Path) -> dict[str, float]:
    path = root / "thresholds.json"
    if not path.exists():
        raise FileNotFoundError("Full clean calibration must finish before formal evaluation")
    data = _read_json(path)
    if data.get("config_sha256") != config.fingerprint() or data.get("calibration_size") != config.calibration_size:
        raise ValueError("Frozen calibration does not match this configuration")
    if set(data.get("thresholds", {})) != set(EVIDENCE_ORIGINS):
        raise ValueError("Calibration is missing an evidence origin")
    return {key: float(value) for key, value in data["thresholds"].items()}


def _case_result_exists(case_dir: Path, config: ExperimentConfig, retry_errors: bool = False) -> bool:
    path = case_dir / "result.json"
    if not path.exists():
        return False
    existing = _read_json(path)
    if existing.get("config_sha256") != config.fingerprint():
        raise ValueError(f"Existing case result uses another configuration: {path}")
    bad_origins = any((_read_json(case_dir / origin / 'result.json').get('error_blocks') or
                       _read_json(case_dir / origin / 'result.json').get('verdict') == 'ERROR')
                      for origin in existing.get('origins', [])
                      if (case_dir / origin / 'result.json').is_file())
    if retry_errors and (existing.get("status_error") or bad_origins):
        diagnostics = case_dir.parent / "_diagnostics"
        diagnostics.mkdir(parents=True, exist_ok=True)
        archive = diagnostics / f"{case_dir.name}_{int(time.time_ns())}"
        shutil.move(str(case_dir), str(archive))
        return False
    return True


def _run_non_model_case(
    config: ExperimentConfig, root: Path, seed: int, condition: str, index: int,
    request: dict[str, Any], positive: bool, model: Any, tokenizer: Any,
    reference: Any, thresholds: dict[str, float], donor_request: dict[str, Any],
    retry_errors: bool = False,
) -> None:
    from .generation import generate_llama, llama_retokenize_response

    case_dir = _case_dir(root, seed, condition, index)
    if _case_result_exists(case_dir, config, retry_errors):
        return
    if (case_dir / "records").exists():
        shutil.rmtree(case_dir / "records")
    _progress(root, "case_started", seed=seed, condition=condition, index=index, positive=positive)
    started = time.monotonic()
    spec = _injection_spec(config, seed, condition, index) if positive else None
    kind = 'clean' if condition == 'shared_clean' else _condition_kind(condition)
    origins = EVIDENCE_ORIGINS if condition == 'shared_clean' else ('generation_live',)
    donor_inputs = None
    injection_success: bool | None = None if positive else True
    parked = False
    stage_seconds: dict[str, float] = {}
    proof_bytes: int | None = None
    try:
        stage_start = time.monotonic()
        if kind == "kv_cache" and positive:
            donor_started = time.monotonic()
            donor = generate_llama(
                model, tokenizer, donor_request["input_ids"], steps=config.generated_tokens,
            )
            donor_inputs = donor.full_ids
            stage_seconds['donor_generation'] = time.monotonic() - donor_started
        generated = generate_llama(
            model, tokenizer, request["input_ids"], steps=config.generated_tokens,
            spec=spec,
            forward=_recorded_forward(model, _reference_dir(root), case_dir),
            donor_inputs=donor_inputs,
        )
        injection_success = _generation_success(kind, generated.events) if positive else True
        stage_seconds["generation_and_recording"] = time.monotonic() - stage_start
        _write_oracle(case_dir, {
            "planned_positive": positive,
            "injection_success": injection_success,
            "kind": kind,
            "spec": vars(spec) if spec else None,
            "source_request": request,
            "donor_request": donor_request if donor_inputs is not None else None,
            "source_model": config.declared_model_path,
            "generation": _generation_payload(generated),
        })
        stage_start = time.monotonic()
        if condition == 'shared_clean':
            bridge = llama_retokenize_response(request['input_ids'], generated.response_text, tokenizer)
            _record_recompute(model, bridge, _reference_dir(root), case_dir, 'model_recompute_live')
            stage_seconds['model_recompute_recording'] = time.monotonic() - stage_start
            truth = _read_json(case_dir / 'oracle' / 'truth.json')
            truth['declared_retokenized_ids'] = bridge
            _write_oracle(case_dir, truth)
        _freeze_evidence(case_dir)
        proof_bytes = _record_bytes(case_dir)
        stage_start = time.monotonic()
        parked = _park_generation_model(model, config)
        stage_seconds["model_parking"] = time.monotonic() - stage_start
        active_reference = None
        try:
            active_reference = reference or _verified_reference(config, root)
            for origin in origins:
                stage_start = time.monotonic()
                blocks = _verify_evidence(
                    case_dir, _reference_dir(root), config, origin, active_reference
                )
                stage_seconds[f"verify_{origin}"] = time.monotonic() - stage_start
                _save_case_result(
                    case_dir / origin, condition=condition, seed=seed, index=index,
                    positive=positive, injection_success=injection_success,
                    origin=origin, blocks=blocks, thresholds=thresholds,
                    status_error=None, config=config,
                    stage_seconds=stage_seconds,
                    proof_bytes=sum(p.stat().st_size for p in (case_dir / 'records' / origin).rglob('*') if p.is_file()),
                )
        finally:
            del active_reference
            _collect_device()
            _restore_generation_model(model, config, parked)
            parked = False
        _atomic_json(case_dir / "result.json", {
            "condition": condition,
            "seed": seed,
            "index": index,
            "config_sha256": config.fingerprint(),
            "origins": list(origins),
            "stage_seconds": stage_seconds,
            "proof_bytes": proof_bytes,
            "elapsed_seconds": time.monotonic() - started,
        })
        _purge_unretained_records(case_dir, config, index)
        _progress(root, "case_finished", seed=seed, condition=condition, index=index)
    except Exception as exc:
        if parked:
            _collect_device()
            _restore_generation_model(model, config, parked)
        # Preserve partial proof and oracle for diagnosis, with a terminal
        # error result so the run does not silently omit a scheduled sample.
        _write_oracle(case_dir, {
            "planned_positive": positive,
            "injection_success": injection_success,
            "kind": kind,
            "spec": vars(spec) if spec else None,
            "source_request": request,
            "source_model": config.declared_model_path,
            "exception": repr(exc),
        })
        for origin in origins:
            result_dir = case_dir / origin
            if not (result_dir / "result.json").exists():
                _save_case_result(
                    result_dir, condition=condition, seed=seed, index=index,
                    positive=positive, injection_success=injection_success,
                    origin=origin, blocks=None, thresholds=None,
                    status_error=repr(exc), config=config,
                )
        _atomic_json(case_dir / "result.json", {
            "condition": condition,
            "seed": seed,
            "index": index,
            "config_sha256": config.fingerprint(),
            "origins": list(origins),
            "status_error": repr(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        _progress(root, "case_error", seed=seed, condition=condition, index=index, error=repr(exc))


def _prepare_model_sources(
    config: ExperimentConfig, root: Path, seed: int, requests: list[dict[str, Any]],
    positives: set[int], indices: range, retry_errors: bool,
) -> None:
    """Generate all missing Qwen positives before bringing Llama onto the GPU."""
    from .generation import generate_qwen

    missing = [
        index for index in indices
        if index in positives
        and not _case_result_exists(_case_dir(root, seed, "model_mismatch", index), config, retry_errors)
        and not (_case_dir(root, seed, "model_mismatch", index) / "oracle" / "model_source.json").exists()
    ]
    if not missing:
        return
    qwen, qwen_tokenizer = _load_qwen(config)
    try:
        for index in missing:
            case_dir = _case_dir(root, seed, "model_mismatch", index)
            request = requests[index]
            _progress(root, "model_source_started", seed=seed, index=index)
            source_start = time.monotonic()
            source = generate_qwen(
                qwen,
                qwen_tokenizer,
                request["request_text"],
                steps=config.generated_tokens,
                spec=_injection_spec(config, seed, "model_mismatch", index),
            )
            if not _generation_success("model", source.events):
                raise RuntimeError(f"Qwen substitution did not trigger for case {seed}/{index}")
            payload = _generation_payload(source)
            payload["source_generation_seconds"] = time.monotonic() - source_start
            _atomic_json(case_dir / "oracle" / "model_source.json", payload)
            _progress(root, "model_source_finished", seed=seed, index=index)
    finally:
        del qwen
        _collect_device()


def _run_model_case(
    config: ExperimentConfig, root: Path, seed: int, index: int,
    request: dict[str, Any], positive: bool, model: Any, tokenizer: Any,
    reference: Any, thresholds: dict[str, float], retry_errors: bool = False,
) -> None:
    from .generation import generate_llama, llama_retokenize_response

    condition = "model_mismatch"
    origin = "model_recompute_live"
    case_dir = _case_dir(root, seed, condition, index)
    if _case_result_exists(case_dir, config, retry_errors):
        return
    if (case_dir / "records").exists():
        shutil.rmtree(case_dir / "records")
    _progress(root, "case_started", seed=seed, condition=condition, index=index, positive=positive)
    started = time.monotonic()
    injection_success: bool | None = None if positive else True
    source_model = config.substitute_model_path if positive else config.declared_model_path
    parked = False
    stage_seconds: dict[str, float] = {}
    proof_bytes: int | None = None
    try:
        if positive:
            source_path = case_dir / "oracle" / "model_source.json"
            if not source_path.is_file():
                raise FileNotFoundError("Qwen generation stage did not produce a model source")
            source = _read_json(source_path)
            injection_success = _generation_success("model", source["events"])
        else:
            # The clean controls use the same text bridge and proof stage as
            # positives. No generation trace is presented to the verifier.
            stage_start = time.monotonic()
            source = _generation_payload(generate_llama(
                model, tokenizer, request["input_ids"], steps=config.generated_tokens,
            ))
            stage_seconds["source_generation"] = time.monotonic() - stage_start
        if positive:
            stage_seconds["source_generation"] = float(source.get("source_generation_seconds", 0.0))
        bridged_ids = llama_retokenize_response(
            request["input_ids"], source["response_text"], tokenizer
        )
        _write_oracle(case_dir, {
            "planned_positive": positive,
            "injection_success": injection_success,
            "kind": "model",
            "source_request": request,
            "source_model": source_model,
            "source_generation": source,
            "declared_retokenized_ids": bridged_ids,
        })
        stage_start = time.monotonic()
        _record_recompute(model, bridged_ids, _reference_dir(root), case_dir, origin)
        stage_seconds["model_recompute_recording"] = time.monotonic() - stage_start
        _freeze_evidence(case_dir)
        proof_bytes = _record_bytes(case_dir)
        stage_start = time.monotonic()
        parked = _park_generation_model(model, config)
        stage_seconds["model_parking"] = time.monotonic() - stage_start
        active_reference = None
        try:
            active_reference = reference or _verified_reference(config, root)
            stage_start = time.monotonic()
            blocks = _verify_evidence(case_dir, _reference_dir(root), config, origin, active_reference)
            stage_seconds["verify_model_recompute_live"] = time.monotonic() - stage_start
            _save_case_result(
                case_dir / origin, condition=condition, seed=seed, index=index,
                positive=positive, injection_success=injection_success,
                origin=origin, blocks=blocks, thresholds=thresholds,
                status_error=None, config=config,
                stage_seconds=stage_seconds, proof_bytes=proof_bytes,
            )
        finally:
            del active_reference
            _collect_device()
            _restore_generation_model(model, config, parked)
            parked = False
        _atomic_json(case_dir / "result.json", {
            "condition": condition,
            "seed": seed,
            "index": index,
            "config_sha256": config.fingerprint(),
            "origins": [origin],
            "stage_seconds": stage_seconds,
            "proof_bytes": proof_bytes,
            "elapsed_seconds": time.monotonic() - started,
        })
        _purge_unretained_records(case_dir, config, index)
        _progress(root, "case_finished", seed=seed, condition=condition, index=index)
    except Exception as exc:
        if parked:
            _collect_device()
            _restore_generation_model(model, config, parked)
        _write_oracle(case_dir, {
            "planned_positive": positive,
            "injection_success": injection_success,
            "kind": "model",
            "source_request": request,
            "source_model": source_model,
            "exception": repr(exc),
        })
        _save_case_result(
            case_dir / origin, condition=condition, seed=seed, index=index,
            positive=positive, injection_success=injection_success,
            origin=origin, blocks=None, thresholds=None,
            status_error=repr(exc), config=config,
        )
        _atomic_json(case_dir / "result.json", {
            "condition": condition,
            "seed": seed,
            "index": index,
            "config_sha256": config.fingerprint(),
            "origins": [origin],
            "status_error": repr(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        _progress(root, "case_error", seed=seed, condition=condition, index=index, error=repr(exc))


def _check_disk_budget(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < 5 * 1024 ** 3:
        raise RuntimeError(f"Fewer than 5 GiB free before the next sequence: {free} bytes")


def _gpu_memory_status(config: ExperimentConfig) -> dict[str, int] | None:
    if not (config.generation_device.startswith("cuda") or config.verification_device.startswith("cuda")):
        return None
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU generation/verification requested but CUDA is unavailable")
    index = torch.device(config.generation_device).index or 0
    free, total = torch.cuda.mem_get_info(index)
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
    }


def _check_gpu_budget(config: ExperimentConfig) -> dict[str, int] | None:
    status = _gpu_memory_status(config)
    if status and status["free_bytes"] < 768 * 1024 ** 2:
        raise RuntimeError(
            f"Fewer than 768 MiB GPU memory free before the next sequence: {status['free_bytes']} bytes"
        )
    return status


def run_cases(
    config: ExperimentConfig, root: Path, *, conditions: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None, limit: int | None = None,
    retry_errors: bool = False,
) -> None:
    """Generate the shared controls once, then only the condition positives."""
    conditions = CONDITIONS if conditions is None else conditions
    if any(condition not in CONDITIONS for condition in conditions):
        raise ValueError(f"Unsupported condition in {conditions}")
    seeds = config.seeds if seeds is None else seeds
    if any(seed not in config.seeds for seed in seeds):
        raise ValueError('Unknown seed')
    count = config.m if limit is None else min(config.m, limit)
    if count < 1:
        raise ValueError('Case limit must be positive')
    corpus = _load_corpus(config, root)
    prepare_reference(config, root)
    thresholds = _load_thresholds(config, root)
    reference = _verified_reference(config, root) if config.verification_device == 'cpu' else None
    for seed in seeds:
        requests = corpus['evaluation'][str(seed)]
        positives = _positive_indices(config, seed)
        for condition in ('shared_clean', *conditions):
            configure_determinism(_derive_seed('formal', seed, condition, config.fingerprint()))
            indices = [i for i in range(count)
                       if (i not in positives if condition == 'shared_clean' else i in positives)]
            if condition == 'model_mismatch':
                _prepare_model_sources(config, root, seed, requests, positives, range(count), retry_errors)
            if not indices or all(_case_result_exists(_case_dir(root, seed, condition, i), config, retry_errors)
                                  for i in indices):
                continue
            model, tokenizer = _load_llama(config)
            try:
                prior_error = None
                same_error_count = 0
                for index in indices:
                    _check_disk_budget(root)
                    gpu_status = _check_gpu_budget(config)
                    _progress(root, 'resource_check', seed=seed, condition=condition, index=index, gpu=gpu_status)
                    if condition == 'model_mismatch':
                        _run_model_case(config, root, seed, index, requests[index], True,
                                        model, tokenizer, reference, thresholds, retry_errors)
                    else:
                        _run_non_model_case(config, root, seed, condition, index, requests[index],
                                            condition != 'shared_clean', model, tokenizer, reference,
                                            thresholds, requests[(index + 1) % config.m], retry_errors)
                    case_dir = _case_dir(root, seed, condition, index)
                    result = _read_json(case_dir / 'result.json')
                    current_error = result.get('status_error')
                    if not current_error:
                        for origin in result['origins']:
                            row = _read_json(case_dir / origin / 'result.json')
                            if row.get('error_blocks'):
                                current_error = json.dumps(row['error_blocks'][:1], sort_keys=True)
                                break
                    if current_error and 'out of memory' in current_error.lower():
                        raise RuntimeError(f'GPU OOM in {condition}/{seed}/{index}; evidence retained')
                    same_error_count = (same_error_count + 1 if current_error == prior_error else 1) if current_error else 0
                    prior_error = current_error
                    if same_error_count >= 3:
                        raise RuntimeError(f'Three consecutive identical failures in {condition}/{seed}: {current_error}')
            finally:
                del model
                _collect_device()
    _progress(root, 'formal_run_complete', conditions=conditions, seeds=seeds, case_limit=count)


def summarize(config: ExperimentConfig, root: Path, *, allow_partial: bool = False) -> dict[str, Any]:
    """Condition views reference the same immutable negative evidence results."""
    prepare_corpus(config, root)
    thresholds = _load_thresholds(config, root)
    groups = {}
    incomplete = []
    unique_results = set()
    unique_tasks = {}
    for condition in CONDITIONS:
        origin = _condition_origin(condition)
        for seed in config.seeds:
            positives = _positive_indices(config, seed)
            rows = []
            strict_rows = []
            sources = {}
            for index in range(config.m):
                case_dir = _source_case_dir(root, seed, condition, index, positives)
                result_path = case_dir / origin / 'result.json'
                if not result_path.is_file() or not (case_dir / 'result.json').is_file():
                    incomplete.append(f'{seed}/{condition}/{index:04d}/{origin}')
                    continue
                row = _read_json(result_path)
                if row.get('config_sha256') != config.fingerprint():
                    raise ValueError(f'Case result has another configuration: {result_path}')
                if row['planned_positive'] != (index in positives):
                    raise ValueError(f'Wrong shared-control label: {result_path}')
                rows.append(row)
                strict_rows.append({**row, 'verdict': row.get('strict_hash_verdict', 'ERROR')})
                sources[str(index)] = str(result_path.relative_to(root))
                unique_results.add(str(result_path.relative_to(root)))
                unique_tasks[str(case_dir.relative_to(root))] = _read_json(case_dir / 'result.json')
            group_key = f'{seed}/{condition}/{origin}'
            proof_sizes = [float(row['proof_bytes']) for row in rows if row.get('proof_bytes') is not None]
            groups[group_key] = {
                'numeric': summarize_cases(rows),
                'strict_hash': summarize_cases(strict_rows),
                'sampling': summarize_sampling(rows, config.sample_rates, config.sample_repeats),
                'observed_cases': len(rows), 'expected_cases': config.m,
                'threshold': thresholds[origin], 'sources': sources,
                'mean_proof_bytes': sum(proof_sizes) / len(proof_sizes) if proof_sizes else None,
            }
    if incomplete and not allow_partial:
        raise RuntimeError(f'Formal experiment incomplete: {len(incomplete)} references missing; first={incomplete[0]}')
    stage_totals = {}
    for task in unique_tasks.values():
        for stage, seconds in task.get('stage_seconds', {}).items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + float(seconds)
    result = {
        'generated_at': _timestamp(), 'protocol_version': 'v3',
        'config_sha256': config.fingerprint(), 'complete': not bool(incomplete),
        'missing_count': len(incomplete), 'missing_first_20': incomplete[:20], 'groups': groups,
        'unique_generation_tasks': len(unique_tasks),
        'unique_sequence_evidence': len(unique_results),
        'condition_sample_references': sum(group['observed_cases'] for group in groups.values()),
        'actual_costs': {'proof_bytes': sum(task.get('proof_bytes') or 0 for task in unique_tasks.values()),
                         'stage_seconds': stage_totals,
                         'note': 'Shared tasks counted once; calibration and failed retries excluded; donor stage is auxiliary.'},
        'model_case_counting': 'Model origin counted once; five other conditions share the same D negatives',
    }
    _atomic_json(root / ('summary.json' if not incomplete else 'summary.partial.json'), result)
    _progress(root, 'summary_created', complete=result['complete'], missing_count=len(incomplete))
    return result
