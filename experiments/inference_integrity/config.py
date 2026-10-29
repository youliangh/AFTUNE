"""Frozen, dependency-free configuration for inference-integrity experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


CONDITIONS = (
    "kv_cache",
    "serving_config",
    "model_mismatch",
    "token_selection",
    "computation",
    "input_mismatch",
)

EVIDENCE_ORIGINS = (
    "generation_live",
    "model_recompute_live",
)

SAMPLE_RATES = (0.08,)
DEFAULT_DATASET_REVISION = "5f60cd380cfc58f0f12d44892bed41ee3670a70a"
DEFAULT_DATASET_PATH = str(
    Path(__file__).resolve().parents[3]
    / ".cache/experiment-datasets/mentalchat16k"
    / DEFAULT_DATASET_REVISION
    / "inputs.jsonl"
)


@dataclass(frozen=True)
class ExperimentConfig:
    protocol_version: str = "v3"
    m: int = 256
    n: int = 32
    prefill_tokens: int = 32
    max_length: int = 128
    seeds: tuple[int, ...] = (0,)
    calibration_size: int = 32
    dataset_name: str = "ShenLab/MentalChat16K"
    dataset_revision: str | None = DEFAULT_DATASET_REVISION
    dataset_jsonl: str | None = DEFAULT_DATASET_PATH
    declared_model_path: str = "../models/Llama-3.2-1B-Instruct"
    substitute_model_path: str = "../models/Qwen3.5-0.8B"
    generation_device: str = "cuda:0"
    verification_device: str = "cpu"
    cpu_threads: int = 8
    dtype: str = "float32"
    attention_backend: str = "sdpa"
    layers_per_block: int = 2
    activation_interval: int = 4
    serving_round_bits: int = 6
    serving_round_bits_sweep: tuple[int, ...] = ()
    computation_relative_l2: float = 1e-2
    sample_rates: tuple[float, ...] = SAMPLE_RATES
    sample_repeats: int = 1000
    retain_record_indices: tuple[int, ...] = (0, 64, 128, 192)

    def __post_init__(self) -> None:
        if not (0 < self.n < self.m):
            raise ValueError("Require 0 < n < m")
        if self.prefill_tokens < 2 or self.max_length <= self.prefill_tokens + 1:
            raise ValueError("Invalid token lengths")
        if len(self.seeds) == 0 or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Seeds must be nonempty and unique")
        if self.calibration_size < 1:
            raise ValueError("calibration_size must be positive")
        if self.dtype != "float32":
            raise ValueError("The preregistered primary experiment requires float32")
        if self.layers_per_block < 1 or self.activation_interval < 1:
            raise ValueError("Block size and activation interval must be positive")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        if self.protocol_version != "v3" or self.serving_round_bits_sweep:
            raise ValueError("The v3 protocol has one serving configuration and shared controls")
        if any(not 0 < rate <= 1 for rate in self.sample_rates):
            raise ValueError("Sample rates must be in (0, 1]")
        if self.sample_repeats < 1:
            raise ValueError("sample_repeats must be positive")
        if self.serving_round_bits not in range(1, 24):
            raise ValueError("serving_round_bits must be in [1, 23]")
        if any(bit not in range(1, 24) for bit in self.serving_round_bits_sweep):
            raise ValueError("serving_round_bits_sweep must contain valid fraction-bit counts")
        if self.computation_relative_l2 <= 0:
            raise ValueError("computation_relative_l2 must be positive")

    @property
    def generated_tokens(self) -> int:
        return self.max_length - self.prefill_tokens

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    allowed = {field.name for field in fields(ExperimentConfig)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for key in ("seeds", "serving_round_bits_sweep", "sample_rates", "retain_record_indices"):
        if key in data:
            data[key] = tuple(data[key])
    for key in ("dataset_jsonl", "declared_model_path", "substitute_model_path"):
        if data.get(key):
            resolved = Path(data[key]).expanduser()
            data[key] = str((path.parent / resolved).resolve() if not resolved.is_absolute() else resolved)
    return ExperimentConfig(**data)
