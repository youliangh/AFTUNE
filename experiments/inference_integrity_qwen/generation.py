"""Full-prefix text generation for the AFTUNE integrity experiment.

The ``forward`` callback lets a caller record a *live* step before the token
is selected. It runs while the injection hook is installed. For Model
Mismatch, the Substitute generation remains private and the caller records only the
subsequent declared-model recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .injections import (
    InjectionSpec,
    kv_context_fault,
    mlp_fault,
    next_legal_token,
    swap_input_spans,
)


Forward = Callable[[torch.Tensor, int], torch.Tensor]


@dataclass
class GenerationResult:
    input_ids: list[int]
    generated_ids: list[int]
    full_ids: list[int]
    response_text: str
    events: list[dict[str, Any]]


def _last_logits(logits: torch.Tensor) -> torch.Tensor:
    if hasattr(logits, "logits"):
        logits = logits.logits
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Forward callback must return a logits tensor")
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported")
        return logits[0, -1, :]
    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported")
        return logits[0, :]
    if logits.ndim == 1:
        return logits
    raise ValueError("Logits must have 1, 2, or 3 dimensions")


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _plain_forward(model: Any, prefix: torch.Tensor, _step: int) -> torch.Tensor:
    return model(input_ids=prefix.unsqueeze(0), use_cache=False, logits_to_keep=1).logits


def _tokenizer_eos_ids(tokenizer: Any) -> set[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        return {eos}
    if isinstance(eos, (list, tuple, set)):
        return {int(token) for token in eos}
    return set()


def _greedy_with_fixed_length(logits: torch.Tensor, tokenizer: Any) -> tuple[int, torch.Tensor]:
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("Model returned non-finite logits")
    allowed = logits.clone()
    # Output head rows without tokenizer tokens are never legal, in either role.
    cache = getattr(tokenizer, '_aftune_vocab_masks', None)
    if cache is None:
        cache = {}
        tokenizer._aftune_vocab_masks = cache
    key = (str(logits.device), logits.numel())
    if key not in cache:
        valid = sorted(set(int(index) for index in tokenizer.get_vocab().values()
                           if 0 <= int(index) < logits.numel()))
        mask = torch.zeros(logits.numel(), dtype=torch.bool, device=logits.device)
        mask[torch.tensor(valid, dtype=torch.long, device=logits.device)] = True
        cache[key] = mask
    allowed.masked_fill_(~cache[key], -torch.inf)
    for token in _tokenizer_eos_ids(tokenizer):
        if 0 <= token < allowed.numel():
            allowed[token] = -torch.inf
    if not bool(torch.isfinite(allowed).any().item()):
        raise ValueError("No legal next token after excluding EOS")
    return int(torch.argmax(allowed).item()), allowed


@torch.no_grad()
def generate_declared(
    model: Any,
    tokenizer: Any,
    input_ids: list[int],
    *,
    steps: int = 96,
    spec: InjectionSpec | None = None,
    forward: Forward | None = None,
    donor_inputs: list[int] | None = None,
) -> GenerationResult:
    """Generate declared-model tokens with an optional single external fault event.

    ``forward(prefix, step)`` receives a 1D long tensor and a one-based token
    index. To produce D-track evidence it should call ``native.record_forward``
    exactly once and return that call's logits. The active fault hook is already
    installed on the model during that call.
    """
    if steps < 1 or len(input_ids) != 32:
        raise ValueError("Expected 32 input tokens and at least one generation step")
    if spec is not None and spec.kind == "model":
        raise ValueError("Model mismatch uses generate_substitute_source and declared-model recomputation")
    if spec is not None and spec.step is not None and spec.step > steps:
        raise ValueError("Injection step exceeds the generation length")
    if spec is not None and spec.kind == "kv_cache" and donor_inputs is None:
        raise ValueError("KV mismatch requires a clean donor sequence")

    model.eval()
    actual_input = list(input_ids)
    events: list[dict[str, Any]] = []
    if spec is not None and spec.kind == "input":
        actual_input, event = swap_input_spans(
            actual_input, seed=spec.seed, positions=spec.swap_positions
        )
        events.append(event)

    generated: list[int] = []
    run_forward = forward or (lambda prefix, step: _plain_forward(model, prefix, step))
    for step in range(1, steps + 1):
        prefix_ids = actual_input + generated
        prefix = torch.tensor(prefix_ids, dtype=torch.long, device=_model_device(model))
        actions: list[dict[str, Any]] = []
        if spec is not None and spec.kind == "kv_cache" and step == spec.step:
            if donor_inputs is None or len(donor_inputs) < len(prefix_ids):
                raise ValueError("KV donor has fewer tokens than the target prefix")
            fault = kv_context_fault(model, donor_inputs[: len(prefix_ids)], spec, step, actions)
        elif spec is not None and spec.kind in {"computation", "serving"}:
            fault = mlp_fault(model, spec, step, actions)
        else:
            from contextlib import nullcontext
            fault = nullcontext()
        with fault:
            logits = _last_logits(run_forward(prefix, step))
        if spec is not None and spec.kind == "kv_cache" and step == spec.step:
            if len(actions) != 1:
                raise RuntimeError("KV mismatch must affect exactly one attention call")
            events.extend(actions)
        elif spec is not None and spec.kind == "computation" and step == spec.step:
            if len(actions) != 1 or actions[0]["changed_elements"] == 0:
                raise RuntimeError("Computation injection did not change one MLP output")
            events.extend(actions)
        elif spec is not None and spec.kind == "serving" and step >= spec.step:
            if len(actions) != len(model.model.layers):
                raise RuntimeError("Serving perturbation did not run in every decoder layer")
            if step == spec.step:
                events.append({
                    "kind": "serving",
                    "event": "route_switch",
                    "step": step,
                    "rounding_bits": spec.rounding_bits,
                })
            events.extend(actions)

        normal_token, legal_logits = _greedy_with_fixed_length(logits, tokenizer)
        if spec is not None and spec.kind == "token_selection" and step == spec.step:
            normal_token, replacement = next_legal_token(
                legal_logits, set(getattr(tokenizer, "all_special_ids", []))
            )
            events.append({
                "kind": "token_selection",
                "step": step,
                "normal_token": normal_token,
                "replacement_token": replacement,
                "normal_logit": float(logits[normal_token].item()),
                "replacement_logit": float(logits[replacement].item()),
                "triggered": True,
            })
            chosen = replacement
        else:
            chosen = normal_token
        generated.append(chosen)

    if spec is not None and spec.kind == "serving":
        affected = [e for e in events if e.get("kind") == "serving" and e.get("event") != "route_switch"]
        if not any(e["changed_elements"] > 0 for e in affected):
            raise RuntimeError("Serving rounding never changed an activation")
    return GenerationResult(
        input_ids=actual_input,
        generated_ids=generated,
        full_ids=actual_input + generated,
        response_text=tokenizer.decode(
            generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        events=events,
    )


@torch.no_grad()
def generate_substitute_source(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    *,
    steps: int = 96,
    spec: InjectionSpec | None = None,
    forward: Forward | None = None,
) -> GenerationResult:
    """Generate substitute text, keeping its token space private to the evaluator."""
    if spec is not None and spec.kind != "model":
        raise ValueError("Substitute generation is only used for Model Mismatch")
    if steps < 1:
        raise ValueError("Expected at least one generation step")
    source_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    if not source_ids:
        raise ValueError("Substitute tokenizer returned an empty input")
    model.eval()
    generated: list[int] = []
    run_forward = forward or (lambda prefix, step: _plain_forward(model, prefix, step))
    for step in range(1, steps + 1):
        prefix = torch.tensor(source_ids + generated, dtype=torch.long, device=_model_device(model))
        logits = _last_logits(run_forward(prefix, step))
        next_token, _ = _greedy_with_fixed_length(logits, tokenizer)
        generated.append(next_token)
    events = []
    if spec is not None:
        events.append({"kind": "model", "event": "request_model_substitution", "triggered": True})
    return GenerationResult(
        input_ids=source_ids,
        generated_ids=generated,
        full_ids=source_ids + generated,
        response_text=tokenizer.decode(
            generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        events=events,
    )


def retokenize_response(
    original_prompt_ids: list[int], response_text: str, declared_tokenizer: Any,
) -> list[int]:
    """Append the declared-model tokenization of response text to the exact 32-token prompt."""
    if len(original_prompt_ids) != 32:
        raise ValueError("Expected the original 32-token declared prompt")
    response_ids = declared_tokenizer.encode(response_text, add_special_tokens=False)
    if not response_ids:
        raise ValueError("Response text retokenized to no declared-model tokens")
    return list(original_prompt_ids) + [int(token) for token in response_ids]

