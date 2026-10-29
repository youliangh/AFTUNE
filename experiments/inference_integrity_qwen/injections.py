"""Controlled, single-event inference faults used by the integrity experiment.

Only the returned ``events`` are ground truth. They must not be passed to the
AFTUNE verifier, whose input is the Recorder artifact from the actual forward.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterator

import torch


KINDS = frozenset(
    {"kv_cache", "serving", "model", "token_selection", "computation", "input"}
)


@dataclass(frozen=True)
class InjectionSpec:
    kind: str
    step: int | None = None  # One-based generated-token index; input/model are request-level.
    layer: int | None = None  # Zero-based declared-model decoder layer.
    seed: int = 0
    epsilon: float = 1e-2
    rounding_bits: int = 6
    swap_positions: tuple[int, int] | None = None
    donor_position: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown injection kind: {self.kind}")
        if self.kind in {"kv_cache", "serving", "token_selection", "computation"}:
            if self.step is None or self.step < 2:
                raise ValueError(f"{self.kind} requires a decode step >= 2")
        if self.kind in {"kv_cache", "computation"}:
            if self.layer is None or self.layer < 0:
                raise ValueError(f"{self.kind} requires a zero-based layer")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")
        if not 1 <= self.rounding_bits <= 22:
            raise ValueError("rounding_bits must be in 1..22")


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash the exact tensor bytes for private ground-truth logging."""
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def fp32_round_fraction_bits(value: torch.Tensor, bits: int) -> torch.Tensor:
    """Round finite FP32 to nearest even after dropping fraction low bits.

    This is an artificial device-rounding proxy. It preserves the FP32 dtype
    and exponent range instead of converting to FP16 or BF16.
    """
    if value.dtype != torch.float32:
        raise TypeError("FP32 rounding injection requires float32 activations")
    if not 1 <= bits <= 22:
        raise ValueError("bits must be in 1..22")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("Cannot round a non-finite activation")

    sign_and_magnitude = value.contiguous().view(torch.int32)
    magnitude = sign_and_magnitude & 0x7FFFFFFF
    sign = sign_and_magnitude & -0x80000000
    low_mask = (1 << bits) - 1
    base = magnitude & ~low_mask
    remainder = magnitude & low_mask
    halfway = 1 << (bits - 1)
    increment = (remainder > halfway) | (
        (remainder == halfway) & (((base >> bits) & 1) == 1)
    )
    rounded_magnitude = base + increment.to(torch.int32) * (1 << bits)
    rounded = (sign | rounded_magnitude).view(torch.float32).reshape_as(value)
    if not bool(torch.isfinite(rounded).all().item()):
        raise OverflowError("FP32 rounding produced a non-finite activation")
    return rounded


def swap_input_spans(
    input_ids: list[int], *, seed: int, positions: tuple[int, int] | None = None,
    span_width: int = 4,
) -> tuple[list[int], dict[str, Any]]:
    """Swap two disjoint spans while protecting the first content token."""
    if len(input_ids) != 32:
        raise ValueError("Input mismatch experiment requires exactly 32 input tokens")
    if span_width < 1 or span_width * 2 > 31:
        raise ValueError("Invalid span width")
    candidates = [
        (i, j)
        for i in range(1, 33 - span_width)
        for j in range(i + span_width, 33 - span_width)
        if input_ids[i : i + span_width] != input_ids[j : j + span_width]
    ]
    if positions is None:
        if not candidates:
            raise ValueError("No distinct non-overlapping input spans available")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        positions = candidates[int(torch.randint(len(candidates), (1,), generator=generator))]
    if positions not in candidates:
        raise ValueError("Selected positions are not distinct valid spans")
    i, j = positions
    changed = list(input_ids)
    changed[i : i + span_width], changed[j : j + span_width] = (
        changed[j : j + span_width], changed[i : i + span_width]
    )
    return changed, {
        "kind": "input",
        "triggered": True,
        "positions": [i, j],
        "span_width": span_width,
        "before": input_ids,
        "after": changed,
    }


def next_legal_token(logits: torch.Tensor, excluded_token_ids: set[int]) -> tuple[int, int]:
    """Return (greedy token, highest-ranked different non-special token)."""
    if logits.ndim != 1:
        raise ValueError("Expected one-dimensional last-position logits")
    # Fixed-length greedy decoding masks EOS with -inf before this function is
    # called. That mask is expected; NaN or +inf are not valid model scores.
    if bool(torch.isnan(logits).any().item()) or bool(torch.isposinf(logits).any().item()):
        raise ValueError("Invalid logits")
    if not bool(torch.isfinite(logits).any().item()):
        raise ValueError("No finite logits")
    greedy = int(torch.argmax(logits).item())
    candidates = logits.clone()
    for index in excluded_token_ids | {greedy}:
        if 0 <= index < candidates.numel():
            candidates[index] = -torch.inf
    if not bool(torch.isfinite(candidates).any().item()):
        raise ValueError("No legal replacement token")
    return greedy, int(torch.argmax(candidates).item())


def _mlp_event(step: int, layer: int, before: torch.Tensor, after: torch.Tensor) -> dict[str, Any]:
    delta = after - before
    return {
        "step": step,
        "layer": layer,
        "before_sha256": tensor_digest(before),
        "after_sha256": tensor_digest(after),
        "before_l2": float(torch.linalg.vector_norm(before).item()),
        "delta_l2": float(torch.linalg.vector_norm(delta).item()),
        "changed_elements": int(torch.count_nonzero(delta).item()),
    }


@contextmanager
def mlp_fault(
    model: Any, spec: InjectionSpec, step: int, actions: list[dict[str, Any]],
) -> Iterator[None]:
    """Install one-step MLP hooks before Recorder sees outer decoder outputs."""
    if spec.kind not in {"computation", "serving"}:
        yield
        return
    if spec.kind == "computation" and step != spec.step:
        yield
        return
    if spec.kind == "serving" and step < spec.step:
        yield
        return

    layers = model.model.layers
    indexes = [spec.layer] if spec.kind == "computation" else list(range(len(layers)))
    handles = []

    def make_hook(index: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor):
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise TypeError("Expected an MLP output shaped [batch, tokens, hidden]")
            before = output[:, -1, :].detach().clone()
            if spec.kind == "serving":
                after = fp32_round_fraction_bits(before, spec.rounding_bits)
            else:
                # A CPU seed makes the direction independent of the generation
                # device and of all other sampling random streams.
                generator = torch.Generator(device="cpu").manual_seed(spec.seed)
                signs = torch.randint(0, 2, before.shape, generator=generator)
                direction = signs.to(device=before.device, dtype=before.dtype).mul_(2).sub_(1)
                direction /= torch.linalg.vector_norm(direction)
                norm = torch.linalg.vector_norm(before)
                if not bool((norm > 0).item()):
                    raise ValueError("Computation injection encountered a zero-norm activation")
                after = before + direction * (spec.epsilon * norm)
            changed = output.clone()
            changed[:, -1, :] = after
            event = _mlp_event(step, index, before, changed[:, -1, :].detach())
            event["kind"] = spec.kind
            if spec.kind == "serving":
                event["rounding_bits"] = spec.rounding_bits
            else:
                event["epsilon"] = spec.epsilon
            actions.append(event)
            return changed
        return hook

    try:
        for index in indexes:
            if index is None or index >= len(layers):
                raise ValueError(f"Decoder layer index out of range: {index}")
            handles.append(layers[index].mlp.register_forward_hook(make_hook(index)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def kv_context_fault(
    model: Any, donor_ids: list[int], spec: InjectionSpec, step: int,
    actions: list[dict[str, Any]], *, allow_identical_donor: bool = False,
) -> Iterator[None]:
    """Replace one historical K/V slot seen by the last query in one layer.

    This is an uncached full-prefix simulation. The original attention computes
    all historical query rows. The attention recompute uses the same full
    sequence shape and backend as that original call; only its final output
    row is substituted. There is no DynamicCache object.
    """
    if spec.kind != "kv_cache" or step != spec.step:
        yield
        return

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, eager_attention_forward

    if spec.layer is None or spec.layer >= len(model.model.layers):
        raise ValueError(f"Decoder layer index out of range: {spec.layer}")
    layer = model.model.layers[spec.layer]
    if not hasattr(layer, 'self_attn'):
        raise ValueError('KV history replacement only supports full-attention layers')
    attention = layer.self_attn
    seq_len = len(donor_ids)
    if seq_len < 3:
        raise ValueError("KV donor must contain at least three positions")
    if spec.donor_position is None:
        generator = torch.Generator(device="cpu").manual_seed(spec.seed)
        requested_position = int(torch.randint(1, seq_len - 1, (1,), generator=generator))
    else:
        requested_position = spec.donor_position
    if not 1 <= requested_position < seq_len - 1:
        raise ValueError("Donor position must be historical and exclude the protected first token")

    donor_capture: dict[str, Any] = {}

    def capture_attention_input(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        donor_capture["hidden"] = (kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]).detach()
        donor_capture["position_embeddings"] = kwargs["position_embeddings"]

    capture_handle = attention.register_forward_pre_hook(capture_attention_input, with_kwargs=True)
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            model(
                input_ids=torch.tensor([donor_ids], device=device),
                use_cache=False,
                logits_to_keep=1,
            )
    finally:
        capture_handle.remove()
    donor_hidden = donor_capture["hidden"]
    donor_cos, donor_sin = donor_capture["position_embeddings"]
    if donor_hidden.shape[1] != seq_len:
        raise RuntimeError("Donor layer capture length mismatch")
    donor_shape = (*donor_hidden.shape[:-1], -1, attention.head_dim)
    donor_key = attention.k_norm(attention.k_proj(donor_hidden).view(donor_shape)).transpose(1, 2)
    donor_value = attention.v_proj(donor_hidden).view(donor_shape).transpose(1, 2)
    _, donor_key = apply_rotary_pos_emb(donor_key, donor_key, donor_cos, donor_sin)

    original_forward = attention.forward

    def injected_forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
        if hidden_states.shape[1] != seq_len:
            raise ValueError("Target and donor prefix lengths differ")
        if kwargs.get("past_key_values") is not None:
            raise ValueError("KV context simulation requires use_cache=False")
        original_output, original_weights = original_forward(hidden_states, *args, **kwargs)
        cos, sin = kwargs["position_embeddings"]
        head_shape = (*hidden_states.shape[:-1], -1, attention.head_dim)
        # Qwen's projection interleaves each query head with its output gate.
        query, gate = torch.chunk(attention.q_proj(hidden_states).view(
            *hidden_states.shape[:-1], -1, attention.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*hidden_states.shape[:-1], -1)
        query = attention.q_norm(query.reshape(head_shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
        value = attention.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if donor_key.shape != key.shape or donor_value.shape != value.shape:
            raise ValueError("Donor K/V shape mismatch")
        donor_position = requested_position
        if not allow_identical_donor:
            # The requested slot may lie in a shared prompt prefix. Select
            # the first genuinely different history slot in a fixed cyclic
            # order, keeping the attack to exactly one K/V position.
            for offset in range(seq_len - 2):
                candidate = 1 + ((requested_position - 1 + offset) % (seq_len - 2))
                if not (
                    torch.equal(key[:, :, candidate, :], donor_key[:, :, candidate, :])
                    and torch.equal(value[:, :, candidate, :], donor_value[:, :, candidate, :])
                ):
                    donor_position = candidate
                    break
            else:
                raise ValueError("No differing historical donor K/V slot exists")
        changed_key = key.clone()
        changed_value = value.clone()
        before_k = changed_key[:, :, donor_position, :].detach().clone()
        before_v = changed_value[:, :, donor_position, :].detach().clone()
        changed_key[:, :, donor_position, :] = donor_key[:, :, donor_position, :]
        changed_value[:, :, donor_position, :] = donor_value[:, :, donor_position, :]
        attention_mask = kwargs.get("attention_mask")
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attention.config._attn_implementation, eager_attention_forward
        )
        interface_kwargs = {
            name: value for name, value in kwargs.items()
            if name not in {"attention_mask", "position_embeddings", "past_key_values"}
        }
        full_output, _ = interface(
            attention,
            query,
            changed_key,
            changed_value,
            attention_mask,
            dropout=0.0 if not attention.training else attention.attention_dropout,
            scaling=attention.scaling,
            **interface_kwargs,
        )
        full_output = full_output.reshape(*hidden_states.shape[:-1], -1).contiguous()
        full_output = attention.o_proj(full_output * torch.sigmoid(gate))
        last_output = full_output[:, -1:, :]
        delta_l2 = float(torch.linalg.vector_norm(last_output - original_output[:, -1:, :]).item())
        if not allow_identical_donor and delta_l2 <= 0:
            raise ValueError('KV replacement did not change the current output row')
        changed = original_output.clone()
        changed[:, -1:, :] = last_output
        actions.append({
            "kind": "kv_cache",
            "step": step,
            "layer": spec.layer,
            "donor_position": donor_position,
            "requested_donor_position": requested_position,
            "actual_donor_position": donor_position,
            "before_k_sha256": tensor_digest(before_k),
            "after_k_sha256": tensor_digest(changed_key[:, :, donor_position, :]),
            "before_v_sha256": tensor_digest(before_v),
            "after_v_sha256": tensor_digest(changed_value[:, :, donor_position, :]),
            "output_delta_l2": delta_l2,
            "historical_output_delta_l2": float(torch.linalg.vector_norm(
                changed[:, :-1, :] - original_output[:, :-1, :]).item()),
            "attention_backend": attention.config._attn_implementation,
            "mechanism": "qwen_full_attention_single_history_slot",
        })
        return changed, original_weights

    attention.forward = injected_forward
    try:
        yield
    finally:
        attention.forward = original_forward

