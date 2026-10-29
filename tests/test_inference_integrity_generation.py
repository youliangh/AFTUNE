import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from experiments.inference_integrity.generation import (
    generate_llama,
    llama_retokenize_response,
)
from experiments.inference_integrity.injections import (
    InjectionSpec,
    fp32_round_fraction_bits,
    kv_context_fault,
    next_legal_token,
    swap_input_spans,
    tensor_digest,
)


class TinyTokenizer:
    eos_token_id = 0
    all_special_ids = [0, 1]

    def decode(self, tokens, **_kwargs):
        return " ".join(map(str, tokens))

    def encode(self, text, **_kwargs):
        return [2 + (ord(ch) % 100) for ch in text]


class InjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
            )
        ).float().eval()
        cls.tokenizer = TinyTokenizer()
        cls.input_ids = [1] + list(range(2, 33))

    def test_fp32_rounding_ties_even_and_sign(self):
        ulp = 2**-23
        values = torch.tensor([1.0 + 2 * ulp, 1.0 + 6 * ulp, -1.0 - 2 * ulp])
        rounded = fp32_round_fraction_bits(values, 2)
        torch.testing.assert_close(
            rounded, torch.tensor([1.0, 1.0 + 8 * ulp, -1.0]), rtol=0, atol=0
        )
        self.assertEqual(rounded.dtype, torch.float32)

    def test_input_swap_preserves_bos_and_length(self):
        changed, event = swap_input_spans(self.input_ids, seed=10, positions=(4, 20))
        self.assertEqual(len(changed), 32)
        self.assertEqual(changed[0], self.input_ids[0])
        self.assertEqual(changed[4:8], self.input_ids[20:24])
        self.assertEqual(changed[20:24], self.input_ids[4:8])
        self.assertEqual(event["positions"], [4, 20])

    def test_token_replacement_and_text_bridge(self):
        original, replacement = next_legal_token(torch.tensor([0.0, 5.0, 4.0, 3.0]), {0, 2})
        self.assertEqual((original, replacement), (1, 3))
        # Fixed-length generation has already masked EOS to -inf.
        original, replacement = next_legal_token(torch.tensor([-torch.inf, 5.0, 4.0, 3.0]), {0, 2})
        self.assertEqual((original, replacement), (1, 3))
        result = generate_llama(
            self.model, self.tokenizer, self.input_ids, steps=2,
            spec=InjectionSpec("token_selection", step=2),
        )
        self.assertEqual(len(result.events), 1)
        self.assertNotEqual(result.events[0]["normal_token"], result.events[0]["replacement_token"])
        bridged = llama_retokenize_response(self.input_ids, "ab", self.tokenizer)
        self.assertEqual(bridged[:32], self.input_ids)
        self.assertEqual(bridged[32:], self.tokenizer.encode("ab"))

    def test_computation_and_serving_use_actual_mlp_outputs(self):
        computation = generate_llama(
            self.model, self.tokenizer, self.input_ids, steps=2,
            spec=InjectionSpec("computation", step=2, layer=0, seed=5),
        )
        self.assertEqual(len(computation.events), 1)
        self.assertEqual(computation.events[0]["step"], 2)
        self.assertGreater(computation.events[0]["delta_l2"], 0)
        serving = generate_llama(
            self.model, self.tokenizer, self.input_ids, steps=3,
            spec=InjectionSpec("serving", step=2, rounding_bits=6),
        )
        self.assertEqual(
            len(serving.events), 1 + 2 * len(self.model.model.layers)
        )
        self.assertEqual(serving.events[0]["event"], "route_switch")
        self.assertTrue(all(
            event["changed_elements"] > 0 for event in serving.events[1:]
        ))

    def test_kv_simulation_changes_only_final_attention_query(self):
        target = self.input_ids + [35]
        donor = self.input_ids.copy() + [35]
        donor[5] = 70
        attention = self.model.model.layers[0].self_attn
        observed = []
        hook = attention.register_forward_hook(
            lambda _module, _inputs, output: observed.append(output[0].detach().clone())
        )
        try:
            with torch.no_grad():
                self.model(input_ids=torch.tensor([target]), use_cache=False)
            original = observed.pop()
            actions = []
            with kv_context_fault(
                self.model, donor,
                InjectionSpec("kv_cache", step=2, layer=0, donor_position=5),
                step=2, actions=actions,
            ):
                with torch.no_grad():
                    self.model(input_ids=torch.tensor([target]), use_cache=False)
            changed = observed.pop()
            observed.clear()  # The donor forward also passed through this hook.
            torch.testing.assert_close(original[:, :-1], changed[:, :-1], rtol=0, atol=0)
            self.assertGreater(torch.linalg.vector_norm(original[:, -1] - changed[:, -1]).item(), 0)
            self.assertEqual(len(actions), 1)
            with torch.no_grad():
                self.model(input_ids=torch.tensor([target]), use_cache=False)
            torch.testing.assert_close(original, observed.pop(), rtol=0, atol=0)
        finally:
            hook.remove()

    def test_kv_same_shape_sham_is_bitwise_clean(self):
        target = self.input_ids + [35]
        attention = self.model.model.layers[0].self_attn
        observed = []
        hook = attention.register_forward_hook(
            lambda _module, _inputs, output: observed.append(output[0].detach().clone())
        )
        try:
            with torch.no_grad():
                self.model(input_ids=torch.tensor([target]), use_cache=False)
            original = observed.pop()
            actions = []
            with kv_context_fault(
                self.model, target,
                InjectionSpec("kv_cache", step=2, layer=0, donor_position=5),
                step=2, actions=actions, allow_identical_donor=True,
            ):
                with torch.no_grad():
                    self.model(input_ids=torch.tensor([target]), use_cache=False)
            changed = observed.pop()
            torch.testing.assert_close(original, changed, rtol=0, atol=0)
            self.assertEqual(actions[0]["before_k_sha256"], actions[0]["after_k_sha256"])
            self.assertEqual(actions[0]["before_v_sha256"], actions[0]["after_v_sha256"])
            self.assertEqual(actions[0]["requested_donor_position"], 5)
            self.assertEqual(actions[0]["actual_donor_position"], 5)
            self.assertEqual(actions[0]["output_delta_l2"], 0.0)
        finally:
            hook.remove()

    def test_kv_shared_prefix_uses_first_different_history_slot(self):
        target = self.input_ids + [35]
        donor = target.copy()
        donor[8] = 70  # Positions 1..7, including requested slot 5, match.
        actions = []
        with kv_context_fault(
            self.model, donor,
            InjectionSpec("kv_cache", step=2, layer=0, donor_position=5),
            step=2, actions=actions,
        ):
            with torch.no_grad():
                self.model(input_ids=torch.tensor([target]), use_cache=False)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["requested_donor_position"], 5)
        self.assertEqual(actions[0]["actual_donor_position"], 8)
        self.assertEqual(actions[0]["donor_position"], 8)
        self.assertNotEqual(actions[0]["before_k_sha256"], actions[0]["after_k_sha256"])

    def test_kv_identical_donor_fails_and_restores_attention(self):
        target = self.input_ids + [35]
        attention = self.model.model.layers[0].self_attn
        original_forward = attention.forward
        with self.assertRaisesRegex(ValueError, "No differing historical donor K/V"):
            with kv_context_fault(
                self.model, target,
                InjectionSpec("kv_cache", step=2, layer=0, donor_position=5),
                step=2, actions=[],
            ):
                with torch.no_grad():
                    self.model(input_ids=torch.tensor([target]), use_cache=False)
        self.assertIs(attention.forward, original_forward)

    def test_kv_donor_is_actual_clean_attention_kv(self):
        target = self.input_ids + [35]
        donor = self.input_ids.copy() + [35]
        donor[5] = 70
        layer = self.model.model.layers[0]
        attention = layer.self_attn
        raw_inputs = []
        attention_inputs = []

        raw_hook = layer.register_forward_pre_hook(
            lambda _module, args, _kwargs: raw_inputs.append(args[0].detach().clone()),
            with_kwargs=True,
        )
        attention_hook = attention.register_forward_pre_hook(
            lambda _module, _args, kwargs: attention_inputs.append(
                (
                    kwargs["hidden_states"].detach().clone(),
                    tuple(t.detach().clone() for t in kwargs["position_embeddings"]),
                )
            ),
            with_kwargs=True,
        )
        try:
            actions = []
            with kv_context_fault(
                self.model, donor,
                InjectionSpec("kv_cache", step=2, layer=0, donor_position=5),
                step=2, actions=actions,
            ):
                with torch.no_grad():
                    self.model(input_ids=torch.tensor([target]), use_cache=False)

            self.assertEqual(len(actions), 1)
            # The first pass inside kv_context_fault computes clean donor
            # activations. The actual attention input is the normalized layer
            # input; projecting the raw residual would fail these hash checks.
            donor_hidden, (cos, sin) = attention_inputs[0]
            torch.testing.assert_close(
                donor_hidden, layer.input_layernorm(raw_inputs[0]), rtol=0, atol=0
            )
            shape = (*donor_hidden.shape[:-1], -1, attention.head_dim)
            projected_key = attention.k_proj(donor_hidden).view(shape).transpose(1, 2)
            projected_value = attention.v_proj(donor_hidden).view(shape).transpose(1, 2)
            _, projected_key = apply_rotary_pos_emb(
                projected_key, projected_key, cos, sin
            )
            self.assertEqual(
                actions[0]["after_k_sha256"], tensor_digest(projected_key[:, :, 5, :])
            )
            self.assertEqual(
                actions[0]["after_v_sha256"], tensor_digest(projected_value[:, :, 5, :])
            )
        finally:
            attention_hook.remove()
            raw_hook.remove()


if __name__ == "__main__":
    unittest.main()
