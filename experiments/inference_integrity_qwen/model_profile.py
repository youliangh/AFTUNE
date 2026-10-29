"""Text-only model structure and exact decoding/backend identities."""
from hashlib import sha256
import json
from pathlib import Path


def text_config(config):
    return getattr(config, 'text_config', config)


def qwen_backend():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    names = ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
             'fused_recurrent_gated_delta_rule', 'FusedRMSNormGated')
    available = {name: getattr(qwen, name, None) is not None for name in names}
    if qwen.is_fast_path_available or any(available.values()):
        raise RuntimeError('The frozen v4 linear attention requires the pure Torch fallback')
    return {'full_attention': 'sdpa', 'linear_attention': 'torch_fallback',
            'fast_path_available': False, 'optional_kernels': available}


def decoding_policy(tokenizer):
    ids = sorted(set(int(value) for value in tokenizer.get_vocab().values()))
    payload = json.dumps(ids, separators=(',', ':')).encode()
    return {'valid_vocab_count': len(ids), 'valid_vocab_ids_sha256': sha256(payload).hexdigest(),
            'tokenizer_eos_token_id': tokenizer.eos_token_id,
            'tokenizer_pad_token_id': tokenizer.pad_token_id,
            'tokenizer_bos_token_id': tokenizer.bos_token_id,
            'ordinary_greedy_mask': 'unassigned_ids_and_tokenizer_eos_only',
            'replacement_mask': 'ordinary_mask_plus_greedy_and_all_special_ids',
            'all_special_ids': sorted(set(tokenizer.all_special_ids))}


def build_profile(config, tokenizer, model_name):
    cfg = text_config(config)
    count = int(cfg.num_hidden_layers)
    types = list(getattr(cfg, 'layer_types', ['full_attention'] * count))
    if len(types) != count or any(t not in ('linear_attention', 'full_attention') for t in types):
        raise ValueError('Unsupported decoder attention layout')
    tracked = ['embedding'] + [f'layer_{index}' for index in range(count)] + ['norm']
    blocks = {str(index // 2): tracked[index:index + 2] for index in range(0, len(tracked), 2)}
    return {'model_name': model_name, 'model_type': cfg.model_type,
            'decoder_count': count, 'layer_types': types,
            'kv_eligible_layers': [i for i, kind in enumerate(types) if kind == 'full_attention'],
            'tracked_layers': tracked, 'block_to_layers': blocks, 'block_count': len(blocks),
            'base_output_blocks': list(range(0, len(blocks), 4)),
            'revealed_output_blocks': [i for i in range(len(blocks)) if i % 4],
            'config_eos_token_id': cfg.eos_token_id, 'head_vocab_size': cfg.vocab_size,
            'decoding': decoding_policy(tokenizer)}


def checkpoint_profile(path):
    from transformers import AutoConfig, AutoTokenizer
    path = Path(path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    profile = build_profile(cfg, tokenizer, path.name)
    if profile['model_type'] != 'qwen3_5_text' or profile['decoder_count'] != 24:
        raise ValueError('The declared checkpoint must be the pinned 24-layer Qwen3.5 text model')
    profile['backends'] = qwen_backend()
    return profile
