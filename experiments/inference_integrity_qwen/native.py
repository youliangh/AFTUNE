"""AFTUNE's existing one-forward recorder and verifier as experiment primitives.

Each call to ``record_forward`` records the *actual* forward that produces its
logits. The caller owns autoregressive generation and any fault injection.
This module does not add token, input-identity, cache, or route checks.
"""

from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import json
import time
from typing import Callable, Iterable

import torch

from recorder import Recorder
from verify_inference import VerifiedReference, verify_inference


ForwardHook = tuple[torch.nn.Module, Callable]


def _tracked_layers(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """Match the boundaries that inference.py actually records.

    ``lm_head`` is intentionally absent: inference.py never installs its hook
    and verify_inference.py does not compare its output.
    """
    backbone = model.model
    result = {'embedding': backbone.embed_tokens}
    result.update({f'layer_{i}': layer for i, layer in enumerate(backbone.layers)})
    if hasattr(backbone, 'norm'):
        result['norm'] = backbone.norm
    return result


def prepare_reference(model: torch.nn.Module, reference_dir: str | Path,
                      model_name: str, *, layers_per_block: int = 1,
                      activation_interval: int = 1) -> None:
    """Export a declared-model snapshot in the layout expected by AFTUNE.

    A populated destination is never overwritten. The experiment controller
    should associate this directory with the declared checkpoint hash.
    """
    destination = Path(reference_dir).resolve()
    if destination.exists():
        raise FileExistsError(f'Reference destination already exists: {destination}')
    if not model_name:
        raise ValueError('A non-empty declared model name is required')
    destination.mkdir(parents=True)
    tracked = _tracked_layers(model)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError('Trusted Qwen reference requires FP32 parameters')
    rec = Recorder(save_dir=str(destination), module_structure_dir=str(destination),
                   model_name=model_name, steps_per_block=1, layers_per_block=layers_per_block,
                   activation_interval=activation_interval, checkpoint_interval=1, async_save=False,
                   compress_storage=False)
    rec.layer_order = list(tracked)
    rec.layer_order_finalized = True
    rec.finalize_layer_blocks()
    for name, module in tracked.items():
        rec.save_module_structure(name, module)
    if hasattr(model.model, 'rotary_emb'):
        rec.save_module_structure('rotary_emb', model.model.rotary_emb)
    rec.save_final_model_parameters(model, tracked)
    rec.save_all_records()


def record_forward(model: torch.nn.Module, input_ids: torch.Tensor,
                   record_dir: str | Path, reference_dir: str | Path,
                   model_name: str, extra_hooks: Iterable[ForwardHook] = ()) -> torch.Tensor:
    """Run one exact-token full-prefix forward and persist its native step 0.

    Extra hooks are installed before the forward. Hooks on an MLP submodule
    therefore change its parent decoder output before Recorder captures it.
    The record path must be unique; this prevents silent overwrite of evidence.
    Returns only the last-position logits, with shape ``[1, vocabulary_size]``.
    """
    destination = Path(record_dir).resolve()
    reference = Path(reference_dir).resolve()
    metadata = Recorder(save_dir=str(reference), enabled=False).load_metadata()
    if not metadata or metadata.get('model_name') != model_name:
        raise ValueError('Trusted reference missing or model name differs')
    if destination.exists():
        raise FileExistsError(f'Record destination already exists: {destination}')
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError('input_ids must have shape [1, positive_length]')
    if input_ids.dtype != torch.long:
        raise TypeError('input_ids must be torch.long token IDs')
    if list(_tracked_layers(model)) != metadata['layer_order']:
        raise ValueError('Live model boundaries differ from declared reference')
    destination.mkdir(parents=True)
    rec = Recorder(save_dir=str(destination), module_structure_dir=str(reference),
                   model_name=model_name, steps_per_block=1,
                   layers_per_block=metadata['layers_per_block'],
                   activation_interval=metadata['activation_interval'], checkpoint_interval=1,
                   chunk_size=metadata['chunk_size'], use_blake3=metadata['use_blake3'],
                   async_save=False, compress_storage=False)
    tracked = _tracked_layers(model)
    rec.layer_order = list(tracked)
    rec.layer_order_finalized = True
    rec.finalize_layer_blocks()
    handles = []
    responses = {}
    def retain_response(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            responses[name] = tensor.detach().cpu().clone()
        return hook
    try:
        for module, hook in extra_hooks:
            handles.append(module.register_forward_hook(hook))
        for name, module in tracked.items():
            def require_fp32(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if tensor.dtype != torch.float32:
                    raise TypeError('Recorded Qwen activations must be FP32')
            handles.append(module.register_forward_hook(require_fp32))
            handles.append(module.register_forward_hook(rec.create_forward_hook(name)))
            block = rec.layer_to_block[name]
            if rec.is_block_last_layer(name) and block % rec.activation_interval:
                handles.append(module.register_forward_hook(retain_response(name)))
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        rec.dataset_records[0] = {
            'input_ids': input_ids.detach().cpu(),
            'label': int(torch.argmax(logits, dim=-1).item()),
        }
        rec.save_all_records()
        if responses:
            _prepare_activation_responses(rec, destination, responses)
        return logits
    finally:
        for handle in reversed(handles):
            handle.remove()


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_activation_responses(rec: Recorder, destination: Path,
                                  responses: dict[str, torch.Tensor]) -> None:
    """Reveal original live tensors after freezing the sparse base record.

    Values are retained transiently from the real forward, including any live
    injection. They are never replaced by a clean replay. All boundary hashes
    were already saved by Recorder; preparing responses cannot change them.
    This full-audit experiment reveals every missing boundary, and records the
    resulting storage overhead separately from the initial sparse evidence.
    """
    started = time.monotonic()
    base = {str(p.relative_to(destination)): _file_hash(p)
            for p in sorted(destination.rglob('*')) if p.is_file()}
    marker = destination / 'activation_preparation.json'
    payload = {'method': 'original_live_tensor_response', 'base_files': base,
               'responses': [], 'complete': False}
    marker.write_text(json.dumps(payload, indent=2) + '\n')
    for name, tensor in responses.items():
        expected = rec.load_layer_hashes(name, 0)['output_hash']
        if rec.get_hash(tensor, 'hash_output') != expected:
            raise ValueError(f'{name}: activation response differs from original commitment')
        block = rec.layer_to_block[name]
        output = destination / f'block_{block}_0' / name / 'step_0' / 'output.pt'
        if output.exists():
            raise FileExistsError(f'Activation response would overwrite evidence: {output}')
        rec.save_tensor_compressed(tensor, str(output))
        payload['responses'].append({'layer': name, 'block_id': block,
                                     'path': str(output.relative_to(destination)),
                                     'output_hash': expected,
                                     'sha256': _file_hash(output),
                                     'bytes': output.stat().st_size})
    if any(_file_hash(destination / name) != digest for name, digest in base.items()):
        raise ValueError('Sparse base evidence changed during activation preparation')
    payload.update(complete=True, elapsed_seconds=time.monotonic() - started,
                   base_bytes=sum((destination / name).stat().st_size for name in base),
                   response_bytes=sum(row['bytes'] for row in payload['responses']))
    marker.write_text(json.dumps(payload, indent=2) + '\n')


def verify_record(record_dir: str | Path, reference_dir: str | Path,
                  device: str = 'cpu', strict_hash: bool = False,
                  verified_reference: VerifiedReference | None = None,
                  report_strict_mismatch: bool = False) -> list[dict]:
    """Verify every actually recorded layer block and retain raw native scores."""
    destination = str(Path(record_dir).resolve())
    reference = str(Path(reference_dir).resolve())
    metadata = Recorder(save_dir=destination, enabled=False).load_metadata()
    if not metadata:
        return [{'block_id': None, 'layer_name': None,
                 'error': f'Missing inference metadata: {destination}/metadata.json'}]
    reference_metadata = Recorder(save_dir=reference, enabled=False).load_metadata()
    if not reference_metadata:
        return [{'block_id': None, 'layer_name': None,
                 'error': f'Missing trusted reference metadata: {reference}/metadata.json'}]
    if (metadata.get('layer_order') != reference_metadata.get('layer_order') or
            metadata.get('block_to_layers') != reference_metadata.get('block_to_layers')):
        return [{'block_id': None, 'layer_name': None,
                 'error': 'Inference record does not cover the declared layer blocks'}]
    if metadata.get('activation_interval', 1) > 1:
        preparation = Path(destination) / 'activation_preparation.json'
        try:
            prepared = json.loads(preparation.read_text())
            if not prepared.get('complete') or not prepared.get('base_files'):
                raise ValueError('Incomplete sparse activation preparation')
            for name, digest in prepared['base_files'].items():
                if _file_hash(Path(destination) / name) != digest:
                    return [{'block_id': None, 'layer_name': None,
                             'error': 'Disk data hash mismatch in sparse base evidence'}]
        except (OSError, ValueError, KeyError) as exc:
            return [{'block_id': None, 'layer_name': None, 'error': str(exc)}]
    results = []
    blocks = {int(k): v for k, v in metadata['block_to_layers'].items()}
    for block_id in sorted(blocks):
        try:
            native = verify_inference(block_id, destination, reference, device,
                                      strict_hash, '', True, False,
                                      verified_reference=verified_reference,
                                      report_strict_mismatch=report_strict_mismatch)
        except Exception as exc:
            native = {'error': str(exc), 'exception_type': type(exc).__name__}
        results.append({'block_id': block_id, 'layer_name': blocks[block_id][-1], **native})
    return results


def verify_record_both_modes(record_dir: str | Path, reference_dir: str | Path,
                             device: str = 'cpu',
                             verified_reference: VerifiedReference | None = None) -> dict[str, list[dict]]:
    """Project numeric and native strict-hash results from one replay per block.

    The underlying verifier already computes the output hash in numerical
    mode. A reported mismatch has the exact message that a strict replay
    would return. Disk and trusted-reference errors retain their native
    precedence because they arise before the computed-output comparison.
    """
    numeric = verify_record(record_dir, reference_dir, device=device,
                            strict_hash=False, verified_reference=verified_reference,
                            report_strict_mismatch=True)
    strict = []
    for block in numeric:
        mismatch = block.get('computed_output_hash_mismatch')
        projected = {k: v for k, v in block.items()
                     if k != 'computed_output_hash_mismatch'}
        if mismatch is not None and 'error' not in projected:
            projected = {'block_id': block['block_id'],
                         'layer_name': block['layer_name'], 'error': mismatch}
        strict.append(projected)
    return {'numeric': numeric, 'strict_hash': strict}


__all__ = ['VerifiedReference', 'prepare_reference', 'record_forward',
           'verify_record', 'verify_record_both_modes']
