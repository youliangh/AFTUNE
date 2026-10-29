"""Reuse only preregistered rows, labels and model-independent fault positions."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from .config import CONDITIONS


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def reference_inputs(reference_run):
    root = Path(reference_run).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    corpus_path = root / 'corpus.json'
    if file_hash(corpus_path) != manifest['corpus_sha256']:
        raise ValueError('Reference corpus differs from its frozen manifest')
    corpus = json.loads(corpus_path.read_text())
    if not json.loads((root / 'summary.json').read_text())['complete']:
        raise ValueError('The paired source experiment is incomplete')
    schedule = {}
    source_hashes = {}
    for seed, positive_indices in corpus['positive_indices'].items():
        schedule[seed] = {}
        for condition in CONDITIONS:
            schedule[seed][condition] = {}
            for index in positive_indices:
                if condition == 'model_mismatch':
                    schedule[seed][condition][str(index)] = {}
                    continue
                relative = f'cases/{seed}/{condition}/{index:04d}/oracle/truth.json'
                truth = json.loads((root / relative).read_text())
                if truth['source_request'] != corpus['evaluation'][seed][index]:
                    raise ValueError(f'Reference schedule request mismatch: {relative}')
                if condition == 'input_mismatch':
                    events = truth['generation']['events']
                    if len(events) != 1 or events[0]['span_width'] != 4:
                        raise ValueError('Invalid paired input-swap schedule')
                    entry = {'swap_positions': events[0]['positions']}
                else:
                    entry = {'step': truth['spec']['step']}
                    if not 2 <= entry['step'] <= 96:
                        raise ValueError('Invalid paired injection step')
                    if condition == 'kv_cache':
                        entry['donor_position'] = truth['generation']['events'][0]['requested_donor_position']
                schedule[seed][condition][str(index)] = entry
                source_hashes[relative] = file_hash(root / relative)
    provenance = {'reference_run': str(root),
                  'reference_manifest_sha256': file_hash(root / 'manifest.json'),
                  'reference_corpus_sha256': manifest['corpus_sha256'],
                  'reference_summary_sha256': file_hash(root / 'summary.json'),
                  'reference_schedule_files': source_hashes,
                  'policy': 'same_source_rows_labels_steps_swap_positions_requested_KV_slot'}
    return manifest, corpus, schedule, provenance


def build_paired_corpus(config, rows, dataset):
    from transformers import AutoTokenizer
    previous_manifest, previous, schedule, provenance = reference_inputs(config.reference_run)
    if dataset['sha256'] != previous_manifest['dataset']['sha256']:
        raise ValueError('Paired experiments require the same frozen raw dataset')
    if len(previous['calibration']) != config.calibration_size:
        raise ValueError('Calibration size differs from the paired source experiment')
    tokenizer = AutoTokenizer.from_pretrained(config.declared_model_path, local_files_only=True)
    if tokenizer.bos_token_id is not None:
        raise ValueError('The v4 prompt policy expects the native Qwen tokenizer without BOS')
    seen = set()
    def rebuild(request):
        raw = rows[request['source_row']]
        digest = sha256(raw['input'].encode()).hexdigest()
        if digest != request['source_text_sha256']:
            raise ValueError('Paired raw source text changed')
        ids = tokenizer.encode(raw['input'], add_special_tokens=False)
        if len(ids) < 32 or tuple(ids[:32]) in seen:
            raise ValueError('Paired Qwen input is too short or duplicates another prompt')
        ids = list(map(int, ids[:32]))
        seen.add(tuple(ids))
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return {'source_row': request['source_row'], 'source_file': raw.get('source_file'),
                'source_text_sha256': digest, 'input_ids': ids, 'request_text': text,
                'request_text_sha256': sha256(text.encode()).hexdigest(),
                'reference_request_text_sha256': request['request_text_sha256']}
    corpus = {'calibration': [rebuild(request) for request in previous['calibration']],
              'evaluation': {}, 'positive_indices': {}}
    for seed in config.seeds:
        key = str(seed)
        if len(previous['evaluation'][key]) != config.m:
            raise ValueError('Evaluation size differs from the paired source experiment')
        positive = previous['positive_indices'][key]
        if len(positive) != config.n or len(set(positive)) != config.n:
            raise ValueError('Paired positive labels have another size')
        corpus['positive_indices'][key] = list(positive)
        corpus['evaluation'][key] = [rebuild(request) for request in previous['evaluation'][key]]
        for index in positive:
            ids = corpus['evaluation'][key][index]['input_ids']
            a, b = schedule[key]['input_mismatch'][str(index)]['swap_positions']
            if not (1 <= a < b <= 28 and b - a >= 4 and ids[a:a + 4] != ids[b:b + 4]):
                raise ValueError('Paired swap positions are not valid for the Qwen prompt')
    return corpus, {str(seed): schedule[str(seed)] for seed in config.seeds}, provenance
