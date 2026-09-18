import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from prepare_data import load


AEST = timezone(timedelta(hours=10))
REGIONS = {'NSW': 'NSW1', 'QLD': 'QLD1', 'SA': 'SA1', 'TAS': 'TAS1', 'VIC': 'VIC1'}
SLOTS = 8


def timestamp(text):
    return datetime.fromisoformat(str(text)).astimezone(AEST).timestamp()


def prepare(data, snapshots_path, output):
    output.mkdir(parents=True, exist_ok=False)
    snapshots = [json.loads(line) for line in snapshots_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    snapshots.sort(key=lambda row: timestamp(row['available_at_aest']))
    released = np.asarray([timestamp(row['available_at_aest']) for row in snapshots])
    texts, text_index, events = [''], {'': 0}, []
    for snapshot in snapshots:
        assert snapshot['availability_anchor'] == 'csv_directory_published_at_aest'
        parsed = []
        for event in snapshot['events']:
            region = REGIONS.get(event.get('region', '').strip().upper())
            start, finish = event.get('start_at_aest'), event.get('finish_at_aest')
            if not region or not start or not finish or re.search('withdraw|cancel', event.get('status', '') + ' ' + event.get('impact', ''), re.I):
                continue
            text = 'Published equipment and network record. ' + ' '.join(
                f'{label}: {event[field]}.' for label, field in (
                    ('Region', 'region'), ('Asset', 'network_asset'), ('Status', 'status'),
                    ('Recall', 'recall'), ('Cross-region', 'inter_regional'), ('Reported impact', 'impact'))
                if event.get(field))
            if text not in text_index:
                text_index[text] = len(texts)
                texts.append(text)
            parsed.append((region, timestamp(start), timestamp(finish), text_index[text], bool(event.get('inter_regional'))))
        events.append(parsed)
    report = {'input': str(snapshots_path), 'selection': 'latest archived publication not after origin; regional event overlap',
              'clock': 'fixed UTC+10', 'slots': SLOTS, 'ranking': 'decreasing overlap duration, then event start, then stable text id',
              'text_fields': ['region', 'asset', 'status', 'recall', 'cross-region', 'reported impact'],
              'exact_fields': ['snapshot_age_days/30', 'start_lead_hours/168', 'finish_lead_hours/168',
                               'cross_region', 'overlap_fraction', 'snapshot_age_over_14_days'],
              'all_models': 'same event text lexical vectors, numeric fields, validity and snapshot metadata',
              'target_values_used': False, 'splits': {}}
    fit_ids = set()
    for h in (2, 24, 168):
        folder = output / f'h{h}'
        folder.mkdir()
        cadence = 5 if h == 2 else 30
        steps = h * 60 // cadence
        for split in ('train', 'train_quick', 'val', 'calibration'):
            with np.load(data / f'h{h}/{split}.npz') as rows:
                origins, regions = rows['forecast_start'], rows['node_id']
            ids = np.zeros((len(origins), SLOTS), np.int64)
            exact = np.zeros((len(origins), SLOTS, 6), np.float32)
            valid = np.zeros((len(origins), steps, SLOTS), np.float32)
            meta = np.zeros((len(origins), 2), np.float32)
            excluded, covered = 0, 0
            used = set()
            for i, (origin, region) in enumerate(zip(origins.astype('datetime64[s]').astype(str), regions.astype(str))):
                time = datetime.fromisoformat(origin).replace(tzinfo=AEST).timestamp()
                index = int(np.searchsorted(released, time, side='right')) - 1
                if index < 0:
                    continue
                assert released[index] <= time
                age = (time - released[index]) / 86400
                meta[i] = (1, age / 30)
                used.add(snapshots[index]['snapshot_date'])
                selected = [(min(end, time + h * 3600) - max(start, time), start, end, key, cross)
                            for r, start, end, key, cross in events[index]
                            if r == region and start < time + h * 3600 and end > time]
                selected.sort(key=lambda item: (-item[0], item[1], item[3]))
                excluded += max(0, len(selected) - SLOTS)
                covered += int(bool(selected))
                leads = time + np.arange(1, steps + 1) * cadence * 60
                for slot, (overlap, start, end, key, cross) in enumerate(selected[:SLOTS]):
                    ids[i, slot] = key
                    exact[i, slot] = (age / 30, (start - time) / 604800, (end - time) / 604800,
                                       cross, overlap / (h * 3600), age > 14)
                    valid[i, :, slot] = (leads >= start) & (leads < end)
                    if split == 'train_quick':
                        fit_ids.add(key)
            np.savez_compressed(folder / f'{split}.npz', event_ids=ids, event_exact=exact,
                                event_valid=valid, event_meta=meta, forecast_start=origins, node_id=regions)
            report['splits'][f'h{h}_{split}'] = {'n': len(ids), 'with_events': covered,
                'discarded_event_appearances_due_to_slot_limit': excluded,
                'used_snapshot_dates': sorted(used)}
            print('EVENT_INPUT', h, split, len(ids), covered, excluded, flush=True)
    (output / 'texts.json').write_text(json.dumps({'texts': texts, 'fit_ids': sorted(fit_ids)}, ensure_ascii=False))
    report['unique_texts'] = len(texts) - 1
    report['training_projection_texts'] = len(fit_ids)
    (output / 'manifest.json').write_text(json.dumps(report, indent=2))


def encode(root, model_path):
    import torch
    from sklearn.feature_extraction.text import HashingVectorizer
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.30)
    records = json.loads((root / 'texts.json').read_text())
    texts = records['texts']
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenized = tokenizer(texts[1:], truncation=False, add_special_tokens=False)['input_ids']
    shared_texts = [''] + [tokenizer.decode(ids[:512], skip_special_tokens=True) for ids in tokenized]
    fields = HashingVectorizer(n_features=128, ngram_range=(1, 2), alternate_sign=False,
                              norm='l2').transform(shared_texts).toarray().astype(np.float32)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    embeddings = [np.zeros((1, model.config.hidden_size), np.float32)]
    with torch.inference_mode():
        for start in range(1, len(shared_texts), 4):
            token = tokenizer(shared_texts[start:start + 4], return_tensors='pt', padding=True).to('cuda')
            hidden = model(**token, use_cache=False).last_hidden_state
            last = token.attention_mask.sum(-1) - 1
            embeddings.append(hidden[torch.arange(len(last), device='cuda'), last].float().cpu().numpy())
            print('EVENT_ENCODE', min(start + 4, len(shared_texts)), len(shared_texts), flush=True)
    np.savez_compressed(root / 'codebook.npz', embedding=np.concatenate(embeddings), fields=fields,
                        fit_ids=np.asarray(records['fit_ids'], np.int64), texts=np.asarray(shared_texts))
    (root / 'encoding.json').write_text(json.dumps({'model': str(model_path), 'pooling': 'last nonpadding token',
        'max_content_tokens': 512, 'texts_truncated': sum(len(ids) > 512 for ids in tokenized),
        'lexical_and_qwen_texts_identical': True, 'fields': '128-dimensional fixed unigram/bigram hashing; no target fitting'}, indent=2))


def event_tables(path):
    import torch
    from torch.nn.functional import normalize
    rows = load(path)
    vectors = torch.from_numpy(rows['embedding']).cuda()
    fitting = vectors[torch.from_numpy(rows['fit_ids']).cuda()]
    mean = fitting.mean(0)
    _, _, vh = torch.linalg.svd(fitting - mean, full_matrices=False)
    qwen = normalize((vectors - mean) @ vh[:128].T, dim=-1) * np.sqrt(6)
    fields = torch.from_numpy(rows['fields']).cuda() * np.sqrt(6)
    assert qwen.shape == fields.shape and qwen.shape[1] == 128
    qwen[0] = 0
    return qwen, fields


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path)
    p.add_argument('--snapshots', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path)
    a = p.parse_args()
    if a.model:
        encode(a.output, a.model)
    else:
        prepare(a.data, a.snapshots, a.output)
