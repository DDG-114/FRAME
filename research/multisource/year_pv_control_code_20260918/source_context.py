import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from prepare_data import TASKS, REGIONS, SOURCES, load

LEVELS = ('low', 'middle', 'high')


def vocabulary(feedback=False, weather=False):
    texts, fields, keys = [], [], []

    def add(key, text, source, region=None, month=None, dow=None, levels=()):
        field = np.zeros(64, np.float32)
        field[source] = 1
        if region is not None:
            field[10 + region] = 1
        if month is not None:
            field[15 + month - 1] = 1
            field[27 + dow] = 1
        for n, level in enumerate(levels):
            field[34 + 3 * n + level] = 1
        keys.append(key)
        texts.append(text)
        fields.append(field)

    for i, region in enumerate(REGIONS):
        add(('region', region), f'Region metadata: AEMO NEM region {region}.', 0, region=i)
    for month, dow in itertools.product(range(1, 13), range(7)):
        add(('calendar', month, dow), f'Calendar known before forecasting: month {month}; day of week {dow}, '
            f'where Monday is 0 and Sunday is 6.', 1, month=month, dow=dow)
    for q, task in enumerate(TASKS):
        for levels in itertools.product(range(3), repeat=3):
            words = [LEVELS[x] for x in levels]
            add(('history', q, *levels), f'Observed {task} history available at the forecast origin: '
                f'power level {words[0]}; variation {words[1]}; recent change {words[2]}. '
                'Categories use training-period thresholds; exact observations are numerical inputs.',
                2 + q, levels=levels)
        for levels in itertools.product(range(3), repeat=4):
            words = [LEVELS[x] for x in levels]
            add(('issued', q, *levels), f'Official issued forecast for {task}: valid horizon coverage '
                f'{words[0]}; forecast power level {words[1]}; forecast variation {words[2]}; '
                f'difference from recent observations {words[3]}. '
                'This is a forecast, not a future measurement. Exact valid-lead masks and values are numerical inputs.',
                6 + q, levels=levels)
    if feedback:
        for q, task in enumerate(TASKS):
            for bias, error in itertools.product(range(3), repeat=2):
                add(('feedback', q, bias, error),
                    f'Recent realized performance of the official {task} forecast: '
                    f'signed forecast minus observation bias {LEVELS[bias]}; '
                    f'absolute error {LEVELS[error]}. These categories summarize the previous 28 days '
                    'using only forecasts whose complete target windows have already been observed. '
                    'They describe recent forecast reliability, not future realized errors.',
                    46 + q, levels=(bias, error))
    if weather:
        from weather_context import vocabulary as weather_vocabulary
        weather_texts, weather_fields, weather_keys = weather_vocabulary()
        texts.extend(weather_texts)
        fields.extend(weather_fields)
        keys.extend(weather_keys)
    return texts, np.stack(fields), {tuple(k): i for i, k in enumerate(keys)}


def summaries(rows, history):
    past = rows['x'][:, :history, :4]
    future = rows['x'][:, history:]
    period = 288 if history == 2016 else 48
    recent = past[:, -period:]
    trend = recent.mean(1) - past[:, -2 * period:-period].mean(1)
    hist = np.stack([recent.mean(1), recent.std(1), trend], -1)
    mask = future[..., 12:16]
    values = future[..., 8:12]
    count = mask.sum(1).clip(1)
    mean = (mask * values).sum(1) / count
    std = np.sqrt((mask * (values - mean[:, None]) ** 2).sum(1) / count)
    issued = np.stack([mask.mean(1), mean, std, mean - recent.mean(1)], -1)
    return hist.astype(np.float32), issued.astype(np.float32)


def threshold_fit(rows, history):
    hist, issued = summaries(rows, history)
    result = {'history': np.quantile(hist, [1/3, 2/3], axis=0).tolist(),
              'issued': np.quantile(issued, [1/3, 2/3], axis=0).tolist()}
    if 'feedback' in rows:
        result['feedback'] = np.stack([np.quantile(rows['feedback'][rows['feedback_available'][:, q] > 0, q],
                                                  [1/3, 2/3], axis=0) for q in range(4)], axis=1).tolist()
    return result


def build(rows, history, thresholds):
    use_feedback = 'feedback' in rows
    _, _, lookup = vocabulary(use_feedback)
    hist, issued = summaries(rows, history)
    hcat = (hist[None] > np.asarray(thresholds['history'])[:, None]).sum(0)
    icat = (issued[None] > np.asarray(thresholds['issued'])[:, None]).sum(0)
    n, horizon = len(hist), rows['y'].shape[1]
    source_count = len(SOURCES) + (4 if use_feedback else 0)
    ids = np.zeros((n, source_count), np.int64)
    exact = np.zeros((n, source_count, 6), np.float32)
    exact[:, 2:6, :3], exact[:, 6:10, :4] = hist, issued
    origins = rows['forecast_start'].astype('datetime64[s]')
    months = origins.astype('datetime64[M]').astype(int) % 12 + 1
    dow = (origins.astype('datetime64[D]').astype(int) + 3) % 7
    regions = np.asarray([REGIONS.index(str(x)) for x in rows['node_id']], np.int64)
    if use_feedback:
        fcat = (rows['feedback'][None] > np.asarray(thresholds['feedback'])[:, None]).sum(0)
        exact[:, 10:14, :2] = rows['feedback']
    for i in range(n):
        ids[i, 0] = lookup[('region', REGIONS[regions[i]])]
        ids[i, 1] = lookup[('calendar', int(months[i]), int(dow[i]))]
        for q in range(4):
            ids[i, 2 + q] = lookup[('history', q, *hcat[i, q])]
            ids[i, 6 + q] = lookup[('issued', q, *icat[i, q])]
            if use_feedback:
                ids[i, 10 + q] = lookup[('feedback', q, *fcat[i, q])]
    valid = np.ones((n, horizon, source_count), np.float32)
    valid[:, :, 6:10] = rows['x'][:, history:, 12:16]
    if use_feedback:
        valid[:, :, 10:14] = rows['feedback_available'][:, None]
    return {'source_ids': ids, 'source_exact': exact, 'source_valid': valid, 'region': regions}


def encode(model_path, output, batch, feedback=False, weather=False, base_codebook=None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.25)
    texts, fields, _ = vocabulary(feedback, weather)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                     attn_implementation='sdpa').cuda().eval()
    embeddings, first = [], 0
    if base_codebook:
        base = load(base_codebook)
        first = len(base['texts'])
        np.testing.assert_array_equal(base['texts'], np.asarray(texts[:first]))
        np.testing.assert_array_equal(base['fields'], fields[:first])
        embeddings.append(base['embedding'])
    with torch.inference_mode():
        for start in range(first, len(texts), batch):
            group = texts[start:start + batch]
            token = tokenizer(group, return_tensors='pt', padding=True, truncation=True, max_length=160).to('cuda')
            hidden = model(**token, use_cache=False).last_hidden_state
            last = token.attention_mask.sum(-1) - 1
            pooled = hidden[torch.arange(len(group), device='cuda'), last].float()
            embeddings.append(pooled.cpu().numpy())
            print('ENCODE', start + len(group), len(texts), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, texts=np.asarray(texts), fields=fields, embedding=np.concatenate(embeddings))
    print('ENCODE_COMPLETE', output, torch.cuda.get_device_name(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--feedback', action='store_true')
    parser.add_argument('--weather', action='store_true')
    parser.add_argument('--base-codebook', type=Path)
    a = parser.parse_args()
    encode(a.model, a.output, a.batch, a.feedback, a.weather, a.base_codebook)
