"""TRAIN-only usual facts and metadata-preserving wrong-fact interventions."""
from collections import Counter, defaultdict
import hashlib
import json
import re
import numpy as np
from cache_tools import HERE, read, save, sha, token_rows, input_text, row_positions

PATTERN = re.compile(r'(observed_operation:)(level=(?:low|middle|high|unavailable); trend=(?:low|middle|high|unavailable); weather=(?:low|middle|high|unavailable))(?=; available_groups=)')
FIELDS = ('level', 'trend', 'weather')
BANDS = ('low', 'middle', 'high', 'unavailable')


def separate(text):
    match = PATTERN.search(text)
    assert match, text
    values = tuple(x.split('=', 1)[1] for x in match.group(2).split('; '))
    return text[:match.start(2)], values, text[match.end(2):]


def replace(text, facts):
    prefix, _, suffix = separate(text)
    assert all(x in BANDS for x in facts)
    return prefix + '; '.join(k+'='+v for k, v in zip(FIELDS, facts)) + suffix


def calendar_keys(task, asset, origin):
    if task == 'wind':
        minute = int(origin.astype('datetime64[m]').astype(np.int64)) % 1440
        return [(task, asset, 'phase', str(minute)), (task, asset, 'asset')]
    day = int(origin.astype('datetime64[D]').astype(np.int64))
    month = str(origin.astype('datetime64[M]'))[-2:]
    weekday = str((day+3) % 7)
    return [(task, asset, 'month_weekday', month, weekday),
            (task, asset, 'month', month), (task, asset, 'weekday', weekday),
            (task, asset, 'asset')]


def prepare(data, corpus):
    ids, mapping = row_positions()
    assert ids == [x['id'] for x in corpus]
    buckets = defaultdict(lambda: [Counter() for _ in FIELDS])
    origins = defaultdict(set)
    row_keys, facts = {}, {}
    for task, positions in mapping.items():
        a = data.arrays[task]
        facts[task] = [separate(corpus[p]['text'])[1] for p in positions]
        row_keys[task] = [calendar_keys(task, str(a['asset_id'][i]), a['origin_time'][i]) for i in range(len(positions))]
        for i in data.splits[task]['train']:
            for key in row_keys[task][i]:
                origins[key].add(str(a['origin_time'][i]))
                for j, value in enumerate(facts[task][i]): buckets[key][j][value] += 1
    usual, shifted = [dict(x) for x in corpus], [dict(x) for x in corpus]
    provenance, donors, levels = {}, {}, Counter()
    for task, positions in mapping.items():
        a = data.arrays[task]
        selected = []
        for i, p in enumerate(positions):
            keys = row_keys[task][i]
            key = next(k for k in keys if len(origins[k]) >= 16 or k[-1] == 'asset')
            value = []
            for j, current in enumerate(facts[task][i]):
                # A missing observation remains missing, even if the TRAIN bin is populated.
                counts = buckets[key][j]
                available = [v for v in BANDS[:-1] if counts[v]]
                value.append('unavailable' if current == 'unavailable' or not available else min(available, key=lambda v: (-counts[v], BANDS.index(v))))
            usual[p]['text'] = replace(corpus[p]['text'], value)
            levels[key[2]] += 1
            selected.append(dict(key=list(key), unique_TRAIN_origins=len(origins[key])))
        provenance[task] = selected
        donor = np.arange(len(positions))
        for split in ('train', 'val'):
            groups = defaultdict(list)
            for i in data.splits[task][split]:
                missing = tuple(v == 'unavailable' for v in facts[task][i])
                groups[(str(a['asset_id'][i]), missing)].append(int(i))
            for group in groups.values():
                group = sorted(group, key=lambda i: (str(a['origin_time'][i]), str(a['sample_id'][i])))
                for j, i in enumerate(group): donor[i] = group[(j+len(group)//2) % len(group)]
        donors[task] = donor.tolist()
        for i, p in enumerate(positions):
            shifted[p]['text'] = replace(corpus[p]['text'], facts[task][donor[i]])
    for original, u, w in zip(corpus, usual, shifted):
        p, _, s = separate(original['text'])
        assert separate(u['text'])[::2] == (p, s) == separate(w['text'])[::2]
        assert u['id'] == w['id'] == original['id']
    fit = dict(status='passed', minimum_origins=16, tie_order=list(BANDS),
               buckets={json.dumps(k): dict(origins=sorted(origins[k]), counts=[dict(c) for c in v]) for k, v in buckets.items()},
               recipient_bins=provenance, backoff_counts=dict(levels),
               fitted_rows={t:list(map(int, data.splits[t]['train'])) for t in mapping},
               shifted_donor_rows=donors, fitted_from='TRAIN input categories only; no future targets')
    return usual, shifted, fit


def schema_features(tokenizer, corpus):
    vectors = np.zeros((len(corpus), 4096), np.float32)
    digest = hashlib.sha256()
    for start in range(0, len(corpus), 256):
        rows = token_rows(tokenizer, [input_text(x['text']) for x in corpus[start:start+256]])
        for i, tokens in enumerate(rows, start):
            assert len(tokens) <= 512
            digest.update(np.asarray([len(tokens), *tokens], dtype='<i8').tobytes())
            count = np.bincount(np.asarray(tokens) % 4096, minlength=4096).astype(np.float32)
            vectors[i] = count/np.linalg.norm(count)
    return vectors, digest.hexdigest()


def main():
    from data import Dataset
    data = Dataset(context_stats=read(HERE/'INPUT_CONTRACT.json')['context_stats'])
    corpus = read(HERE/'TEXT_CORPUS.json')
    usual, shifted, fit = prepare(data, corpus)
    save(HERE/'TEXT_USUAL.json', usual)
    save(HERE/'TEXT_SHIFTED.json', shifted)
    save(HERE/'TRAIN_USUAL_FIT.json', fit)
    save(HERE/'FACT_PREPARATION.json', dict(status='passed', windows=len(corpus),
        original_corpus_sha256=sha(HERE/'TEXT_CORPUS.json'), usual_sha256=sha(HERE/'TEXT_USUAL.json'),
        shifted_sha256=sha(HERE/'TEXT_SHIFTED.json'), fit_sha256=sha(HERE/'TRAIN_USUAL_FIT.json'),
        metadata_calendar_availability_age_byte_exact=True, no_new_information_source=True,
        changed_usual=sum(x['text']!=y['text'] for x,y in zip(corpus, usual)),
        changed_shifted=sum(x['text']!=y['text'] for x,y in zip(corpus, shifted)),
        optimizer_updates=0))


if __name__ == '__main__': main()
