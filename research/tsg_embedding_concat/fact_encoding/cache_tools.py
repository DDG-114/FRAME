"""Matched frozen readout and TRAIN-only static representations."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def input_text(text):
    return 'Instruct: ' + read(HERE / 'PLAN.json')['corpus']['instruction'] + '\nQuery:' + text


def token_rows(tokenizer, texts):
    # The shared token sequence ends with one EOS.
    rows = tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)['input_ids']
    terminal = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    assert isinstance(terminal, int) and terminal != tokenizer.unk_token_id
    return [list(row) + [terminal] for row in rows]


def pool_last_l2(hidden, mask):
    import torch
    positions = torch.arange(mask.shape[1], device=mask.device)[None].expand_as(mask)
    last = positions.masked_fill(~mask.bool(), -1).max(1).values
    assert (last >= 0).all()
    selected = hidden[torch.arange(len(hidden), device=hidden.device), last].float()
    return torch.nn.functional.normalize(selected, p=2, dim=-1)


def row_positions():
    corpus = read(HERE / 'TEXT_CORPUS.json')
    ids = [r['id'] for r in corpus]
    assert len(ids) == len(set(ids)) == 18612
    lookup = {key: i for i, key in enumerate(ids)}
    with np.load(HERE / 'TEXT_ROWS.npz', allow_pickle=False) as rows:
        mapping = {t: np.asarray([lookup[key] for key in rows[t]], np.int64) for t in ('load', 'wind', 'pv', 'net_load')}
    return ids, mapping


def train_centroids(vectors, arrays, splits, mapping):
    result = {}
    for task in mapping:
        train = np.asarray(splits[task]['train'], np.int64)
        assets = arrays[task]['asset_id'].astype(str)
        assert not set(train) & set(splits[task]['val'])
        for asset in np.unique(assets):
            rows = train[assets[train] == asset]
            assert len(rows), (task, asset, 'missing TRAIN asset')
            mean = np.asarray(vectors[mapping[task][rows]], np.float64).mean(0)
            norm = np.linalg.norm(mean)
            assert np.isfinite(mean).all() and norm > 0
            result[task + '/' + asset] = (mean / norm).astype(np.float32)
    return result


def static_vectors(vectors, centroids, arrays, mapping):
    out = np.empty_like(vectors)
    assigned = np.zeros(len(vectors), bool)
    for task, positions in mapping.items():
        assets = arrays[task]['asset_id'].astype(str)
        for pos, asset in zip(positions, assets):
            value = centroids[task + '/' + asset]
            if assigned[pos]:
                assert np.array_equal(out[pos], value), 'Cross-asset text collision'
            out[pos] = value
            assigned[pos] = True
    assert assigned.all()
    return out
