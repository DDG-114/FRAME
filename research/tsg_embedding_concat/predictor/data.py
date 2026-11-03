"""Same native arrays and row split; deterministic fixed-slot paper interface."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import math
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
SOURCE = Path(os.environ.get('PAPER_SOURCE_ROOT', str(WORKSPACE/'experiments/windfm_mechanism_screen_s2027_20261024')))
TASKS = ('load', 'wind', 'pv', 'net_load')
BATCH = dict(load=16, wind=2, pv=16, net_load=16)
AVAILABILITY = ('task', 'asset', 'geography', 'calendar', 'capacity', 'current_weather',
                'current_operation', 'older_weather', 'older_operation', 'observation_age')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def protocol(source=SOURCE):
    root = Path(source)/'protocol'
    name = '_paper_aligned_protocol'
    if name not in sys.modules:
        s = importlib.util.spec_from_file_location(name, root/'__init__.py', submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(s)
        sys.modules[name] = module
        s.loader.exec_module(module)
    from importlib import import_module
    return import_module(name+'.trainer'), import_module(name+'.evaluate')


def calendars(times, weekly=True):
    minutes = times.astype('datetime64[m]').astype(np.int64)
    daily = (minutes % 1440) / 1440
    weekday = (minutes // 1440 + 3) % 7
    x = np.stack((np.sin(2*np.pi*daily), np.cos(2*np.pi*daily),
                  np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)), -1).astype(np.float32)
    mask = np.ones(x.shape, bool)
    if not weekly:
        x[..., 2:] = 0
        mask[..., 2:] = False
    return x, mask


def summary(values, mask, cadence, weather_indices):
    daily = int(round(1440/cadence))
    def mean(x, m):
        return float(x[m].mean()) if m.any() else None
    target, valid = values[:, 0], mask[:, 0]
    level = mean(target[-daily:], valid[-daily:])
    a, b = mean(target[-3:], valid[-3:]), mean(target[-6:-3], valid[-6:-3])
    trend = a-b if a is not None and b is not None else None
    weather = mean(values[-daily:, weather_indices], mask[-daily:, weather_indices]) if weather_indices else None
    return [level, trend, weather]


def terciles(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return np.quantile(values, [1/3, 2/3]).tolist() if values else None


def band(value, cuts):
    return 'unavailable' if value is None or cuts is None else ('low', 'middle', 'high')[int(np.searchsorted(cuts, value, side='right'))]


def train_mean_absolute(a, rows):
    values = np.abs(a['target_raw'][rows])[a['target_mask'][rows]].astype(np.float64)
    return max(math.fsum(map(float, values.ravel())) / values.size, 1e-8)


class Dataset:
    def __init__(self, source=SOURCE, context_stats=None, condition='complete', omit_availability=False):
        if condition not in ('complete', 'remove', 'delay24h'):
            raise ValueError(condition)
        self.condition, self.omit_availability = condition, omit_availability
        self.source = Path(source)
        self.splits = json.loads((self.source/'bundle/SPLIT_ROWS.json').read_text(encoding='utf-8'))
        self.arrays, self.manifests, self.weather, self.kappa, self.stats = {}, {}, {}, {}, {}
        for task in TASKS:
            with np.load(self.source/'bundle/native'/(task+'_trainval.npz'), allow_pickle=False) as z:
                a = {k: z[k].copy() for k in z.files if k != 'manifest_json'}
                manifest = json.loads(str(z['manifest_json']))
            self.arrays[task], self.manifests[task] = a, manifest
            train = np.asarray(self.splits[task]['train'], np.int64)
            val = np.asarray(self.splits[task]['val'], np.int64)
            assert not set(train) & set(val)
            assert np.all(a['split'][train] == 'train') and np.all(a['split'][val] == 'val')
            assert len(train)+len(val) == len(a['sample_id'])
            if task == 'load':
                assert len(train) == 1819 and len(val) == 365
                assert np.all(a['origin_time'][val].astype('datetime64[Y]').astype(str) == '2010')
            self.kappa[task] = train_mean_absolute(a, train)
            names = [x['name'] for x in manifest['history_variables']]
            self.weather[task] = [i for i,n in enumerate(names) if i and ('temperature' in n or n.endswith(('_Wspd', '_Wdir', '_Etmp')))]
            if context_stats is None:
                rows = [summary(a['history_values'][i], a['history_mask'][i], int(a['cadence_minutes'][i]), self.weather[task]) for i in train]
                self.stats[task] = [terciles([r[j] for r in rows]) for j in range(3)]
            else:
                self.stats[task] = context_stats[task]
            if condition != 'complete':
                start = 0 if condition == 'remove' else -int(round(1440/int(a['cadence_minutes'][0])))
                a['history_mask'][:, start:, 1:] = False
                a['history_values'][:, start:, 1:] = 0

    def availability(self, task, row):
        a = self.arrays[task]
        hm = a['history_mask'][row]
        daily = int(round(1440/int(a['cadence_minutes'][row])))
        w = self.weather[task]
        # Geographic and capacity slots use the default fill values.
        return np.asarray([1, 1, 0, a['known_future_mask'][row, :, :4].any(), 0,
            hm[-daily:, w].any() if w else False, hm[-daily:, 0].any(),
            hm[:-daily, w].any() if w else False, hm[:-daily, 0].any(), 1], np.float32)

    def text(self, task, row):
        a = self.arrays[task]
        stats = summary(a['history_values'][row], a['history_mask'][row], int(a['cadence_minutes'][row]), self.weather[task])
        names = ('level', 'trend', 'weather')
        descriptions = '; '.join(n+'='+band(v,c) for n,v,c in zip(names, stats, self.stats[task]))
        available = self.availability(task, row)
        groups = ', '.join(n for n, v in zip(AVAILABILITY, available) if v)
        valid = np.flatnonzero(a['history_mask'][row,:,0])
        age = float((a['origin_time'][row]-a['history_time'][row,valid[-1]])/np.timedelta64(1,'m')) if len(valid) else None
        origin = ('ordinal_source_minutes='+str(int(a['origin_time'][row].astype('datetime64[m]').astype(np.int64)))
                  +'; weekday=unavailable') if task=='wind' else 'origin='+str(a['origin_time'][row])
        text = (f"task={task}; asset={a['asset_id'][row]}; cadence_minutes={int(a['cadence_minutes'][row])}; "
                f"{origin}; horizon={a['target'].shape[1]}; latest_operation_age_minutes={age}; "
                f"known_future_fields={','.join(v['name'] for j,v in enumerate(self.manifests[task]['future_variables']) if a['known_future_mask'][row,:,j].any())}; "
                f"observed_operation:{descriptions}; available_groups={groups}")
        if self.condition == 'delay24h' and self.weather[task]:
            daily = int(round(1440/int(a['cadence_minutes'][row])))
            old = summary(a['history_values'][row, :-daily], a['history_mask'][row, :-daily],
                          int(a['cadence_minutes'][row]), self.weather[task])[2]
            text += '; older_weather=' + band(old, self.stats[task][2]) + '; weather_age_minutes=1440'
        if self.omit_availability:
            prefix, rest = text.split('; available_groups=', 1)
            suffix = rest[rest.index(';'):] if ';' in rest else ''
            text = prefix + suffix
        return text

    def batch(self, task, rows, semantic=None, device='cpu'):
        rows = np.asarray(rows, np.int64)
        a = self.arrays[task]
        values, mask = a['history_values'][rows], a['history_mask'][rows]
        b, length, fields = values.shape
        if fields > 26:
            raise ValueError('Native historical fields exceed target plus25 slots')
        history, hm = np.zeros((b, length, 30), np.float32), np.zeros((b, length, 30), bool)
        history[..., 0], hm[..., 0] = values[..., 0], mask[..., 0]
        cal, cm = calendars(a['history_time'][rows], weekly=task != 'wind')
        history[..., 1:5], hm[..., 1:5] = cal, cm
        history[..., 5:5+fields-1], hm[..., 5:5+fields-1] = values[..., 1:], mask[..., 1:]
        future = np.zeros((b, a['target'].shape[1], 6), np.float32)
        fm = np.zeros(future.shape, bool)
        width = a['known_future_values'].shape[-1]
        future[..., :width], fm[..., :width] = a['known_future_values'][rows], a['known_future_mask'][rows]
        cadence = np.asarray(a['cadence_minutes'][rows])
        ids = np.asarray([{60:0, 10:1, 30:2}[int(x)] for x in cadence], np.int64)
        result = dict(history=history, history_mask=hm, future=future, future_mask=fm,
            availability=np.zeros((b, 10), np.float32) if self.omit_availability else np.stack([self.availability(task, i) for i in rows]),
            task_id=np.full(b, TASKS.index(task), np.int64), cadence_id=ids,
            target=a['target'][rows], target_mask=a['target_mask'][rows], target_scale=a['target_scale'][rows])
        result = {k: torch.as_tensor(v, device=device) for k,v in result.items()}
        if semantic is not None:
            result['semantic'] = torch.as_tensor(semantic, device=device, dtype=torch.float32)
        return result

    def fingerprints(self):
        return dict(split_sha256=sha(self.source/'bundle/SPLIT_ROWS.json'),
            native_sha256={t:sha(self.source/'bundle/native'/(t+'_trainval.npz')) for t in TASKS},
            protocol_sha256={n:sha(self.source/'protocol'/n) for n in ('trainer.py','evaluate.py')},
            counts={t:{s:len(self.splits[t][s]) for s in ('train','val')} for t in TASKS}, kappa=self.kappa,
            context_stats=self.stats)
