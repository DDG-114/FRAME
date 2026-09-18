import argparse
import json
from pathlib import Path

import numpy as np

TASKS = ('load', 'wind', 'pv', 'net_load')
REGIONS = ('NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1')
SOURCES = ('region', 'calendar', *(f'history_{x}' for x in TASKS),
           *(f'issued_{x}' for x in TASKS))


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def prepare(root, output):
    output.mkdir(parents=True, exist_ok=True)
    report = {'fit_end': '2024-10-01', 'validation': '2024-10-01/2024-11-24',
              'calibration': '2024-12-01/2024-12-25', 'embargo_days': 7,
              'sampling': 'training origins on days 3,10,17,24; full histories and targets',
              'sources': list(SOURCES), 'unavailable': ['archived weather forecasts',
              'equipment state', 'forecast issue ages and previous vintages'], 'horizons': {}}
    for hours in (2, 24, 168):
        folder = root / 'shared_input/full' / f'h{hours}'
        protocol = json.loads((folder / 'protocol.json').read_text())
        parts = [load(folder / f'{s}.npz') for s in ('train', 'val', 'calibration')]
        rows = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        origins = rows['forecast_start'].astype('datetime64[s]')
        end = origins + np.timedelta64(hours, 'h')
        splits = {
            'train': end < np.datetime64('2024-09-24'),
            'val': (origins >= np.datetime64('2024-10-01')) & (end < np.datetime64('2024-11-24')),
            'calibration': (origins >= np.datetime64('2024-12-01')) & (end < np.datetime64('2024-12-25')),
        }
        out = output / f'h{hours}'
        out.mkdir(exist_ok=True)
        details = {}
        for split, mask in splits.items():
            selected = {k: v[mask] for k, v in rows.items()}
            dates = selected['forecast_start'].astype('datetime64[s]')
            assert len(dates) > 0
            np.savez_compressed(out / f'{split}.npz', **selected)
            details[split] = {'n': len(dates), 'first_origin': str(dates.min()),
                              'last_origin': str(dates.max()),
                              'last_target': str(dates.max() + np.timedelta64(hours, 'h'))}
            if split == 'train':
                day = (dates.astype('datetime64[D]') - dates.astype('datetime64[M]')).astype(int) + 1
                quick = np.isin(day, [3, 10, 17, 24])
                np.savez_compressed(out / 'train_quick.npz', **{k: v[quick] for k, v in selected.items()})
                details['train_quick'] = {'n': int(quick.sum())}
        for split in ('test', 'confirmation'):
            source = load(folder / f'{split}.npz')
            dates = source['forecast_start'].astype('datetime64[s]')
            details[split] = {'n': len(dates), 'first_origin': str(dates.min()),
                              'last_origin': str(dates.max()), 'role': 'previously used chronological backtest'}
        train = load(out / 'train.npz')
        mismatch = train['raw'][..., 3] - (train['raw'][..., 0] - train['raw'][..., 1] - train['raw'][..., 2])
        details['net_identity_max_mw'] = float(np.max(np.abs(mismatch)))
        details['net_identity_mean_mw'] = float(np.mean(np.abs(mismatch)))
        protocol.update(splits=details, new_time_protocol=report['validation'],
                        original_normalization='unchanged; original fit period is a subset of new training')
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        report['horizons'][str(hours)] = details
        print(json.dumps({'hours': hours, **details}), flush=True)
    (output / 'data_audit.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    prepare(a.root, a.output)
