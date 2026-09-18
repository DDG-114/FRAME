import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

FIELDS = ('temperature_2m', 'wind_u_10m', 'wind_v_10m', 'wind_u_100m', 'wind_v_100m',
          'cloud_cover', 'wind_speed_10m', 'wind_speed_100m', 'downward_shortwave')
STATS = ('area_weighted_mean', 'area_weighted_std', 'grid_minimum', 'grid_maximum')


def align_window(records, window, region):
    origin = datetime.fromisoformat(window['origin_utc'])
    cadence = 5 if window['hours'] == 2 else 30
    horizon = window['hours'] * 60 // cadence
    values = np.full((horizon, len(FIELDS), 4), np.nan, np.float32)
    support = np.full((horizon, len(FIELDS), 2), np.nan, np.float32)
    available = np.zeros((horizon, len(FIELDS)), bool)
    ready = sorted(records, key=lambda r: r['lead_hours'])
    if not ready:
        return values, available, support, np.nan
    cycle = datetime.fromisoformat(ready[0]['cycle_utc'])
    age = (origin - cycle).total_seconds() / 3600
    indexed = {r['lead_hours']: r for r in ready}
    for row in ready:
        assert row['cycle_utc'] == ready[0]['cycle_utc']
        assert all(datetime.fromisoformat(f['object_last_modified_utc']) <= origin for f in row['fields'])
        assert all(f['stepUnits'] == 1 for f in row['fields'])
    for step in range(horizon):
        lead = age + (step + 1) * cadence / 60
        left, right = int(np.floor(lead / 3)) * 3, int(np.ceil(lead / 3)) * 3
        if left in indexed and right in indexed:
            alpha = 0 if left == right else (lead - left) / (right - left)
            for q, field in enumerate(FIELDS[:-1]):
                for endpoint in (indexed[left], indexed[right]):
                    original = [f for f in endpoint['fields'] if f['field'] == field]
                    assert not original or original[0]['stepType'] == 'instant'
                a = np.asarray(indexed[left]['regions'][region][field])
                b = np.asarray(indexed[right]['regions'][region][field])
                values[step, q] = (1 - alpha) * a + alpha * b
                support[step, q] = [left - age, right - age]
                available[step, q] = True
        intervals = []
        for row in ready:
            radiation = next(f for f in row['fields'] if f['field'] == FIELDS[-1])
            assert radiation['stepType'] == 'avg'
            start, end = radiation['startStep'], radiation['endStep']
            if start < lead <= end:
                intervals.append((end - start, end, start, row))
        if intervals:
            _, end, start, row = min(intervals, key=lambda item: item[:2])
            values[step, -1] = row['regions'][region][FIELDS[-1]]
            support[step, -1] = [start - age, end - age]
            available[step, -1] = True
    assert np.isfinite(values[available]).all()
    return values, available, support, age


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data_fixed_wind'))
    parser.add_argument('--cache', type=Path, default=Path('weather_forecast_cache'))
    parser.add_argument('--manifest', type=Path, default=Path('weather_request_manifest.json'))
    parser.add_argument('--output', type=Path, default=Path('weather_inputs_staging'))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = {key: json.loads(path.read_text()) for key in manifest['requests']
               if (path := args.cache / key / 'complete.json').exists()}
    windows = {(w['hours'], w['split'], w['origin_aest']): w for w in manifest['windows']}
    report = {'complete_objects': len(records), 'requested_objects': len(manifest['requests']),
              'fields': FIELDS, 'statistics': STATS,
              'time_basis': 'AEMO AEST converted to UTC; forecast targets retained in AEST',
              'instantaneous': 'linear interpolation between adjacent available 3-hour forecast endpoints',
              'radiation': 'shortest available interval average covering the target; no interpolation or instantaneous relabeling',
              'support_hours': 'source interval endpoints relative to forecast origin; supplied with each field',
              'spatial': 'state land-area grid-center statistics, NSW includes ACT',
              'missing': 'NaN values and false validity; no energy windows removed',
              'test_targets_read': False, 'training_started': False, 'splits': {}}
    for hours in (2, 24, 168):
        for split in ('train_quick', 'val'):
            with np.load(args.data / f'h{hours}/{split}.npz') as data:
                origins, regions = data['forecast_start'], data['node_id']
            aligned = []
            for origin, region in zip(origins, regions):
                window = windows[hours, split, str(origin.astype('datetime64[s]'))]
                selected = [records[key] for key in window['requested_objects'] if key in records]
                aligned.append(align_window(selected, window, str(region)))
            values, valid, support, age = map(np.stack, zip(*aligned))
            destination = args.output / f'h{hours}'
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / f'{split}.npz', forecast_start=origins, node_id=regions,
                                values=values, valid=valid, support_hours=support, cycle_age_hours=age)
            report['splits'][f'h{hours}/{split}'] = {'windows': len(origins),
                'complete_windows': int(valid.all((1, 2)).sum()),
                'valid_fraction_by_field': valid.mean((0, 1)).tolist()}
    report['all_requested_objects_complete'] = len(records) == len(manifest['requests'])
    report['all_windows_complete'] = all(v['windows'] == v['complete_windows'] for v in report['splits'].values())
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
