import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bounds = {'train': ('2022-01-01', '2024-01-01'),
              'val': ('2024-01-01', '2024-07-01'),
              'calibration': ('2024-07-01', '2025-01-01'),
              'test': ('2025-01-01', '2026-01-01')}
    report = {'periods': bounds, 'selection': 'forecast origin and all target steps inside each period',
              'training': 'all available training origins; no monthly subsampling',
              'test_role': '2025 chronological evaluation; previously used in project development',
              'horizons': {}}
    scales = None
    for hours in (24, 2, 168):
        source = args.source / f'h{hours}'
        protocol = json.loads((source / 'protocol.json').read_text())
        history = protocol['history_length']
        destination = args.output / f'h{hours}'
        destination.mkdir()
        details = {}
        for split, (begin, end) in bounds.items():
            with np.load(source / f'{split}.npz') as f:
                rows = {k: f[k] for k in f.files}
            origins = rows['forecast_start'].astype('datetime64[s]')
            target_end = origins + np.timedelta64(hours, 'h')
            selected = (origins >= np.datetime64(begin)) & (target_end < np.datetime64(end))
            rows = {k: v[selected] for k, v in rows.items()}
            assert len(rows['raw']) > 0
            if scales is None:
                assert hours == 24 and split == 'train'
                scales = {str(r): float(rows['raw'][rows['node_id'] == r, :, 1].max())
                          for r in np.unique(rows['node_id'])}
            issued = rows['x'][:, history:, 9] * rows['scale'][..., 1] + rows['offset'][..., 1]
            scale = np.asarray([scales[str(r)] for r in rows['node_id']], np.float32)[:, None]
            rows['x'][:, :history, 1] = rows['history'][..., 1] / scale
            rows['x'][:, history:, 9] = np.where(rows['x'][:, history:, 13] > .5, issued / scale, 0)
            rows['y'][..., 1] = rows['raw'][..., 1] / scale
            rows['scale'][..., 1] = scale
            rows['offset'][..., 1] = 0
            np.testing.assert_allclose(rows['y'] * rows['scale'] + rows['offset'], rows['raw'], atol=.004, rtol=1e-6)
            assert not rows['x'][:, history:, :4].any()
            np.savez_compressed(destination / f'{split}.npz', **rows)
            times = rows['forecast_start'].astype('datetime64[s]')
            details[split] = {'n': len(times), 'first_origin': str(times.min()),
                              'last_origin': str(times.max()),
                              'last_target': str(times.max() + np.timedelta64(hours, 'h'))}
        for a, b in [('train', 'val'), ('val', 'calibration'), ('calibration', 'test')]:
            assert np.datetime64(details[a]['last_target']) < np.datetime64(details[b]['first_origin'])
        protocol.update(splits=details, wind_training_scales_mw=scales,
                        wind_normalization='fixed regional maximum from 2022-2023 training targets')
        (destination / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        report['horizons'][str(hours)] = details
        print(json.dumps({'hours': hours, 'splits': details}), flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2))
    print('YEAR_SPLIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
