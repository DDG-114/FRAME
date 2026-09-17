from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TASKS = ('load', 'wind', 'pv', 'net_load')
REGIONS = ('NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1')


@lru_cache(maxsize=1)
def tables():
    folders = sorted((ROOT / 'data' / 'native5min').glob('2024-*'))
    assert len(folders) == 12
    actual = pd.concat([pd.read_csv(p / 'actuals.csv') for p in folders], ignore_index=True)
    issued = pd.concat([pd.read_csv(p / 'issued.csv') for p in folders], ignore_index=True)
    actual['timestamp'] = pd.to_datetime(actual.timestamp)
    for field in ('forecast_origin', 'valid_time', 'available_at'):
        issued[field] = pd.to_datetime(issued[field])
    issued = issued.sort_values('available_at').drop_duplicates(['forecast_origin', 'valid_time', 'REGIONID'], keep='last')
    timestamps = pd.DatetimeIndex(sorted(actual.timestamp.unique()))
    assert ((timestamps[1:] - timestamps[:-1]) == pd.Timedelta(minutes=5)).all()
    assert (issued.available_at <= issued.forecast_origin).all()
    days = len(timestamps) // 288
    a, b = int(days * .6) * 288, int(days * .8) * 288
    splits = dict(train=np.arange(a), val=np.arange(a, b), test=np.arange(b, len(timestamps)))
    train = actual.loc[actual.timestamp <= timestamps[a-1]]
    scales = train.groupby('REGIONID')[list(TASKS)].agg(lambda x: max(float(x.abs().max()), 1.0))
    origins = timestamps[(timestamps.minute % 30 == 0) & (timestamps < timestamps[-1]-pd.Timedelta(hours=2))]
    # Explicitly retain missing forecast leads, so the lookup and the availability mask agree.
    grid = pd.MultiIndex.from_tuples([(o, r, o + pd.Timedelta(minutes=5*k))
                                     for o in origins for r in REGIONS for k in range(1, 25)],
                                    names=['forecast_origin', 'REGIONID', 'valid_time'])
    issued = issued.set_index(['forecast_origin', 'REGIONID', 'valid_time']).reindex(grid).reset_index()
    raw_issued = issued.copy()
    for task in TASKS:
        issued['issued_' + task] = issued[task] / issued.REGIONID.map(scales[task])
    issued = issued.rename(columns={'REGIONID': 'node_id'})
    return actual, issued, raw_issued, timestamps, splits, scales, origins


def load_native(task):
    actual, issued, _, timestamps, splits, scales, origins = tables()
    values = actual.pivot(index='timestamp', columns='REGIONID', values=task).reindex(index=timestamps, columns=REGIONS)
    normalized = values.ffill().div(scales[task], axis=1).to_numpy(np.float32)
    meta = pd.DataFrame(index=REGIONS)
    meta['region'] = REGIONS
    meta['market'] = 'AEMO NEM'
    meta['site_type'] = 'aemo_' + task
    meta['latitude'] = [-32.2, -22.5, -30., -42., -36.5]
    meta['longitude'] = [147., 144.5, 135., 146.5, 144.5]
    columns = ['issued_' + t for t in TASKS]
    return dict(occupancy=normalized, timestamps=timestamps, node_ids=list(REGIONS), node_meta=meta,
                norm_min=0., norm_max=1., node_norm={r: dict(min=0., max=float(scales.loc[r, task])) for r in REGIONS},
                split_indices=splits, forecast_origin_from_history_end=True, forecast_lead_steps=1,
                forecast_start_offset_steps=0, forecast_origin_filter=origins,
                issued_exogenous=issued[['forecast_origin', 'valid_time', 'node_id', *columns]],
                issued_exogenous_columns=columns, covariate_labels=columns,
                issued_forecast_metadata={'source': 'P5MIN_REGIONSOLUTION', 'valid_time_resolution_minutes': 5,
                                          'availability': 'max(run_time+5min, LASTCHANGED) <= origin',
                                          'missing_leads': 'explicit per-lead availability mask'})
