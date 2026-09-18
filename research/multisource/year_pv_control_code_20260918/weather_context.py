import itertools

import numpy as np

from assemble_weather_inputs import FIELDS

DESCRIPTIONS = (
    'air temperature at 2 m in kelvin', 'eastward wind component at 10 m in metres per second',
    'northward wind component at 10 m in metres per second',
    'eastward wind component at 100 m in metres per second',
    'northward wind component at 100 m in metres per second', 'total cloud cover in percent',
    'wind speed at 10 m in metres per second', 'wind speed at 100 m in metres per second',
    'downward shortwave radiation in watts per square metre')


def attach(rows, weather):
    for key in ('forecast_start', 'node_id'):
        np.testing.assert_array_equal(weather[key], rows[key])
    rows.update(weather_values=weather['values'], weather_valid=weather['valid'],
                weather_support=weather['support_hours'], weather_age=weather['cycle_age_hours'])


def vocabulary():
    texts, fields, keys = [], [], []
    levels = ('low', 'middle', 'high')
    for variable, description in enumerate(DESCRIPTIONS):
        support = 'interval average' if variable == 8 else 'instantaneous forecast'
        for category in itertools.product(range(3), repeat=3):
            words = [levels[i] for i in category]
            texts.append(f'NOAA GFS {description}, {support}, issued before the energy forecast origin. '
                         f'Area-weighted state forecast: level {words[0]}; temporal variation {words[1]}; '
                         f'end-minus-start change {words[2]}. Categories use training-period thresholds. '
                         'Exact forecast trajectories, publication age and temporal support are numerical inputs.')
            field = np.zeros(64, np.float32)
            field[50 + variable] = 1
            for i, value in enumerate(category):
                field[34 + 3 * i + value] = 1
            fields.append(field)
            keys.append(('weather', variable, *category))
    return texts, np.stack(fields), keys


def summaries(rows):
    values = rows['weather_values'][..., 0]
    valid = rows['weather_valid']
    count = valid.sum(1)
    mean = np.where(valid, values, 0).sum(1) / count.clip(1)
    variance = np.where(valid, (values - mean[:, None]) ** 2, 0).sum(1) / count.clip(1)
    first = valid.argmax(1)
    last = valid.shape[1] - 1 - valid[:, ::-1].argmax(1)
    initial = np.take_along_axis(values, first[:, None], axis=1)[:, 0]
    final = np.take_along_axis(values, last[:, None], axis=1)[:, 0]
    delta = np.where(count > 0, final - initial, 0)
    return np.stack((mean, np.sqrt(variance), delta), -1).astype(np.float32), count


def fit(rows):
    summary, count = summaries(rows)
    mean, std, cutoffs = [], [], []
    for q in range(len(FIELDS)):
        selected = rows['weather_values'][:, :, q][rows['weather_valid'][:, :, q]]
        assert len(selected)
        mean.append(selected.mean(0))
        std.append(selected.std(0).clip(.05))
        cutoffs.append(np.quantile(summary[count[:, q] > 0, q], [1/3, 2/3], axis=0))
    return {'mean': np.asarray(mean).tolist(), 'std': np.asarray(std).tolist(),
            'cutoffs': np.stack(cutoffs, axis=1).tolist(), 'variables': FIELDS}


def build(rows, fitted, first_id):
    valid = rows['weather_valid']
    summary, count = summaries(rows)
    category = (summary[None] > np.asarray(fitted['cutoffs'])[:, None]).sum(0)
    local_ids = category[..., 0] * 9 + category[..., 1] * 3 + category[..., 2]
    source_ids = first_id + np.arange(len(FIELDS))[None] * 27 + local_ids
    source_exact = np.zeros((len(valid), len(FIELDS), 6), np.float32)
    source_exact[..., :3] = summary
    source_exact[..., 3] = np.where(valid, rows['weather_values'][..., 1], 0).sum(1) / count.clip(1)
    source_exact[..., 4] = np.nan_to_num(rows['weather_age'])[:, None]
    source_exact[..., 5] = valid.mean(1)
    normalized = (rows['weather_values'] - np.asarray(fitted['mean'])) / np.asarray(fitted['std'])
    normalized = np.where(valid[..., None], normalized, 0).astype(np.float32)
    support = np.where(valid[..., None], rows['weather_support'] / 168, 0).astype(np.float32)
    age = np.broadcast_to(np.nan_to_num(rows['weather_age'])[:, None, None] / 24, (*valid.shape[:2], 1))
    numeric = np.concatenate((normalized.reshape(*valid.shape[:2], -1),
                              support.reshape(*valid.shape[:2], -1), valid, age), axis=-1).astype(np.float32)
    assert numeric.shape[-1] == 64 and np.isfinite(numeric).all()
    return {'weather_numeric': numeric, 'weather_available': valid.astype(np.float32),
            'source_ids': source_ids.astype(np.int64), 'source_exact': source_exact,
            'source_valid': valid.astype(np.float32)}
