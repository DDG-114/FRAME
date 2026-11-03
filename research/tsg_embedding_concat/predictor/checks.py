"""Exact common initialization and cold-restoration checks."""
from pathlib import Path
import numpy as np
import torch


def exact(a, b, path='state'):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for key in a: exact(a[key], b[key], path + '/' + str(key))
    elif isinstance(a, (list, tuple)):
        assert type(a) == type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): exact(x, y, path + '/' + str(i))
    else:
        assert a == b, path


def checkpoint_initial(a, b):
    assert a['step'] == b['step'] == 0 and a['seed'] == b['seed'] == 2027
    keys = ['state_dict', 'optimizer', 'optimizer_parameter_names', 'optimizer_ages',
            'optimizer_grad_update_counts', 'sampler_states', 'rng']
    for key in keys: exact(a[key], b[key], key)
    assert not a['optimizer']['state'] and not b['optimizer']['state']
    assert not a['EMA'] and not b['EMA'] and not a['language_LoRA'] and not b['language_LoRA']
    return dict(status='passed', complete_model=True, empty_optimizer=True, optimizer_names_and_ages=True,
        sampler_order_position_RNG_digest=True, Python_NumPy_Torch_CUDA_RNG=True, no_EMA_or_LoRA=True)


def raw_equal(first, second):
    count = 0
    for p in sorted(Path(first).glob('*_predictions.npz')):
        with np.load(p, allow_pickle=False) as a, np.load(Path(second) / p.name, allow_pickle=False) as b:
            assert set(a.files) == set(b.files), p.name
            for key in a.files:
                if a[key].ndim: assert np.array_equal(a[key], b[key]), (p.name, key)
        count += 1
    assert count == 8, count
    return count
