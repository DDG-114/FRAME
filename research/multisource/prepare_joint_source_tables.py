import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize, pad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, default=Path('data_fixed_wind/source_codebook_feedback.npz'))
    parser.add_argument('--events', type=Path, default=Path('event_inputs/codebook.npz'))
    parser.add_argument('--output', type=Path, default=Path('joint_source_tables.pt'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.04)
    with np.load(args.sources) as data:
        sources = torch.from_numpy(data['embedding']).cuda()
        source_fields = torch.from_numpy(data['fields']).cuda()
    with np.load(args.events) as data:
        events = torch.from_numpy(data['embedding']).cuda()
        event_fields = torch.from_numpy(data['fields']).cuda()
        fit_ids = data['fit_ids'].copy()
    fitted = torch.cat((sources, events[torch.from_numpy(fit_ids).cuda()]))
    mean = fitted.mean(0)
    _, singular, vh = torch.linalg.svd(fitted.double() - mean.double(), full_matrices=False, driver='gesvd')
    projection = vh[:128].T.float().contiguous()
    all_vectors = torch.cat((sources, events))
    projected = normalize((all_vectors - mean) @ projection, dim=-1) * math.sqrt(6)
    fields = torch.cat((normalize(pad(source_fields, (0, 128 - source_fields.shape[-1])), dim=-1),
                        normalize(event_fields, dim=-1))) * math.sqrt(6)
    projected[len(sources)] = 0
    assert projected.shape == fields.shape and torch.isfinite(projected).all()
    torch.testing.assert_close(projection.double().T @ projection.double(),
                               torch.eye(128, device='cuda', dtype=torch.float64), atol=2e-5, rtol=2e-5)
    for original, combined in ((sources[:5], projected[:5]), (events[1:6], projected[len(sources) + 1:len(sources) + 6])):
        torch.testing.assert_close(normalize((original - mean) @ projection, dim=-1) * math.sqrt(6), combined,
                                   atol=2e-5, rtol=2e-5)
    record = {'sources': str(args.sources), 'events': str(args.events), 'source_rows': len(sources),
              'event_rows': len(events), 'training_event_fit_ids': fit_ids.tolist(),
              'basis': 'single centered PCA fitted on predeclared source templates and training-only event texts',
              'output': '128 components, per-row L2 norm sqrt(6), empty event zero',
              'fields': 'original categorical and hashed lexical facts, unchanged',
              'retained_fit_variance': float(singular[:128].square().sum() / singular.square().sum()),
              'validation_or_backtest_targets_used': False}
    torch.save({'qwen_table': projected.cpu(), 'fields_table': fields.cpu(), 'projection_mean': mean.cpu(),
                'projection': projection.cpu(), 'record': record}, args.output)
    args.output.with_suffix('.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({key: value for key, value in record.items() if key != 'training_event_fit_ids'}))


if __name__ == '__main__':
    main()
