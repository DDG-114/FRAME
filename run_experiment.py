import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'code' / 'scripts')]
from src.energyca_paper.config import ModelConfig, TaskSpec
from src.energyca_paper.data import load_task_bundle, collate_task_batch
from src.energyca_paper.model import ForwardControls, PhysicalLeadTimeEncoding
from src.energyca_paper.training import TrainingConfig, train_joint_model
from src.energy_context.llm_context import context_text_id, apply_llm_context_embeddings, write_llm_context_cache
from native_data import TASKS, REGIONS, tables, load_native


def model_config():
    return ModelConfig(num_cadences=4, d_model=128, d_context=128, d_residual=128, rank=8,
                       use_physical_lead_time=True, use_issued_target_anchor=True, issued_anchor_init=.98,
                       seasonal_anchor_mode='gated_daily_weekly', decoder_variant='contextual_mlp',
                       method_variant='current', use_quantile_head=True,
                       task_output_modes=('relu', 'unit_interval', 'relu', 'identity'))


def prepare(args):
    from src.data.loaders import DATASET_LOADERS
    cfg = model_config()
    print('Preparing six-month native data and forecast availability', flush=True)
    _, issued, _, _, _, _, _ = tables()
    mask_lookup = {(pd.Timestamp(o), str(r)): g.set_index('valid_time')[[f'issued_{t}' for t in TASKS]].notna()
                   for (o, r), g in issued.groupby(['forecast_origin', 'node_id'], sort=False)}
    bundles, manifest = {}, {}
    for i, task in enumerate(TASKS):
        print('Building windows: ' + task, flush=True)
        dataset = 'aemo_' + task
        DATASET_LOADERS[dataset] = lambda t=task: load_native(t)
        spec = TaskSpec(task, dataset, i, 3, 5, 2016, 24, 6, 5,
                        args.train_windows, args.eval_windows)
        bundle = load_task_bundle(spec, model_config=cfg, llm_cache=None, require_llm=False,
                                  future_exog_policy='issued', split_embargo_steps=2016,
                                  calibration_fraction=.5, calibration_embargo_days=1)
        for split, ds in bundle.datasets.items():
            assert len(ds), (task, split)
            for ex in ds.examples:
                future = pd.DatetimeIndex(ex.future_timestamps)
                assert (future[-1]-pd.Timestamp(ex.forecast_start)) == pd.Timedelta(hours=2)
                ex.future_exog_mask = mask_lookup[(pd.Timestamp(ex.forecast_start), str(ex.node_id))].reindex(future).to_numpy(np.float32)
        bundles[task + '_h2hours'] = bundle
        manifest[task] = bundle.manifest
        print(json.dumps({'prepared': task, 'examples': {s: len(d) for s, d in bundle.datasets.items()}}), flush=True)
    for split in ('train', 'val', 'calibration', 'test'):
        pairs = [{(x.forecast_start, x.node_id) for x in b.datasets[split].examples} for b in bundles.values()]
        assert all(p == pairs[0] for p in pairs)
    artifact = ROOT / 'bundles.pkl'
    with artifact.open('wb') as f:
        pickle.dump(bundles, f, protocol=5)
    (ROOT / 'data_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'PREPARED {artifact}', flush=True)


def load_bundles():
    with (ROOT / 'bundles.pkl').open('rb') as f:
        return pickle.load(f)


def cache(args):
    from transformers import AutoModel, AutoTokenizer
    from build_paper_context_caches import encode_task
    bundles = load_bundles()
    text_by_id = {}
    for b in bundles.values():
        for ds in b.datasets.values():
            for ex in ds.examples:
                text_by_id[context_text_id(ex.context_text)] = ex.context_text
    ids = sorted(text_by_id)
    texts = [text_by_id[i] for i in ids]
    path = str(ROOT / 'models' / 'Qwen3-8B')
    print(f'CONTEXTS {len(ids)}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16,
                                      device_map={'': 0}, attn_implementation='sdpa').eval()
    emb = encode_task(texts, tokenizer=tokenizer, model=model, batch_size=16, max_length=384,
                       pool_normalization='l2', text_schema_variant='current', pooling='last_token')
    write_llm_context_cache(path=ROOT / 'context_cache.npz', ids=ids, texts=texts, embeddings=emb,
                            metadata={'model': path, 'pooling': 'last_token', 'max_length': 384,
                                      'split': '2024 Jan-Jun chronological', 'frozen': True})
    print('CACHE_COMPLETE', flush=True)


def check(args):
    enc = PhysicalLeadTimeEncoding(4)
    enc.projection = torch.nn.Identity()
    for cadence, steps, hours in [(3, 24, 2), (2, 48, 24), (2, 336, 168)]:
        features = enc(torch.tensor([cadence]), steps, dtype=torch.float32)
        assert torch.isclose(features[0, -1, 0], torch.tensor(hours / 24))
        assert torch.isclose(features[0, -1, 1], torch.log1p(torch.tensor(float(hours))))
    bundles = load_bundles()
    from src.energyca_paper.model import SharedEnergyCALLM
    from src.energyca_paper.training import _model_inputs
    model = SharedEnergyCALLM(model_config()).cuda().eval()
    batch = collate_task_batch([b.datasets['train'][0] for b in bundles.values()])
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        result = model(**_model_inputs(batch), controls=ForwardControls(context_mode='capacity_matched_schema'))
    print({k: tuple(v.shape) for k, v in result.items() if torch.is_tensor(v)}, flush=True)
    assert torch.isfinite(result['prediction']).all()
    assert result['prediction'].shape == (4, 24)
    assert result['quantiles'].shape == (4, 24, 23)
    assert torch.all(torch.diff(result['quantiles'], dim=-1) >= 0)
    assert torch.all(batch['tokens'][:, 2016+10:, 30:34] == 0)
    print('GPU_CHECK_PASS: 4 tasks, native 5 min, 24 outputs, 2/24/168 h physical encoding', flush=True)


def train(args):
    bundles = load_bundles()
    if args.variant == 'full':
        for b in bundles.values():
            for ds in b.datasets.values():
                apply_llm_context_embeddings(ds.examples, cache_path=ROOT / 'context_cache.npz', mode='concat')
                ds.require_llm = True
    if args.stage == 'smoke':
        for b in bundles.values():
            for ds in b.datasets.values():
                ds.examples = ds.examples[:8]
                ds.schema_contexts = ds.schema_contexts[:8]
    cfg = TrainingConfig(seed=args.seed, epochs=args.epochs, steps_per_epoch=args.steps, patience=4,
                         learning_rate=.0003, batch_size_load=8, batch_size_wind=8,
                         batch_size_pv=8, batch_size_net_load=8, eval_batch_multiplier=4,
                         semantic_warmup_epochs=2)
    out = ROOT / 'results' / f'{args.stage}_{args.variant}_seed{args.seed}'
    assert torch.cuda.is_available()
    train_joint_model(bundles=bundles, model_config=model_config(), training_config=cfg,
                       controls=ForwardControls(context_mode='full' if args.variant == 'full' else 'capacity_matched_schema'),
                       output_dir=out, run_name=out.name, validation_interventions=('full',),
                       test_interventions=('full',), evaluate_test=args.stage != 'smoke')
    print(f'COMPLETE {out}', flush=True)
    if args.stage == 'train':
        import subprocess
        subprocess.run([sys.executable, str(ROOT / 'summarize_results.py')], check=True)


def baselines(args):
    import run_tste_point_baselines as base
    bundles = load_bundles()
    device = torch.device('cuda')
    base_args = argparse.Namespace(seed=args.seed, strict_determinism=False, learning_rate=.0003,
                                    batch_size_load=8, batch_size_wind=8, batch_size_pv=8, batch_size_net_load=8)
    def reference(batch, spec):
        future = batch['tokens'][:, spec.history_len:]
        values = future[:, :, 5 + spec.task_id]
        masks = future[:, :, 5 + 25 + spec.task_id]
        last = batch['tokens'][:, spec.history_len-1:spec.history_len, 0]
        return masks * values + (1-masks) * last
    base.issued_target_from_batch = reference
    rows = []
    for label, b in bundles.items():
        loaders = base.make_loaders(b, base_args)
        for method in args.methods:
            out = ROOT / 'results' / f'baselines_seed{args.seed}' / method / label
            out.mkdir(parents=True, exist_ok=True)
            model = None
            if method not in ('persistence', 'seasonal_naive', 'issued_official'):
                model, history, best = base.train_direct_model(bundle=b, loaders=loaders, args=base_args,
                                epochs=args.epochs, steps=args.steps, patience=4, output_dir=out, device=device, method=method)
                (out / 'training.json').write_text(json.dumps({'history': history, 'best_epoch': best}), encoding='utf-8')
            metrics, artifact = base.evaluate(method=method, loader=loaders['test'], spec=b.spec,
                                        model=model, issued_bias=None, scale=base.task_scale(b), device=device)
            np.savez_compressed(out / 'predictions.npz', **artifact)
            (out / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
            rows.append(dict(task=label, method=method, **metrics))
            print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / 'results' / f'baselines_seed{args.seed}' / ('metrics_' + args.methods[0] + '.csv'), index=False)
    print('BASELINES_COMPLETE', flush=True)
    import subprocess
    subprocess.run([sys.executable, str(ROOT / 'summarize_results.py')], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'cache', 'check', 'smoke', 'train', 'baselines'])
    p.add_argument('--variant', choices=['numeric', 'full'], default='numeric')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-windows', type=int, default=128)
    p.add_argument('--eval-windows', type=int, default=32)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--methods', nargs='+', default=['persistence', 'seasonal_naive', 'issued_official',
                                                   'issued_residual_linear', 'patchtst_issued', 'timexer_issued', 'tft_issued'])
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.stage in ('train', 'smoke'):
        train(a)
    else:
        globals()[a.stage](a)
