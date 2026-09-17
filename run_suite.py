import argparse
from copy import copy
from dataclasses import replace
from functools import lru_cache
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT
sys.path[:0] = [str(ROOT), str(ROOT / 'code'), str(ROOT / 'code/scripts'), str(SOURCE)]

import numpy as np
import pandas as pd
import torch
from src.energyca_paper.config import TaskSpec, aemo_task_specs
from src.energyca_paper.data import load_task_bundle
from src.energyca_paper.model import ForwardControls
from src.energyca_paper.training import TrainingConfig, train_joint_model
from src.energy_context.llm_context import apply_llm_context_embeddings, context_text_id, write_llm_context_cache
import src.energy_context.llm_context as llm_context
llm_context.load_llm_context_cache = lru_cache(maxsize=1)(llm_context.load_llm_context_cache)
from native_data import TASKS, load_native, tables
from run_experiment import model_config
from multiyear_data import load_multiyear_task, split_plan
from shared_input_contract import apply_shared_input_contract, contract_manifest, write_baseline_arrays


def _subset(dataset, predicate):
    selected = [index for index, example in enumerate(dataset.examples) if predicate(pd.Timestamp(example.forecast_start))]
    result = copy(dataset)
    result.examples = [dataset.examples[index] for index in selected]
    result.schema_contexts = [dataset.schema_contexts[index] for index in selected]
    return result


def _subset_examples(dataset, predicate):
    selected = [index for index, example in enumerate(dataset.examples) if predicate(example)]
    result = copy(dataset)
    result.examples = [dataset.examples[index] for index in selected]
    result.schema_contexts = [dataset.schema_contexts[index] for index in selected]
    return result


def _artifact(name, mode):
    return ROOT / f'{name}_{mode}.pkl'


def _build_bundle(task, hours, mode):
    from src.data.loaders import DATASET_LOADERS
    os.environ['AEMO_SAMPLE_MODE'] = mode
    cfg = model_config()
    split_entries = {item['name']: item for item in split_plan()['splits']}
    dataset = f'aemo_{task}_h{hours}'
    DATASET_LOADERS[dataset] = lambda t=task, h=hours: load_multiyear_task(t, h)
    task_id = TASKS.index(task)
    if hours == 2:
        spec = TaskSpec(task, dataset, task_id, 3, 5, 2016, 24, 72, 5, 0, 0)
    else:
        spec = TaskSpec(task, dataset, task_id, 2, 30, 336, hours * 2, 48, 5, 0, 0)
    bundle = load_task_bundle(spec, model_config=cfg, llm_cache=None, require_llm=False,
                              future_exog_policy='issued', split_embargo_steps=0,
                              calibration_fraction=0, text_schema_variant='aaai_exact')
    combined = bundle.datasets['val']
    bundle.datasets['val'] = _subset(
        combined, lambda value: pd.Timestamp('2024-01-01') <= value < pd.Timestamp('2024-06-24'))
    bundle.datasets['calibration'] = _subset(
        combined, lambda value: pd.Timestamp('2024-07-01') <= value < pd.Timestamp('2024-12-25'))
    if mode in ('full', 'monthly_quick'):
        combined_test = bundle.datasets['test']
        test_start = pd.Timestamp(split_entries['test']['origin_start'])
        test_end = pd.Timestamp(split_entries['test']['origin_end'])
        confirmation_start = pd.Timestamp(split_entries['confirmation']['origin_start'])
        confirmation_end = pd.Timestamp(split_entries['confirmation']['origin_end'])
        bundle.datasets['test'] = _subset(
            combined_test, lambda value: test_start <= value < test_end)
        bundle.datasets['confirmation'] = _subset(
            combined_test, lambda value: confirmation_start <= value < confirmation_end)
    for split, ds in bundle.datasets.items():
        if not len(ds):
            raise RuntimeError(f'Empty {task} H{hours} {split} split')
        for ex in ds.examples:
            if pd.Timestamp(ex.future_timestamps[-1]) - pd.Timestamp(ex.forecast_start) != pd.Timedelta(hours=hours):
                raise RuntimeError(f'Incorrect target horizon for {task} H{hours}')
    return f'{task}_h{hours}hours', bundle


def _write_bundle_part(job):
    task, hours, mode, directory = job
    key, bundle = _build_bundle(task, hours, mode)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{key}.pkl'
    temporary = directory/f'{key}.{os.getpid()}.tmp'
    with temporary.open('wb') as handle:
        pickle.dump(bundle, handle, protocol=5)
    os.replace(temporary, path)
    return key, str(path), {split: len(dataset) for split, dataset in bundle.datasets.items()}


def prepare(args):
    workers = max(1, int(os.environ.get('AEMO_PREPARE_WORKERS', '1')))
    jobs = [(task, hours) for hours in (2, 24, 168) for task in TASKS]
    bundles = {}
    part_paths = {}
    if workers == 1:
        for task, hours in jobs:
            key, bundle = _build_bundle(task, hours, args.mode)
            bundles[key] = bundle
            print('prepared', task, hours, {s:len(d) for s,d in bundle.datasets.items()}, flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        part_dir = ROOT/f'prepare_parts_{args.mode}'
        payloads = [(task, hours, args.mode, str(part_dir)) for task, hours in jobs]
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
            futures = {executor.submit(_write_bundle_part, payload): payload for payload in payloads}
            for future in as_completed(futures):
                task, hours, _, _ = futures[future]
                key, path, counts = future.result()
                part_paths[key] = Path(path)
                print('prepared', task, hours, counts, flush=True)
        for task, hours in jobs:
            key = f'{task}_h{hours}hours'
            with part_paths[key].open('rb') as handle:
                bundles[key] = pickle.load(handle)
    for hours in (2, 24, 168):
        group = [b for k,b in bundles.items() if k.endswith(f'_h{hours}hours')]
        splits = ('train', 'val', 'calibration', 'test', 'confirmation') if args.mode in ('full', 'monthly_quick') else (
            'train', 'val', 'calibration', 'test')
        for split in splits:
            keys = [{(ex.forecast_start, ex.node_id) for ex in b.datasets[split].examples} for b in group]
            common = set.intersection(*keys)
            assert common, split
            for b in group:
                ds = b.datasets[split]
                ids = [i for i, ex in enumerate(ds.examples) if (ex.forecast_start, ex.node_id) in common]
                ds.examples = [ds.examples[i] for i in ids]
                ds.schema_contexts = [ds.schema_contexts[i] for i in ids]
            print('aligned', hours, split, len(common), flush=True)
    apply_shared_input_contract(bundles)
    with _artifact('bundles', args.mode).open('wb') as f:
        pickle.dump(bundles, f, protocol=5)
    write_baseline_arrays(bundles, ROOT/'shared_input'/args.mode)
    manifest = {k: {'spec':b.spec.to_dict(), 'splits':{s:len(d) for s,d in b.datasets.items()}} for k,b in bundles.items()}
    manifest['configuration'] = {'sample_mode': args.mode, 'text_schema_variant': 'aaai_exact',
                                 'pooling': 'mean', 'pool_normalization': 'none',
                                 'shared_input_contract': contract_manifest()}
    (ROOT/f'protocol_{args.mode}.json').write_text(json.dumps(manifest, indent=2))
    for path in part_paths.values():
        path.unlink()
    if part_paths:
        next(iter(part_paths.values())).parent.rmdir()


def load(mode='quick'):
    with _artifact('bundles', mode).open('rb') as f:
        return apply_shared_input_contract(pickle.load(f))


def cache(args):
    from transformers import AutoModel, AutoTokenizer
    from build_paper_context_caches import encode_task
    texts = {}
    for b in load(args.mode).values():
        for ds in b.datasets.values():
            for ex in ds.examples:
                texts[context_text_id(ex.context_text)] = ex.context_text
    ids = sorted(texts)
    path = os.environ.get('QWEN_MODEL_PATH', str(SOURCE/'models/Qwen3-8B'))
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16,
                                     device_map={'':0}, attn_implementation='sdpa').eval()
    vectors = encode_task([texts[k] for k in ids], tokenizer=tok, model=model, batch_size=16,
                          max_length=384, pool_normalization='none', text_schema_variant='aaai_exact', pooling='mean')
    write_llm_context_cache(path=ROOT/f'context_cache_{args.mode}.npz', ids=ids, texts=[texts[k] for k in ids],
                            embeddings=vectors, metadata={'encoder':path, 'frozen':True, 'pooling':'mean',
                                                         'pool_normalization':'none', 'text_schema_variant':'aaai_exact'})
    print('CACHE_COMPLETE', len(ids), flush=True)


def train(args):
    bundles = load(args.mode)
    run_variant = args.variant + (f'_holdout_{args.held_out_region}' if args.held_out_region else '')
    existing = ROOT/'results'/args.mode/f'{run_variant}_seed{args.seed}'
    if not args.smoke and not args.group and (existing/'summary.json').exists() and all(
            (existing/'calibration_artifacts'/key/'full.npz').exists() for key in bundles):
        if not (existing/'calibrated_metrics.csv').exists():
            calibrate(existing,bundles)
        print('REUSED_COMPLETE',existing,flush=True)
        return
    if args.group:
        kind, value = args.group.split(':', 1)
        bundles = {k:b for k,b in bundles.items() if
                   (kind == 'task' and b.spec.key == value) or
                   (kind == 'horizon' and k.endswith(f'_h{value}hours')) or
                   (kind == 'cell' and k == value)}
        existing = ROOT/'results'/args.mode/f'full_seed{args.seed}_{args.group.replace(":","_")}'
        if (existing/'calibration_metrics.json').exists() and (existing/'summary.json').exists():
            print('REUSED_COMPLETE',existing,flush=True)
            return
    if args.held_out_region:
        if args.group:
            raise ValueError('--held-out-region and --group are separate experiment controls')
        for bundle in bundles.values():
            for split in ('train', 'val', 'calibration'):
                bundle.datasets[split] = _subset_examples(
                    bundle.datasets[split], lambda example: str(example.node_id) != args.held_out_region)
            for split in ('test', 'confirmation'):
                bundle.datasets[split] = _subset_examples(
                    bundle.datasets[split], lambda example: str(example.node_id) == args.held_out_region)
            if any(not len(bundle.datasets[split]) for split in ('train', 'val', 'calibration', 'test', 'confirmation')):
                raise RuntimeError(f'empty leave-one-region split for {bundle.spec.key} {args.held_out_region}')
    if args.variant not in ('numeric', 'no_context'):
        for b in bundles.values():
            for ds in b.datasets.values():
                apply_llm_context_embeddings(ds.examples, cache_path=ROOT/f'context_cache_{args.mode}.npz', mode='concat')
                ds.require_llm = True
    cfg = model_config()
    context_mode = {
        'numeric': 'capacity_matched_schema',
        'no_context': 'no_context',
    }.get(args.variant, 'full')
    controls = ForwardControls(context_mode=context_mode,
                               use_adapters=args.variant != 'no_adapters', rank_gate=args.variant != 'no_rank_gate',
                               strength_gate=args.variant != 'no_strength_gate')
    if args.variant == 'no_anchor':
        cfg = replace(cfg, use_issued_target_anchor=False)
    quick_training = args.mode != 'full'
    training = TrainingConfig(seed=args.seed,
                               epochs=12 if quick_training else 20,
                               steps_per_epoch=96 if quick_training else 128,
                               patience=3 if quick_training else 5,
                              batch_size_load=16, batch_size_wind=16, batch_size_pv=16, batch_size_net_load=16,
                              eval_batch_multiplier=8)
    if args.smoke:
        training = replace(training, epochs=1, steps_per_epoch=2)
        for b in bundles.values():
            for ds in b.datasets.values():
                ds.examples, ds.schema_contexts = ds.examples[:8], ds.schema_contexts[:8]
    suffix = 'smoke' if args.smoke else f'seed{args.seed}' + ('_'+args.group.replace(':','_') if args.group else '')
    output = ROOT/'results'/args.mode/f'{run_variant}_{suffix}'
    assert torch.cuda.is_available()
    interventions = ('full',)
    if args.variant == 'full' and not args.group and not args.smoke and not args.held_out_region:
        interventions += ('missing_fields_masked','missing_fields_unmasked','delayed_context','wrong_release_time','shuffle_context')
    train_joint_model(bundles=bundles, model_config=cfg, training_config=training, controls=controls,
                      output_dir=output, run_name=output.name, validation_interventions=('full',),
                      test_interventions=interventions, evaluate_test=not args.smoke)
    if not args.smoke:
        from src.energyca_paper.training import evaluate_tasks, compute_training_scales
        from src.energyca_paper.model import SharedEnergyCALLM
        from evaluate_joint_checkpoint import make_calibration_loaders
        model = SharedEnergyCALLM(cfg).cuda()
        checkpoint = torch.load(output/'best_checkpoint.pt', map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        metrics, artifacts = evaluate_tasks(model, make_calibration_loaders(bundles, training),
                                             task_scales=compute_training_scales(bundles), controls=controls,
                                             intervention='full', device=torch.device('cuda'), use_bf16=True)
        for key, artifact in artifacts.items():
            folder = output/'calibration_artifacts'/key
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/'full.npz', **artifact)
        (output/'calibration_metrics.json').write_text(json.dumps(metrics, indent=2))
        if all('confirmation' in bundle.datasets for bundle in bundles.values()):
            from torch.utils.data import DataLoader
            from src.energyca_paper.data import collate_task_batch
            confirmation_loaders = {
                key: DataLoader(
                    bundle.datasets['confirmation'],
                    batch_size=training.eval_batch_size(key),
                    shuffle=False,
                    collate_fn=collate_task_batch,
                    num_workers=training.num_workers,
                    pin_memory=True,
                )
                for key, bundle in bundles.items()
            }
            confirmation_metrics, confirmation_artifacts = evaluate_tasks(
                model,
                confirmation_loaders,
                task_scales=compute_training_scales(bundles),
                controls=controls,
                intervention='full',
                device=torch.device('cuda'),
                use_bf16=True,
            )
            for key, artifact in confirmation_artifacts.items():
                folder = output/'confirmation_artifacts'/key
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder/'full.npz', **artifact)
            (output/'confirmation_metrics.json').write_text(json.dumps(confirmation_metrics, indent=2))
    print('COMPLETE', output, flush=True)
    if not args.smoke and not args.group:
        calibrate(output, bundles)


def reference(batch, spec):
    future = batch['tokens'][:, spec.history_len:]
    values = future[:, :, 5+spec.task_id]
    mask = future[:, :, 30+spec.task_id]
    last = batch['tokens'][:, spec.history_len-1:spec.history_len, 0]
    return mask*values + (1-mask)*last


def issued_raw(dataset, index):
    item = dataset[index]
    tokens = item['tokens'].unsqueeze(0)
    prediction = reference({'tokens':tokens}, dataset.spec)[0]
    return (prediction*item['target_scale']+item['target_offset']).numpy()


def calibrate(output, bundles):
    from run_tste_conformal_calibration import cqr_quantiles, aci_quantiles
    from run_tste_issued_anchor_calibration import weighted_median_coefficient
    from run_tste_probability_baselines import artifact_metrics
    rows = []
    for label,b in bundles.items():
        cal = dict(np.load(output/'calibration_artifacts'/label/'full.npz'))
        test = dict(np.load(output/'artifacts'/label/'full.npz'))
        artifacts = {'calibration': cal, 'test': test}
        confirmation_path = output/'confirmation_artifacts'/label/'full.npz'
        if confirmation_path.exists():
            artifacts['confirmation'] = dict(np.load(confirmation_path))
        issued = {}
        for split,artifact in artifacts.items():
            ds = b.datasets[split]
            ids = {(pd.Timestamp(ex.forecast_start),str(ex.node_id)):i for i,ex in enumerate(ds.examples)}
            issued[split] = np.stack([issued_raw(ds, ids[(pd.Timestamp(str(o)),str(n))])
                                      for o,n in zip(artifact['forecast_start'],artifact['node_id'])])
        beta = weighted_median_coefficient(cal['prediction_raw'], issued['calibration'], cal['target_raw'])
        levels = test['quantile_levels']
        for split, artifact in artifacts.items():
            shift = (1-beta)*(issued[split]-artifact['prediction_raw'])
            artifact['quantiles_raw'] = artifact['quantiles_raw']+shift[...,None]
            if b.spec.key != 'net_load':
                artifact['quantiles_raw'] = np.maximum(artifact['quantiles_raw'],0)
            artifact['prediction_raw'] = artifact['quantiles_raw'][...,np.argmin(abs(levels-.5))]
        for split in ('test', 'confirmation'):
            if split not in artifacts:
                continue
            evaluation = artifacts[split]
            cqr, qhats = cqr_quantiles(cal['quantiles_raw'],cal['target_raw'],evaluation['quantiles_raw'],
                                      levels=levels,cadence_minutes=b.spec.cadence_minutes,nonnegative=b.spec.key!='net_load')
            aci = aci_quantiles(evaluation['quantiles_raw'],evaluation['target_raw'],levels=levels,
                                cadence_minutes=b.spec.cadence_minutes,cqr_qhats=qhats,eta=.01,
                                scale=max(float(abs(cal['target_raw']).mean()),1e-6),nonnegative=b.spec.key!='net_load',
                                forecast_start=evaluation['forecast_start'])
            for method,q in [('center',evaluation['quantiles_raw']),('cqr',cqr),('aci',aci)]:
                artifact = dict(evaluation,quantiles_raw=q,prediction_raw=q[...,np.argmin(abs(levels-.5))])
                folder=output/('calibrated' if split == 'test' else 'calibrated_confirmation')/label
                folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/f'{method}.npz',**artifact)
                rows.append({'task':label,'split':split,'method':method,'beta':beta,
                             **artifact_metrics(artifact,cadence_minutes=b.spec.cadence_minutes)})
    pd.DataFrame(rows).to_csv(output/'calibrated_metrics.csv',index=False)


def probability(args):
    import run_tste_probability_baselines as base
    base.issued_target_from_batch = reference
    base.issued_point = issued_raw
    base.set_seed(args.seed)
    levels=np.asarray(base.DEFAULT_QUANTILE_LEVELS)
    output=ROOT/'results'/args.mode/('probability_smoke' if args.smoke else f'probability_seed{args.seed}')
    rows=[]
    for label,b in load(args.mode).items():
        if args.smoke and label != 'load_h2hours':
            continue
        loaders=base.loaders(b,8,args.seed)
        opts=argparse.Namespace(learning_rate=.0003,protocol='aemo' if b.spec.cadence_minutes==30 else 'perform')
        for method in ('seasonal_empirical','issued_empirical','patchtst_issued_quantile'):
            folder=output/method/label
            folder.mkdir(parents=True,exist_ok=True)
            model=None
            if method=='patchtst_issued_quantile':
                model,history=base.train_model(method,b,loaders,args=opts,epochs=1 if args.smoke else 20,steps=2 if args.smoke else 128,patience=5,
                                               output_dir=folder,device=torch.device('cuda'),levels=levels)
                (folder/'training.json').write_text(json.dumps(history))
            splits = ('calibration','test','confirmation') if 'confirmation' in b.datasets else ('calibration','test')
            for split in splits:
                if model is None:
                    fn=base.empirical_artifact if method=='seasonal_empirical' else base.issued_empirical_artifact
                    artifact=fn(b,split=split,levels=levels)
                else:
                    artifact=base.predict_model(model,loaders[split],task=b.spec.key,method=method,levels=levels,
                                                device=torch.device('cuda'),aemo_capacity_factor=opts.protocol=='aemo')
                np.savez_compressed(folder/f'{split}.npz',**artifact)
                rows.append({'task':label,'method':method,'split':split,**base.artifact_metrics(artifact,cadence_minutes=b.spec.cadence_minutes)})
            print('PROBABILITY',label,method,flush=True)
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)


def baselines(args):
    import run_tste_point_baselines as base
    base.issued_target_from_batch = reference
    opts = argparse.Namespace(seed=args.seed, strict_determinism=False, learning_rate=.0003,
                              batch_size_load=8, batch_size_wind=8, batch_size_pv=8, batch_size_net_load=8)
    rows = []
    for label,b in load(args.mode).items():
        if args.smoke and label != 'load_h2hours':
            continue
        loaders = base.make_loaders(b, opts)
        for method in ('persistence','issued_official','issued_residual_linear','patchtst_issued','timexer_issued','tft_issued'):
            out = ROOT/'results'/args.mode/('baselines_smoke' if args.smoke else f'baselines_seed{args.seed}')/method/label
            out.mkdir(parents=True, exist_ok=True)
            model = None
            if method not in ('persistence','issued_official'):
                model, history, best = base.train_direct_model(bundle=b, loaders=loaders, args=opts,
                    epochs=1 if args.smoke else 20, steps=2 if args.smoke else 128, patience=5, output_dir=out, device=torch.device('cuda'), method=method)
                (out/'training.json').write_text(json.dumps({'history':history,'best_epoch':best}))
            metrics, artifact = base.evaluate(method=method, loader=loaders['test'], spec=b.spec,
                model=model, issued_bias=None, scale=base.task_scale(b), device=torch.device('cuda'))
            np.savez_compressed(out/'predictions.npz', **artifact)
            (out/'metrics.json').write_text(json.dumps(metrics))
            rows.append({'task':label,'method':method,**metrics})
            print('BASELINE', label, method, metrics.get('nmae'), flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results'/args.mode/('baselines_smoke' if args.smoke else f'baselines_seed{args.seed}')/'metrics.csv', index=False)


def recalibrate(args):
    calibrate(ROOT/'results'/args.mode/f'{args.variant}_seed{args.seed}', load(args.mode))
    print('CALIBRATION_COMPLETE',args.variant,args.seed,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','cache','train','baselines','probability','recalibrate'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variant', choices=['full','numeric','no_context','no_adapters','no_rank_gate','no_strength_gate','no_anchor'], default='full')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--group')
    parser.add_argument('--held-out-region', choices=['NSW1', 'QLD1', 'VIC1', 'SA1', 'TAS1'])
    parser.add_argument('--mode', choices=['quick','monthly_quick','full'], default='quick')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'train':
        import fcntl
        lock_name=f'{args.mode}_{args.variant}_{args.seed}_{args.group}_{args.held_out_region}_{args.smoke}'.replace(':','_')
        with (ROOT/f'{lock_name}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            train(args)
    elif args.stage in ('baselines','probability'):
        import fcntl
        with (ROOT/f'{args.stage}_{args.seed}_{args.smoke}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            completed=ROOT/'results'/args.mode/f'{args.stage}_seed{args.seed}'/'metrics.csv'
            if completed.exists() and not args.smoke:
                print('REUSED_COMPLETE',completed,flush=True)
            else:
                globals()[args.stage](args)
    elif args.stage == 'recalibrate':
        globals()[args.stage](args)
    else:
        globals()[args.stage](args)
