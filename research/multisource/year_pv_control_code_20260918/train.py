import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from model import FRAME
from prepare_data import TASKS, REGIONS, load
from source_context import build, threshold_fit, vocabulary


def recency_weights(origins, half_life_days):
    dates = np.asarray(origins).astype('datetime64[s]')
    age = (dates.max() - dates) / np.timedelta64(1, 'D')
    weights = np.exp2(-age / half_life_days)
    return (weights / weights.mean()).astype(np.float32)


def weighted_task_error(error, weights, rmse_weight=.5):
    weights = weights[:, None, None]
    total = weights.sum() * error.shape[1]
    mae = (error.abs() * weights).sum((0, 1)) / total
    mse = (error.square() * weights).sum((0, 1)) / total
    return mae + rmse_weight * (mse + 1e-8).sqrt()


def point_task_error(error, rmse_weight=.5):
    return error.abs().mean((0, 1)) + rmse_weight * (error.square().mean((0, 1)) + 1e-8).sqrt()


def reference_skill_weights(reference, alternatives, target):
    official_mse = (reference - target).square().mean((0, 1))
    alternative_mse = (alternatives - target).square().mean((1, 2)).amin(0)
    return (1 - official_mse / alternative_mse.clamp_min(1e-8)).clamp(0, 1)


def merge_task_gradients(gradients, normalize=False):
    original = gradients.detach().clone()
    if normalize:
        norms = original.norm(dim=1, keepdim=True)
        original = original / norms.clamp_min(1e-12) * norms.mean()
    projected = original.clone()
    for i in range(len(original)):
        for j in torch.randperm(len(original)).tolist():
            if i != j:
                dot = projected[i].dot(original[j])
                projected[i] -= dot.clamp_max(0) / original[j].square().sum().clamp_min(1e-12) * original[j]
    return projected.mean(0)


def projected_backward(task_losses, parameters, auxiliary_loss=None, normalize=False):
    def flatten(gradients):
        return torch.cat([torch.zeros_like(parameter).flatten() if gradient is None else gradient.flatten()
                          for parameter, gradient in zip(parameters, gradients)])
    vectors = []
    for i, loss in enumerate(task_losses):
        gradients = torch.autograd.grad(loss, parameters,
            retain_graph=i < len(task_losses) - 1 or auxiliary_loss is not None, allow_unused=True)
        vectors.append(flatten(gradients))
    merged = merge_task_gradients(torch.stack(vectors), normalize)
    if auxiliary_loss is not None:
        merged = merged + flatten(torch.autograd.grad(auxiliary_loss, parameters, allow_unused=True))
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = merged[offset:offset + count].view_as(parameter)
        offset += count


def previous_origin_indices(origins, regions):
    indices = np.arange(len(origins))
    for region in np.unique(regions):
        selected = np.flatnonzero(regions == region)
        ordered = selected[np.argsort(origins[selected])]
        assert np.all(origins[ordered[1:]] > origins[ordered[:-1]])
        indices[ordered[1:]] = ordered[:-1]
    return indices


def tables(path, normalization='whiten'):
    data = load(path)
    vectors = torch.from_numpy(data['embedding']).cuda()
    raw_fields = torch.from_numpy(data['fields']).cuda()
    if normalization == 'source_centered':
        source_columns = list(range(10)) + list(range(46, 50))
        source_columns += [i for i in range(50, 59) if raw_fields[:, i].any()]
        assert (raw_fields[:, source_columns].sum(-1) == 1).all()
        groups = raw_fields[:, source_columns].argmax(-1)
        identity = torch.nn.functional.one_hot(groups, len(source_columns)).float() * math.sqrt(3)
        residual_vectors, residual_fields = torch.empty_like(vectors), torch.empty_like(raw_fields)
        for group in range(len(source_columns)):
            selected = groups == group
            residual_vectors[selected] = vectors[selected] - vectors[selected].mean(0)
            residual_fields[selected] = raw_fields[selected] - raw_fields[selected].mean(0)
        u, s, _ = torch.linalg.svd(residual_vectors, full_matrices=False)
        content_width = 128 - len(source_columns)
        qwen_content = torch.nn.functional.normalize(u[:, :content_width] * s[:content_width], dim=-1) * math.sqrt(3)
        field_content = torch.nn.functional.pad(torch.nn.functional.normalize(residual_fields, dim=-1),
                                                (0, content_width - raw_fields.shape[-1])) * math.sqrt(3)
        qwen, fields = torch.cat((identity, qwen_content), -1), torch.cat((identity, field_content), -1)
        assert qwen.shape == fields.shape == (len(vectors), 128)
        torch.testing.assert_close(qwen.square().sum(-1), torch.full_like(qwen[:, 0], 6))
        torch.testing.assert_close(fields.square().sum(-1), torch.full_like(fields[:, 0], 6))
        return qwen, fields
    centered = vectors - vectors.mean(0)
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    qwen = u[:, :128] * s[:128]
    fields = raw_fields
    fields = torch.nn.functional.pad(fields, (0, 64))
    if normalization == 'whiten':
        qwen = qwen / qwen.std(0, correction=0).clamp_min(.01)
    else:
        qwen = torch.nn.functional.normalize(qwen, dim=-1) * math.sqrt(6)
        fields = torch.nn.functional.normalize(fields, dim=-1) * math.sqrt(6)
    return qwen, fields


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['no_context', 'structured', 'shared_weights', 'shared_structured', 'full'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--patience', type=int, default=0)
    p.add_argument('--adapter-ablation', choices=['none', 'no_adapter', 'static_gate'], default='none')
    p.add_argument('--pv-long-context-scale', type=float, default=1.)
    p.add_argument('--scheduler-epochs', type=int, default=60)
    p.add_argument('--width', type=int, default=96)
    p.add_argument('--full-data', action='store_true')
    p.add_argument('--evaluation-split', choices=['calibration', 'test'])
    p.add_argument('--reference-state', action='store_true')
    p.add_argument('--feedback-root', type=Path)
    p.add_argument('--branch-residual', action='store_true')
    p.add_argument('--centered-semantic', action='store_true')
    p.add_argument('--initialize', type=Path)
    p.add_argument('--semantic-lr', type=float)
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--table-normalization', choices=['whiten', 'unit_geometry', 'source_centered'], default='whiten')
    p.add_argument('--gpu-memory-fraction', type=float)
    p.add_argument('--oof', action='store_true')
    p.add_argument('--aux-labels', type=Path)
    p.add_argument('--aux-weight', type=float, default=.05)
    p.add_argument('--error-control', action='store_true')
    p.add_argument('--context-placement', choices=['internal', 'output'], default='internal')
    p.add_argument('--coherent-output', action='store_true')
    p.add_argument('--event-root', type=Path)
    p.add_argument('--weather-root', type=Path)
    p.add_argument('--affine-adapter', action='store_true')
    p.add_argument('--adapter-after-ff', action='store_true')
    p.add_argument('--issued-residual', action='store_true')
    p.add_argument('--local-dynamics', action='store_true')
    p.add_argument('--ordered-history', action='store_true')
    p.add_argument('--task-readout', action='store_true')
    p.add_argument('--future-trajectory', action='store_true')
    p.add_argument('--gradient-audit', action='store_true')
    p.add_argument('--reference-audit', action='store_true')
    p.add_argument('--task-gradients', choices=['mean', 'project', 'balanced_project'], default='mean')
    p.add_argument('--audit-attention', action='store_true')
    p.add_argument('--context-dropout', type=float, default=0.)
    p.add_argument('--recent-history', action='store_true')
    p.add_argument('--native-history', action='store_true')
    p.add_argument('--origin-bridge', action='store_true')
    p.add_argument('--task-history', action='store_true')
    p.add_argument('--semantic-routing-only', action='store_true')
    p.add_argument('--hierarchical-query', action='store_true')
    p.add_argument('--conditioned-query', action='store_true')
    p.add_argument('--task-semantic-prior', action='store_true')
    p.add_argument('--trajectory-mixer', action='store_true')
    p.add_argument('--joint-source-tables', type=Path)
    p.add_argument('--reference-balanced-loss', action='store_true')
    p.add_argument('--residual-regularization', type=float, default=0.)
    p.add_argument('--skill-weighted-regularization', action='store_true')
    p.add_argument('--reference-scaled-residual', action='store_true')
    p.add_argument('--adaptive-residual-scale', action='store_true')
    p.add_argument('--recency-half-life', type=float, default=0.)
    p.add_argument('--numerical-dropout', type=float)
    p.add_argument('--origin-anchored-reference', action='store_true')
    p.add_argument('--lead-weather-context', action='store_true')
    p.add_argument('--task-adapter', action='store_true')
    p.add_argument('--loss-scale', choices=['task_mean', 'regional_standard'], default='task_mean')
    p.add_argument('--rmse-weight', type=float, default=.5)
    p.add_argument('--ema-decay', type=float, default=0.)
    p.add_argument('--relative-readout', action='store_true')
    p.add_argument('--exclude-context-sources', nargs='+', default=[],
                   choices=['region', 'calendar', 'history', 'issued', 'feedback', 'weather', 'events'])
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    assert not a.evaluation_split or a.initialize
    assert not a.skill_weighted_regularization or a.residual_regularization > 0
    if a.weather_root:
        weather_manifest = json.loads((a.weather_root / 'manifest.json').read_text())
        assert a.smoke or weather_manifest['all_windows_complete']
        assert a.joint_source_tables
    assert torch.cuda.is_available()
    if a.gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    started = time.monotonic()
    previous = torch.load(a.initialize, map_location='cuda', weights_only=False) if a.initialize else None
    raw, data, norms, thresholds, specs = {}, {}, {}, {}, {}
    auxiliary_labels = {}
    residual_scales = {}
    smoke_indices = {}
    for hours in (2, 24, 168):
        protocol = json.loads((a.data / f'h{hours}/protocol.json').read_text())
        specs[hours] = protocol
        raw[hours], data[hours], norms[hours] = {}, {}, {}
        for split in (('train', 'val', 'oof') if a.oof else ('train', 'val')):
            name = 'train_quick' if split == 'train' and not a.full_data else split
            if split == 'val' and a.evaluation_split:
                name = a.evaluation_split
            rows = load(a.data / f'h{hours}/{name}.npz')
            if a.coherent_output:
                np.testing.assert_allclose(rows['raw'][..., 3],
                    rows['raw'][..., 0] - rows['raw'][..., 1] - rows['raw'][..., 2], atol=.004, rtol=1e-6)
            if a.feedback_root:
                feedback = load(a.feedback_root / f'h{hours}/{name}.npz')
                assert np.array_equal(feedback['forecast_start'], rows['forecast_start'])
                assert np.array_equal(feedback['node_id'], rows['node_id'])
                rows['feedback'] = feedback['feedback']
                rows['feedback_available'] = feedback['available']
                rows['context'] = np.concatenate((rows['context'], feedback['feedback'].reshape(len(rows['x']), -1),
                                                   feedback['available']), axis=1)
            if a.event_root:
                events = load(a.event_root / f'h{hours}/{name}.npz')
                for key in ('forecast_start', 'node_id'):
                    np.testing.assert_array_equal(events[key], rows[key])
                rows.update({key: events[key] for key in ('event_ids', 'event_exact', 'event_valid', 'event_meta')})
            if a.weather_root:
                from weather_context import attach as attach_weather
                weather = load(a.weather_root / f'h{hours}/{name}.npz')
                attach_weather(rows, weather)
                rows['context'][:, protocol['baseline_static_fields'].index('has_weather')] = weather['valid'].any((1, 2))
            if split == 'train' and a.reference_scaled_residual:
                history = protocol['history_length']
                future = rows['x'][:, history:]
                reference = np.where(future[..., 12:16] > .5, future[..., 8:12], rows['x'][:, history - 1:history, :4])
                squared_error = np.square(reference.astype(np.float64) - rows['y'])
                residual_scales[hours] = np.stack([
                    np.sqrt(squared_error[rows['node_id'] == region].mean((0, 1))).clip(1e-6)
                    for region in REGIONS]).astype(np.float32)
                assert np.isfinite(residual_scales[hours]).all()
            if a.smoke:
                selected = np.arange(min(8, len(rows['x'])))
                if split == 'train' and a.aux_labels:
                    available = load(a.aux_labels / f'h{hours}/{name}.npz')['available']
                    if a.event_root:
                        available = available & (rows['event_valid'].sum((1, 2)) > 0)
                    selected = np.flatnonzero(available)[:8]
                    if a.weather_root:
                        weather_selected = np.flatnonzero(rows['weather_valid'].all((1, 2)))[:4]
                        selected = np.unique(np.concatenate((selected[:4], weather_selected)))
                    smoke_indices[hours] = selected
                rows = {k: v[selected] for k, v in rows.items()}
            if split == 'train':
                thresholds[hours] = previous['thresholds'][hours] if previous else threshold_fit(rows, protocol['history_length'])
            rows.update(build(rows, protocol['history_length'], thresholds[hours]))
            if a.weather_root:
                from weather_context import fit as fit_weather, build as build_weather
                if split == 'train' and not previous:
                    thresholds[hours]['weather'] = fit_weather(rows)
                weather = build_weather(rows, thresholds[hours]['weather'], len(vocabulary(bool(a.feedback_root))[0]))
                for key in ('source_ids', 'source_exact'):
                    rows[key] = np.concatenate((rows[key], weather[key]), axis=1)
                rows['source_valid'] = np.concatenate((rows['source_valid'], weather['source_valid']), axis=2)
                rows.update({key: weather[key] for key in ('weather_numeric', 'weather_available')})
            if a.event_root:
                rows['event_numeric_ids'] = rows['event_ids'] + len(vocabulary(bool(a.feedback_root), bool(a.weather_root))[0])
                rows['source_ids'] = np.concatenate((rows['source_ids'], rows['event_numeric_ids']), axis=1)
                rows['source_exact'] = np.concatenate((rows['source_exact'], rows['event_exact']), axis=1)
                rows['source_valid'] = np.concatenate((rows['source_valid'], rows['event_valid']), axis=2)
            raw[hours][split] = rows
            needed = ('x', 'context', 'y', 'raw', 'scale', 'offset', 'source_ids', 'source_exact', 'source_valid', 'region')
            if a.feedback_root:
                needed += ('feedback', 'feedback_available')
            if a.event_root:
                needed += ('event_numeric_ids', 'event_meta')
            if a.weather_root:
                needed += ('weather_numeric', 'weather_available')
            data[hours][split] = {k: torch.from_numpy(rows[k]).cuda() for k in needed}
            previous_ids = previous_origin_indices(rows['forecast_start'], rows['node_id'])
            data[hours][split]['past_source_ids'] = data[hours][split]['source_ids'][previous_ids]
        train = data[hours]['train']
        if a.aux_labels:
            name = 'train' if a.full_data else 'train_quick'
            auxiliary = load(a.aux_labels / f'h{hours}/{name}.npz')
            if a.smoke:
                auxiliary = {key: value[smoke_indices[hours]] for key, value in auxiliary.items()}
            for key in ('forecast_start', 'node_id'):
                np.testing.assert_array_equal(auxiliary[key], raw[hours]['train'][key])
            auxiliary_labels[hours] = {key: torch.from_numpy(auxiliary[key]).cuda() for key in ('target', 'available')}
        for key in ('context', 'source_exact') + (('event_meta',) if a.event_root else ()):
            if previous:
                mean = previous['normalization'][hours][key]['mean'].to(train[key].device)
                std = previous['normalization'][hours][key]['std'].to(train[key].device)
            else:
                mean, std = train[key].mean(0, keepdim=True), train[key].std(0, keepdim=True, correction=0).clamp_min(.05)
            norms[hours][key] = {'mean': mean.cpu(), 'std': std.cpu()}
            for rows in data[hours].values():
                rows[key] = (rows[key] - mean) / std
    codebook = (a.weather_root / 'source_codebook.npz' if a.weather_root else
                a.data / ('source_codebook_feedback.npz' if a.feedback_root else 'source_codebook.npz'))
    qwen, fields = tables(codebook, a.table_normalization)
    if a.event_root:
        from event_context import event_tables
        event_qwen, event_fields = event_tables(a.event_root / 'codebook.npz')
        qwen, fields = torch.cat((qwen, event_qwen)), torch.cat((fields, event_fields))
    if a.joint_source_tables:
        joint = torch.load(a.joint_source_tables, weights_only=False, map_location='cuda')
        assert joint['qwen_table'].shape == qwen.shape
        torch.testing.assert_close(joint['fields_table'], fields, atol=1e-6, rtol=1e-6)
        qwen = joint['qwen_table']
    task_source_ids = None
    if a.task_semantic_prior:
        source_fields = vocabulary(bool(a.feedback_root), bool(a.weather_root))[1]
        task_source_ids = [np.flatnonzero(source_fields[:, [2 + q, 6 + q, 46 + q]].sum(1) > 0).tolist()
                           for q in range(4)]
    model_args = {'static_dim': data[2]['train']['context'].shape[-1], 'mode': a.mode,
                  'width': a.width, 'reference_state': a.reference_state, 'feedback_state': bool(a.feedback_root),
                  'branch_residual': a.branch_residual, 'centered_semantic': a.centered_semantic,
                  'auxiliary': bool(a.aux_labels), 'error_control': a.error_control,
                  'context_placement': a.context_placement, 'affine_adapter': a.affine_adapter,
                  'context_dropout': a.context_dropout,
                  'recent_history': a.recent_history,
                  'native_history': a.native_history,
                  'origin_bridge': a.origin_bridge,
                  'task_history': a.task_history,
                  'adaptive_residual_scale': a.adaptive_residual_scale,
                  'semantic_routing_only': a.semantic_routing_only,
                  'hierarchical_query': a.hierarchical_query,
                  'conditioned_query': a.conditioned_query,
                  'task_source_ids': task_source_ids,
                  'trajectory_mixer': a.trajectory_mixer,
                  'weather_state': bool(a.weather_root),
                  'adapter_after_ff': a.adapter_after_ff,
                  'adapter_ablation': a.adapter_ablation,
                  'pv_long_context_scale': a.pv_long_context_scale,
                  'issued_residual': a.issued_residual,
                  'local_dynamics': a.local_dynamics,
                  'ordered_history': a.ordered_history,
                  'task_readout': a.task_readout,
                  'future_trajectory': a.future_trajectory,
                  'origin_anchored_reference': a.origin_anchored_reference,
                  'lead_weather_context': a.lead_weather_context,
                  'task_adapter': a.task_adapter,
                  'relative_readout': a.relative_readout,
                  'residual_scale': ([residual_scales[h].tolist() for h in (2, 24, 168)]
                                     if a.reference_scaled_residual else None),
                  'exclude_context_sources': a.exclude_context_sources,
                  'event_slots': raw[2]['train']['event_ids'].shape[1] if a.event_root else 0}
    torch.manual_seed(a.seed)
    model = FRAME(**model_args, qwen_table=qwen, fields_table=fields).cuda()
    if a.numerical_dropout is not None:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = a.numerical_dropout
            elif isinstance(module, torch.nn.MultiheadAttention):
                module.dropout = a.numerical_dropout
    initialization = None
    if a.initialize:
        previous_protocol = json.loads((a.initialize.parent / 'protocol.json').read_text())
        previous_table_normalization = previous_protocol['arguments'].get('table_normalization', 'whiten')
        replacement_tables = ('qwen_table', 'fields_table') if a.table_normalization != previous_table_normalization else ()
        weights = {name: value for name, value in previous['model'].items() if name not in replacement_tables}
        absent = model.load_state_dict(weights, strict=False)
        assert not absent.unexpected_keys
        assert all('semantic_control' in key or 'semantic_strength' in key or key in replacement_tables
                   for key in absent.missing_keys)
        new_semantic_branch = a.branch_residual and not previous['model_args'].get('branch_residual', False)
        if new_semantic_branch and a.affine_adapter:
            for block in model.blocks:
                torch.nn.init.zeros_(block.context_shift.weight)
        initialization = {'checkpoint': str(a.initialize), 'epoch': previous['epoch'],
                          'new_parameters': absent.missing_keys,
                          'affine_context_shift_zero_initialized': bool(new_semantic_branch and a.affine_adapter),
                          'normalization_and_source_thresholds': 'retained from shared initialization'}
    if a.semantic_lr is not None:
        semantic_names = ('source_proj', 'select_query', 'select_key', 'null_source', 'shared_query',
                          'semantic_control', 'semantic_strength', 'context_shift')
        semantic, numerical = [], []
        for name, parameter in model.named_parameters():
            (semantic if any(part in name for part in semantic_names) else numerical).append(parameter)
        optimizer = torch.optim.AdamW([{'params': numerical, 'lr': a.lr},
                                       {'params': semantic, 'lr': a.semantic_lr}], weight_decay=a.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.scheduler_epochs, eta_min=a.lr * .1)
    denominators = {h: rows['train']['raw'].abs().mean((0, 1)).clamp_min(1e-6) for h, rows in data.items()}
    recency = {h: torch.from_numpy(recency_weights(rows['train']['forecast_start'], a.recency_half_life)).cuda()
               for h, rows in raw.items()} if a.recency_half_life else {}
    loss_weights, reference_rms, reference_skill = {}, {}, {}
    if a.reference_balanced_loss or a.residual_regularization:
        for h, splits in data.items():
            rows = splits['train']
            history = specs[h]['history_length']
            future = rows['x'][:, history:]
            reference = torch.where(future[..., 12:16] > .5, future[..., 8:12], rows['x'][:, history - 1:history, :4])
            reference = reference * rows['scale'] + rows['offset']
            reference_rms[h] = (reference - rows['raw']).square().mean((0, 1)).sqrt().clamp_min(1e-6)
            if a.skill_weighted_regularization:
                horizon = future.shape[1]
                period = 288 if h == 2 else 48
                past = rows['x'][:, :history, :4]
                steps = torch.arange(horizon, device=past.device)
                alternatives = torch.stack((past[:, -1:].expand(-1, horizon, -1),
                    past[:, history - period + steps % period], past[:, steps % history]))
                alternatives = alternatives * rows['scale'] + rows['offset']
                reference_skill[h] = reference_skill_weights(reference, alternatives, rows['raw'])
            if a.reference_balanced_loss:
                relative_error = (reference - rows['raw']).abs().mean((0, 1)) / denominators[h]
                weights = relative_error.clamp_min(1e-4).reciprocal()
                loss_weights[h] = weights / weights.mean()
                assert torch.isfinite(loss_weights[h]).all() and (loss_weights[h] > 0).all()
    residual_output = []
    if a.residual_regularization:
        model.output.register_forward_hook(lambda module, inputs, output: residual_output.__setitem__(slice(None), [output, inputs[0]]))

    def residual_penalty(rows, hours, ids):
        residual = residual_output[0]
        residual = residual.diagonal(dim1=-2, dim2=-1) if model.task_readout else residual.squeeze(-1)
        if model.residual_scale is not None:
            residual = residual * model.residual_multiplier(residual_output[1], hours, rows['region'][ids])
        normalized = residual.float() * rows['scale'][ids] / reference_rms[hours]
        penalty = normalized.square()
        if a.skill_weighted_regularization:
            penalty = penalty * reference_skill[hours]
        if a.coherent_output:
            penalty = penalty[..., :3]
        return a.residual_regularization * penalty.mean()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'code').mkdir()
    for filename in ('train.py', 'model.py', 'prepare_data.py', 'source_context.py') + (('event_context.py',) if a.event_root else ()) + \
                    (('weather_context.py', 'assemble_weather_inputs.py') if a.weather_root else ()):
        shutil.copyfile(Path(__file__).parent / filename, a.output / 'code' / filename)
    protocol = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                'model': model_args, 'parameters': sum(v.numel() for v in model.parameters()),
                'task_order': TASKS, 'joint_horizons_hours': [2, 24, 168],
                'training': 'equal update counts per horizon, one shared model and checkpoint',
                'information': 'same raw inputs and facts; representation and selection mechanism differ',
                'selection': 'fixed 12-cell macro mean of validation NMAE plus NRMSE',
                'calibration': 'none; raw point forecasts', 'test_evaluated': False,
                'data': specs, 'initialization': initialization}
    protocol['loss_weights'] = {str(h): w.tolist() for h, w in loss_weights.items()}
    protocol['point_loss'] = {'scale': a.loss_scale, 'rmse_weight': a.rmse_weight,
        'regional_standard': 'existing per-region target transform scale; wind uses fixed training-period maximum power',
        'validation': 'unchanged global NMAE plus NRMSE, equal weights for the 12 cells'}
    protocol['weight_averaging'] = {'decay': a.ema_decay,
        'update': 'after every optimizer step; validation selects one averaged network checkpoint'}
    protocol['history_encoder'] = ('shared two-layer GRU over seven days at original cadence; no hourly aggregation'
                                   if a.native_history else 'hourly Transformer with optional native-cadence recent day')
    protocol['history_retrieval'] = ('four task-conditioned queries over the same memory and shared attention parameters'
                                    if a.task_history else 'shared temporal retrieval before task embeddings')
    protocol['origin_bridge'] = ('jointly trained, zero-initialized state gate and task decay; latest observation anchors '
                                 'origin extrapolated from first two predicted steps' if a.origin_bridge else None)
    protocol['recency_weighting'] = {'half_life_days': a.recency_half_life,
        'reference': 'latest training origin per horizon; all training windows retained',
        'effective_window_count': {str(h): float(w.sum().square() / w.square().sum()) for h, w in recency.items()},
        'application': 'power prediction loss only; auxiliary loss and validation selection unchanged'}
    protocol['readout_regularization'] = {
        'weight': a.residual_regularization,
        'normalizer': 'official forecast RMSE on training windows only; raw power units',
        'rms_by_horizon_task': {str(h): value.tolist() for h, value in reference_rms.items()},
        'skill_weighting': ('max(0, 1 - official MSE / minimum MSE of last-value, daily and weekly references); training windows only'
                            if a.skill_weighted_regularization else None),
        'skill_by_horizon_task': {str(h): value.tolist() for h, value in reference_skill.items()},
        'scope': 'neural readout residual; load/wind/PV only when net load is derived'}
    protocol['readout_scaling'] = ('per-region/task/horizon official forecast RMSE in normalized target units, fitted on training windows only'
                                   if a.reference_scaled_residual else 'original normalized target units')
    protocol['adaptive_readout_scaling'] = ('state-conditioned sigmoid interpolation between unit scale and training-fitted '
                                           'official forecast RMSE scale; initialized at their midpoint'
                                           if a.adaptive_residual_scale else None)
    protocol['gradient_aggregation'] = {'mode': a.task_gradients,
        'scope': 'four tasks within each horizon update; auxiliary loss gradient added afterwards',
        'reference': 'https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html',
        'balanced_project': 'normalize each task gradient to their mean norm before conflict projection'}
    protocol['loss_weight_source'] = ('inverse normalized official-forecast MAE on training windows, normalized to mean one per horizon'
                                     if a.reference_balanced_loss else 'uniform task weights')
    protocol['output_definition'] = ('nonnegative load, wind, PV; net load = load - wind - PV in MW; joint target loss'
                                     if a.coherent_output else 'four independent task outputs')
    protocol['sources'] = ['region', 'calendar', 'four historical operating summaries', 'four issued forecast summaries']
    if a.joint_source_tables:
        protocol['semantic_projection'] = joint['record']
    if a.weather_root:
        protocol['sources'].append('nine regional weather forecast variables, temporal support, age and validity')
        protocol['weather'] = weather_manifest
        protocol['weather_information'] = 'identical 64 numerical forecast features and availability supplied to every context control'
    if a.feedback_root:
        protocol['sources'].append('four rolling past forecast performance records, also supplied numerically to every model')
    if a.event_root:
        protocol['sources'].append('published equipment/network records; per-step validity and snapshot age')
        protocol['event_information'] = json.loads((a.event_root / 'manifest.json').read_text())
        protocol['event_encoding'] = json.loads((a.event_root / 'encoding.json').read_text())
        protocol['representation_controls'] = 'every model receives event lexical/numeric summaries; Structured Fields additionally selects individual lexical event vectors; Full uses Qwen vectors for the identical text'
    (a.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    torch.save({'normalization': norms, 'thresholds': thresholds}, a.output / 'input_transform.pt')

    def predict(rows, hours, ids, intervention=None, return_aux=False):
        inputs = {k: v[ids] for k, v in rows.items() if k not in ('raw', 'scale', 'offset', 'y')}
        if a.semantic_routing_only:
            inputs['source_value_ids'] = inputs['source_ids']
        if intervention == 'global_semantic_mismatch':
            inputs['source_ids'] = rows['source_ids'].roll(len(rows['x']) // 2, 0)[ids]
            intervention = None
        elif intervention == 'past_semantic_mismatch':
            inputs['source_ids'] = rows['past_source_ids'][ids]
            intervention = None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(inputs, hours, intervention, return_aux=return_aux)
        if return_aux:
            prediction, auxiliary = prediction
        prediction = prediction.float() * rows['scale'][ids] + rows['offset'][ids]
        prediction = torch.cat((prediction[..., :3].clamp_min(0), prediction[..., 3:]), -1)
        if a.coherent_output:
            net_load = prediction[..., 0] - prediction[..., 1] - prediction[..., 2]
            prediction = torch.cat((prediction[..., :3], net_load[..., None]), -1)
        return (prediction, auxiliary.float()) if return_aux else prediction

    @torch.no_grad()
    def evaluate(intervention=None, save=False, split='val'):
        model.eval()
        metrics = {}
        for h in (2, 24, 168):
            rows = data[h][split]
            prediction = torch.cat([predict(rows, h, slice(i, i + a.batch), intervention)
                                    for i in range(0, len(rows['x']), a.batch)])
            error = prediction - rows['raw']
            denominator = rows['raw'].abs().mean((0, 1)).clamp_min(1e-6)
            mae = error.abs().mean((0, 1)) / denominator
            rmse = error.square().mean((0, 1)).sqrt() / denominator
            metrics[str(h)] = {task: {'nmae': float(mae[q]), 'nrmse': float(rmse[q])} for q, task in enumerate(TASKS)}
            if save:
                save_split = a.evaluation_split if a.evaluation_split and split == 'val' else split
                np.savez_compressed(a.output / f'{save_split}_h{h}.npz', prediction_raw=prediction.cpu().numpy(),
                                    target_raw=raw[h][split]['raw'], forecast_start=raw[h][split]['forecast_start'],
                                    node_id=raw[h][split]['node_id'])
        return metrics

    if a.evaluation_split:
        metrics = evaluate(save=True)
        (a.output / 'complete.json').write_text(json.dumps({
            'evaluation_only': True, 'split': a.evaluation_split,
            'checkpoint': str(a.initialize), 'metrics': metrics,
            'test_evaluated': a.evaluation_split == 'test',
            'optimizer_steps': 0}, indent=2))
        print('EVALUATION_COMPLETE', a.evaluation_split, flush=True)
        return

    if a.reference_audit:
        assert a.initialize
        model.eval()
        results = {}
        original_issued = model.issued_residual
        with torch.no_grad():
            for label in ('normal', 'without_residual', 'issued_with_residual', 'issued_only', 'initial_reference'):
                model.issued_residual = label in ('issued_with_residual', 'issued_only') or original_issued
                handles = []
                if label in ('without_residual', 'issued_only', 'initial_reference'):
                    handles.append(model.output.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output)))
                if label == 'initial_reference':
                    model.issued_residual = False
                    handles.append(model.reference_weights.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output)))
                results[label] = {split: evaluate(split=split) for split in ('train', 'val')}
                for handle in handles:
                    handle.remove()
        model.issued_residual = original_issued
        for h in ('2', '24', '168'):
            for task in TASKS:
                for metric in ('nmae', 'nrmse'):
                    np.testing.assert_allclose(results['normal']['val'][h][task][metric],
                        previous['validation'][h][task][metric], rtol=1e-5, atol=1e-7)
        result = {'scope': 'fixed checkpoint component diagnostic on training and validation; no parameter update or test evaluation',
                  'checkpoint': str(a.initialize), 'coherent_output': a.coherent_output,
                  'normal_reproduced': True, 'results': results}
        (a.output / 'reference_audit.json').write_text(json.dumps(result, indent=2))
        print('REFERENCE_AUDIT_COMPLETE', str(a.output), flush=True)
        return

    if a.gradient_audit:
        assert a.initialize
        model.eval()
        parameters = [parameter for name, parameter in model.named_parameters()
                      if parameter.requires_grad and not name.startswith('error_auxiliary')]
        labels = [f'{task}_h{h}' for h in (2, 24, 168) for task in TASKS]
        rng = np.random.default_rng(a.seed)
        similarities, magnitudes = [], []
        for _ in range(8):
            vectors = []
            for h in (2, 24, 168):
                rows = data[h]['train']
                ids = torch.as_tensor(rng.choice(len(rows['x']), size=a.batch, replace=False), device='cuda')
                error = (predict(rows, h, ids) - rows['raw'][ids]) / denominators[h]
                losses = error.abs().mean((0, 1)) + .5 * (error.square().mean((0, 1)) + 1e-8).sqrt()
                for qi in range(4):
                    gradients = torch.autograd.grad(losses[qi], parameters, retain_graph=qi < 3, allow_unused=True)
                    vectors.append(torch.cat([torch.zeros_like(parameter).flatten() if gradient is None
                                              else gradient.float().flatten()
                                              for parameter, gradient in zip(parameters, gradients)]))
            vectors = torch.stack(vectors)
            norms_gradient = vectors.norm(dim=1)
            normalized = torch.nn.functional.normalize(vectors, dim=1)
            similarities.append((normalized @ normalized.T).cpu().numpy())
            magnitudes.append(norms_gradient.cpu().numpy())
        cosine = np.stack(similarities)
        np.testing.assert_allclose(cosine, cosine.transpose(0, 2, 1), atol=1e-6)
        assert np.isfinite(cosine).all()
        result = {'scope': 'training minibatches only; fixed checkpoint; dropout disabled; no parameter updates',
                  'checkpoint': str(a.initialize), 'cell_order': labels, 'batches': 8, 'batch_size': a.batch,
                  'sampling': 'independent uniform training windows for each horizon; identical four-task batch within a horizon',
                  'mean_cosine': cosine.mean(0).tolist(), 'negative_fraction': (cosine < 0).mean(0).tolist(),
                  'mean_gradient_norm': np.stack(magnitudes).mean(0).tolist()}
        (a.output / 'gradient_audit.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return

    if a.evaluate_only:
        assert a.initialize and a.mode != 'no_context'
        interventions = ('zero', 'semantic_zero', 'shuffle', 'global_semantic_mismatch', 'past_semantic_mismatch', 'source_identity_mismatch', 'without_region', 'without_calendar',
                         'without_history', 'without_issued')
        if a.feedback_root:
            interventions += ('without_feedback',)
        if a.event_root:
            interventions += ('without_events',)
        if a.weather_root:
            interventions += ('without_weather',)
        result = {'scope': 'fixed checkpoint, fixed validation, common numerical inputs unchanged',
                  'checkpoint': str(a.initialize), 'epoch': previous['epoch'], 'test_evaluated': False,
                  'past_semantic_mismatch': 'preceding validation origin in the same region, source slots unchanged; first regional origin retained; not a forecast-vintage comparison',
                  'normal': evaluate(), 'interventions': {kind: evaluate(kind) for kind in interventions}}
        result['semantic_id_change_fraction'] = {}
        for h in (2, 24, 168):
            ids = data[h]['val']['source_ids']
            batch_mismatch = torch.cat([ids[i:i + a.batch].roll(1, 0) for i in range(0, len(ids), a.batch)])
            global_mismatch = ids.roll(len(ids) // 2, 0)
            result['semantic_id_change_fraction'][str(h)] = {
                'past_origin_by_source': (data[h]['val']['past_source_ids'] != ids).float().mean(0).tolist(),
                'batch_shuffle_by_source': (batch_mismatch != ids).float().mean(0).tolist(),
                'global_mismatch_by_source': (global_mismatch != ids).float().mean(0).tolist()}
        (a.output / 'interventions.json').write_text(json.dumps(result, indent=2))
        if a.audit_attention:
            attention = {}
            with torch.no_grad():
                for h in (2, 24, 168):
                    rows = data[h]['val']
                    summaries = {}
                    for start in range(0, len(rows['x']), a.batch):
                        inputs = {k: v[start:start + a.batch] for k, v in rows.items()
                                  if k not in ('raw', 'scale', 'offset', 'y')}
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            _, weights = model(inputs, h, return_weights=True)
                        weights = weights.float()
                        active = (inputs['source_valid'][..., -model.event_slots:].sum(-1) > 0
                                  if model.event_slots else torch.zeros(weights.shape[:2], device='cuda', dtype=torch.bool))
                        for name, mask in [('all', torch.ones_like(active)), ('active_event', active), ('no_active_event', ~active)]:
                            selected = weights[mask]
                            if not len(selected):
                                continue
                            entropy = -(selected * selected.clamp_min(1e-8).log()).sum(-1)
                            record = summaries.setdefault(name, {'count': 0, 'weights': torch.zeros_like(selected[0]),
                                                                  'entropy': torch.zeros_like(entropy[0])})
                            record['count'] += len(selected)
                            record['weights'] += selected.sum(0)
                            record['entropy'] += entropy.sum(0)
                    attention[str(h)] = {name: {'target_steps': r['count'],
                        'mean_weights_by_task_source': (r['weights'] / r['count']).cpu().tolist(),
                        'mean_entropy_by_task': (r['entropy'] / r['count']).cpu().tolist()}
                        for name, r in summaries.items()}
            (a.output / 'attention.json').write_text(json.dumps({'task_order': TASKS,
                'source_order': ['region', 'calendar', *[f'history_{t}' for t in TASKS], *[f'issued_{t}' for t in TASKS],
                                 *([f'feedback_{t}' for t in TASKS] if a.feedback_root else []),
                                 *([f'weather_{i}' for i in range(9)] if a.weather_root else []),
                                 *[f'event_slot_{i}' for i in range(model.event_slots)], 'null'],
                'scope': 'fixed validation and checkpoint, diagnostics without retraining', 'results': attention}, indent=2))
        print(json.dumps(result), flush=True)
        return

    if a.smoke:
        if a.initialize and (a.branch_residual or a.mode == 'no_context'):
            baseline = FRAME(**previous['model_args'], qwen_table=qwen, fields_table=fields).cuda().eval()
            baseline.load_state_dict(previous['model'])
            model.eval()
            for h in (2, 24, 168):
                rows = data[h]['train']
                normal = predict(rows, h, slice(0, 4)).detach()
                disabled = predict(rows, h, slice(0, 4), 'zero').detach()
                assert torch.equal(normal, disabled)
                inputs = {k: v[:4] for k, v in rows.items() if k not in ('raw', 'scale', 'offset', 'y')}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    expected = baseline(inputs, h).float() * rows['scale'][:4] + rows['offset'][:4]
                expected = torch.cat((expected[..., :3].clamp_min(0), expected[..., 3:]), -1)
                assert torch.equal(normal, expected)
            del baseline
        model.train()
        losses = []
        for h in (2, 24, 168):
            rows = data[h]['train']
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                if a.aux_labels:
                    prediction, auxiliary = predict(rows, h, slice(0, 4), return_aux=True)
                    assert auxiliary.shape == (4, 4, 2)
                else:
                    prediction = predict(rows, h, slice(0, 4))
                error = (prediction - rows['raw'][:4]) / denominators[h]
                loss = error.abs().mean() + error.square().mean()
                if a.residual_regularization:
                    loss = loss + residual_penalty(rows, h, slice(0, 4))
                if a.aux_labels:
                    loss = loss + a.aux_weight * torch.nn.functional.smooth_l1_loss(
                        auxiliary, auxiliary_labels[h]['target'][:4])
                loss.backward()
                assert torch.isfinite(loss)
                optimizer.step()
            losses.append(float(loss.detach()))
        if a.mode != 'no_context':
            assert model.source_proj[0].weight.grad.abs().sum() > 0
            if a.semantic_routing_only:
                assert model.source_value[0].weight.grad.abs().sum() > 0
            if a.hierarchical_query and a.mode not in ('shared_weights', 'shared_structured'):
                assert model.query_residual.weight.grad.abs().sum() > 0
            if a.conditioned_query and a.mode in ('full', 'structured'):
                assert model.query_condition.weight.grad.abs().sum() > 0
            if a.task_semantic_prior:
                assert model.semantic_prior_strength.grad.abs() > 0
            if a.affine_adapter and a.adapter_ablation != 'no_adapter':
                if a.context_placement == 'internal':
                    assert all(block.context_shift.weight.grad.abs().sum() > 0 for block in model.blocks)
                else:
                    assert all(block.context_shift.weight.grad is None for block in model.blocks)
                    assert model.output_context[0].weight.grad.abs().sum() > 0
        if a.aux_labels:
            assert model.error_auxiliary[-1].weight.grad.abs().sum() > 0
        if a.error_control:
            assert model.error_control_projection.weight.grad.abs().sum() > 0
        if a.event_root:
            assert model.event_numeric[0].weight.grad.abs().sum() > 0
        if a.recent_history:
            assert model.recent_proj.weight.grad.abs().sum() > 0
        if a.native_history:
            assert model.encoder.weight_ih_l0.grad.abs().sum() > 0
        if a.origin_bridge:
            assert model.origin_bridge.weight.grad.abs().sum() > 0
            assert model.bridge_log_decay.grad.abs().sum() > 0
        if a.adaptive_residual_scale:
            assert model.residual_scale_gate.weight.grad.abs().sum() > 0
        if a.local_dynamics:
            assert model.dynamics.weight_ih_l0.grad.abs().sum() > 0
        if a.ordered_history:
            assert model.ordered_history_projection.weight.grad.abs().sum() > 0
        if a.future_trajectory:
            assert model.future_strength.grad.abs() > 0
            assert model.future_trajectory.self_attn.in_proj_weight.grad.abs().sum() > 0
        if a.origin_anchored_reference:
            assert model.reference_anchor.weight.grad.abs().sum() > 0
            assert model.reference_log_decay.grad.abs().sum() > 0
        if a.relative_readout:
            assert model.relative_gain.weight.grad.abs().sum() > 0
        if a.weather_root:
            assert model.weather_proj[0].weight.grad.abs().sum() > 0
        if a.lead_weather_context and a.mode != 'no_context':
            assert model.lead_weather_value[0].weight.grad.abs().sum() > 0
        if a.task_adapter:
            assert all(block.task_up.grad.abs().sum() > 0 and block.task_down.grad.abs().sum() > 0
                       for block in model.blocks)
        if a.trajectory_mixer:
            assert all(layer.local.weight.grad.abs().sum() > 0 and layer.wide.weight.grad.abs().sum() > 0
                       for layer in model.trajectory_mixers)
        model.eval()
        before = predict(data[2]['train'], 2, slice(0, 4)).detach()
        if a.semantic_routing_only and a.mode != 'no_context':
            values = []
            handle = model.source_value.register_forward_hook(lambda module, inputs, output: values.append(output.detach().clone()))
            predict(data[2]['train'], 2, slice(0, 4))
            predict(data[2]['train'], 2, slice(0, 4), 'global_semantic_mismatch')
            handle.remove()
            assert torch.equal(values[0], values[1])
        if a.exclude_context_sources and a.mode != 'no_context':
            inputs = {k: v[:4] for k, v in data[2]['train'].items()
                      if k not in ('raw', 'scale', 'offset', 'y')}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, weights = model(inputs, 2, return_weights=True)
            slices = {'region': slice(0, 1), 'calendar': slice(1, 2), 'history': slice(2, 6),
                      'issued': slice(6, 10), 'feedback': slice(10, 14),
                      'weather': slice(10 + 4 * bool(a.feedback_root), 19 + 4 * bool(a.feedback_root)),
                      'events': slice(-model.event_slots - 1, -1)}
            for source in a.exclude_context_sources:
                assert torch.count_nonzero(weights[..., slices[source]]) == 0
        if a.event_root:
            numerical_states = []
            handle = model.event_numeric.register_forward_hook(lambda module, inputs, output: numerical_states.append(output.detach().clone()))
            predict(data[2]['train'], 2, slice(0, 4))
            mismatched = predict(data[2]['train'], 2, slice(0, 4), 'global_semantic_mismatch')
            handle.remove()
            assert torch.equal(numerical_states[0], numerical_states[1])
            if a.mode == 'no_context':
                assert torch.equal(before, mismatched)
        if a.coherent_output:
            assert torch.equal(before[..., 3], before[..., 0] - before[..., 1] - before[..., 2])
        if a.weather_root:
            weather_states = []
            handle = model.weather_proj.register_forward_hook(
                lambda module, inputs, output: weather_states.append(output.detach().clone()))
            for intervention in (None, 'zero', 'semantic_zero', 'source_identity_mismatch', 'without_weather'):
                predict(data[2]['train'], 2, slice(0, 4), intervention)
            handle.remove()
            assert all(torch.equal(weather_states[0], value) for value in weather_states[1:])
            if a.mode != 'no_context':
                inputs = {k: v[:4] for k, v in data[2]['train'].items()
                          if k not in ('raw', 'scale', 'offset', 'y')}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, weights = model(inputs, 2, 'without_weather', return_weights=True)
                start = 10 + 4 * bool(a.feedback_root)
                assert torch.count_nonzero(weights[..., start:start + 9]) == 0
        torch.save(model.state_dict(), a.output / 'smoke.pt')
        model.load_state_dict(torch.load(a.output / 'smoke.pt', weights_only=True))
        assert torch.equal(before, predict(data[2]['train'], 2, slice(0, 4)))
        (a.output / 'smoke.json').write_text(json.dumps({'losses': losses, 'gpu': torch.cuda.get_device_name(),
            'positive_context_gradient': a.mode != 'no_context', 'three_horizon_shared_checkpoint': True,
            'initialization_identity_verified': bool(a.initialize and a.branch_residual)}, indent=2))
        print('GPU_SMOKE_PASSED', losses, flush=True)
        return

    print('TRAINING', a.mode, torch.cuda.get_device_name(), {h: len(v['train']['x']) for h, v in data.items()},
          'parameters', protocol['parameters'], 'setup_seconds', time.monotonic() - started, flush=True)
    best = float('inf')
    stale_epochs = 0
    def save_checkpoint(epoch, score, metrics):
        torch.save({'model': model.state_dict(), 'model_args': model_args, 'epoch': epoch, 'score': score,
                    'validation': metrics, 'normalization': norms, 'thresholds': thresholds,
                    'coherent_output': a.coherent_output}, a.output / 'best.pt')
    if a.initialize:
        initial = evaluate()
        best = float(np.mean([v['nmae'] + v['nrmse'] for cells in initial.values() for v in cells.values()]))
        record = {'epoch': 0, 'score': best, 'validation': initial, 'seconds': 0,
                  'stage': 'shared numerical initialization'}
        (a.output / 'training.jsonl').write_text(json.dumps(record) + '\n')
        save_checkpoint(0, best, initial)
        print('INITIALIZATION', best, flush=True)
    steps = max(math.ceil(len(v['train']['x']) / a.batch) for v in data.values())
    averaged = (torch.optim.swa_utils.AveragedModel(model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema_decay)) if a.ema_decay else None)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for epoch in range(1, a.epochs + 1):
        begin = time.monotonic()
        model.train()
        total = 0.
        permutations = {h: torch.cat([torch.randperm(len(v['train']['x']), device='cuda')
            for _ in range(math.ceil(steps * a.batch / len(v['train']['x'])))])[:steps * a.batch]
                        for h, v in data.items()}
        for step in range(steps):
            for h in (2, 24, 168):
                rows = data[h]['train']
                ids = permutations[h][step * a.batch:(step + 1) * a.batch]
                optimizer.zero_grad(set_to_none=True)
                if a.aux_labels:
                    prediction, auxiliary = predict(rows, h, ids, return_aux=True)
                else:
                    prediction = predict(rows, h, ids)
                loss_scale = rows['scale'][ids] if a.loss_scale == 'regional_standard' else denominators[h]
                error = (prediction - rows['raw'][ids]) / loss_scale
                task_loss = point_task_error(error, a.rmse_weight)
                if a.recency_half_life:
                    task_loss = weighted_task_error(error, recency[h][ids], a.rmse_weight)
                if a.reference_balanced_loss:
                    task_loss = task_loss * loss_weights[h]
                if a.reference_balanced_loss or a.task_gradients != 'mean' or a.recency_half_life:
                    loss = task_loss.mean()
                else:
                    loss = error.abs().mean() + a.rmse_weight * (error.square().mean((0, 1)) + 1e-8).sqrt().mean()
                auxiliary_loss = residual_penalty(rows, h, ids) if a.residual_regularization else None
                if a.aux_labels:
                    available = auxiliary_labels[h]['available'][ids]
                    if available.any():
                        label_loss = a.aux_weight * torch.nn.functional.smooth_l1_loss(
                            auxiliary[available], auxiliary_labels[h]['target'][ids][available])
                        auxiliary_loss = label_loss if auxiliary_loss is None else auxiliary_loss + label_loss
                if auxiliary_loss is not None:
                    loss = loss + auxiliary_loss
                if a.task_gradients == 'mean':
                    loss.backward()
                else:
                    projected_backward(task_loss, trainable_parameters, auxiliary_loss,
                                       normalize=a.task_gradients == 'balanced_project')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                if averaged is not None:
                    averaged.update_parameters(model)
                total += float(loss.detach())
        scheduler.step()
        training_model = model
        if averaged is not None:
            model = averaged.module
        metrics = evaluate()
        score = float(np.mean([v['nmae'] + v['nrmse'] for cells in metrics.values() for v in cells.values()]))
        record = {'epoch': epoch, 'loss': total / (steps * 3), 'score': score, 'validation': metrics,
                  'seconds': time.monotonic() - begin, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2**30}
        with (a.output / 'training.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if score < best:
            best = score
            save_checkpoint(epoch, score, metrics)
            stale_epochs = 0
        else:
            stale_epochs += 1
        model = training_model
        if a.patience and stale_epochs >= a.patience:
            print('EARLY_STOP', epoch, flush=True)
            break
    checkpoint = torch.load(a.output / 'best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    metrics = evaluate(save=True)
    interventions = {kind: evaluate(kind) for kind in ('zero', 'shuffle')} if a.mode != 'no_context' else {}
    complete = {'best_epoch': checkpoint['epoch'], 'epochs_completed': epoch, 'validation': metrics, 'interventions': interventions,
                'test_evaluated': False, 'seconds': time.monotonic() - started}
    if a.oof:
        complete['oof_training_period'] = evaluate(save=True, split='oof')
    (a.output / 'complete.json').write_text(json.dumps(complete, indent=2))
    print('COMPLETE', a.mode, checkpoint['epoch'], checkpoint['score'], flush=True)


if __name__ == '__main__':
    main()
