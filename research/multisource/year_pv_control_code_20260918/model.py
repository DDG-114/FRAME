import math

import torch
from torch import nn
from torch.nn import functional as F


def scale_pv_control(control, hours, scale):
    if hours == 2 or scale == 1.:
        return control
    adjusted = control.clone()
    adjusted[:, :, 2] *= scale
    return adjusted


class InternalAdapter(nn.Module):
    def __init__(self, width, rank=12, branch_residual=False, centered_semantic=False, affine_adapter=False,
                 adapter_after_ff=False, task_residual=False):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.control = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, rank))
        self.branch_residual = branch_residual
        self.centered_semantic = centered_semantic
        self.adapter_after_ff = adapter_after_ff
        if branch_residual:
            self.semantic_control = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, rank))
            nn.init.zeros_(self.semantic_control[-1].weight)
            nn.init.zeros_(self.semantic_control[-1].bias)
            self.semantic_strength = nn.Parameter(torch.tensor(-2.2))
        self.up = nn.Linear(rank, width, bias=False)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
                                nn.Dropout(.1), nn.Linear(width * 2, width))
        nn.init.zeros_(self.up.weight)
        self.context_shift = nn.Linear(width, rank, bias=False) if affine_adapter else None
        self.task_up = None
        if task_residual:
            with torch.random.fork_rng(devices=[]):
                self.task_down = nn.Parameter(torch.empty(4, width, rank))
                nn.init.xavier_uniform_(self.task_down)
                self.task_up = nn.Parameter(torch.zeros(4, rank, width))

    def forward(self, state, context, use_context=True):
        if getattr(self, 'ablation', 'none') == 'no_adapter':
            return state + self.ff(state)
        normalized = self.norm(state)
        if self.branch_residual:
            gate = self.control(torch.cat((normalized, torch.zeros_like(context)), -1)).tanh()
            if use_context:
                semantic = self.semantic_control(torch.cat((normalized, context), -1))
                if self.centered_semantic:
                    semantic = semantic - self.semantic_control(torch.cat((normalized, torch.zeros_like(context)), -1))
                gate = gate + self.semantic_strength.sigmoid() * semantic.tanh()
        else:
            gate = self.control(torch.cat((normalized, context), -1)).tanh()
        if getattr(self, 'ablation', 'none') == 'static_gate':
            gate = self.control(torch.zeros_like(torch.cat((normalized, context), -1))).tanh()
        residual = self.down(normalized) * gate
        if self.context_shift is not None and use_context:
            residual = residual + self.context_shift(context).tanh()
        correction = self.up(residual)
        if self.task_up is not None:
            task_residual = torch.einsum('bhqd,qdr->bhqr', normalized, self.task_down) * gate
            if self.context_shift is not None and use_context:
                task_residual = task_residual + self.context_shift(context).tanh()
            correction = correction + torch.einsum('bhqr,qrd->bhqd', task_residual, self.task_up)
        if self.adapter_after_ff:
            return state + self.ff(state) + correction
        state = state + correction
        return state + self.ff(state)


class TrajectoryMixer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.local = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.wide = nn.Conv1d(width, width, 7, padding=6, dilation=2, groups=width)
        self.project = nn.Sequential(nn.Linear(width * 2, width * 2), nn.GLU(), nn.Linear(width, width))
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, state):
        batch, horizon, tasks, width = state.shape
        series = self.norm(state).permute(0, 2, 3, 1).reshape(batch * tasks, width, horizon)
        features = torch.cat((self.local(series), self.wide(series)), 1)
        features = features.reshape(batch, tasks, width * 2, horizon).permute(0, 3, 1, 2)
        return state + self.project(features)


class FRAME(nn.Module):
    def __init__(self, static_dim, qwen_table, fields_table, mode, width=96, reference_state=False, feedback_state=False,
                 branch_residual=False, centered_semantic=False, auxiliary=False,
                 error_control=False, context_placement='internal', event_slots=0, affine_adapter=False,
                 context_dropout=0., recent_history=False, exclude_context_sources=(), semantic_routing_only=False,
                 hierarchical_query=False, conditioned_query=False, task_source_ids=None, trajectory_mixer=False,
                 weather_state=False, adapter_after_ff=False, issued_residual=False, local_dynamics=False,
                 ordered_history=False, task_readout=False, future_trajectory=False, residual_scale=None,
                 origin_anchored_reference=False, lead_weather_context=False, task_adapter=False,
                 relative_readout=False, native_history=False, origin_bridge=False, task_history=False,
                 adaptive_residual_scale=False, adapter_ablation='none', pv_long_context_scale=1.):
        super().__init__()
        self.mode = mode
        self.pv_long_context_scale = pv_long_context_scale
        self.width = width
        self.reference_state = reference_state
        self.feedback_state = feedback_state
        self.error_control = error_control
        self.context_placement = context_placement
        self.event_slots = event_slots
        assert 0 <= context_dropout < 1
        self.context_dropout = context_dropout
        self.recent_history = recent_history
        self.native_history = native_history
        self.task_history = task_history
        assert not (native_history and (recent_history or ordered_history))
        self.exclude_context_sources = tuple(exclude_context_sources)
        self.semantic_routing_only = semantic_routing_only
        self.hierarchical_query = hierarchical_query
        self.conditioned_query = conditioned_query
        self.weather_state = weather_state
        self.issued_residual = issued_residual
        self.task_readout = task_readout
        self.register_buffer('residual_scale', torch.tensor(residual_scale, dtype=torch.float32) if residual_scale is not None else None)
        assert not (hierarchical_query and conditioned_query)
        self.register_buffer('qwen_table', qwen_table)
        self.register_buffer('fields_table', fields_table)
        self.history_proj = nn.Linear(20, width)
        self.history_time = nn.Linear(6, width)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.static = nn.Sequential(nn.Linear(static_dim, width), nn.GELU(), nn.Linear(width, width))
        self.region = nn.Embedding(5, width)
        self.future_proj = nn.Linear(14, width)
        self.future_time = nn.Linear(6, width)
        if weather_state:
            self.weather_proj = nn.Sequential(nn.Linear(64, width), nn.GELU(), nn.Linear(width, width))
        self.attend = nn.MultiheadAttention(width, 4, dropout=.1, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.task = nn.Parameter(torch.randn(4, width) * .02)
        self.local = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
        if reference_state:
            self.task_numeric = nn.Sequential(nn.Linear(15 if feedback_state else 12, width), nn.GELU(), nn.Linear(width, width))
        self.source_proj = nn.Sequential(nn.Linear(134, width), nn.GELU(), nn.LayerNorm(width))
        self.select_query = nn.Linear(width, width, bias=False)
        self.select_key = nn.Linear(width, width, bias=False)
        self.null_source = nn.Parameter(torch.zeros(1, 1, width))
        self.shared_query = nn.Sequential(nn.Linear(width, width), nn.Tanh())
        self.blocks = nn.ModuleList([InternalAdapter(width, branch_residual=branch_residual, centered_semantic=centered_semantic,
                                                     affine_adapter=affine_adapter,
                                                     adapter_after_ff=adapter_after_ff, task_residual=task_adapter) for _ in range(2)])
        for block in self.blocks:
            block.ablation = adapter_ablation
        self.reference_weights = nn.Linear(width, 4)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.reference_weights.weight)
        nn.init.zeros_(self.reference_weights.bias)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        if auxiliary:
            self.error_auxiliary = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width // 2),
                                                 nn.GELU(), nn.Linear(width // 2, 2))
        if error_control:
            assert auxiliary
            self.error_control_projection = nn.Linear(2, width)
        if context_placement == 'output':
            self.output_context = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
            nn.init.zeros_(self.output_context[-1].weight)
            nn.init.zeros_(self.output_context[-1].bias)
        if event_slots:
            self.event_numeric = nn.Sequential(nn.Linear(137, width), nn.GELU(), nn.Linear(width, width))
        if recent_history:
            self.recent_proj = nn.Linear(8, width)
        self.dynamics = nn.GRU(12, width, batch_first=True) if local_dynamics else None
        if local_dynamics:
            self.dynamics_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        if semantic_routing_only:
            self.source_value = nn.Sequential(nn.Linear(134, width), nn.GELU(), nn.LayerNorm(width))
        if hierarchical_query:
            self.query_residual = nn.Linear(width, width, bias=False)
            nn.init.zeros_(self.query_residual.weight)
        if conditioned_query and mode in ('full', 'structured'):
            self.query_condition = nn.Linear(width + 6, width, bias=False)
            nn.init.zeros_(self.query_condition.weight)
        self.task_source_ids = task_source_ids
        self.trajectory_mixers = nn.ModuleList([TrajectoryMixer(width) for _ in self.blocks]) if trajectory_mixer else None
        if task_source_ids is not None:
            table = fields_table if mode in ('structured', 'shared_structured') else qwen_table
            prototypes = torch.stack([table[ids].mean(0) for ids in task_source_ids])
            self.register_buffer('task_semantic_prototypes', F.normalize(prototypes, dim=-1))
            self.semantic_prior_strength = nn.Parameter(torch.tensor(1.))
        self.ordered_history_projection = nn.Linear(48, width, bias=False) if ordered_history else None
        if ordered_history:
            nn.init.zeros_(self.ordered_history_projection.weight)
        if task_readout:
            with torch.random.fork_rng(devices=[]):
                self.output[-1] = nn.Linear(width, 4)
                nn.init.zeros_(self.output[-1].weight)
                nn.init.zeros_(self.output[-1].bias)
        self.future_trajectory = None
        if future_trajectory:
            with torch.random.fork_rng(devices=[]):
                self.future_trajectory = nn.TransformerEncoderLayer(
                    width, 4, width * 2, dropout=.1, batch_first=True, norm_first=True)
            self.future_strength = nn.Parameter(torch.tensor(0.))
        self.reference_anchor = None
        if origin_anchored_reference:
            with torch.random.fork_rng(devices=[]):
                self.reference_anchor = nn.Linear(width, 1)
                nn.init.zeros_(self.reference_anchor.weight)
                nn.init.zeros_(self.reference_anchor.bias)
            self.reference_log_decay = nn.Parameter(torch.full((4,), math.log(24.)))
        self.lead_weather_value = None
        if lead_weather_context:
            assert weather_state
            with torch.random.fork_rng(devices=[]):
                self.lead_weather_value = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
                nn.init.zeros_(self.lead_weather_value[-1].weight)
                nn.init.zeros_(self.lead_weather_value[-1].bias)
        self.relative_gain = None
        if relative_readout:
            with torch.random.fork_rng(devices=[]):
                self.relative_gain = nn.Linear(width, 1)
                nn.init.zeros_(self.relative_gain.weight)
                nn.init.zeros_(self.relative_gain.bias)
        if native_history:
            with torch.random.fork_rng(devices=[]):
                self.history_proj = nn.Linear(8, width)
                self.encoder = nn.GRU(width, width, num_layers=2, dropout=.1, batch_first=True)
                self.history_norm = nn.LayerNorm(width)
        self.origin_bridge = None
        if origin_bridge:
            with torch.random.fork_rng(devices=[]):
                self.origin_bridge = nn.Linear(width, 1)
                nn.init.zeros_(self.origin_bridge.weight)
                nn.init.zeros_(self.origin_bridge.bias)
            self.bridge_log_decay = nn.Parameter(torch.full((4,), math.log(2.)))
        self.residual_scale_gate = None
        if adaptive_residual_scale:
            assert residual_scale is not None
            with torch.random.fork_rng(devices=[]):
                self.residual_scale_gate = nn.Linear(width, 1)
                nn.init.zeros_(self.residual_scale_gate.weight)
                nn.init.zeros_(self.residual_scale_gate.bias)

    def residual_multiplier(self, state, hours, region):
        scale = self.residual_scale[(2, 24, 168).index(hours), region][:, None]
        if self.residual_scale_gate is not None:
            weight = self.residual_scale_gate(state).squeeze(-1).sigmoid()
            scale = 1 + weight * (scale - 1)
        return scale

    @staticmethod
    def ordered_hour_values(bins):
        centered = bins - bins.mean(2, keepdim=True)
        return centered.repeat_interleave(12 // bins.shape[2], dim=2).flatten(2)

    @staticmethod
    def time_features(hours, cadence):
        return torch.stack((hours / 168, torch.sin(hours * math.pi / 12), torch.cos(hours * math.pi / 12),
                            torch.sin(hours * math.pi / 84), torch.cos(hours * math.pi / 84),
                            torch.full_like(hours, cadence / 30)), -1)

    def forward(self, rows, hours, intervention=None, return_weights=False, return_aux=False):
        x = rows['x']
        horizon = 24 if hours == 2 else hours * 2
        cadence = 5 if hours == 2 else 30
        history = x.shape[1] - horizon
        period = 1440 // cadence
        past, future = x[:, :history], x[:, history:]
        if self.native_history:
            historical_hours = torch.arange(1 - history, 1, device=x.device, dtype=x.dtype) * cadence / 60
            memory = self.history_proj(past[..., :8]) + self.history_time(self.time_features(historical_hours, cadence))
            memory = self.history_norm(self.encoder(memory)[0])
        else:
            bins = past[..., :4].reshape(len(x), 168, 60 // cadence, 4)
            calendar = past[..., 4:8].reshape(len(x), 168, 60 // cadence, 4)[:, :, -1]
            summary = torch.cat((bins.mean(2), bins.std(2, correction=0), bins.amin(2), bins.amax(2), calendar), -1)
            historical_hours = torch.arange(-167, 1, device=x.device, dtype=x.dtype)
            memory = self.history_proj(summary) + self.history_time(self.time_features(historical_hours, cadence))
            if self.ordered_history_projection is not None:
                memory = memory + self.ordered_history_projection(self.ordered_hour_values(bins))
            if self.recent_history:
                recent_hours = torch.arange(1 - period, 1, device=x.device, dtype=x.dtype) * cadence / 60
                recent_memory = self.recent_proj(past[:, -period:, :8]) + self.history_time(self.time_features(recent_hours, cadence))
                memory = torch.cat((memory[:, :-24], recent_memory), 1)
            memory = self.encoder(memory)
        leads = torch.arange(1, horizon + 1, device=x.device, dtype=x.dtype) * cadence / 60
        static = self.static(rows['context']) + self.region(rows['region'])
        query = self.future_proj(future[..., 4:]) + self.future_time(self.time_features(leads, cadence)) + static[:, None]
        if self.weather_state:
            query = query + self.weather_proj(rows['weather_numeric']) * rows['weather_available'].any(-1, keepdim=True)
        if self.future_trajectory is not None:
            query = query + self.future_strength.tanh() * (self.future_trajectory(query) - query)
        if self.task_history:
            task_query = query[:, :, None] + self.task[None, None]
            retrieved = self.attend(task_query.flatten(1, 2), memory, memory, need_weights=False)[0]
            state = self.norm(task_query + retrieved.reshape_as(task_query))
        else:
            state = self.norm(query + self.attend(query, memory, memory, need_weights=False)[0])
        if self.dynamics is not None:
            recent = past[:, -(360 // cadence):, :8]
            changes = torch.diff(recent[..., :4], dim=1, prepend=recent[:, :1, :4])
            _, hidden = self.dynamics(torch.cat((recent, changes), -1))
            delta = self.dynamics_projection(hidden[-1])[:, None]
            state = state + (delta[:, :, None] if self.task_history else delta)
        if self.event_slots:
            numeric_events = torch.cat((self.fields_table[rows['event_numeric_ids']],
                                        rows['source_exact'][:, -self.event_slots:]), -1)
            event_valid = rows['source_valid'][..., -self.event_slots:]
            count = event_valid.sum(-1, keepdim=True)
            pooled = torch.einsum('bhs,bsd->bhd', event_valid, numeric_events) / count.clamp_min(1)
            metadata = rows['event_meta'][:, None].expand(-1, horizon, -1)
            delta = self.event_numeric(torch.cat((pooled, count / self.event_slots, metadata), -1))
            state = state + (delta[:, :, None] if self.task_history else delta)
        local = torch.cat((past[:, -1, :4], past[:, -max(2, 60 // cadence):, :4].mean(1)), -1)
        if not self.task_history:
            state = state[:, :, None] + self.task[None, None]
        state = state + self.local(local)[:, None, None]
        steps = torch.arange(horizon, device=x.device)
        last = past[:, -1:, :4].expand(-1, horizon, -1)
        daily = past[:, history - period + steps % period, :4]
        weekly = past[:, steps % history, :4]
        issued = torch.where(future[..., 12:16] > .5, future[..., 8:12], last)
        references = torch.stack((issued, last, daily, weekly), -1)
        if self.reference_state:
            recent = past[:, -period:, :4]
            statistics = torch.stack((recent.mean(1), recent.std(1, correction=0),
                                      recent[:, -1] - recent[:, 0]), -1)
            statistics = statistics[:, None].expand(-1, horizon, -1, -1)
            differences = torch.stack((issued - last, issued - daily, issued - weekly,
                                       daily - weekly), -1)
            task_numeric = torch.cat((references, differences, statistics,
                                      future[..., 12:16, None]), -1)
            if self.feedback_state:
                feedback = torch.cat((rows['feedback'], rows['feedback_available'][..., None]), -1)
                task_numeric = torch.cat((task_numeric, feedback[:, None].expand(-1, horizon, -1, -1)), -1)
            state = state + self.task_numeric(task_numeric)

        context = torch.zeros_like(state)
        weights = None
        if self.mode != 'no_context' and intervention != 'zero':
            table = self.fields_table if self.mode in ('structured', 'shared_structured') else self.qwen_table
            ids = rows['source_ids']
            embedding = table[ids]
            if intervention == 'shuffle':
                embedding = embedding.roll(1, 0)
            elif intervention == 'source_identity_mismatch':
                embedding = embedding.roll(1, 1)
            elif intervention == 'semantic_zero':
                embedding = torch.zeros_like(embedding)
            sources = self.source_proj(torch.cat((embedding, rows['source_exact']), -1))
            sources = torch.cat((sources, self.null_source.expand(len(x), -1, -1)), 1)
            if self.semantic_routing_only:
                value_ids = rows.get('source_value_ids', rows['source_ids'])
                values = self.source_value(torch.cat((self.fields_table[value_ids], rows['source_exact']), -1))
                values = torch.cat((values, torch.zeros_like(self.null_source).expand(len(x), -1, -1)), 1)
            else:
                values = sources
            if self.mode in ('shared_weights', 'shared_structured'):
                selector = self.shared_query(memory.mean(1) + static)[:, None, None].expand_as(state)
            elif self.conditioned_query:
                shared = self.shared_query(memory.mean(1) + static)[:, None, None]
                task = self.task[None].expand(horizon, -1, -1)
                time = self.time_features(leads, cadence)[:, None].expand(-1, 4, -1)
                selector = shared + self.query_condition(torch.cat((task, time), -1))[None]
            elif self.hierarchical_query:
                shared = self.shared_query(memory.mean(1) + static)[:, None, None].expand_as(state)
                selector = shared + self.query_residual(state - shared)
            else:
                selector = state
            score = torch.einsum('bhqd,bsd->bhqs', self.select_query(selector), self.select_key(sources)) / math.sqrt(self.width)
            if self.task_source_ids is not None:
                similarity = torch.einsum('qd,bsd->bqs', self.task_semantic_prototypes,
                                          F.normalize(embedding, dim=-1))
                if self.mode in ('shared_weights', 'shared_structured'):
                    similarity = similarity.mean(1, keepdim=True).expand(-1, 4, -1)
                score = score + self.semantic_prior_strength * F.pad(similarity[:, None], (0, 1))
            mask = F.pad(rows['source_valid'], (0, 1), value=1).bool()
            source_groups = {'without_region': slice(0, 1), 'without_calendar': slice(1, 2),
                             'without_history': slice(2, 6), 'without_issued': slice(6, 10),
                             'without_feedback': slice(10, 14)}
            if self.weather_state:
                first_weather = 10 + 4 * self.feedback_state
                source_groups['without_weather'] = slice(first_weather, first_weather + 9)
            for source in self.exclude_context_sources:
                if source == 'events':
                    if self.event_slots:
                        mask[..., -self.event_slots - 1:-1] = False
                else:
                    mask[..., source_groups['without_' + source]] = False
            if intervention in source_groups:
                mask[..., source_groups[intervention]] = False
            if intervention == 'without_events' and self.event_slots:
                mask[..., -self.event_slots - 1:-1] = False
            score = score.masked_fill(~mask[:, :, None], -1e4)
            weights = score.softmax(-1)
            context = torch.einsum('bhqs,bsd->bhqd', weights, values)
            if self.lead_weather_value is not None:
                weather = rows['weather_numeric']
                local = torch.cat((weather[..., :36].reshape(len(x), horizon, 9, 4),
                                   weather[..., 36:54].reshape(len(x), horizon, 9, 2),
                                   weather[..., 54:63, None],
                                   weather[..., 63:64, None].expand(-1, -1, 9, -1)), -1)
                first_weather = 10 + 4 * self.feedback_state
                context = context + torch.einsum('bhqv,bhvd->bhqd',
                    weights[..., first_weather:first_weather + 9], self.lead_weather_value(local))
        if self.training and self.context_dropout:
            retained = torch.rand((len(x), 1, 1, 1), device=x.device) >= self.context_dropout
            context = context * retained / (1 - self.context_dropout)
        error_state = self.error_auxiliary(state + context) if return_aux or self.error_control else None
        auxiliary = error_state.mean(1) if return_aux else None
        control = self.error_control_projection(error_state) if self.error_control else context
        internal_control = control if self.context_placement == 'internal' else torch.zeros_like(control)
        internal_control = scale_pv_control(internal_control, hours, self.pv_long_context_scale)
        for index, block in enumerate(self.blocks):
            if self.trajectory_mixers is not None:
                state = self.trajectory_mixers[index](state)
            state = block(state, internal_control, use_context=self.context_placement == 'internal' and
                          (self.error_control or (self.mode != 'no_context' and intervention != 'zero')))

        prior = x.new_tensor((3., 1., 0., 0.) if hours == 2 else (4., 0., 0., 0.))
        mix = (self.reference_weights(state) + prior).softmax(-1)
        if self.reference_anchor is not None:
            decay = torch.exp(-leads[None, :, None] / self.reference_log_decay.exp()[None, None])
            strength = self.reference_anchor(state).squeeze(-1).tanh()
            issued = issued + strength * (last - issued[:, :1]) * decay
            references = torch.stack((issued, last, daily, weekly), -1)
        reference = issued if self.issued_residual else (mix * references).sum(-1)
        residual = self.output(state)
        residual = residual.diagonal(dim1=-2, dim2=-1) if self.task_readout else residual.squeeze(-1)
        if self.residual_scale is not None:
            residual = residual * self.residual_multiplier(state, hours, rows['region'])
        prediction = reference + residual
        if self.relative_gain is not None:
            prediction = prediction + reference * self.relative_gain(state).squeeze(-1).tanh()
        if self.context_placement == 'output':
            prediction = prediction + self.output_context(torch.cat((state, control), -1)).squeeze(-1)
        if self.origin_bridge is not None:
            origin = 2 * prediction[:, :1] - prediction[:, 1:2]
            strength = self.origin_bridge(state[:, :1]).squeeze(-1).tanh()
            decay = torch.exp(-leads[None, :, None] / self.bridge_log_decay.exp()[None, None])
            prediction = prediction + strength * (last[:, :1] - origin) * decay
        if return_aux:
            return prediction, auxiliary
        return (prediction, weights) if return_weights else prediction
