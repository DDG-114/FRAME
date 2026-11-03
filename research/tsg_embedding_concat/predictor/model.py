"""Paper equations: patch backbone, joint-state controller, four layer adapters."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('dual', 'mlp_schema', 'mlp_schema_concat', 'frozen_random', 'numeric_state', 'semantic_state',
         'concat_only', 'none', 'fixed', 'task_specific', 'rank_only',
         'strength_only', 'no_availability')


@dataclass(frozen=True)
class Config:
    width: int = 128
    channels: int = 30
    patch: int = 16
    stride: int = 8
    layers: int = 4
    heads: int = 4
    rank: int = 8
    semantic_width: int = 4096
    future_fields: int = 6
    dropout: float = .1
    mode: str = 'dual'

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(self.mode)


class TemporalBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.width
        self.norm_self, self.norm_cross, self.norm_ffn = (nn.LayerNorm(d) for _ in range(3))
        self.self_attention = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(c.dropout), nn.Linear(4*d, d))
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, hidden, known):
        x = self.norm_self(hidden)
        hidden = hidden + self.dropout(self.self_attention(x, x, x, need_weights=False)[0])
        hidden = hidden + self.dropout(self.cross_attention(self.norm_cross(hidden), known, known, need_weights=False)[0])
        return hidden + self.dropout(self.ffn(self.norm_ffn(hidden)))


class Adapter(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.A = nn.Parameter(torch.empty(c.width, c.rank))
        self.B = nn.Parameter(torch.zeros(c.rank, c.width))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.rank_gate = nn.Linear(c.width, c.rank, bias=False)
        self.strength = nn.Linear(c.width, 1)
        nn.init.zeros_(self.strength.weight)
        nn.init.constant_(self.strength.bias, -2.)
        if c.mode == 'task_specific':
            self.A = nn.Parameter(self.A.detach()[None].repeat(4, 1, 1))
            self.B = nn.Parameter(self.B.detach()[None].repeat(4, 1, 1))

    def forward(self, hidden, code, mode, task_id=None):
        gamma = torch.tanh(self.rank_gate(code))
        alpha = torch.sigmoid(self.strength(code))
        if mode in ('fixed', 'none', 'strength_only'):
            gamma = torch.ones_like(gamma)
        if mode in ('fixed', 'rank_only', 'none'):
            alpha = torch.ones_like(alpha)
        if mode == 'task_specific':
            candidate = torch.bmm(torch.bmm(hidden, self.A[task_id]) * gamma[:, None], self.B[task_id])
        else:
            candidate = ((hidden @ self.A) * gamma[:, None]) @ self.B
        if mode == 'none':
            candidate = torch.zeros_like(candidate)
        applied = alpha[:, None] * candidate
        # The manuscript regularizes the candidate BEFORE the strength gate.
        penalty = candidate.float().square().mean((1, 2))
        return hidden + applied, penalty, gamma, alpha, candidate, applied


class PaperForecaster(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        c, d = config, config.width
        self.patch_projection = nn.Linear(2*c.channels*c.patch, d, bias=False)
        self.stem_norm = nn.LayerNorm(d)
        self.position = nn.Embedding(256, d)
        self.task = nn.Embedding(4, d)
        self.cadence = nn.Embedding(3, d)
        self.known_projection = nn.Linear(2*c.future_fields+10, d, bias=False)
        self.semantic_projection = nn.Linear(c.semantic_width, d, bias=False)
        self.semantic_norm = nn.LayerNorm(d)
        self.pool_query = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.pool_query, std=.02)
        self.availability_projection = nn.Linear(10, d, bias=False)
        self.W1 = nn.Linear(3*d, d, bias=False)
        self.W2 = nn.Linear(d, d, bias=False)
        self.W3 = nn.Linear(d, d, bias=False)
        self.W4 = nn.Linear(2*d, d, bias=False)
        self.code_norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(TemporalBlock(c) for _ in range(c.layers))
        self.adapters = nn.ModuleList(Adapter(c) for _ in range(c.layers))
        self.decoder = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.output = nn.Linear(d, 1)
        if c.mode in ('mlp_schema', 'mlp_schema_concat'):
            # Preserve all common numerical initialization and the subsequent RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.schema_output = nn.Linear(d, d, bias=False)

    def forward(self, batch):
        c, d = self.config, self.config.width
        history, mask = batch['history'], batch['history_mask'].bool()
        values = torch.where(mask, history, torch.zeros_like(history))
        # [B,N,P,2C], flattening preserves the order in vec([X,M]).
        patches = torch.cat((values, mask.to(values.dtype)), -1).unfold(1, c.patch, c.stride)
        patches = patches.permute(0, 1, 3, 2).flatten(2)
        n = patches.shape[1]
        task = self.task(batch['task_id'])[:, None]
        cadence = self.cadence(batch['cadence_id'])[:, None]
        stem = self.stem_norm(self.patch_projection(patches))
        stem = stem + self.position(torch.arange(n, device=history.device))[None] + task + cadence
        availability = batch['availability'].to(stem.dtype)
        if c.mode == 'no_availability':
            availability = torch.zeros_like(availability)
        known_values = torch.where(batch['future_mask'], batch['future'], torch.zeros_like(batch['future']))
        horizon = known_values.shape[1]
        known = self.known_projection(torch.cat((known_values, batch['future_mask'].to(known_values.dtype),
            availability[:, None].expand(-1, horizon, -1)), -1))
        known = known + self.position(torch.arange(horizon, device=history.device))[None] + task + cadence
        # Frozen encoder features are detached; their projection remains trainable.
        semantic = self.semantic_projection(batch['semantic'].detach())
        if c.mode in ('mlp_schema', 'mlp_schema_concat'):
            semantic = self.schema_output(F.silu(semantic))
        semantic = self.semantic_norm(semantic)
        weights = (stem @ self.pool_query / math.sqrt(d)).softmax(1)
        pooled = (stem * weights[..., None]).sum(1)
        if c.mode == 'numeric_state':
            semantic = torch.zeros_like(semantic)
        if c.mode == 'semantic_state':
            pooled = torch.zeros_like(pooled)
        v = self.availability_projection(availability)
        concatenate = F.silu(self.W1(torch.cat((semantic, pooled, v), -1)))
        interaction = self.W2(semantic) * self.W3(pooled)
        if c.mode in ('concat_only', 'mlp_schema_concat'):
            interaction = torch.zeros_like(interaction)
        code = self.code_norm(self.W4(torch.cat((concatenate, interaction), -1)))
        hidden, penalties, diagnostics = stem, [], []
        for block, adapter in zip(self.blocks, self.adapters):
            hidden = block(hidden, known)
            hidden, penalty, gamma, alpha, candidate, applied = adapter(hidden, code, c.mode, batch['task_id'])
            penalties.append(penalty)
            diagnostics.append(dict(gamma=gamma, alpha=alpha, candidate=candidate, applied=applied))
        query = self.position(torch.arange(horizon, device=history.device))[None] + task + cadence
        query = query.expand(len(history), -1, -1)
        decoded = query + self.decoder(query, hidden, hidden, need_weights=False)[0]
        prediction = self.output(decoded).float()
        return dict(prediction=prediction, residual_penalty_per_sample=torch.stack(penalties).sum(0),
                    diagnostics=diagnostics, code=code, pool_weights=weights, patch_count=n)

    def configuration(self):
        return asdict(self.config)


def objective(result, target, mask, target_scale, kappa):
    valid = mask.bool()
    counts = valid.sum((1, 2))
    if bool((counts == 0).any()):
        raise ValueError('A training window has no valid targets')
    error = (torch.where(valid, result['prediction'], 0.) - torch.where(valid, target, 0.))
    error = error * target_scale / kappa
    point = F.huber_loss(error, torch.zeros_like(error), reduction='none', delta=1.)
    prediction_loss = ((point * valid).sum((1, 2)) / counts).mean()
    energy = result['residual_penalty_per_sample'].mean()
    return prediction_loss + 1e-4*energy, prediction_loss, energy


def masked_token_mean(hidden, attention_mask):
    weights = attention_mask[..., None].to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
