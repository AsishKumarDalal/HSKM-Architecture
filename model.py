"""
Hierarchical Sparse Kernel Memory (HSKM) Architecture - V3.1 (Production)
--------------------------------------------------------------------------
Changes from V3:
  - RoPE on queries for positional awareness.
  - Optional causal gate on kernels (default OFF).
  - Cleaned up LTM write logic with EMA-style updates.
  - Gradient checkpointing added for memory efficiency.
  - Sparse top_k clamped to min(top_k, n_kernels).
  - Kernel init fallback from orthogonal to xavier_uniform.
  - Device/dtype consistency enforced throughout.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from dataclasses import dataclass, field
from typing import Optional, Tuple
from torch import Tensor


# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

@dataclass
class HSKMConfig:
    vocab_size: int = 50257
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_medium: int = 128
    n_kernels: int = 64
    top_k: int = 16
    window: int = 512
    n_patterns: int = 8192
    mtm_decay: float = 0.95
    max_seq_len: int = 1024
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    # V3.1 additions
    kernel_causal: bool = False
    use_gradient_checkpointing: bool = False


# ─────────────────────────────────────────────
#  1. Stability Components
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RoPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, x: Tensor, seq_len: int) -> Tensor:
        cos = self.cos_cached[:, :, :seq_len, :x.size(-1)].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :, :seq_len, :x.size(-1)].to(dtype=x.dtype, device=x.device)
        return (x * cos) + (self._rotate_half(x) * sin)


# ─────────────────────────────────────────────
#  2. Multi-Head Kernel Attention
# ─────────────────────────────────────────────

class MultiHeadKernelAttention(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads
        self.n_kernels = config.n_kernels
        self.top_k = config.top_k
        self.window = config.window
        self.scale = self.d_head ** -0.5
        self.kernel_causal = config.kernel_causal

        self.kernel_centers = nn.Parameter(
            torch.empty(config.n_heads, config.n_kernels, self.d_head)
        )
        self._init_kernels()

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RoPE(self.d_head, max_len=config.max_seq_len)

    def _init_kernels(self):
        try:
            for h in range(self.kernel_centers.size(0)):
                nn.init.orthogonal_(self.kernel_centers.data[h])
        except RuntimeError:
            nn.init.xavier_uniform_(self.kernel_centers)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = self.rope.apply_rotary(q, seq_len=T)

        sims = torch.einsum("bhtd,hmd->bhtm", q, self.kernel_centers) * self.scale

        if self.kernel_causal:
            pos_fraction = torch.arange(T, device=x.device, dtype=x.dtype) / max(T, 1)
            kernel_positions = torch.linspace(0, 1, self.n_kernels, device=x.device, dtype=x.dtype)
            causal_gate = torch.sigmoid(10.0 * (pos_fraction[:, None] - kernel_positions[None, :]))
            sims = sims + causal_gate[None, None, :, :].log().clamp(min=-10)

        effective_k = min(self.top_k, self.n_kernels)
        if effective_k < self.n_kernels:
            topk_vals, topk_idx = torch.topk(sims, effective_k, dim=-1)
            sparse_sims = torch.full_like(sims, float('-inf'))
            sparse_sims.scatter_(-1, topk_idx, topk_vals)
            attn = F.softmax(sparse_sims, dim=-1)
        else:
            attn = F.softmax(sims, dim=-1)

        attn = self.dropout(attn)
        kernel_context = torch.einsum("bhtm,hmd->bhtd", attn, self.kernel_centers)
        out = (v * kernel_context).transpose(1, 2).reshape(B, T, D)
        return self.o_proj(self._causal_window_pool(out))

    def _causal_window_pool(self, h: Tensor) -> Tensor:
        B, T, D = h.shape
        cs = torch.cumsum(h, dim=1)
        cs_shifted = F.pad(cs, (0, 0, self.window, 0))[:, :T, :]
        window_sum = cs - cs_shifted
        counts = torch.arange(1, T + 1, device=h.device, dtype=h.dtype).clamp(max=self.window).view(1, T, 1)
        return window_sum / counts


# ─────────────────────────────────────────────
#  3. MLP & Gating
# ─────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


class HierarchicalGating(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.gate_gen = nn.Linear(config.d_model * 4, 3)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, h_s, h_m, h_l, x):
        combined = torch.cat([h_s, h_m, h_l, x], dim=-1)
        gates = F.softmax(self.gate_gen(combined), dim=-1)
        fused = (
            gates[..., 0:1] * h_s +
            gates[..., 1:2] * h_m +
            gates[..., 2:3] * h_l
        )
        return self.proj(fused)


# ─────────────────────────────────────────────
#  4. Memory Modules
# ─────────────────────────────────────────────

class MediumTermMemory(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.decay = config.mtm_decay
        self.compress = nn.Linear(config.d_model, config.d_medium, bias=False)
        self.decompress = nn.Linear(config.d_medium, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        c = self.compress(x)
        alpha = 1.0 - self.decay
        powers = self.decay ** torch.arange(T, device=x.device, dtype=x.dtype)
        c_scaled = c * alpha
        inv_powers = (1.0 / (powers + 1e-12)).unsqueeze(0).unsqueeze(-1)
        c_undecayed = c_scaled * inv_powers
        c_cumsum = torch.cumsum(c_undecayed, dim=1)
        ema_out = c_cumsum * powers.unsqueeze(0).unsqueeze(-1)
        return self.decompress(ema_out)


class LongTermMemory(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.n_patterns = config.n_patterns
        self.d_model = config.d_model
        self.patterns = nn.Parameter(torch.randn(config.n_patterns, config.d_model) * 0.02)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.scale = config.d_model ** -0.5
        self.write_key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.write_val = nn.Linear(config.d_model, config.d_model, bias=False)
        self.write_gate = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        context = x.mean(dim=1)
        write_key = self.write_key(context)
        write_value = self.write_val(context)
        gate = self.write_gate(context)
        write_attn = F.softmax(torch.matmul(write_key, self.patterns.t()) * self.scale, dim=-1)
        pattern_delta = write_attn.unsqueeze(-1) * (gate * write_value).unsqueeze(1)
        adapted = self.patterns.unsqueeze(0) + pattern_delta
        q = self.q_proj(x)
        scores = torch.matmul(q, adapted.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, adapted)


# ─────────────────────────────────────────────
#  5. HSKM Block & Model
# ─────────────────────────────────────────────

class HSKMBlock(nn.Module):
    def __init__(self, config: HSKMConfig, layer_idx: int):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = MultiHeadKernelAttention(config)
        self.mtm = MediumTermMemory(config)
        self.ltm = LongTermMemory(config)
        self.gate = HierarchicalGating(config)
        self.ln2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, 4 * config.d_model, config.dropout)
        self.residual_scale = 1.0 / math.sqrt(2.0 * config.n_layers)
        self.use_checkpoint = config.use_gradient_checkpointing

    def _forward_body(self, x: Tensor) -> Tensor:
        h = self.ln1(x)
        h_s = self.attn(h)
        h_m = self.mtm(h_s)
        h_l = self.ltm(h)
        x = x + self.residual_scale * self.gate(h_s, h_m, h_l, h)
        x = x + self.residual_scale * self.mlp(self.ln2(x))
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.use_checkpoint and self.training:
            return grad_checkpoint(self._forward_body, x, use_reentrant=False)
        return self._forward_body(x)


class HSKM(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([HSKMBlock(config, i) for i in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None) -> Tuple[Optional[Tensor], Tensor]:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        return loss, logits

    @torch.inference_mode()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 100, temperature: float = 1.0, top_p: float = 0.9):
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.config.max_seq_len:]
            _, logits = self.forward(ctx)
            next_logits = logits[:, -1, :] / (temperature + 1e-8)
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[0, indices_to_remove] = -float('Inf')
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            if next_tok.item() == 50256: break
        return input_ids
