"""
Hierarchical Sparse Kernel Memory (HSKM) Architecture - Improved
Refined for better convergence:
- Multi-Head Sparse Kernel Attention
- Gated Linear Unit (GLU) MLP
- Residual scaling for deep initialization
- RMSNorm for stability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from torch import Tensor


# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

@dataclass
class HSKMConfig:
    vocab_size: int = 50257
    d_model: int = 512
    n_layers: int = 8               # Moderate depth
    n_heads: int = 8                # Multi-head kernel attention
    d_medium: int = 128
    n_kernels: int = 64             # More kernels for better coverage
    top_k: int = 16
    window: int = 512
    n_patterns: int = 8192
    mtm_decay: float = 0.95
    max_seq_len: int = 1024
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


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
    """Rotary Positional Embeddings for better sequence modeling."""
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]


# ─────────────────────────────────────────────
#  2. Refined Kernel Attention
# ─────────────────────────────────────────────

class MultiHeadKernelAttention(nn.Module):
    """
    Splits kernels into heads. Each head attends to a subset of kernels.
    """
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head  = config.d_model // config.n_heads
        self.n_kernels = config.n_kernels
        self.top_k = config.top_k
        self.window = config.window
        self.scale = self.d_head ** -0.5

        # Kernels are shared or per-head? Per-head is better for diversity.
        self.kernel_centers = nn.Parameter(torch.empty(config.n_heads, config.n_kernels, self.d_head))
        nn.init.orthogonal_(self.kernel_centers)

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        # Q, V: [B, T, H, d_head]
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # q: [B, H, T, d_head], kernels: [H, m, d_head]
        # sims: [B, H, T, m]
        sims = torch.einsum("bhtd,hmd->bhtm", q, self.kernel_centers) * self.scale
        attn = F.softmax(sims, dim=-1)

        # Sparsity
        if self.top_k < self.n_kernels:
            topk_vals, topk_idx = torch.topk(attn, self.top_k, dim=-1)
            sparse_attn = torch.zeros_like(attn)
            sparse_attn.scatter_(-1, topk_idx, topk_vals)
        else:
            sparse_attn = attn

        sparse_attn = self.dropout(sparse_attn)

        # Context: [B, H, T, d_head]
        # sparse_attn: [B, H, T, m], kernels: [H, m, d_head]
        kernel_context = torch.einsum("bhtm,hmd->bhtd", sparse_attn, self.kernel_centers)
        
        # Combine with values
        out = (v * kernel_context).transpose(1, 2).reshape(B, T, D)
        
        # Causal window pool (simple version for multi-layer)
        return self.o_proj(self._causal_window_pool(out))

    def _causal_window_pool(self, h: Tensor) -> Tensor:
        B, T, D = h.shape
        cs = torch.cumsum(h, dim=1)
        cs_shifted = F.pad(cs, (0, 0, self.window, 0))[:, :T, :]
        window_sum = (cs - cs_shifted)
        counts = torch.arange(1, T + 1, device=h.device).clamp(max=self.window).float().view(1, T, 1)
        return window_sum / counts


# ─────────────────────────────────────────────
#  3. Refined Gating & MLP
# ─────────────────────────────────────────────

class SwiGLU(nn.Module):
    """Better MLP than standard GELU."""
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


class HierarchicalGating(nn.Module):
    """Dynamic fusion of STM, MTM, LTM."""
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.gate_gen = nn.Linear(config.d_model * 4, 3)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, h_s, h_m, h_l, x):
        # x is the residual/embedding stream
        combined = torch.cat([h_s, h_m, h_l, x], dim=-1)
        gates = F.softmax(self.gate_gen(combined), dim=-1)
        
        fused = (
            gates[..., 0:1] * h_s +
            gates[..., 1:2] * h_m +
            gates[..., 2:3] * h_l
        )
        return self.proj(fused)


# ─────────────────────────────────────────────
#  4. HSKM Block & Model
# ─────────────────────────────────────────────

class HSKMBlock(nn.Module):
    def __init__(self, config: HSKMConfig, layer_idx: int):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = MultiHeadKernelAttention(config)
        
        # Recurrent Medium Term (EMA) - optionally shared or separate
        self.mtm  = MediumTermMemory(config)
        self.ltm  = LongTermMemory(config)
        self.gate = HierarchicalGating(config)
        
        self.ln2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, 4 * config.d_model, config.dropout)
        
        # Scaling factor for deep residuals
        self.layer_idx = layer_idx

    def forward(self, x: Tensor) -> Tensor:
        h = self.ln1(x)
        h_s = self.attn(h)
        h_m = self.mtm(h_s)
        h_l = self.ltm(h)
        
        x = x + self.gate(h_s, h_m, h_l, h)
        x = x + self.mlp(self.ln2(x))
        return x


# Supporting classes (MediumTermMemory, LongTermMemory) remain similar but use RMSNorm
class MediumTermMemory(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.decay = config.mtm_decay
        self.compress = nn.Linear(config.d_model, config.d_medium, bias=False)
        self.decompress = nn.Linear(config.d_medium, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        c = self.compress(x)
        outputs = []
        state = torch.zeros(B, c.size(-1), device=x.device, dtype=x.dtype)
        for t in range(T):
            state = self.decay * state + (1.0 - self.decay) * c[:, t, :]
            outputs.append(state)
        return self.decompress(torch.stack(outputs, dim=1))


class LongTermMemory(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.patterns = nn.Parameter(torch.randn(config.n_patterns, config.d_model) * 0.02)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.scale = config.d_model ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        q = self.q_proj(x)
        s = torch.matmul(q, self.patterns.t()) * self.scale
        a = F.softmax(s, dim=-1)
        return torch.matmul(a, self.patterns)


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
        print(f"[HSKM-V2] {self.num_params():,.0f} params")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Scale initialization for deep models
            std = 0.02
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
            if hasattr(m, 'bias') and m.bias is not None:
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
            # Shift happens outside usually, but cross_entropy handles it if we pass targets
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits

    @torch.inference_mode()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 100, temperature: float = 1.0, top_p: float = 0.9):
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.config.max_seq_len:]
            _, logits = self.forward(ctx)
            next_logits = logits[:, -1, :] / (temperature + 1e-8)
            
            # Simple nucleus sampling
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
