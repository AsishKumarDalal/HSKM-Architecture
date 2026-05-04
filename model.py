"""
Hierarchical Sparse Kernel Memory (HSKM) Architecture - Scaling Up
Multi-layer implementation with BPE compatibility.
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
    """Configuration for a larger HSKM model."""
    vocab_size: int = 50257        # GPT-2 default BPE vocab size
    d_model: int = 768             # Increased from 256
    n_layers: int = 12             # Added layers
    n_heads: int = 12              # For multi-head kernel attention if needed
    d_medium: int = 128            # Increased medium-memory compressed dim
    n_kernels: int = 32            # Increased prototype kernels
    top_k: int = 16                # Sparse selection
    window: int = 512              # Increased short-term window
    n_patterns: int = 8192         # Increased long-term pattern bank size
    mtm_decay: float = 0.95        # EMA decay for medium-term memory
    max_seq_len: int = 1024        # GPT-2 standard
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


# ─────────────────────────────────────────────
#  1. Components
# ─────────────────────────────────────────────

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        return x + self.pe(positions)


class SparseKernelAttention(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_kernels = config.n_kernels
        self.top_k = config.top_k
        self.window = config.window
        self.scale = config.d_model ** -0.5

        self.kernel_centers = nn.Parameter(torch.empty(config.n_kernels, config.d_model))
        nn.init.orthogonal_(self.kernel_centers)

        self.value_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj   = nn.Linear(config.d_model, config.d_model)
        self.dropout    = nn.Dropout(config.dropout)

    def forward(self, e_seq: Tensor) -> Tensor:
        B, T, D = e_seq.shape
        V = self.value_proj(e_seq)
        sims = torch.matmul(e_seq, self.kernel_centers.t()) * self.scale
        attn = F.softmax(sims, dim=-1)

        topk_vals, topk_idx = torch.topk(attn, self.top_k, dim=-1)
        sparse_attn = torch.zeros_like(attn)
        sparse_attn.scatter_(-1, topk_idx, topk_vals)
        sparse_attn = self.dropout(sparse_attn)

        kernel_context = torch.matmul(sparse_attn, self.kernel_centers)
        h_short = V * kernel_context
        h_short = self._causal_window_pool(h_short)
        return self.out_proj(h_short)

    def _causal_window_pool(self, h: Tensor) -> Tensor:
        B, T, D = h.shape
        cs = torch.cumsum(h, dim=1)
        cs_shifted = F.pad(cs, (0, 0, self.window, 0))[:, :T, :]
        window_sum = cs - cs_shifted
        counts = torch.arange(1, T + 1, device=h.device).clamp(max=self.window).float().view(1, T, 1)
        return window_sum / counts


class MediumTermMemory(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.decay = config.mtm_decay
        self.compress   = nn.Linear(config.d_model, config.d_medium, bias=False)
        self.decompress = nn.Linear(config.d_medium, config.d_model, bias=False)

    def forward(self, h_short_seq: Tensor) -> Tensor:
        B, T, D = h_short_seq.shape
        compressed = self.compress(h_short_seq)
        
        # Optimized EMA scan
        outputs = []
        mem = torch.zeros(B, compressed.size(-1), device=h_short_seq.device, dtype=h_short_seq.dtype)
        for t in range(T):
            mem = self.decay * mem + (1.0 - self.decay) * compressed[:, t, :]
            outputs.append(mem)
        mem_seq = torch.stack(outputs, dim=1)
        return self.decompress(mem_seq)


class LongTermMemory(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.scale = config.d_model ** -0.5
        self.patterns = nn.Parameter(torch.empty(config.n_patterns, config.d_model))
        nn.init.xavier_uniform_(self.patterns)
        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, e_seq: Tensor) -> Tensor:
        q = self.query_proj(e_seq)
        scores = torch.matmul(q, self.patterns.t()) * self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, self.patterns)


class GatedFusion(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(4 * config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 3),
        )
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, h_short, h_med, h_long, e_t):
        combined = torch.cat([h_short, h_med, h_long, e_t], dim=-1)
        gates = F.softmax(self.gate_net(combined), dim=-1)
        g_s, g_m, g_l = gates[..., 0:1], gates[..., 1:2], gates[..., 2:3]
        h = g_s * h_short + g_m * h_med + g_l * h_long
        return self.out_proj(h)


# ─────────────────────────────────────────────
#  2. HSKM Block (Single Layer)
# ─────────────────────────────────────────────

class HSKMBlock(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.sparse_kernel = SparseKernelAttention(config)
        self.medium_memory = MediumTermMemory(config)
        self.long_memory   = LongTermMemory(config)
        self.gated_fusion  = GatedFusion(config)
        
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Memory/Attention stream
        norm_x = self.ln1(x)
        h_short = self.sparse_kernel(norm_x)
        h_med   = self.medium_memory(h_short)
        h_long  = self.long_memory(norm_x)
        
        # Gated fusion with residual
        x = x + self.gated_fusion(h_short, h_med, h_long, norm_x)
        
        # Feed-forward stream
        x = x + self.mlp(self.ln2(x))
        return x


# ─────────────────────────────────────────────
#  3. Full HSKM Model (Stacked)
# ─────────────────────────────────────────────

class HSKM(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc   = LearnedPositionalEncoding(config.max_seq_len, config.d_model)
        self.dropout   = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([HSKMBlock(config) for _ in range(config.n_layers)])
        
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embedding.weight

        self.apply(self._init_weights)
        print(f"[HSKM-Large] Initialized — {self.num_params():,.0f} parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None) -> Tuple[Optional[Tensor], Tensor]:
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        x = self.pos_enc(x)
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
    def generate(self, input_ids: Tensor, max_new_tokens: int = 100, temperature: float = 0.8, top_p: float = 0.95, top_k: int = 50) -> Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.config.max_seq_len:]
            _, logits = self.forward(ctx)
            next_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[0, indices_to_remove] = -float('Inf')

            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            if next_tok.item() == 50256: # End of text token for GPT-2
                break
        return input_ids
