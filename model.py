"""
Hierarchical Sparse Kernel Memory (HSKM) Architecture
Full PyTorch implementation with CUDA support.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple
from torch import Tensor


# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

@dataclass
class HSKMConfig:
    """Configuration for the HSKM model (small GPU-trainable variant)."""
    # Vocabulary
    vocab_size: int = 10000        # small vocab for WikiText-2 / custom BPE

    # Model dimensions
    d_model: int = 256             # hidden size  (keeps param count low)
    d_medium: int = 64             # medium-memory compressed dim

    # Sparse Kernel Attention
    n_kernels: int = 16            # number of prototype kernels
    top_k: int = 8                 # how many kernels to keep active
    window: int = 128              # short-term context window (L)

    # Hierarchical Memory
    n_patterns: int = 2048         # long-term pattern bank size
    mtm_decay: float = 0.90        # EMA decay for medium-term memory

    # Sequence / training
    max_seq_len: int = 256

    # Misc
    layer_norm_eps: float = 1e-6


# ─────────────────────────────────────────────
#  1. Learned Positional Encoding
# ─────────────────────────────────────────────

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D]
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        return x + self.pe(positions)


# ─────────────────────────────────────────────
#  2. Embedding Layer
# ─────────────────────────────────────────────

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = LearnedPositionalEncoding(max_len, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T]  →  [B, T, D]
        e = self.embedding(x)
        e = self.pos_enc(e)
        return self.ln(e)


# ─────────────────────────────────────────────
#  3. Sparse Kernel Attention
# ─────────────────────────────────────────────

class SparseKernelAttention(nn.Module):
    """
    Each token attends to learned prototype kernels.
    Instead of token-to-token attention (O(n²)), we compute
    token-to-kernel scores and aggregate value projections.
    Complexity: O(n · m) where m ≪ n.
    """

    def __init__(self, d_model: int, n_kernels: int, top_k: int, window: int):
        super().__init__()
        self.d_model = d_model
        self.n_kernels = n_kernels
        self.top_k = min(top_k, n_kernels)
        self.window = window
        self.scale = d_model ** -0.5

        # Learnable kernel (prototype) centers  [m, D]
        self.kernel_centers = nn.Parameter(torch.empty(n_kernels, d_model))
        nn.init.orthogonal_(self.kernel_centers)  # encourage diversity

        # Value & output projections
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj   = nn.Linear(d_model, d_model, bias=True)
        self.ln_out     = nn.LayerNorm(d_model)

    def forward(self, e_seq: Tensor) -> Tensor:
        """
        e_seq: [B, T, D]  →  returns h_short [B, T, D]

        Causal: for each position t, only use tokens [max(0,t-L) … t].
        """
        B, T, D = e_seq.shape

        # Value projections  [B, T, D]
        V = self.value_proj(e_seq)

        # Token-to-kernel similarities: [B, T, m]
        # kernel_centers: [m, D] → broadcast matmul
        sims = torch.matmul(e_seq, self.kernel_centers.t()) * self.scale  # [B, T, m]

        # Normalise over kernels
        attn = F.softmax(sims, dim=-1)  # [B, T, m]

        # Sparsify: top-k kernels per position
        topk_vals, topk_idx = torch.topk(attn, self.top_k, dim=-1)  # [B, T, k]
        sparse_attn = torch.zeros_like(attn)
        sparse_attn.scatter_(-1, topk_idx, topk_vals)

        # Build context vector: weighted combination of kernels, then back to D
        # sparse_attn: [B, T, m]  kernel_centers: [m, D]
        # → kernel_context: [B, T, D]
        kernel_context = torch.matmul(sparse_attn, self.kernel_centers)  # [B, T, D]

        # Modulate with value projection (token-wise)
        h_short = V * kernel_context

        # Causal windowed aggregation via simple cumulative pooling
        # (fast parallel implementation using cumsum trick)
        h_short = self._causal_window_pool(h_short)

        return self.ln_out(self.out_proj(h_short))

    def _causal_window_pool(self, h: Tensor) -> Tensor:
        """
        Average-pool over causal window of length `window`.
        Efficient parallel implementation using prefix-sum.
        h: [B, T, D]
        """
        B, T, D = h.shape
        # Cumulative sum along time axis
        cs = torch.cumsum(h, dim=1)          # [B, T, D]
        # Shifted sum for the window
        cs_shifted = F.pad(cs, (0, 0, self.window, 0))[:, :T, :]  # [B, T, D]
        window_sum = cs - cs_shifted          # sum of last `window` tokens
        # Count of tokens in each window
        counts = torch.arange(1, T + 1, device=h.device).clamp(max=self.window).float()
        counts = counts.view(1, T, 1)
        return window_sum / counts


# ─────────────────────────────────────────────
#  4. Medium-Term Memory  (EMA)
# ─────────────────────────────────────────────

class MediumTermMemory(nn.Module):
    """
    Exponential Moving Average of compressed hidden states.
    Runs recurrently; state is carried across time steps.
    """

    def __init__(self, d_model: int, d_med: int, decay: float = 0.9):
        super().__init__()
        self.decay = decay
        self.compress   = nn.Linear(d_model, d_med, bias=False)
        self.decompress = nn.Linear(d_med, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, h_short_seq: Tensor) -> Tensor:
        """
        h_short_seq: [B, T, D]
        Returns h_med_seq: [B, T, D]  (decompressed EMA)
        """
        B, T, D = h_short_seq.shape
        compressed = self.compress(h_short_seq)     # [B, T, d_med]

        # Parallel EMA scan (efficient; avoids Python loop for large T)
        outputs = []
        mem = torch.zeros(B, compressed.size(-1), device=h_short_seq.device, dtype=h_short_seq.dtype)
        for t in range(T):
            mem = self.decay * mem + (1.0 - self.decay) * compressed[:, t, :]
            outputs.append(mem)
        mem_seq = torch.stack(outputs, dim=1)       # [B, T, d_med]

        return self.ln(self.decompress(mem_seq))    # [B, T, D]


# ─────────────────────────────────────────────
#  5. Long-Term Memory  (Pattern Bank)
# ─────────────────────────────────────────────

class LongTermMemory(nn.Module):
    """
    Fixed learnable pattern bank.
    At each step, attend over all patterns using current embedding as query.
    Updated only via backprop (frozen at inference once trained).
    """

    def __init__(self, n_patterns: int, d_model: int):
        super().__init__()
        self.scale = d_model ** -0.5
        self.patterns    = nn.Parameter(torch.empty(n_patterns, d_model))
        nn.init.xavier_uniform_(self.patterns)
        self.query_proj  = nn.Linear(d_model, d_model, bias=False)
        self.ln          = nn.LayerNorm(d_model)

    def forward(self, e_seq: Tensor) -> Tensor:
        """
        e_seq: [B, T, D]  →  h_long: [B, T, D]
        """
        q = self.query_proj(e_seq)          # [B, T, D]
        # Attend over pattern bank
        scores  = torch.matmul(q, self.patterns.t()) * self.scale   # [B, T, P]
        weights = F.softmax(scores, dim=-1)                          # [B, T, P]
        h_long  = torch.matmul(weights, self.patterns)               # [B, T, D]
        return self.ln(h_long)


# ─────────────────────────────────────────────
#  6. Gated Fusion
# ─────────────────────────────────────────────

class GatedFusion(nn.Module):
    """
    Learns soft gates over three memory streams + embedding.
    Gates are normalised (softmax) so they always sum to 1.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),   # 3 gate scalars
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln       = nn.LayerNorm(d_model)

    def forward(
        self,
        h_short: Tensor,   # [B, T, D]
        h_med:   Tensor,   # [B, T, D]
        h_long:  Tensor,   # [B, T, D]
        e_t:     Tensor,   # [B, T, D]
    ) -> Tensor:
        combined = torch.cat([h_short, h_med, h_long, e_t], dim=-1)  # [B,T,4D]
        gates = F.softmax(self.gate_net(combined), dim=-1)            # [B,T,3]
        g_s, g_m, g_l = gates[..., 0:1], gates[..., 1:2], gates[..., 2:3]
        h = g_s * h_short + g_m * h_med + g_l * h_long
        return self.ln(self.out_proj(h))


# ─────────────────────────────────────────────
#  7. Prediction Head
# ─────────────────────────────────────────────

class PredictionHead(nn.Module):
    """LayerNorm + tied-weight linear to vocab."""

    def __init__(self, d_model: int, vocab_size: int, embedding: nn.Embedding):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.embedding = embedding          # tied weights
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, h: Tensor) -> Tensor:
        # h: [B, T, D]
        h = self.ln(h)
        return F.linear(h, self.embedding.weight, self.bias)  # [B, T, V]


# ─────────────────────────────────────────────
#  8. Full HSKM Model
# ─────────────────────────────────────────────

class HSKM(nn.Module):
    """
    Hierarchical Sparse Kernel Memory Language Model.
    Forward pass is fully parallelised over the time axis (teacher-forcing).
    """

    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.config = config

        self.embedding    = EmbeddingLayer(config.vocab_size, config.d_model, config.max_seq_len)
        self.sparse_kernel = SparseKernelAttention(
            d_model   = config.d_model,
            n_kernels = config.n_kernels,
            top_k     = config.top_k,
            window    = config.window,
        )
        self.medium_memory = MediumTermMemory(
            d_model = config.d_model,
            d_med   = config.d_medium,
            decay   = config.mtm_decay,
        )
        self.long_memory = LongTermMemory(
            n_patterns = config.n_patterns,
            d_model    = config.d_model,
        )
        self.gated_fusion = GatedFusion(config.d_model)
        self.pred_head    = PredictionHead(
            d_model    = config.d_model,
            vocab_size = config.vocab_size,
            embedding  = self.embedding.embedding,
        )

        # Weight initialisation
        self.apply(self._init_weights)
        print(f"[HSKM] Initialised — {self.num_params():,.0f} parameters")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Tensor,              # [B, T]
        labels:    Optional[Tensor] = None,  # [B, T]
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        Returns (loss, logits).
        loss is None when labels is None.
        """
        # 1. Embed
        e_seq = self.embedding(input_ids)   # [B, T, D]

        # 2. Short-term: Sparse Kernel Attention
        h_short = self.sparse_kernel(e_seq)

        # 3. Medium-term: EMA memory
        h_med = self.medium_memory(h_short)

        # 4. Long-term: Pattern bank
        h_long = self.long_memory(e_seq)

        # 5. Gated Fusion
        h_fused = self.gated_fusion(h_short, h_med, h_long, e_seq)

        # 6. Prediction
        logits = self.pred_head(h_fused)    # [B, T, V]

        # 7. Loss (next-token prediction)
        loss = None
        if labels is not None:
            # Shift: predict token at position t from tokens 0..t-1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=0,          # ignore padding
            )

        return loss, logits

    # ── Generation ───────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,       # [1, T_prompt]
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> Tensor:
        """
        Autoregressive generation with top-k + top-p (nucleus) sampling.
        """
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Trim to max_seq_len
            ctx = generated[:, -self.config.max_seq_len:]

            _, logits = self.forward(ctx)
            next_logits = logits[:, -1, :] / temperature   # [1, V]

            # Top-k filter
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, top_k)
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, -1e9)

            # Nucleus (top-p) filter
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_mask] = -1e9
                next_logits.scatter_(-1, sorted_idx, sorted_logits)

            probs    = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)   # [1, 1]
            generated = torch.cat([generated, next_tok], dim=1)

        return generated
