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
    n_patterns: int = 4096
    mtm_decay: float = 0.9
    max_seq_len: int = 1024
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    kernel_causal: bool = False
    use_gradient_checkpointing: bool = False


# ─────────────────────────────────────────────
#  Components
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_rope_freqs(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    # x: [B, T, H, D] -> [B, T, H, D/2] as complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Match T dimension
    freqs_cis = freqs_cis[:x.shape[1]].to(x_complex.device)
    # Broadast over batch and heads
    x_out = torch.view_as_real(x_complex * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(3)
    return x_out.type_as(x)


class MultiHeadKernelAttention(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads
        self.n_kernels = config.n_kernels
        self.top_k = min(config.top_k, config.n_kernels)
        self.window = config.window

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Learned Kernels (Position-Invariant Prototypes)
        self.kernels = nn.Parameter(torch.randn(config.n_heads, config.n_kernels, self.d_head))
        self.kernel_causal = config.kernel_causal

        nn.init.xavier_uniform_(self.kernels)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head)

        # RoPE on Queries and Keys
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Sparse Kernel Matching: [B, T, H, K]
        # Query similarity to learned prototypes
        sim = torch.einsum("bthd,hkd->bthk", q, self.kernels) / math.sqrt(self.d_head)

        if self.kernel_causal:
            # Mask to prevent looking at kernels that might represent "future"
            # But kernels are global prototypes, so causal masking is usually OFF.
            mask = torch.triu(torch.ones(T, self.n_kernels, device=x.device), diagonal=1).bool()
            sim.masked_fill_(mask.unsqueeze(0).unsqueeze(2), -float('inf'))

        # Top-K Sparsity
        topk_sim, topk_idx = torch.topk(sim, self.top_k, dim=-1)
        sparse_sim = torch.full_like(sim, -float('inf'))
        sparse_sim.scatter_(-1, topk_idx, topk_sim)

        attn = F.softmax(sparse_sim, dim=-1)

        # Kernel-Weighted Values: [B, T, H, D]
        # This acts as a global retrieval from the kernel prototypes
        k_out = torch.einsum("bthk,hkd->bthd", attn, self.kernels)
        
        # Local Context injection (Standard V logic)
        v_out = v # Local context through residual
        
        out = (k_out + v_out).reshape(B, T, C)
        return self.o_proj(out)


class MediumTermMemory(nn.Module):
    """
    Vectorized EMA-based memory using parallel prefix scan.
    """
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_medium = config.d_medium
        self.decay = config.mtm_decay

        self.in_proj = nn.Linear(config.d_model, config.d_medium)
        self.out_proj = nn.Linear(config.d_medium, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        h = self.in_proj(x) # [B, T, D_m]

        # Parallel EMA scan: h_t = decay * h_{t-1} + (1-decay) * input_t
        # Using geometric series power trick for parallelization
        powers = torch.arange(T - 1, -1, -1, device=x.device).float()
        decay_weights = self.decay ** powers # [T]
        
        # We use a simple causal convolution as a proxy for vectorized EMA 
        # to ensure O(N) and stability in high-dim.
        # But for HSKM-V3.1, we use the geometric power cumsum.
        
        exp_weights = (self.decay ** torch.arange(T, device=x.device)).unsqueeze(0).unsqueeze(-1)
        h_scaled = h * (self.decay ** -torch.arange(T, device=x.device)).unsqueeze(0).unsqueeze(-1)
        h_sum = torch.cumsum(h_scaled, dim=1)
        h_ema = h_sum * exp_weights * (1 - self.decay)

        return self.out_proj(h_ema)


class LongTermMemory(nn.Module):
    """
    Gated Read/Write patterns for long-term conceptual retrieval.
    """
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.n_patterns = config.n_patterns
        self.d_model = config.d_model
        
        # Persistent Pattern Store (Global Keys)
        self.patterns = nn.Parameter(torch.randn(config.n_patterns, config.d_model))
        nn.init.orthogonal_(self.patterns)

        self.write_gate = nn.Linear(config.d_model, 1)
        self.read_gate = nn.Linear(config.d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        
        # Gated Write: Tokens "update" the global patterns contextually (per batch)
        # We compute a context-dependent updated_pattern on the fly
        w_g = torch.sigmoid(self.write_gate(x)) # [B, T, 1]
        write_val = x * w_g # [B, T, C]
        
        # Gated Read: Tokens retrieve from the global pattern store
        r_g = torch.sigmoid(self.read_gate(x)) # [B, T, 1]
        
        # Cross-attention to patterns
        sim = torch.einsum("btc,pc->btp", x, self.patterns) / math.sqrt(C)
        attn = F.softmax(sim, dim=-1)
        read_val = torch.einsum("btp,pc->btc", attn, self.patterns)

        return read_val * r_g


class HSKMBlock(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = MultiHeadKernelAttention(config)
        
        self.ln2 = RMSNorm(config.d_model)
        self.mtm = MediumTermMemory(config)
        
        self.ln3 = RMSNorm(config.d_model)
        self.ltm = LongTermMemory(config)
        
        # Hierarchical Fusion Gate
        self.gate = nn.Linear(config.d_model * 3, config.d_model)
        
        # Feed Forward (SwiGLU)
        self.ln_ffn = RMSNorm(config.d_model)
        self.ffn_in = nn.Linear(config.d_model, config.d_model * 4)
        self.ffn_out = nn.Linear(config.d_model * 2, config.d_model) # SwiGLU reduction
        
        self.use_checkpoint = config.use_gradient_checkpointing

    def _forward_inner(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        # 1. Short-Term
        s_out = self.attn(self.ln1(x), freqs_cis)
        
        # 2. Medium-Term
        m_out = self.mtm(self.ln2(x))
        
        # 3. Long-Term
        l_out = self.ltm(self.ln3(x))
        
        # Hierarchical Fusion
        combined = torch.cat([s_out, m_out, l_out], dim=-1)
        fused = self.gate(combined)
        x = x + fused
        
        # SwiGLU FFN
        res = x
        x = self.ln_ffn(x)
        x = self.ffn_in(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.ffn_out(x)
        return res + x

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        if self.use_checkpoint and self.training:
            return grad_checkpoint(self._forward_inner, x, freqs_cis)
        return self._forward_inner(x, freqs_cis)


# ─────────────────────────────────────────────
#  Core Model
# ─────────────────────────────────────────────

class HSKM(nn.Module):
    def __init__(self, config: HSKMConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([HSKMBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.head.weight
        
        self.freqs_cis = precompute_rope_freqs(
            config.d_model // config.n_heads, config.max_seq_len * 2
        )
        
        # Initial Scaling
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None) -> Tuple[Optional[Tensor], Tensor]:
        B, T = input_ids.shape
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        freqs_cis = self.freqs_cis.to(x.device)

        for block in self.blocks:
            x = block(x, freqs_cis)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1
            )
        return loss, logits

    @torch.inference_mode()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_p: float = 0.9,
                 repetition_penalty: float = 1.2):
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.config.max_seq_len:]
            _, logits = self.forward(ctx)
            next_logits = logits[:, -1, :] / (temperature + 1e-8)

            # Apply repetition penalty
            for token_id in set(input_ids[0].tolist()):
                if next_logits[0, token_id] > 0:
                    next_logits[0, token_id] /= repetition_penalty
                else:
                    next_logits[0, token_id] *= repetition_penalty

            # Nucleus sampling
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
            
            # EOS check
            if next_tok.item() == 50256:
                break
        return input_ids
