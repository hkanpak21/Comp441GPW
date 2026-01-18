"""
GPW Watermarking - Gaussian Pancakes Watermarking

Implements 3 variants:
1. GPW - Static cosine score with secret direction w
2. GPW-SP - Salted Phase (context-keyed phase φ)
3. GPW-SP+SR - Semantic Representation Coupling (hidden-state dependent direction)

Based on the working implementation in gpw_sp_contextual_cluster_wm.py
"""

import math
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F

from .base import TokenLevelWatermarker, DetectionResult


@dataclass
class GPWConfig:
    """Configuration for GPW watermarking."""
    alpha: float = 1.2          # Logit bias strength
    omega: float = 10.0         # Cosine frequency parameter
    salted: bool = True         # Enable context-keyed phase (GPW-SP)
    ctx_mode: str = "prev_token"  # "prev_token" | "ngram" | "rolling"
    ngram: int = 4              # n-gram size for ctx_mode="ngram"


@dataclass
class SRConfig:
    """Configuration for Semantic Representation coupling."""
    enabled: bool = False       # Enable SR coupling
    lambda_couple: float = 0.2  # Coupling strength
    rank: int = 8               # Low-rank approximation dimension


def _hmac_like(key: bytes, msg: bytes) -> bytes:
    """PRF-like keyed hash."""
    return hashlib.sha256(key + b"||" + msg).digest()


def prf_to_uniform01(key: bytes, msg: bytes) -> float:
    """Map PRF output to [0,1)."""
    digest = _hmac_like(key, msg)
    x = int.from_bytes(digest[:8], "big")
    return (x % (2**53)) / float(2**53)


def seeded_torch_generator(key: bytes, tag: bytes, device: str) -> torch.Generator:
    """Create seeded torch Generator."""
    seed_bytes = _hmac_like(key, tag)
    seed = int.from_bytes(seed_bytes[:8], "big") % (2**63 - 1)
    # Generator must be on CPU for most operations
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    return g


@torch.no_grad()
def get_token_embedding_matrix(model) -> torch.Tensor:
    """Get token embedding matrix from model."""
    return model.get_input_embeddings().weight  # [V, d]


@torch.no_grad()
def derive_secret_direction_w(key: bytes, d: int, device: str) -> torch.Tensor:
    """Derive secret direction w from key. Always uses float32."""
    g = seeded_torch_generator(key, b"GPW_w", device='cpu')
    # Generate in float32 on CPU, then move to device
    v = torch.randn(d, generator=g, dtype=torch.float32)
    v = v / (v.norm() + 1e-12)
    return v.to(device=device)


@torch.no_grad()
def precompute_projections(E: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Precompute projections s = E @ w. Always computes in float32."""
    # Convert to float32 for computation
    E_f32 = E.float()
    w_f32 = w.float()
    s = E_f32 @ w_f32
    return s  # Keep as float32


def ctx_fingerprint(input_ids: torch.Tensor, mode: str, ngram: int = 4) -> bytes:
    """Compute context fingerprint for phase calculation."""
    ids = input_ids[0].tolist()
    if len(ids) == 0:
        return b"empty"
    
    if mode == "prev_token":
        return f"prev:{ids[-1]}".encode()
    
    if mode == "ngram":
        tail = ids[-ngram:] if len(ids) >= ngram else ids
        return ("ngram:" + ",".join(map(str, tail))).encode()
    
    if mode == "rolling":
        h = hashlib.sha256(("roll:" + ",".join(map(str, ids[-64:]))).encode()).digest()
        return b"roll:" + h[:16]
    
    raise ValueError(f"Unknown ctx_mode={mode}")


def salted_phase_phi(key: bytes, input_ids: torch.Tensor, cfg: GPWConfig) -> float:
    """Compute salted phase from context."""
    fp = ctx_fingerprint(input_ids, cfg.ctx_mode, cfg.ngram)
    u = prf_to_uniform01(key, b"phi||" + fp)
    return 2.0 * math.pi * u


@torch.no_grad()
def make_low_rank_A(key: bytes, d_embed: int, d_hid: int, rank: int, device: str) -> torch.Tensor:
    """Create low-rank coupling matrix A = B @ C. Always uses float32."""
    gB = seeded_torch_generator(key, b"SR_B", device='cpu')
    gC = seeded_torch_generator(key, b"SR_C", device='cpu')
    # Generate in float32 on CPU
    B = torch.randn(d_embed, rank, generator=gB, dtype=torch.float32) / math.sqrt(d_embed)
    C = torch.randn(rank, d_hid, generator=gC, dtype=torch.float32) / math.sqrt(d_hid)
    A = B @ C
    return A.to(device=device)


@torch.no_grad()
def compute_w_t(w: torch.Tensor, A: torch.Tensor, h_t: torch.Tensor, lambda_couple: float) -> torch.Tensor:
    """Compute position-dependent direction with SR coupling. Always uses float32."""
    # Convert everything to float32
    w_f32 = w.float()
    A_f32 = A.float()
    h_t_f32 = h_t.float()
    v = w_f32 + lambda_couple * (A_f32 @ h_t_f32)
    v_norm = v.norm()
    if v_norm < 1e-12:
        return v
    return v / (v_norm + 1e-12)


@torch.no_grad()
def pancake_score(s: torch.Tensor, omega: float, phi: float) -> torch.Tensor:
    """Compute pancake (cosine) score. Always uses float32."""
    s_f32 = s.float()
    return torch.cos(omega * s_f32 + phi)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    thresh = values[..., -1, None]
    return torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    mask = cum > p
    mask[..., 0] = False  # keep at least 1 token
    sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
    out = torch.full_like(logits, float('-inf'))
    out.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return out


def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, key: bytes, step_tag: bytes) -> int:
    """Sample from logits with seeded generator."""
    # Work in float32 for numerical stability
    logits_f32 = logits.float()
    logits_f32 = logits_f32 / max(temperature, 1e-6)
    logits_f32 = top_k_filter(logits_f32, top_k)
    logits_f32 = top_p_filter(logits_f32, top_p)
    probs = F.softmax(logits_f32, dim=-1)
    
    # Use CPU generator for multinomial
    g = seeded_torch_generator(key, b"SAMPLE||" + step_tag, device='cpu')
    # Move probs to CPU for sampling with CPU generator
    probs_cpu = probs.cpu()
    idx = torch.multinomial(probs_cpu, num_samples=1, generator=g)
    return int(idx.item())


class GPWWatermark(TokenLevelWatermarker):
    """
    Gaussian Pancakes Watermarking (GPW).
    
    Three variants controlled by gpw_cfg and sr_cfg:
    - GPW: gpw_cfg.salted=False, sr_cfg.enabled=False
    - GPW-SP: gpw_cfg.salted=True, sr_cfg.enabled=False
    - GPW-SP+SR: gpw_cfg.salted=True, sr_cfg.enabled=True
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        gpw_cfg: GPWConfig = None,
        sr_cfg: SRConfig = None,
        hash_key: bytes = None,
        device: str = "cuda",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            gamma=0.5,  # Not used in GPW but required by base
            delta=gpw_cfg.alpha if gpw_cfg else 1.2,
            z_threshold=4.0,
            device=device
        )
        
        self.gpw_cfg = gpw_cfg or GPWConfig()
        self.sr_cfg = sr_cfg or SRConfig()
        self.hash_key = hash_key or hashlib.sha256(b"gpw-default-key").digest()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """Generate watermarked text using GPW."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        
        # Get embeddings (convert to float32 for all computations)
        E = get_token_embedding_matrix(self.model).to(self.device).float()
        V, d_embed = E.shape
        
        # Derive secret direction and precompute projections (all in float32)
        w = derive_secret_direction_w(self.hash_key, d_embed, device=self.device)
        s_base = precompute_projections(E, w)
        
        # Setup SR coupling if enabled
        A = None
        if self.sr_cfg.enabled:
            d_hid = getattr(self.model.config, 'n_embd', 
                          getattr(self.model.config, 'hidden_size', d_embed))
            A = make_low_rank_A(self.hash_key, d_embed, d_hid, self.sr_cfg.rank, device=self.device)
        
        for t in range(max_new_tokens):
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=self.sr_cfg.enabled
            )
            # Get logits and convert to float32 immediately
            logits = outputs.logits[0, -1, :].float()  # [V] in float32
            
            # Compute phase
            phi = 0.0
            if self.gpw_cfg.salted:
                phi = salted_phase_phi(self.hash_key, input_ids, self.gpw_cfg)
            
            # Choose projection s depending on SR
            if self.sr_cfg.enabled:
                h_t = outputs.hidden_states[-1][0, -1, :].float()  # Convert to float32
                w_t = compute_w_t(w, A, h_t, self.sr_cfg.lambda_couple)
                s = precompute_projections(E, w_t)
            else:
                s = s_base
            
            # Compute pancake score and apply logit bias (all in float32)
            g = pancake_score(s, self.gpw_cfg.omega, phi)
            logits_wm = logits + self.gpw_cfg.alpha * g
            
            # Sample next token
            next_id = sample_from_logits(
                logits_wm,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                key=self.hash_key,
                step_tag=f"t={t}".encode(),
            )
            
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=self.device)], dim=1)
            
            if next_id == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    @torch.no_grad()
    def detect(self, text: str, **kwargs) -> DetectionResult:
        """Detect watermark in text."""
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        
        if input_ids.size(1) < 2:
            return DetectionResult(
                is_watermarked=False,
                z_score=0.0,
                p_value=1.0,
                num_tokens_scored=0,
                green_fraction=0.0
            )
        
        # Get embeddings (convert to float32)
        E = get_token_embedding_matrix(self.model).to(self.device).float()
        V, d_embed = E.shape
        w = derive_secret_direction_w(self.hash_key, d_embed, device=self.device)
        s_base = precompute_projections(E, w)
        
        A = None
        if self.sr_cfg.enabled:
            d_hid = getattr(self.model.config, 'n_embd', 
                          getattr(self.model.config, 'hidden_size', d_embed))
            A = make_low_rank_A(self.hash_key, d_embed, d_hid, self.sr_cfg.rank, device=self.device)
        
        S = 0.0
        token_scores = []
        
        # Iterate token by token (skip first token because ctx undefined)
        for t in range(1, input_ids.size(1)):
            prefix = input_ids[:, :t]
            token = int(input_ids[0, t].item())
            
            outputs = self.model(
                input_ids=prefix,
                output_hidden_states=self.sr_cfg.enabled
            )
            
            phi = salted_phase_phi(self.hash_key, prefix, self.gpw_cfg) if self.gpw_cfg.salted else 0.0
            
            if self.sr_cfg.enabled:
                h_t = outputs.hidden_states[-1][0, -1, :].float()  # Convert to float32
                w_t = compute_w_t(w, A, h_t, self.sr_cfg.lambda_couple)
                s = precompute_projections(E, w_t)
            else:
                s = s_base
            
            # Get token score (all in float32)
            if token < s.shape[0]:
                s_token_val = float(s[token].cpu())
                score_t = math.cos(self.gpw_cfg.omega * s_token_val + phi)
            else:
                score_t = 0.0
            
            S += score_t
            token_scores.append(score_t)
        
        num_tokens = len(token_scores)
        
        # Normalize score
        # Under null hypothesis (random tokens), E[cos] ≈ 0, Var[cos] ≈ 0.5
        # z = S / sqrt(n * 0.5) = S * sqrt(2/n)
        z_score = S * math.sqrt(2.0 / num_tokens) if num_tokens > 0 else 0.0
        
        # P-value from normal approximation
        from scipy.stats import norm
        p_value = norm.sf(z_score)
        
        is_watermarked = z_score > self.z_threshold
        
        # Green fraction approximation: fraction of tokens with positive score
        green_fraction = sum(1 for s in token_scores if s > 0) / num_tokens if num_tokens > 0 else 0.0
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            z_score=z_score,
            p_value=p_value,
            num_tokens_scored=num_tokens,
            green_fraction=green_fraction,
            confidence=1.0 - p_value if is_watermarked else 0.0,
            metadata={"raw_score": S}
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        variant = "GPW"
        if self.gpw_cfg.salted:
            variant = "GPW-SP"
            if self.sr_cfg.enabled:
                variant = "GPW-SP+SR"
        
        return {
            "method": variant,
            "variant": variant,
            "alpha": self.gpw_cfg.alpha,
            "omega": self.gpw_cfg.omega,
            "salted": self.gpw_cfg.salted,
            "ctx_mode": self.gpw_cfg.ctx_mode,
            "sr_enabled": self.sr_cfg.enabled,
            "sr_lambda": self.sr_cfg.lambda_couple if self.sr_cfg.enabled else None,
            "sr_rank": self.sr_cfg.rank if self.sr_cfg.enabled else None,
            "z_threshold": self.z_threshold,
        }


def create_gpw_variant(
    model,
    tokenizer,
    variant: str = "GPW-SP",
    alpha: float = 1.2,
    omega: float = 10.0,
    hash_key: bytes = None,
    device: str = "cuda"
) -> GPWWatermark:
    """Factory function to create GPW variant.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        variant: "GPW", "GPW-SP", or "GPW-SP+SR"
        alpha: Logit bias strength
        omega: Cosine frequency
        hash_key: Secret key
        device: Device
        
    Returns:
        Configured GPWWatermark instance
    """
    if variant == "GPW":
        gpw_cfg = GPWConfig(alpha=alpha, omega=omega, salted=False)
        sr_cfg = SRConfig(enabled=False)
    elif variant == "GPW-SP":
        gpw_cfg = GPWConfig(alpha=alpha, omega=omega, salted=True)
        sr_cfg = SRConfig(enabled=False)
    elif variant == "GPW-SP+SR":
        gpw_cfg = GPWConfig(alpha=alpha, omega=omega, salted=True)
        sr_cfg = SRConfig(enabled=True, lambda_couple=0.2, rank=8)
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'GPW', 'GPW-SP', or 'GPW-SP+SR'")
    
    return GPWWatermark(
        model=model,
        tokenizer=tokenizer,
        gpw_cfg=gpw_cfg,
        sr_cfg=sr_cfg,
        hash_key=hash_key,
        device=device
    )
