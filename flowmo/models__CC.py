"""Simplified model code for FlowMo.

This file contains the implementation of the FlowMo model, a diffusion-based model
for image generation, which includes components like the Flux transformer,
attention blocks, and quantization schemes.

Sources: https://github.com/feizc/FluxMusic/blob/main/train.py
https://github.com/black-forest-labs/flux/tree/main/src/flux
"""

import ast
import itertools
import math
from dataclasses import dataclass
from typing import List, Tuple

import einops
import numpy as np
import torch
from einops import rearrange, repeat
from mup import MuReadout
from torch import Tensor, nn

import lookup_free_quantize
from pyldpc import make_ldpc, encode, decode, get_message


MUP_ENABLED = True


# --- Attention and Positional Embeddings ---

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    """
    Computes scaled dot-product attention with rotary position embeddings.

    Args:
        q: Query tensor of shape (B, H, L, D).
        k: Key tensor of shape (B, H, L, D).
        v: Value tensor of shape (B, H, L, D).
        pe: Positional embedding tensor.

    Returns:
        Output tensor of shape (B, L, H*D).
    """
    b, h, l, d = q.shape
    q, k = apply_rope(q, k, pe)

    # MUP (μ-Parametrization) requires a specific scaling factor for attention.
    # A temporary workaround is included for a specific PyTorch version.
    if torch.__version__ == "2.0.1+cu117":
        if d != 64:
            print("MUP is broken in this setting! Be careful!")
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    else:
        scale = 8.0 / d if MUP_ENABLED else None
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    assert x.shape == q.shape
    return rearrange(x, "B H L D -> B L (H D)")


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """
    Generates Rotary Position Embeddings (RoPE).

    Args:
        pos: Positions tensor.
        dim: Embedding dimension.
        theta: A hyperparameter for frequency scaling.

    Returns:
        RoPE tensor.
    """
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)],
        dim=-1,
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    """
    Applies Rotary Position Embeddings to query and key tensors.
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    """A module to create N-dimensional positional embeddings using RoPE."""
    def __init__(self, dim: int, theta: int, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Creates sinusoidal timestep embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


# --- VAE-like Quantization Helpers ---

def _get_diagonal_gaussian(parameters):
    """Extracts mean and log-variance from a tensor."""
    mean, logvar = torch.chunk(parameters, 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    return mean, logvar


def _sample_diagonal_gaussian(mean, logvar):
    """Samples from a diagonal Gaussian distribution."""
    std = torch.exp(0.5 * logvar)
    return mean + std * torch.randn(mean.shape, device=mean.device)


def _kl_diagonal_gaussian(mean, logvar):
    """Computes the KL divergence between a diagonal Gaussian and a standard normal."""
    var = torch.exp(logvar)
    return 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar, dim=1).mean()


# --- Core Model Building Blocks ---

class MLPEmbedder(nn.Module):
    """A simple MLP for embedding time or other conditioning signals."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    """Root Mean Square Normalization."""
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    """Normalization for Query and Key tensors in attention."""
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    """A standard self-attention block."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    """Output of the Modulation layer, containing shift, scale, and gate."""
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    """
    Computes modulation parameters (shift, scale, gate) from a conditioning vector.
    Used for adaLN-style conditioning.
    """
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

        # Zero-initialize the gates for better stability at the start of training.
        self.lin.weight[dim * 2 : dim * 3].data.zero_()
        self.lin.bias[dim * 2 : dim * 3].data.zero_()
        if double:
            self.lin.weight[dim * 5 : dim * 6].data.zero_()
            self.lin.bias[dim * 5 : dim * 6].data.zero_()

    def forward(self, vec: Tensor) -> Tuple[ModulationOut, ModulationOut]:
        params = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )
        mod1 = ModulationOut(*params[:3])
        mod2 = ModulationOut(*params[3:]) if self.is_double else None
        return mod1, mod2


class DoubleStreamBlock(nn.Module):
    """
    A transformer block that processes two streams of data (e.g., image and text)
    with shared attention. This is a core component of the Flux model.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # Modules for the 'image' stream
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # Modules for the 'text' stream
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def _prepare_stream(self, x: Tensor, mod: ModulationOut, norm: nn.Module, attn: SelfAttention) -> Tuple[Tensor, Tensor, Tensor]:
        """Prepares a single stream (image or text) for the shared attention."""
        p = 1.0
        x_modulated = norm(x)
        x_modulated = (p + mod.scale) * x_modulated + mod.shift
        qkv = attn.qkv(x_modulated)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        return q, k, v

    def _process_stream_output(
        self, x: Tensor, attn_output: Tensor, mod1: ModulationOut, mod2: ModulationOut,
        proj: nn.Module, mlp: nn.Module, norm2: nn.Module
    ) -> Tensor:
        """Applies the post-attention transformations to a single stream."""
        p = 1.0
        x = x + mod1.gate * proj(attn_output)
        x = x + mod2.gate * mlp((p + mod2.scale) * norm2(x) + mod2.shift)
        return x

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tuple[Tensor, Tensor]):
        _, pe_double = pe

        # Get modulation parameters from the conditioning vector 'vec'
        if vec is None:
            # Use identity modulation if no conditioning vector is provided
            img_mod1, img_mod2 = ModulationOut(0, 0, 1), ModulationOut(0, 0, 1)
            txt_mod1, txt_mod2 = ModulationOut(0, 0, 1), ModulationOut(0, 0, 1)
        else:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

        # Prepare each stream for attention
        img_q, img_k, img_v = self._prepare_stream(img, img_mod1, self.img_norm1, self.img_attn)
        txt_q, txt_k, txt_v = self._prepare_stream(txt, txt_mod1, self.txt_norm1, self.txt_attn)

        # Concatenate and run shared attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        attn_out = attention(q, k, v, pe=pe_double)
        txt_attn_out, img_attn_out = attn_out[:, : txt.shape[1]], attn_out[:, txt.shape[1] :]

        # Process the output of each stream
        img = self._process_stream_output(
            img, img_attn_out, img_mod1, img_mod2, self.img_attn.proj, self.img_mlp, self.img_norm2
        )
        txt = self._process_stream_output(
            txt, txt_attn_out, txt_mod1, txt_mod2, self.txt_attn.proj, self.txt_mlp, self.txt_norm2
        )
        return img, txt


class LastLayer(nn.Module):
    """The final layer of the transformer, which projects back to the patch/token dimension."""
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        readout_zero_init=False,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        if MUP_ENABLED:
            self.linear = MuReadout(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
                readout_zero_init=readout_zero_init,
            )
        else:
            self.linear = nn.Linear(
                hidden_size, patch_size * patch_size * out_channels, bias=True
            )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        if vec is not None:
            shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
            x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        else:
            x = self.norm_final(x)
        x = self.linear(x)
        return x


# --- Flux Transformer Model ---

@dataclass
class FluxParams:
    """Parameters for the Flux model."""
    in_channels: int
    patch_size: int
    context_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    axes_dim: List[int]
    theta: int
    qkv_bias: bool


DIT_ZOO = dict(
    dit_b_4=dict(
        hidden_size=768,
        mlp_ratio=4.0,
        num_heads=12,
        axes_dim=[8, 28, 28],
        theta=10_000,
        qkv_bias=True,
    ),
    # Other DiT configurations can be added here
)


def prepare_idxs(img: Tensor, code_length: int, patch_size: int) -> Tuple[Tensor, Tensor]:
    """Prepares 3D indices for image patches and text tokens for positional embeddings."""
    bs, _, h, w = img.shape
    device = img.device
    
    # Image patch indices
    img_ids = torch.zeros(h // patch_size, w // patch_size, 3, device=device)
    img_ids[..., 1] += torch.arange(h // patch_size, device=device)[:, None]
    img_ids[..., 2] += torch.arange(w // patch_size, device=device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    # Text token indices
    txt_ids = torch.zeros((bs, code_length, 3), device=device)
    txt_ids += torch.arange(code_length, device=device)[None, :, None]
    
    return img_ids, txt_ids


class Flux(nn.Module):
    """
    A transformer model that processes image and text sequences.
    It can be used as an encoder or a decoder in the FlowMo architecture.
    """
    def __init__(self, params: FluxParams, name="", lsg=False):
        super().__init__()
        self.name = name
        self.lsg = lsg
        self.params = params
        self.patch_size = params.patch_size
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Sum of axes_dim {params.axes_dim} must be equal to pe_dim {pe_dim}")

        # Input layers
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.img_in = nn.Linear(params.in_channels, self.hidden_size, bias=True)
        self.txt_in = nn.Linear(params.context_dim, self.hidden_size)

        # Transformer blocks
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size, self.num_heads, params.mlp_ratio, params.qkv_bias
                )
                for _ in range(params.depth)
            ]
        )

        # Output layers
        self.final_layer_img = LastLayer(self.hidden_size, 1, params.in_channels)
        self.final_layer_txt = LastLayer(self.hidden_size, 1, params.context_dim)

    def forward(
        self, img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids: Tensor, timesteps: Tensor
    ) -> Tuple[Tensor, Tensor, dict]:
        b, c, h, w = img.shape

        # 1. Patchify and embed image
        img_patches = rearrange(
            img, "b c (gh ph) (gw pw) -> b (gh gw) (ph pw c)",
            ph=self.patch_size, pw=self.patch_size
        )
        img = self.img_in(img_patches)

        # 2. Embed timesteps
        vec = self.time_in(timestep_embedding(timesteps, 256)) if timesteps is not None else None

        # 3. Embed text
        txt = self.txt_in(txt)

        # 4. Get positional embeddings
        pe_single = self.pe_embedder(txt_ids) # Not used in DoubleStreamBlock, maybe for future use
        pe_double = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))

        # 5. Pass through transformer blocks
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, pe=(pe_single, pe_double), vec=vec)

        # 6. Project back to image and text dimensions
        img_out = self.final_layer_img(img, vec=vec)
        img_out = rearrange(
            img_out, "b (gh gw) (ph pw c) -> b c (gh ph) (gw pw)",
            ph=self.patch_size, pw=self.patch_size, gh=h//self.patch_size, gw=w//self.patch_size
        )
        txt_out = self.final_layer_txt(txt, vec=vec)
        
        return img_out, txt_out, {"final_txt": txt_out}


# --- FlowMo Model ---

def get_weights_to_fix(model):
    """Generator for weights to fix during training (related to MUP)."""
    with torch.no_grad():
        for name, module in model.named_modules():
            if "double_blocks" in name and isinstance(module, torch.nn.Linear):
                yield name, module.weight


class FlowMo(nn.Module):
    """
    The main FlowMo model, composed of a Flux encoder and a Flux decoder.
    It handles the end-to-end process of encoding, quantization, diffusion,
    and reconstruction.
    """
    def __init__(self, width, config):
        super().__init__()
        self.config = config
        self.image_size = config.data.image_size
        self.patch_size = config.model.patch_size
        self.code_length = config.model.code_length
        self.context_dim = config.model.context_dim
        self.dit_mode = "dit_b_4" # Base DiT configuration

        # Determine context dimension for the encoder based on quantization type
        if self.config.model.quantization_type == "kl":
            self.encoder_context_dim = self.context_dim * 2
        else:
            self.encoder_context_dim = self.context_dim

        # Initialize quantizer if specified
        if config.model.quantization_type == "lfq":
            self.quantizer = lookup_free_quantize.LFQ(
                codebook_size=2**config.model.codebook_size_for_entropy,
                dim=config.model.codebook_size_for_entropy,
                num_codebooks=1,
                token_factorization=False,
            )

        # --- Configure Encoder and Decoder Parameters ---
        enc_width = config.model.enc_mup_width if config.model.enc_mup_width is not None else width
        
        encoder_base_params = DIT_ZOO[self.dit_mode]
        decoder_base_params = DIT_ZOO[self.dit_mode]

        # Apply MUP scaling to hidden size and positional embedding dimensions
        enc_hidden_size = enc_width * (encoder_base_params['hidden_size'] // 4)
        dec_hidden_size = width * (decoder_base_params['hidden_size'] // 4)
        enc_axes_dim = [(d // 4) * enc_width for d in encoder_base_params['axes_dim']]
        dec_axes_dim = [(d // 4) * width for d in decoder_base_params['axes_dim']]

        encoder_params = FluxParams(
            in_channels=3 * self.patch_size**2,
            context_dim=self.encoder_context_dim,
            patch_size=self.patch_size,
            depth=config.model.enc_depth,
            **encoder_base_params,
        )
        decoder_params = FluxParams(
            in_channels=3 * self.patch_size**2,
            context_dim=self.context_dim + 1, # +1 for CFG mask
            patch_size=self.patch_size,
            depth=config.model.dec_depth,
            **decoder_base_params,
        )
        # width=4, dit_b_4 is the usual model
        encoder_params.hidden_size = enc_width * (encoder_params.hidden_size // 4)
        decoder_params.hidden_size = width * (decoder_params.hidden_size // 4)
        encoder_params.axes_dim = [
            (d // 4) * enc_width for d in encoder_params.axes_dim
        ]
        decoder_params.axes_dim = [(d // 4) * width for d in decoder_params.axes_dim]        

        self.encoder = Flux(encoder_params, name="encoder")
        self.decoder = Flux(decoder_params, name="decoder")

    @torch.compile
    def encode(self, img: Tensor) -> Tuple[Tensor, dict]:
        """Encodes an image into a latent code."""
        b, _, _, _ = img.shape
        img_idxs, txt_idxs = prepare_idxs(img, self.code_length, self.patch_size)
        # A dummy tensor is used for the text input to the encoder
        dummy_txt = torch.zeros((b, self.code_length, self.encoder_context_dim), device=img.device)
        _, code, aux = self.encoder(img, img_idxs, dummy_txt, txt_idxs, timesteps=None)
        return code, aux

    def _decode(self, img: Tensor, code: Tensor, timesteps: Tensor) -> Tuple[Tensor, dict]:
        """The core decoding function."""
        img_idxs, txt_idxs = prepare_idxs(img, self.code_length, self.patch_size)
        pred, _, decode_aux = self.decoder(img, img_idxs, code, txt_idxs, timesteps=timesteps)
        return pred, decode_aux

    @torch.compile
    def decode(self, *args, **kwargs):
        """Compiled version of the decoding function."""
        return self._decode(*args, **kwargs)

    @torch.compile
    def decode_checkpointed(self, *args, **kwargs):
        """Gradient-checkpointed version of the decoding function for memory efficiency."""
        return torch.utils.checkpoint.checkpoint(self._decode, *args, use_reentrant=False, **kwargs)

    def _quantize(self, code: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Applies the configured quantization method to the latent code.
        """
        b, t, f = code.shape
        quantizer_loss = torch.tensor(0.0, device=code.device)
        indices = None
        quantized_code = code

        q_type = self.config.model.quantization_type
        if q_type == "noop":
            pass # No quantization
        elif q_type == "kl":
            # VAE-style KL-regularized quantization
            mean, logvar = _get_diagonal_gaussian(einops.rearrange(code, "b t f -> b (f t)"))
            quantized_code = _sample_diagonal_gaussian(mean, logvar)
            quantized_code = einops.rearrange(quantized_code, "b (f t) -> b t f", f=f // 2, t=t)
            quantizer_loss = _kl_diagonal_gaussian(mean, logvar)
        elif q_type == "lfq":
            # Lookup-Free Quantization
            code_for_lfq = einops.rearrange(
                code, "b t (fg fh) -> b fg (t fh)",
                fg=self.config.model.codebook_size_for_entropy
            )
            (quantized, entropy_loss, indices), breakdown = self.quantizer(code_for_lfq, return_loss_breakdown=True)
            quantized_code = einops.rearrange(quantized, "b fg (t fh) -> b t (fg fh)", t=t)
            
            quantizer_loss = (
                entropy_loss * self.config.model.entropy_loss_weight
                + breakdown.commitment * self.config.model.commit_loss_weight
            )
        else:
            raise NotImplementedError(f"Quantization type '{q_type}' not implemented.")
        
        return quantized_code, indices, quantizer_loss

    def forward(self, img: Tensor, noised_img: Tensor, timesteps: Tensor, enable_cfg: bool = True) -> Tuple[Tensor, dict]:
        """
        The main forward pass for training.
        """
        aux = {}

        # 1. Encode the clean image to get the latent code
        code, _ = self.encode(img)
        aux["original_code"] = code

        # 2. Quantize the latent code
        quantized_code, _, aux["quantizer_loss"] = self._quantize(code)

        # 3. Prepare code for Classifier-Free Guidance (CFG)
        # Append a mask channel to the code
        mask = torch.ones_like(quantized_code[..., :1])
        code_with_mask = torch.cat([quantized_code, mask], dim=-1)
        
        if self.config.model.enable_cfg and enable_cfg:
            # Randomly drop out some codes for CFG
            cfg_mask = (torch.rand((img.shape[0],), device=code.device) > 0.1)[:, None, None]
            final_code = code_with_mask * cfg_mask
        else:
            final_code = code_with_mask

        # 4. Decode the noised image using the (potentially masked) code
        # v_est, decode_aux = self.decode(noised_img, final_code, timesteps)
        v_est, decode_aux = self.decode_checkpointed(noised_img, final_code, timesteps)
        aux.update(decode_aux)

        # 5. (Optional) Post-training sample for direct evaluation
        if self.config.model.posttrain_sample:
            aux["posttrain_sample"] = self.reconstruct_checkpoint(code_with_mask)

        return v_est, aux

    def reconstruct_checkpoint(self, code: Tensor) -> Tensor:
        """Performs a multi-step reconstruction using gradient checkpointing."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bs = code.shape[0]
            z = torch.randn((bs, 3, self.image_size, self.image_size), device=code.device)
            
            # Generate timesteps for sampling
            k = self.config.model.posttrain_sample_k
            ts = torch.rand((bs, k + 1)).cumsum(dim=1).to(code.device)
            ts = (ts - ts[:, :1]) / ts[:, -1:]
            ts = ts.flip(dims=(1,))
            dts = ts[:, :-1] - ts[:, 1:]

            for t, dt in zip(ts.T, dts.T):
                code_t = code
                if self.config.model.posttrain_sample_enable_cfg:
                    mask = (torch.rand((bs,), device=code.device) > 0.1)[:, None, None].to(code.dtype)
                    code_t = code * mask
                
                vc, _ = self.decode_checkpointed(z, code_t, t)
                z = z - dt[:, None, None, None] * vc
        return z

    @torch.no_grad()
    def reconstruct(self, images: Tensor, dtype=torch.bfloat16, code: Tensor = None) -> Tensor:
        """
        Reconstructs an image from a latent code using the diffusion sampling process.
        If code is not provided, it will be generated from the input images.
        """
        self.eval()
        sampling_config = self.config.eval.sampling
        with torch.autocast("cuda", dtype=dtype):
            if code is None:
                prequantized_code, _ = self.encode(images.cuda())
                code, _, _ = self._quantize(prequantized_code)

            z = torch.randn_like(images, device='cuda')
            
            # Prepare code with CFG mask
            mask = torch.ones_like(code[..., :1])
            code_with_mask = torch.cat([code, mask], dim=-1)

            # Prepare null code for CFG
            null_code = code_with_mask * 0.0 if sampling_config.cfg != 1.0 else None

            samples = rf_sample(
                self, z, code_with_mask, null_code=null_code,
                sample_steps=sampling_config.sample_steps,
                cfg=sampling_config.cfg,
                schedule=sampling_config.schedule,
            )
            return samples[-1].clip(-1, 1).to(torch.float32)

    @torch.no_grad()
    def reconstruct_noise(self, images: Tensor, noise_level: float = 100, dtype=torch.bfloat16, code: Tensor = None) -> Tensor:
        """
        Reconstructs an image from a latent code that has been corrupted by noise.
        """
        self.eval()
        if code is None:
            prequantized_code, _ = self.encode(images.cuda())
            code, _, _ = self._quantize(prequantized_code)
        
        # Apply noise to the latent code
        noisy_code = apply_awgn_noise(code, noise_level)
        
        return self.reconstruct(images, dtype=dtype, code=noisy_code)


# --- Loss and Sampling ---

def rf_loss(config, model, batch, aux_state):
    """Rectified Flow loss function."""
    x = batch["image"]
    b = x.size(0)

    # Sample timesteps
    if config.opt.schedule == "lognormal":
        t = torch.sigmoid(torch.randn((b,), device=x.device))
    elif config.opt.schedule == "uniform":
        t = torch.rand((b,), device=x.device)
    else: # fat_lognormal and others
        nt = torch.randn((b,)).to(x.device)
        t = torch.sigmoid(nt)
        t = torch.where(torch.rand_like(t) <= 0.9, t, torch.rand_like(t))

    # Create noised data
    z1 = torch.randn_like(x)
    t_reshaped = t.view([b, *([1] * (x.ndim - 1))])
    zt = (1 - t_reshaped) * x + t_reshaped * z1
    zt, t = zt.to(x.dtype), t.to(x.dtype)

    # Get model prediction
    v_theta, aux = model(img=x, noised_img=zt, timesteps=t)

    # Calculate diffusion loss
    loss = ((z1 - v_theta - x) ** 2).mean()
    aux["loss_dict"] = {"diffusion_loss": loss, "quantizer_loss": aux["quantizer_loss"]}

    # Calculate LPIPS loss if specified
    if config.opt.lpips_weight > 0.0:
        x_pred = aux.get("posttrain_sample", zt - v_theta * t_reshaped)
        lpips_dist = aux_state["lpips_model"](x, x_pred).mean()
        lpips_loss = config.opt.lpips_weight * lpips_dist
        aux["loss_dict"]["lpips_loss"] = lpips_loss
        loss += lpips_loss

    loss += aux["quantizer_loss"]
    aux["loss_dict"]["total_loss"] = loss
    return loss, aux


def _edm_to_flow_convention(noise_level):
    return noise_level / (1 + noise_level)


def rf_sample(
    model, z, code, null_code=None, sample_steps=25, cfg=2.0, schedule="linear"
):
    """Rectified Flow sampling loop."""
    b = z.size(0)
    if schedule == "linear":
        ts = torch.linspace(1.0, 0.0, sample_steps + 1)
    # Define sampling schedule
    if schedule.startswith("pow"):
        p = float(schedule.split("_")[1])
        ts = torch.arange(0, sample_steps + 1).flip(0) ** (1 / p) / sample_steps ** (1 / p)
    else: 
        raise NotImplementedError
    
    dts = ts[:-1] - ts[1:]
    ts = ts[:-1]

    # Define CFG interval if specified
    cfg_interval = None
    if model.config.eval.sampling.cfg_interval:
        cfg_lo, cfg_hi = ast.literal_eval(model.config.eval.sampling.cfg_interval)
        cfg_interval = _edm_to_flow_convention(cfg_lo), _edm_to_flow_convention(cfg_hi)

    images = []
    for t, dt in zip(ts, dts):
        timesteps = torch.full((b,), t, device=z.device)
        
        # # Get unconditional prediction
        # vu, _ = model.decode(img=z, timesteps=timesteps, code=null_code)
        
        # Get conditional prediction
        vc, _ = model.decode(img=z, timesteps=timesteps, code=code)
        
        # Apply CFG
        # v_pred = vu + cfg * (vc - vu)

        if null_code is not None and (
            cfg_interval is None
            or ((t.item() >= cfg_interval[0]) and (t.item() <= cfg_interval[1]))
        ):
            vu, _ = model.decode(img=z, timesteps=timesteps, code=null_code)
            vc = vu + cfg * (vc - vu)
        
        
        # Update step
        z = z - dt * vc
        images.append(z)
        
    return images


# --- Noise Simulation and Channel Coding ---

def apply_awgn_noise(code: Tensor, noise_level: float) -> Tensor:
    """
    Applies Additive White Gaussian Noise (AWGN) to a latent code.
    """
    if noise_level >= 100: return code
    
    signal_power = torch.mean(code ** 2)
    psnr_linear = 10 ** (noise_level / 10)
    noise_variance = signal_power / psnr_linear
    noise_std = torch.sqrt(noise_variance)
    
    noise = torch.randn_like(code) * noise_std
    return code + noise


# LDPC configuration
n_ldpc = 512
dv_ldpc, dc_ldpc = 2, 4
H_ldpc, G_ldpc = make_ldpc(n_ldpc, dv_ldpc, dc_ldpc, systematic=True, sparse=True)
k_ldpc = G_ldpc.shape[1]

def ldpc_protect_and_send(code: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Protects a latent code with LDPC, simulates transmission over an AWGN
    channel, and decodes it.
    """
    device = code.device
    
    # Quantize to uint8
    code_u8 = (((code.clamp(-1, 1) + 1) / 2) * 255).round().to(torch.uint8)
    bs, *dims = code_u8.shape
    flat_u8 = code_u8.view(bs, -1).cpu().numpy()
    
    # Unpack to bits and pad
    bits = np.unpackbits(flat_u8, axis=1)
    n_bits = bits.shape[1]
    pad = (-n_bits) % k_ldpc
    if pad:
        bits = np.pad(bits, ((0, 0), (0, pad)), 'constant')
    messages = bits.reshape(bs, -1, k_ldpc)
    
    # LDPC encode
    codewords = np.array([[encode(G_ldpc, msg, 100) for msg in batch] for batch in messages])
    
    # BPSK modulation and AWGN channel
    x = 1 - 2 * codewords
    sigma = np.sqrt(1 / (2 * (10**(snr_db / 10))))
    y = x + sigma * np.random.randn(*x.shape)
    
    # LDPC decode
    decoded_msgs = np.array([[decode(H_ldpc, cw, snr_db)[:k_ldpc] for cw in batch] for batch in y])
    decoded_bits = decoded_msgs.reshape(bs, -1)[:, :n_bits]
    
    # Reassemble tensor
    out_bytes = np.packbits(decoded_bits, axis=1).reshape(bs, *dims)
    code_rec = torch.from_numpy(out_bytes).to(device, dtype=torch.float32)
    return (code_rec / 255.0) * 2.0 - 1.0


