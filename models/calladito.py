# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
# Copied from: https://github.com/facebookresearch/DiT/blob/main/models.py

import torch
import torch.nn as nn
import numpy as np
import math
import os
import json

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

#from src.models.edge_conv import EdgeConvBlock
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict


#TODO: Implement EdgeConv from DGCNN as an laternative of PatchEmbedPointTransformer
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@dataclass
class DiTConfig:
    name: str = "calladito"
    input_dim: int = 512
    hidden_size: int = 1024
    depth: int = 13
    num_heads: int = 16
    num_particle_classes: int = 2
    gap_classes: int = 4
    in_features: int = 4 #4 channels: x,y,z,E 
    #OLD
    energy_cond: bool = True
    k: int = 8
    nblocks: int =  4
    num_centroids: int = 128
    transformer_features: int = 32
    out_channels: int = 4    
    #num_points: int = 500
    patch_size: int = 2
    mlp_ratio: int = 4.0
    class_dropout_prob: float = 0.1
    use_long_skip: bool = True
    final_conv: bool = False


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

"""his embedder uses the standard sinusoidal approach (popularized by Transformer and Diffusion models)
 to map a discrete integer timestep t into a continuous vector space that the Transformer can understand."""
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        # We add +1 to the table to act as the "null" token
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels, train, force_drop_ids=None):
        if force_drop_ids is None:
            if train and self.dropout_prob > 0:
                drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            else:
                drop_ids = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.bool)
        else:
            # force_drop_ids comes from the sampling loop (usually 1 for drop, 0 for keep)
            drop_ids = force_drop_ids.bool()

        # Create a copy so we don't modify the original labels tensor
        labels_copy = labels.clone()
        # Replace dropped samples with the 'null' index (the last index)
        labels_copy[drop_ids] = self.num_classes
        
        return self.embedding_table(labels_copy)
    
class EnergyEmbedder(nn.Module):
    """
    Embeds continuous (float) energy class into vector representations.
    Preserves label dropout for Classifier-Free Guidance (CFG).
    """

    def __init__(self, hidden_size, dropout_prob, mlp_layers=3):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size

        # The MLP maps the single float input (dimension 1) to the hidden_size.
        mlp_modules = [
            nn.Linear(1, hidden_size),
            nn.SiLU(), # Swish is a common activation for this
        ]
        for _ in range(mlp_layers - 1):
             mlp_modules.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
            ])
        
        self.embedding_mlp = nn.Sequential(*mlp_modules)

        # This replaces the need for an extra slot in nn.Embedding
        self.null_embedding = nn.Parameter(torch.randn(1, hidden_size))

    def forward(self, labels, train, force_drop_ids=None):
        """
        Args:
            labels (torch.Tensor): Input tensor of continuous labels (B, 1).
            train (bool): Whether in training mode (for dropout).
            force_drop_ids (torch.Tensor, optional): Explicitly define which samples to drop.
        Returns:
            torch.Tensor: Embedded label vectors (B, hidden_size).
        """
        B = labels.shape[0]

        # Ensure labels are of shape (B, 1) for the MLP input
        if labels.ndim == 1:
             labels = labels.unsqueeze(1)
             
        # Generate initial embeddings from the continuous value
        embeddings = self.embedding_mlp(labels)

        # Apply dropout logic for Classifier-Free Guidance (CFG)
        use_dropout = self.dropout_prob > 0
        
        if (train and use_dropout) or (force_drop_ids is not None):
            
            if force_drop_ids is None:
                # Determine which samples to drop based on dropout_prob
                drop_ids = (
                    torch.rand(B, device=labels.device) < self.dropout_prob
                )
            else:
                drop_ids = force_drop_ids.bool()

            # Broadcast the null embedding to the samples marked for dropout
            null_broadcast = self.null_embedding.expand(B, self.hidden_size)
            
            # Replace the generated embeddings with the null embedding
            embeddings[drop_ids] = null_broadcast[drop_ids]

        return embeddings
"""In a standard Transformer, LayerNorm is static. In AdaLN, we use a MLP to predict the gain (γ) and bias (β) parameters from our conditioning
 vector (t+e_init). This allows the normalization to adapt based on the conditioning information, enabling more dynamic feature modulation.
 
 AdaLN: Instead of the model having to "find" the energy in a long sequence of tokens, the energy 
 effectively becomes the global environment (the scale/shift) in which the Transformer operates."""
class AdaLN(nn.Module):
    def __init__(self, latent_dim, cond_emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        # Predicts gamma and beta for each feature
        self.linear = nn.Linear(cond_emb_dim, latent_dim * 2)

    def forward(self, x, cond_emb):
        # x: [B, latent_dim], cond_emb: [B, cond_emb_dim]
        gate = self.linear(cond_emb) # [B, latent_dim * 2]
        gamma, beta = gate.chunk(2, dim=-1) # Split into scale and shift
        
        x = self.norm(x)
        # Apply the predicted scale and shift
        return x * (1 + gamma) + beta
#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A Transformer block that conditions on timestep (t) and optional metadata (y).
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        # AdaLN modulation: 6 parameters per block (scale/shift/gate for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # In the original DiT paper, the adaLN_modulation MLP (the one that predicts shift/scale/gate) must be initialized such that it starts by doing nothing.
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        # c is the combined conditional embedding: (t, e_init) so far
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Attention Branch
        x_norm = self.norm1(x)
        # Apply scale and shift
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP Branch
        x_norm = self.norm2(x)
        # Apply scale and shift
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x
class LatentDiT(nn.Module):
    config_class = DiTConfig
    #TODO add class conditioning embedding and CFG
    #def __init__(self, input_dim=512, hidden_size=1024, depth=12, num_heads=16):
    def __init__(self, 
                 config: DiTConfig,
                 ):
        super().__init__()
        #input_dim = latent_dim
        self.latent_embed = nn.Linear(config.input_dim, config.hidden_size)
        #  Timestep Embedding
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        #  Energy Embedding
        self.e_embedder = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        # Input Projection (z is 512, hidden is 768)
        self.x_proj = nn.Linear(config.input_dim, config.hidden_size)

        if config.num_particle_classes > 0:  # conditional generation on particle labels
            self.y_embedder = LabelEmbedder(
                config.num_particle_classes, config.hidden_size, config.class_dropout_prob
            )
        if config.gap_classes > 0:  # conditional generation on gap
            self.gap_embedder = LabelEmbedder(
                config.gap_classes, config.hidden_size, config.class_dropout_prob
            )
        
        self.blocks = nn.ModuleList([
            DiTBlock(config.hidden_size, config.num_heads) for _ in range(config.depth)
        ])

        self.final_layer = nn.Linear(config.hidden_size, config.input_dim)
        #self.final_layer = FinalLayer(config.hidden_size, config.input_dim)

    def forward(self, z_t, t, e_init=None, y=None, gap=None, mask_condition=None):
        """
        z: [B, 512] (Noisy latent)
        t: [B] (Timestep)
        e_init: [B, 1] (Conditioning)
        mask_condition: [B] Binary mask for CFG (1 to keep condition, 0 to drop)
        """
        if e_init.dim() == 1:
            e_init = e_init.unsqueeze(1) # [B] -> [B, 1]
        # CFG logic: If we drop the condition, we use a learned "null" or zero
        if mask_condition is not None:
            e_init = e_init * mask_condition.view(-1, 1)
        # 1. Timestep embedding (standard sinusoidal)
        # If self.t_embed is nn.Linear(1024, ...), pass 1024 here
        #t_emb = self.t_embed(self.pos_encoding(t, 1024))
        
        c_t = self.t_embedder(t)      # [B, hidden_size]
        # Embed Energy
        c_e = self.e_embedder(e_init) # [B, hidden_size]
        # Combine conditioning
        c = c_t + c_e                 # Merged context
        
        # Prepare input tokens
        # Since z is a single vector, we treat it as 1 token
        x = self.x_proj(z_t).unsqueeze(1) # [B, 1, hidden_size]
        
        # Pass through blocks with AdaLN conditioning
        for block in self.blocks:
            x = block(x, c)
            
        return self.final_layer(x).squeeze(1) # [B, 512]

#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}
