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
import math

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from model.PTransformer import TransformerBlock
#from src.models.edge_conv import EdgeConvBlock

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict


#TODO: Implement EdgeConv from DGCNN as an laternative of PatchEmbedPointTransformer
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@dataclass
class DiTConfig:
    #Point Transformer config
    k: int = 8
    nblocks: int =  4
    num_centroids: int = 128
    hidden_size: int = 256#512
    in_features: int = 4 
    transformer_features: int = 32
    out_channels: int = 4
    name: str = "calopodit"
    energy_cond: bool = True
    num_points: int = 500
    num_classes: int = 2
    gap_classes: int = 4
    input_dim: int = 4
    # Usual DiT
    input_size: int = 32
    patch_size: int = 2
    depth: int = 13
    num_heads: int = 8
    mlp_ratio: int = 4.0
    class_dropout_prob: float = 0.1
    use_long_skip: bool = True
    final_conv: bool = False


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, time_factor: float = 1000.0):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                  t range [0, 1] for recitified flow, scaled by time_factor for numerical stability
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = t * time_factor
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

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

#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None

    def forward(self, x, c, x_skip=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, x_skip], dim=-1))

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
# class FinalLayer(nn.Module):
#     """
#     The final layer of DiT.
#     """

#     def __init__(self, hidden_size, patch_size, out_channels):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.linear = nn.Linear(
#             hidden_size, patch_size * patch_size * out_channels, bias=True
#         )
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
#         )

#     def forward(self, x, c):
#         shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
#         x = modulate(self.norm_final(x), shift, scale)
#         x = self.linear(x)
#         return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    config_class = DiTConfig

    def __init__(
        self,
        config: DiTConfig,
    ):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size
        self.num_heads = config.num_heads

        # self.x_embedder = PatchEmbed(
        #     config.input_size,
        #     config.patch_size,
        #     config.in_channels,
        #     config.hidden_size,
        #     bias=True,
        # )
        #FIXME add flag to  pick between PT, EConv, PFS 
        #NOTE point transformer replaced PatchEmbed
        self.x_embedder = TransformerBlock(
            config.in_features,
            config.hidden_size,#config.transformer_features,
            config.hidden_size,
            config.k,
            )
        #NOTE  EdgeConvBlock replaced point transformer
        # self.x_embedder = EdgeConvBlock(
        #     config.in_features,
        #     config.hidden_size,#config.transformer_features,
        #     config.hidden_size,
        #     config.k,
        #     )
        
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        if config.num_classes > 0:  # conditional generation on particle labels
            self.y_embedder = LabelEmbedder(
                config.num_classes, config.hidden_size, config.class_dropout_prob
            )
        if config.gap_classes > 0:  # conditional generation on gap
            self.gap_embedder = LabelEmbedder(
                config.gap_classes, config.hidden_size, config.class_dropout_prob
            )
        if config.energy_cond:
            self.e_embedder = EnergyEmbedder(
                config.hidden_size, config.class_dropout_prob
            )
        #FIXME initially not post emmbedding. Still need to figure out the equivalent in point transformer
        #num_patches = self.x_embedder.num_centroids
        # Will use fixed sin-cos embedding:
        #self.pos_embed = nn.Parameter(
        #    torch.zeros(1, num_patches, config.hidden_size), requires_grad=False
        #)

        self.in_blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    skip=False,
                )
                for _ in range(config.depth // 2)
            ]
        )

        self.mid_block = DiTBlock(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            skip=False,
        )

        self.out_blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    skip=config.use_long_skip,
                )
                for _ in range(config.depth // 2)
            ]
        )

        self.final_layer = FinalLayer(
            config.hidden_size, self.out_channels
        )
        # self.final_layer = FinalLayer(
        #     config.hidden_size, config.patch_size, self.out_channels
        # )
        self.final_conv = (
            nn.Conv2d(config.out_channels, config.out_channels, 3, padding=1)
            if config.final_conv
            else nn.Identity()
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        #FIXME? Not pos_emb
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(
        #     #self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        #     self.pos_embed.shape[-1], int(self.config.num_points**0.5)
        # )
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))        
        #Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if hasattr(self, "y_embedder"):
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        if hasattr(self, "gap_embedder"):
            nn.init.normal_(self.gap_embedder.embedding_table.weight, std=0.02)
        if hasattr(self, "e_embedder"):
            for layer in self.e_embedder.embedding_mlp:
                if isinstance(layer, nn.Linear):
                    # Use Xavier/Glorot for weights, often preferred for linear layers
                    nn.init.xavier_uniform_(layer.weight)
                    # Initialize bias to zero (if bias exists)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)              
            # Initialize null embedding parameter
            nn.init.normal_(self.e_embedder.null_embedding, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.in_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.mid_block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.mid_block.adaLN_modulation[-1].bias, 0)

        for block in self.out_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def save_pretrained(self, save_directory: str, filename: str = "dit"):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, f"{filename}_config.json")
        config_dict = asdict(self.config)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)
        print(f"Configuration saved to {config_path}")

        model_to_save = self.module if hasattr(self, "module") else self
        state_dict = model_to_save.state_dict()
        state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
        output_model_file = os.path.join(save_directory, f"{filename}_model.pt")
        torch.save(state_dict_cpu, output_model_file)
        print(f"Model weights saved to {output_model_file}")

    @classmethod
    def from_pretrained(
        cls, save_directory: str, filename: str = "dit", use_ema: bool = False
    ):
        config_path = os.path.join(save_directory, f"{filename}_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config = cls.config_class(**config_dict)
        model = cls(config)

        if use_ema:
            model_path = os.path.join(save_directory, f"{filename}_ema.pt")
        else:
            model_path = os.path.join(save_directory, f"{filename}_model.pt")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")

        return model

    # def unpatchify(self, x):
    #     """
    #     x: (N, T, patch_size**2 * C)
    #     imgs: (N, H, W, C)
    #     """
    #     c = self.out_channels
    #     #p = self.x_embedder.patch_size[0]
    #     h = w = int(x.shape[1] ** 0.5)
    #     assert h * w == x.shape[1]

    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    #     x = torch.einsum("nhwpqc->nchpwq", x)
    #     imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    #     return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None, gap=None, energy = None):
        """
        Forward pass of DiT.
        x: (N, HxW,C)=(N, Points, C) tensor of spatial point inputs 
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x= self.x_embedder(x)[0] # (points:(N,P,transformer_features))
        #x = (x_emb + self.pos_embed)  # (N, T, D), where T = H * W / patch_size ** 2
        
        #t = self.t_embedder(t)  # (N, D)
        c = self.t_embedder(t)  # c is (N, D)
        #Sequentially add other embeddings to 'c'
        if hasattr(self, "y_embedder"):
            y_emb = self.y_embedder(y, self.training)  # (N, D)
            # Add the y embedding to the conditional vector 'c'
            c = c + y_emb
        if hasattr(self, "gap_embedder"):
            gap_emb = self.gap_embedder(gap, self.training)  # (N, D)
            # Add the gap embedding to the conditional vector 'c'
            c = c + gap_emb

        if hasattr(self, "e_embedder"):
            energy_emb = self.e_embedder(energy, self.training)  # (N, D)
            # Add the energy embedding conditional vector 'c'
            c = c + energy_emb
        
        #'c' contains t + y (+ gap) (+ energy) 
        skips = []
        for idx, block in enumerate(self.in_blocks):
            x = block(x, c)  # (N, T, D)
            skips.append(x)

        x = self.mid_block(x, c)  # (N, T, D)

        for block in self.out_blocks:
            x = block(x, c, x_skip=skips.pop())  # (N, T, D), with long skip connections
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        #NOTE Not needed for point transformer?
        #x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = self.final_conv(x)  # (N, out_channels, H, W)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


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
