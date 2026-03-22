# from curses import reset_shell_mode
import os
from re import I
# from this import d
import time
import math
import copy
import pickle
import argparse
import functools
from collections import deque

import numpy as np
from scipy import integrate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
from torch import autograd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]





def get_sigmas():
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(np.log(50), np.log(0.01), 1000))

    return sigmas


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class GFDenosier(nn.Module):
    """
    Independent condition feature projection layers for each block
    """
    def __init__(self,
        n_joints=17, joint_dim=2, hidden_dim=64, embed_dim=32, cond_dim=2,
        n_blocks=2):
        super(GFDenosier, self).__init__()

        self.n_joints = n_joints
        self.joint_dim = joint_dim
        self.n_blocks = n_blocks

        self.act = nn.SiLU()

        self.pre_dense= nn.Linear(n_joints * joint_dim, hidden_dim)
        self.pre_dense_t = nn.Linear(embed_dim, hidden_dim)
        self.pre_dense_cond = nn.Linear(hidden_dim, hidden_dim)
        self.pre_gnorm = nn.GroupNorm(32, num_channels=hidden_dim)
        self.dropout = nn.Dropout(p=0.25)

        # time embedding
        # self.time_embedding_type = config.model.embedding_type.lower()
        self.time_embedding_type = 'positional'
        if self.time_embedding_type == 'fourier':
            self.gauss_proj = GaussianFourierProjection(embed_dim=embed_dim)
        elif self.time_embedding_type == 'positional':
            self.posit_proj = functools.partial(get_timestep_embedding, embedding_dim=embed_dim)
        else:
            assert 0

        self.shared_time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
        )
        self.register_buffer('sigmas', torch.tensor(get_sigmas()))

        # conditional embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(n_joints * cond_dim, hidden_dim),
            self.act
        )


        for idx in range(n_blocks):
            setattr(self, f'b{idx+1}_dense1', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense1_t', nn.Linear(embed_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense1_cond', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_gnorm1', nn.GroupNorm(32, num_channels=hidden_dim))

            setattr(self, f'b{idx+1}_dense2', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense2_t', nn.Linear(embed_dim, hidden_dim))
            setattr(self, f'b{idx+1}_dense2_cond', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx+1}_gnorm2', nn.GroupNorm(32, num_channels=hidden_dim))

        # self.post_dense = nn.Linear(hidden_dim, n_joints * joint_dim)
        self.post_dense = nn.Linear(hidden_dim, n_joints * 4)


    def forward(self, batch,condition, t ):
        """
        batch: [B, j, 3] or [B, j, 1]
        t: [B]
        condition: [B, j, 2 or 3]
        mask: [B, j, 2 or 3] only used during evaluation
        Return: [B, j, 3] or [B, j, 1] same dim as batch
        """
        bs = batch.shape[0]


        batch = batch.view(bs, -1)  # [B, j*2]
        condition = condition.view(bs, -1)  # [B, j*2]

        # time embedding
        if self.time_embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = self.gauss_proj(torch.log(used_sigmas))
        elif self.time_embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = t
            used_sigmas = self.sigmas[t.long()]
            temb = self.posit_proj(timesteps)
        else:
            raise ValueError(f'time embedding type {self.time_embedding_type} unknown.')

        temb = self.shared_time_embed(temb)

        # cond embedding
        cond = self.cond_embed(condition)  # [B, hidden]

        h = self.pre_dense(batch)
        h += self.pre_dense_t(temb)
        h += self.pre_dense_cond(cond)
        h = self.pre_gnorm(h)
        h = self.act(h)
        h = self.dropout(h)

        for idx in range(self.n_blocks):
            h1 = getattr(self, f'b{idx+1}_dense1')(h)
            h1 += getattr(self, f'b{idx+1}_dense1_t')(temb)
            h1 += getattr(self, f'b{idx+1}_dense1_cond')(cond)
            h1 = getattr(self, f'b{idx+1}_gnorm1')(h1)
            h1 = self.act(h1)
            # dropout, maybe
            h1 = self.dropout(h1)

            h2 = getattr(self, f'b{idx+1}_dense2')(h1)
            h2 += getattr(self, f'b{idx+1}_dense2_t')(temb)
            h2 += getattr(self, f'b{idx+1}_dense2_cond')(cond)
            h2 = getattr(self, f'b{idx+1}_gnorm2')(h2)
            h2 = self.act(h2)
            # dropout, maybe
            h2 = self.dropout(h2)

            h = h + h2

        res = self.post_dense(h)  # [B, j*4]
        res = res.view(bs, self.n_joints, -1)  # [B, j, 4]
        # print("111",res)

        ''' normalize the output '''
        # if self.config.model.scale_by_sigma:
        #     used_sigmas = used_sigmas.reshape((bs, 1, 1))
        #     res = res / used_sigmas
        # used_sigmas = used_sigmas.reshape((bs, 1, 1))
        # res = res / used_sigmas
        # print("111",res.shape)

        return res
