# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union
import math
import torch
from mmengine.structures import PixelData
from torch import Tensor, nn
import torch.nn.functional as F
from collections import namedtuple
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from mmpose.codecs.utils.gaussian_heatmap import (generate_gaussian_heatmaps,
                                     generate_unbiased_gaussian_heatmaps)
import numpy as np
import cv2
from einops import rearrange
from mmpose.utils.tensor_utils import to_numpy
OptIntSeq = Optional[Sequence[int]]
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas

class DDPM(nn.Module):

    _version = 2

    def __init__(self,
                 denoise,
                 sampling_timesteps=None):
        super().__init__()
        # build diffusion
        timesteps =4
        sampling_timesteps = sampling_timesteps
        self.objective = 'pred_x0'
        # betas = cosine_beta_schedule(timesteps)
        betas = make_beta_schedule('linear', timesteps, 0.1, 0.99)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.denoise=denoise


        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0.  #在反向过程中不加入噪声
        self.scale = 1.0


        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))





    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # 生成alpha_cp
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # 扩散公式
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # 进行前向扩散
    def prepare_diffusion_concat(self, x_0):
        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise = torch.randn_like(x_0)
        x_start = x_0

        x_start = x_start * self.scale
        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # # 这段代码对输入的张量 x 进行了裁剪和归一化处理，使其取值范围在 [min, max] 内，并且保留了原始张量的形状和数据类型。
        # x = torch.clamp(x, min=0.*self.scale, max=1. * self.scale)
        # x = x / self.scale

        return x, noise, t

    def prepare_targets(self, targets):
        diffused_poses = []
        noises = []
        ts = []
        for i in range(0,targets.shape[0]):#batch
            targets_per_sample = targets[i]

            d_poses, d_noise, d_t = self.prepare_diffusion_concat(targets_per_sample)
            diffused_poses.append(d_poses)
            noises.append(d_noise)
            ts.append(d_t)

        return torch.stack(diffused_poses), torch.stack(noises), torch.stack(ts)


    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x_t, cond, t):
        # x_t = torch.clamp(x_t, min=0.*self.scale, max=1. * self.scale)
        # x_t = x_t / self.scale

        pred = self.denoise(x_t,cond, t)
        x_start = pred
        # x_start = x_start * self.scale
        # x_start = torch.clamp(x_start, min=0.*self.scale, max=1. * self.scale)

        pred_noise = self.predict_noise_from_start(x_t, t, x_start)
        pred_noise = pred_noise.float()

        return ModelPrediction(pred_noise, x_start)


    def ddim_sample(self, cond):
        batch = cond.shape[0]
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        #成一个服从标准正态分布（均值为0，标准差为1）的随机张量作为初始噪声
        x_t = torch.randn_like(cond, device='cuda')

        preds_all=[]
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, dtype=torch.long).cuda()

            preds = self.model_predictions(x_t, cond, time_cond)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            preds_all.append(x_start)

            if time_next < 0:
                x_t = x_start
                continue

            #渐进式去噪
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_t)

            #DDIM的采样公式
            x_t = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        # return torch.stack(preds_all, dim=1)
        # 取最后的结果
        return x_t



    def forward(self,cond,x_0=None) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        if self.training:
            # 将真实信息进行扩散加噪
            x_ts, noises, t = self.prepare_targets(x_0)
            x_ts = x_ts.float()
            t = t.squeeze(-1)
            # 条件去噪
            preds = self.denoise(x_ts, cond, t)

        else: #推理
            preds = self.ddim_sample(cond)

        return preds




