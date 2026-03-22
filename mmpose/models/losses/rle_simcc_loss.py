# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS
from ..utils.realnvp import RealNVP


@MODELS.register_module()
class RLE_SimCC_Loss(nn.Module):
    """RLE Loss.

    `Human Pose Regression With Residual Log-Likelihood Estimation
    arXiv: <https://arxiv.org/abs/2107.11291>`_.

    Code is modified from `the official implementation
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
        residual (bool): Option to add L1 loss and let the flow
            learn the residual error distribution.
        q_dis (string): Option for the identity Q(error) distribution,
            Options: "laplace" or "gaussian"
    """

    def __init__(self,
                 use_target_weight=False,
                 size_average=True,
                 residual=True,
                 q_distribution='laplace'):
        super(SimCC_RLE_Loss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.q_distribution = q_distribution

        self.flow_model = RealNVP()

    def forward(self, pred, sigma, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=1)

        Args:
            pred (Tensor[N, K, D]): Output regression.
            sigma (Tensor[N, K, D]): Output sigma.
            target (Tensor[N, K, D]): Target regression.
            target_weight (Tensor[N, K, D]):
                Weights across different joint types.
        """
        sigma = sigma.sigmoid()
        error = (pred - target) / (sigma + 1e-9)
        # (B, K, 1)
        log_phi = self.flow_model.log_prob(error.reshape(-1, 1))
        #log_phi = self.flow_model.log_prob(error.reshape(-1, 2))
        log_phi = log_phi.reshape(target.shape[0], target.shape[1], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], target.shape[1],1)

        nf_loss = log_sigma - log_phi

        if self.residual:
            assert self.q_distribution in ['laplace', 'gaussian']
            if self.q_distribution == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(
                    sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            loss = nf_loss + loss_q
        else:
            loss = nf_loss

        if self.use_target_weight:
            assert target_weight is not None
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()