
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.registry import MODELS


@MODELS.register_module()
class TokenDistilLoss(nn.Module):
    """Tokens Distillation loss."""

    def __init__(self, name,
                 use_this,
                 dist_type='L2',
                 loss_weight=1.):
        super().__init__()
        if dist_type == 'L2':
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            # TO BE IMPLEMENTED
            self.criterion = None
        self.loss_weight = loss_weight

    def forward(self, token_s, token_t):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K
            - embedding dimension: D
        Args:
            token_s (torch.Tensor[N, K, D]): tokens of student.
            token_t (torch.Tensor[N, K, D]): tokens of teacher.
        """
        loss = self.criterion(token_s, token_t)

        return loss * self.loss_weight