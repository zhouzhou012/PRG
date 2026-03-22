# Copyright (c) OpenMMLab. All rights reserved.
from .rtmcc_head import RTMCCHead
from .simcc_head import SimCCHead
from .simcc_soft_argmax_rle_head import SimCC_Soft_Argmax_Rle_Head


__all__ = ['SimCCHead', 'RTMCCHead','SimCC_Soft_Argmax_Rle_Head']
