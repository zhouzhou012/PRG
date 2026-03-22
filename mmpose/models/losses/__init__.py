# Copyright (c) OpenMMLab. All rights reserved.
from .ae_loss import AssociativeEmbeddingLoss
from .classification_loss import BCELoss, JSDiscretLoss, KLDiscretLoss,NMTCritierion
from .heatmap_loss import (AdaptiveWingLoss, KeypointMSELoss,
                           KeypointOHKMMSELoss)
from .loss_wrappers import CombinedLoss, MultipleLossWrapper
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, RLELoss,
                              SemiSupervisionLoss, SmoothL1Loss,
                              SoftWeightSmoothL1Loss, SoftWingLoss, WingLoss,L1_Charbonnier_loss)
from .rle_simcc_loss import RLE_SimCC_Loss
from .fea import FeaLoss,CORAL_loss
from .kd import KDLoss,JointsKLDLoss
from .dist_loss import TokenDistilLoss
from .channel_wise_fea import Channel_FeaLoss


__all__ = [
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'KLDiscretLoss', 'MultipleLossWrapper', 'JSDiscretLoss', 'CombinedLoss',
    'AssociativeEmbeddingLoss', 'SoftWeightSmoothL1Loss','RLE_SimCC_Loss',
    'L1_Charbonnier_loss','FeaLoss','KDLoss','TokenDistilLoss','JointsKLDLoss',
    'Channel_FeaLoss','CORAL_loss','NMTCritierion'
]
