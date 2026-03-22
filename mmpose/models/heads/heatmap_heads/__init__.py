# Copyright (c) OpenMMLab. All rights reserved.
from .ae_head import AssociativeEmbeddingHead
from .cid_head import CIDHead
from .cpm_head import CPMHead
from .heatmap_head import HeatmapHead
from .mspn_head import MSPNHead
from .vipnas_head import ViPNASHead
from .hrnet_model_head import HRNet_Model_Head
from .y_heatmap_head import Y_HeatmapHead
from .cpn_head import CPNHead
from .cpn_gn_head import CPN_GNHead
__all__ = [
    'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
    'AssociativeEmbeddingHead', 'CIDHead','HRNet_Model_Head','Y_HeatmapHead',
    'CPNHead','CPN_GNHead'
]
