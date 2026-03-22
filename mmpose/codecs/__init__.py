# Copyright (c) OpenMMLab. All rights reserved.
from .associative_embedding import AssociativeEmbedding
from .decoupled_heatmap import DecoupledHeatmap
from .image_pose_lifting import ImagePoseLifting
from .integral_regression_label import IntegralRegressionLabel
from .megvii_heatmap import MegviiHeatmap
from .msra_heatmap import MSRAHeatmap
from .img_msra_heatmap import Img_MSRAHeatmap
from .regression_label import RegressionLabel
from .simcc_label import SimCCLabel
from .img_simcc_label import Img_SimCCLabel
from .spr import SPR
from .udp_heatmap import UDPHeatmap
from .video_pose_lifting import VideoPoseLifting
from .img_udp_heatmap import Img_UDPHeatmap
from .cpn_heatmap import CPNHeatmap
__all__ = [
    'MSRAHeatmap', 'MegviiHeatmap', 'UDPHeatmap', 'RegressionLabel',
    'SimCCLabel', 'IntegralRegressionLabel', 'AssociativeEmbedding', 'SPR',
    'DecoupledHeatmap', 'VideoPoseLifting', 'ImagePoseLifting','Img_MSRAHeatmap',
    'Img_UDPHeatmap','Img_SimCCLabel','CPNHeatmap'
]
