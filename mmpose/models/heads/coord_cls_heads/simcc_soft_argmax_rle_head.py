# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple, Union
import torch
from mmcv.cnn import build_conv_layer
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn
import numpy as np

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import keypoint_pck_accuracy,simcc_pck_accuracy
from mmpose.codecs.utils import get_heatmap_maximum, get_simcc_maximum,get_simcc_soft_maximum
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead
from mmengine.model import BaseModule, ModuleDict, Sequential

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class SimCC_Soft_Argmax_Rle_Head(BaseHead):
    """Top-down heatmap head introduced in `SimCC`_ by Li et al (2022). The
    head is composed of a few deconvolutional layers followed by a fully-
    connected layer to generate 1d representation from low-resolution feature
    maps.

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        input_size (tuple): Input image size in shape [w, h]
        in_featuremap_size (int | sequence[int]): Size of input feature map
        simcc_split_ratio (float): Split ratio of pixels
        deconv_type (str, optional): The type of deconv head which should
            be one of the following options:

                - ``'heatmap'``: make deconv layers in `HeatmapHead`
                - ``'vipnas'``: make deconv layers in `ViPNASHead`

            Defaults to ``'Heatmap'``
        deconv_out_channels (sequence[int]): The output channel number of each
            deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        deconv_num_groups (Sequence[int], optional): The group number of each
            deconv layer. Defaults to ``(16, 16, 16)``
        conv_out_channels (sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`SimCC`: https://arxiv.org/abs/2107.03332
    """

    _version = 2

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        deconv_type: str = 'heatmap',
        deconv_out_channels: OptIntSeq = (256, 256, 256),
        deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
        deconv_num_groups: OptIntSeq = (16, 16, 16),
        conv_out_channels: OptIntSeq = None,
        conv_kernel_sizes: OptIntSeq = None,
        final_layer: dict = dict(kernel_size=1),
        #修改
        simcc_loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
        rle_loss: ConfigType = dict(type='RLE_Loss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        if deconv_type not in {'heatmap', 'vipnas'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `deconv_type` value'
                f'{deconv_type}. Should be one of '
                '{"heatmap", "vipnas"}')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        # build losses
        self.loss_module = ModuleDict(
            dict(
                simcc=MODELS.build(simcc_loss),
                rle=MODELS.build(rle_loss),
            ))
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        num_deconv = len(deconv_out_channels) if deconv_out_channels else 0
        if num_deconv != 0:
            self.heatmap_size = tuple(
                [s * (2**num_deconv) for s in in_featuremap_size])

            # deconv layers + 1x1 conv
            self.deconv_head = self._make_deconv_head(
                in_channels=in_channels,
                out_channels=out_channels,
                deconv_type=deconv_type,
                deconv_out_channels=deconv_out_channels,
                deconv_kernel_sizes=deconv_kernel_sizes,
                deconv_num_groups=deconv_num_groups,
                conv_out_channels=conv_out_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                final_layer=final_layer)

            if final_layer is not None:
                in_channels = out_channels
            else:
                in_channels = deconv_out_channels[-1]

        else:
            self.deconv_head = None

            if final_layer is not None:
                cfg = dict(
                    type='Conv2d',
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1)
                cfg.update(final_layer)
                self.final_layer = build_conv_layer(cfg)
            else:
                self.final_layer = None

            self.heatmap_size = in_featuremap_size

        # Define SimCC layers
        flatten_dims = self.heatmap_size[0] * self.heatmap_size[1]

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.mlp_head_x = nn.Linear(flatten_dims, W)
        self.mlp_head_y = nn.Linear(flatten_dims, H)
        self.mlp_sigma = nn.Linear(flatten_dims, 2)#出方差


    def _make_deconv_head(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        deconv_type: str = 'heatmap',
        deconv_out_channels: OptIntSeq = (256, 256, 256),
        deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
        deconv_num_groups: OptIntSeq = (16, 16, 16),
        conv_out_channels: OptIntSeq = None,
        conv_kernel_sizes: OptIntSeq = None,
        final_layer: dict = dict(kernel_size=1)
    ) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        if deconv_type == 'heatmap':
            deconv_head = MODELS.build(
                dict(
                    type='HeatmapHead',
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    deconv_out_channels=deconv_out_channels,
                    deconv_kernel_sizes=deconv_kernel_sizes,
                    conv_out_channels=conv_out_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    final_layer=final_layer))
        else:
            deconv_head = MODELS.build(
                dict(
                    type='ViPNASHead',
                    in_channels=in_channels,
                    out_channels=out_channels,
                    deconv_out_channels=deconv_out_channels,
                    deconv_num_groups=deconv_num_groups,
                    conv_out_channels=conv_out_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    final_layer=final_layer))

        return deconv_head

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network.

        The input is the featuremap extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        if self.deconv_head is None:
            feats = feats[-1]
            if self.final_layer is not None:
                feats = self.final_layer(feats)
        else:
            feats = self.deconv_head(feats)

        x = torch.flatten(feats, 2)  # torch.Size([B, 17, 3072])

        pred_x = self.mlp_head_x(x)  # torch.Size([B, 17, 384])   simcc向量
        pred_y = self.mlp_head_y(x)  # torch.Size([B, 17, 512])
        pred_sigma = self.mlp_sigma(x)

        return pred_x, pred_y,pred_sigma



    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            List[InstanceData]: The pose predictions, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
                - keypoint_x_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the x direction
                - keypoint_y_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the y direction
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_pred_x, _batch_pred_y,_batch_sigma = self.forward(_feats)

            # 单独处理两个sigma
            _batch_sigma = _batch_sigma.sigmoid()

            _batch_pred_x_flip, _batch_pred_y_flip,_ = self.forward(_feats_flip)
            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                _batch_pred_x_flip,
                _batch_pred_y_flip,
                flip_indices=flip_indices)

            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
        else:
            batch_pred_x, batch_pred_y,batch_sigma= self.forward(feats)
            # 单独处理两个sigma
            batch_sigma= batch_sigma.sigmoid()

        preds = self.decode((batch_pred_x, batch_pred_y))  #解码器部分通过soft-argmax进行坐标解码  keypoints+scores


        if test_cfg.get('output_heatmaps', False):
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('The predicted simcc values are normalized for '
                              'visualization. This may cause discrepancy '
                              'between the keypoint scores and the 1D heatmaps'
                              '.')

            # normalize the predicted 1d distribution
            sigma = self.decoder.sigma
            batch_pred_x = get_simcc_normalized(batch_pred_x, sigma[0])
            batch_pred_y = get_simcc_normalized(batch_pred_y, sigma[1])

            B, K, _ = batch_pred_x.shape
            # B, K, Wx -> B, K, Wx, 1
            x = batch_pred_x.reshape(B, K, 1, -1)
            # B, K, Wy -> B, K, 1, Wy
            y = batch_pred_y.reshape(B, K, -1, 1)
            # B, K, Wx, Wy
            batch_heatmaps = torch.matmul(y, x)
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]

            for pred_instances, pred_x, pred_y in zip(preds,
                                                      to_numpy(batch_pred_x),
                                                      to_numpy(batch_pred_y)):

                pred_instances.keypoint_x_labels = pred_x[None]
                pred_instances.keypoint_y_labels = pred_y[None]

            return preds, pred_fields
        else:
            return preds

    #需要修改
    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        #出坐标信息
        pred_x, pred_y, pred_sigma = self.forward(feats)

        # preds = self.decode((pred_x, pred_y))  # 解码器部分通过soft-argmax进行坐标解码  keypoints+scores
        # keypoints_list = [instance_data.keypoints for instance_data in preds]  #得到坐标keypoints
        # #pred_coords=torch.tensor(keypoints_list)   #list转tensor 坐标  torch.Size([2, 1, 17, 2])
        # pred_coords= torch.tensor([item.cpu().detach().numpy() for item in keypoints_list]).cuda()
        # pred_coords=pred_coords.reshape(pred_coords.shape[0],pred_coords.shape[-2],pred_coords.shape[-1])


        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples],
                         dim=0)
        simcc_keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.simcc_keypoint_weights for d in batch_data_samples],
            dim=0,
        )

        keypoint_labels = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])


        rle_keypoint_weights = torch.cat([
            d.gt_instance_labels.rle_keypoint_weights for d in batch_data_samples])



        pred_simcc = (pred_x,  pred_y)
        gt_simcc = (gt_x, gt_y)
        # 通过soft-argmax  图片尺度
        output = to_numpy(pred_simcc)
        pred_x, pred_y = output
        pred_coords, _ = get_simcc_soft_maximum(pred_x, pred_y)
        pred_coords /= self.simcc_split_ratio
        pred_coords = torch.tensor(pred_coords).cuda()

        # calculate losses
        losses = dict()

        simcc_loss = self.loss_module['simcc'](pred_simcc, gt_simcc, simcc_keypoint_weights)

        rle_loss = self.loss_module['rle'](
            pred_coords, pred_sigma, keypoint_labels,rle_keypoint_weights.unsqueeze(-1))   #0

        losses.update({
            'loss/simcc': simcc_loss,
            'loss/rle': rle_loss,
        })

        # print("111",keypoint_labels[0])
        # print("222",pred_coords[0])
        # calculate accuracy
        # _, avg_acc, _ = keypoint_pck_accuracy(
        #     pred=to_numpy(pred_coords),
        #     gt=to_numpy(keypoint_labels),
        #     mask=to_numpy(rle_keypoint_weights) > 0,
        #     thr=0.05,
        #     norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32))

        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(simcc_keypoint_weights) > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
        losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg
