# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union, Dict, Any
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead
import torch.nn.functional as F
OptIntSeq = Optional[Sequence[int]]

@MODELS.register_module()
class Y_HeatmapHead(BaseHead):
    """Top-down heatmap head introduced in `Simple Baselines`_ by Xiao et al
    (2018). The head is composed of a few deconvolutional layers followed by a
    convolutional layer to generate heatmaps from low-resolution feature maps.

    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        deconv_out_channels (Sequence[int], optional): The output channel
            number of each deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        conv_out_channels (Sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KeypointMSELoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
        extra (dict, optional): Extra configurations.
            Defaults to ``None``

    .. _`Simple Baselines`: https://arxiv.org/abs/1804.06208
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 final_layer_unsup:dict=dict(kernel_size=1),
                 loss: ConfigType = dict(type='KeypointMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        # self.supervised_conv=nn.Identity()
        # self.bottleneck=nn.Identity()
        self.pose_upsampler = nn.Upsample(scale_factor=4, mode='bilinear')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()

        if final_layer_unsup is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer_unsup)
            self.final_layer_unsup = build_conv_layer(cfg)
        else:
            self.final_layer_unsup = nn.Identity()

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            #layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.GroupNorm(num_groups=32, num_channels=out_channels))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.LeakyReLU(inplace=True))
            # layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            #layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.GroupNorm(num_groups=32, num_channels=out_channels))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.LeakyReLU(inplace=True))
            # layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    @property
    # def default_init_cfg(self):
    #     init_cfg = [
    #         dict(
    #             type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
    #         dict(type='Constant', layer='BatchNorm2d', val=1)
    #     ]
    #     return init_cfg

    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='GroupNorm2d', val=1)
        ]
        return init_cfg

    # def forward(self, feats: Tuple[Tensor]) -> Tensor:
    #     """Forward the network. The input is multi scale feature maps and the
    #     output is the heatmap.
    #
    #     Args:
    #         feats (Tuple[Tensor]): Multi scale feature maps.
    #
    #     Returns:
    #         Tensor: output heatmap.
    #     """
    #     x = feats[-1]
    #     x = self.deconv_layers(x)
    #     x = self.conv_layers(x)
    #     x = self.final_layer(x)
    #
    #     return x

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        # 获取特征
        x = feats[-1]
        # 共享的卷积层
        x = self.deconv_layers(x)
        x = self.conv_layers(x)

        # 预测目标热图
        predicted_heatmap = self.final_layer(x)
        # unsup_predicted_heatmap = self.final_layer_unsup(x)



        # 自监督分支
        # self_supervised_heatmap=unsup_predicted_heatmap
        # self_supervised_heatmap = self.get_gaussian_map(unsup_predicted_heatmap)
        # self_supervised_heatmap = self.pose_upsampler(self_supervised_heatmap)

        self_supervised_heatmap = self._flat_softmax(predicted_heatmap)

        # 返回两个热图
        return predicted_heatmap,self_supervised_heatmap

    def group_heatmaps(self,gaussian_map):
        """
        gaussian_map: Tensor of shape (B, 14, H, W)
        Return: Tensor of shape (B, len(groups), H, W)
        """
        groups = {
            "head": [12, 13],
            "left_arm": [0, 2, 4],
            "right_arm": [1, 3, 5],
            "left_leg": [6, 8, 10],
            "right_leg": [7, 9, 11],
        }
        binary_masks=[]
        for group_name, indices in groups.items():
            selected = gaussian_map[:, indices, :, :]  # (B, len(indices), H, W)
            group_map = selected.sum(dim=1)  # (B, H, W)
            group_map = group_map / (group_map.amax(dim=(1, 2), keepdim=True) + 1e-6)
            # binary mask: 1 if pixel > threshold, else 0
            #binary_mask = (group_map > 0.3).float()
            binary_mask = torch.sigmoid(group_map - 0.5)
            binary_masks.append(binary_mask)

        binary_masks = torch.stack(binary_masks, dim=1)  # (B, G, H, W)

        return binary_masks

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_heatmaps,_batch_self_supervised_heatmaps= self.forward(_feats)
            _batch_heatmaps_flip,_batch_self_supervised_heatmaps_flip=self.forward(_feats_flip)
            _batch_heatmaps_flip = flip_heatmaps(
                _batch_heatmaps_flip,
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            _batch_self_supervised_heatmaps_flip = flip_heatmaps(
                _batch_self_supervised_heatmaps_flip,
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
            batch_self_supervised_heatmaps = (_batch_self_supervised_heatmaps + _batch_self_supervised_heatmaps_flip) * 0.5
        else:
            batch_heatmaps,batch_self_supervised_heatmaps= self.forward(feats)

        # batch_binary_masks = self.group_heatmaps(batch_self_supervised_heatmaps)
        # batch_binary_masks = self.pose_upsampler(batch_binary_masks)

        preds = self.decode(batch_heatmaps)
        # preds = self.decode(batch_self_supervised_heatmaps)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields,batch_self_supervised_heatmaps
        else:
            return preds,batch_heatmaps,batch_self_supervised_heatmaps

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> Tuple[Dict[str, Union[Tensor, Any]], Tensor, Tensor, Any]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """
        # global num
        pred_fields,self_supervised_heatmap = self.forward(feats)

        # binary_masks = self.group_heatmaps(self_supervised_heatmap)
        # binary_masks = self.pose_upsampler(binary_masks)

        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

       # return losses,self_supervised_heatmap
        return losses,pred_fields,self_supervised_heatmap

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`DeepposeRegressionHead` (before MMPose v1.0.0) to a
        compatible format of :class:`RegressionHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layers, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weights of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            else:
                k_new = k

            state_dict[prefix + k_new] = v

    def get_gaussian_map(self, heatmaps):
        self.sigma=2
        n, c, h, w = heatmaps.size()

        heatmaps_y = F.softmax(heatmaps.sum(dim=3), dim=2).reshape(n, c, h, 1)
        heatmaps_x = F.softmax(heatmaps.sum(dim=2), dim=2).reshape(n, c, 1, w)

        coord_y = heatmaps.new_tensor(range(h)).reshape(1, 1, h, 1)
        coord_x = heatmaps.new_tensor(range(w)).reshape(1, 1, 1, w)

        joints_y = heatmaps_y * coord_y
        joints_x = heatmaps_x * coord_x

        joints_y = joints_y.sum(dim=2)
        joints_x = joints_x.sum(dim=3)

        joints_y = joints_y.reshape(n, c, 1, 1)  # 关键点坐标
        joints_x = joints_x.reshape(n, c, 1, 1)

        gaussian_map = torch.exp(-((coord_y - joints_y) ** 2 + (coord_x - joints_x) ** 2) / (2 * self.sigma ** 2))
        return gaussian_map

    def _flat_softmax(self, featmaps: Tensor) -> Tensor:
        """Use Softmax to normalize the featmaps in depthwise."""

        _, N, H, W = featmaps.shape

        featmaps = featmaps.reshape(-1, N, H * W)
        heatmaps = F.softmax(featmaps, dim=2)

        return heatmaps.reshape(-1, N, H, W)


