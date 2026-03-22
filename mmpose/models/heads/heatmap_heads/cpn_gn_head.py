# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn
from mmengine.model import BaseModule, ModuleDict
from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead
import math
import torch.nn.functional as F

OptIntSeq = Optional[Sequence[int]]


class globalNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(globalNet, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        # layers.append(nn.BatchNorm2d(256))
        layers.append(nn.GroupNorm(num_groups=32, num_channels=256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        #layers.append(nn.BatchNorm2d(256))
        layers.append(nn.GroupNorm(num_groups=32, num_channels=256))

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        #layers.append(nn.BatchNorm2d(256))
        layers.append(nn.GroupNorm(num_groups=32, num_channels=256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        # layers.append(nn.BatchNorm2d(num_class))
        layers.append(nn.GroupNorm(num_groups=7, num_channels=num_class))

        return nn.Sequential(*layers)

    def forward(self, x):
        global_fms, global_outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            global_fms.append(feature)
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
            feature = self.predict[i](feature)
            global_outs.append(feature)

        return global_fms, global_outs


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.GroupNorm(num_groups=32, num_channels=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.GroupNorm(num_groups=32, num_channels=planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        #self.bn3 = nn.BatchNorm2d(planes * 2)
        self.bn3 = nn.GroupNorm(num_groups=32, num_channels=planes*2)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 2,
                      kernel_size=1, stride=stride, bias=False),
            #nn.BatchNorm2d(planes * 2),
            nn.GroupNorm(num_groups=32, num_channels=planes*2),
        )

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class refineNet(nn.Module):
    def __init__(self, lateral_channel, out_shape, num_class):
        super(refineNet, self).__init__()
        cascade = []
        num_cascade = 4
        for i in range(num_cascade):
            cascade.append(self._make_layer(lateral_channel, num_cascade - i - 1, out_shape))
        self.cascade = nn.ModuleList(cascade)
        self.final_predict = self._predict(4 * lateral_channel, num_class)
        #self.final_predict_unsup = self._predict(4 * lateral_channel, num_class)

    def _make_layer(self, input_channel, num, output_shape):
        layers = []
        for i in range(num):
            layers.append(Bottleneck(input_channel, 128))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)

    def _predict(self, input_channel, num_class):
        layers = []
        layers.append(Bottleneck(input_channel, 128))
        layers.append(nn.Conv2d(256, num_class,
                                kernel_size=3, stride=1, padding=1, bias=False))
        #layers.append(nn.BatchNorm2d(num_class))
        layers.append(nn.GroupNorm(num_groups=7, num_channels=num_class))
        return nn.Sequential(*layers)

    def forward(self, x):
        refine_fms = []
        for i in range(4):
            refine_fms.append(self.cascade[i](x[i]))
        x = torch.cat(refine_fms, dim=1)
        out = self.final_predict(x)
        #unsup_out = self.final_predict_unsup(x)
        # return out, unsup_out
        return out


@MODELS.register_module()
class CPN_GNHead(BaseHead):
    def __init__(self, output_shape, num_class,
                 global_loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 refine_loss: ConfigType = dict(
                     type='KeypointOKHMMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        self.loss_module = ModuleDict(
            dict(
                global_loss=MODELS.build(global_loss),
                refine_loss=MODELS.build(refine_loss),
            ))


        channel_settings = [2048, 1024, 512, 256]
        self.global_net = globalNet(channel_settings, output_shape, num_class)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Constant', layer='GroupNorm2d', val=1)
        ]
        return init_cfg

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """

        # 反转元组
        feats = tuple(reversed(feats))

        global_fms, global_outs = self.global_net(feats)
        refine_out= self.refine_net(global_fms)
        self_supervised_heatmap = self._flat_softmax(refine_out)
        # self_supervised_heatmap = self.get_gaussian_map(refine_out)
        # refine_out,unsup_refine_out = self.refine_net(global_fms)
        # self_supervised_heatmap = self.get_gaussian_map(unsup_refine_out)

        return global_outs,refine_out,self_supervised_heatmap

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
            _,_batch_heatmaps,_batch_self_supervised_heatmaps = self.forward(_feats)
            _, _batch_heatmaps_flip,_batch_self_supervised_heatmaps_flip = self.forward(_feats_flip)
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
            _,batch_heatmaps,batch_self_supervised_heatmaps= self.forward(feats)

        preds = self.decode(batch_heatmaps)
        # preds = self.decode(batch_self_supervised_heatmaps)
        # print("111", preds)

        # batch_binary_masks = torch.randn(batch_heatmaps.shape[0],5,256,192)  # (B, G, H, W)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds,batch_heatmaps,batch_self_supervised_heatmaps

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
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
        global_outs,pred_fields,self_supervised_heatmaps = self.forward(feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        gt_heatmaps = torch.chunk(gt_heatmaps, chunks=4, dim=1)
        # 使用 torch.stack 将它们沿新维度堆叠
        gt_heatmaps = torch.stack(gt_heatmaps, dim=1)
        global_outputs = torch.stack(
            [global_out for global_out in global_outs])
        # 使用 permute 来重新排列维度
        global_outputs = global_outputs.permute(1, 0, 2, 3, 4)
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        losses = dict()

        # # GlobalNet Loss
        # global_loss = 0
        # for i in range(4):  # S是你想要迭代的维度
        #     global_output = global_outputs[:, i:i + 1, :, :, :].squeeze(1)   # 切片得到 (B, 1, K, H, W)
        #     gt_heatmap = gt_heatmaps[:, i:i + 1, :, :, :].squeeze(1)  # 切片得到 (B, 1, K, H, W)
        #
        #     global_loss += self.loss_module['global_loss'](global_output, gt_heatmap,keypoint_weights)
        #
        # refine_loss = self.loss_module['refine_loss'](pred_fields, gt_heatmaps[:, 3:4, :, :, :].squeeze(1) ,keypoint_weights)

        # 生成用于global_loss的掩码，只保留visible == 2的点
        global_mask = (keypoint_weights > 1.1).float()
        global_loss = 0.
        for i in range(4):
            global_output = global_outputs[:, i:i + 1, :, :, :].squeeze(1)  # 切片得到 (B, 1, K, H, W)
            gt_heatmap = gt_heatmaps[:, i:i + 1, :, :, :].squeeze(1)  # 切片得到 (B, 1, K, H, W)

            stage_loss = self.loss_module['global_loss'](global_output, gt_heatmap, global_mask)
            stage_loss = stage_loss.mean() / 2.0
            global_loss += stage_loss

        # 生成用于refine_loss的掩码，只保留visible == 1或2的点
        refine_mask = (keypoint_weights > 0.1).float()

        refine_loss = self.loss_module['refine_loss'](pred_fields, gt_heatmaps[:, 3:4, :, :, :].squeeze(1), refine_mask)

        losses.update({
            'global_loss': global_loss,
            'refine_loss':refine_loss
        })

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps[:, 3:4, :, :, :].squeeze(1)),
                mask=to_numpy(keypoint_weights) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        #binary_masks = torch.randn(pred_fields.shape[0], 5, 256, 192)  # (B, G, H, W)

        return losses,pred_fields,self_supervised_heatmaps

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
        self.sigma = 2
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





