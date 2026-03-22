from abc import ABCMeta, abstractmethod
from typing import Tuple, Union
from itertools import zip_longest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.models.utils import check_and_update_config
from mmengine.model import BaseModel
from mmpose.models import build_pose_estimator, build_head
from mmpose.registry import MODELS
from mmengine.config import Config
from mmengine.logging import MessageHub
from mmpose.utils.typing import (ConfigType, ForwardResults, InstanceList, OptConfigType,
                                 Optional, OptMultiConfig, PixelDataList, OptSampleList,
                                 SampleList)
from mmpose.evaluation.functional import pose_pck_accuracy
from mmengine.runner.checkpoint import load_checkpoint, _load_checkpoint, load_state_dict
from mmpose.utils.tensor_utils import to_numpy
from collections import OrderedDict
from mmengine.utils import is_seq_of
import math
import cv2
from einops import rearrange
from mmpose.utils.tensor_utils import to_numpy
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
num=0
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.LeakyReLU(0.1, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class CPEN(nn.Module):
    def __init__(self,n_feats = 64, n_encoder_res = 6):
        super(CPEN, self).__init__()


        E1=[nn.Conv2d(48, n_feats, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True)]

        E2=[
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            # nn.AdaptiveAvgPool2d(1),
        ]
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        # self.mlp = nn.Sequential(
        #     nn.Linear(n_feats * 4, n_feats * 4),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(n_feats * 4, n_feats * 4),
        #     nn.LeakyReLU(0.1, True)
        # )
        self.mlp=nn.Identity()

        self.pixel_unshuffle1 = nn.PixelUnshuffle(2)
        self.pixel_unshuffle2 = nn.PixelUnshuffle(4)
        self.pixel_unshuffle3 = nn.PixelUnshuffle(8)
        self.pixel_unshuffle4 = nn.PixelUnshuffle(16)


    # def forward(self,gt,x=None):
    #
    #     # x = self.pixel_unshuffle1(gt)  # (B,2*2*C,H/2,W/2) 只输入HR(2 scale)
    #
    #     # x = self.pixel_unshuffle2(gt)  #(B,4*4*C,H/4,W/4)  (B,48,64,48)  只输入HR  (4scale)
    #
    #     x = self.pixel_unshuffle3(gt)#(B,8*8*C,H/8,W/8) 只输入HR(8 scale)
    #
    #     # x = self.pixel_unshuffle4(gt)#(B,16*16*C,H/16,W/16) 只输入HR(16 scale)  不用
    #
    #
    #     # x=gt #heatmap #(B,17,64,48) 只输入gt
    #
    #     # x = self.pixel_unshuffle(x) #(B,4*4*C,H/4,W/4)  (B,48,64,48) #输入gt+LR(Bicubic)
    #     # x = torch.cat([x, gt], dim=1)  #(B,48+17,64,48)
    #
    #     # gt = self.pixel_unshuffle(gt)  # (B,4*4*C,H/4,W/4)  (B,48,64,64) #输入HR+LR
    #     # x = torch.cat([x, gt], dim=1)  # (B,48+3,64,64)
    #
    #     # x = self.pixel_unshuffle(x) # (B,4*4*C,H/4,W/4)  (B,48,64,64) #输入HR+Blind_LR
    #     # gt = self.pixel_unshuffle(gt)
    #     # x = torch.cat([x, gt], dim=1)  # (B,48+48,64,64)
    #
    #     fea = self.E(x).squeeze(-1).squeeze(-1)
    #
    #     fea = torch.mean(fea, dim=1, keepdim=True) #(1,H,W)
    #
    #     prior = self.mlp(fea)
    #     return prior  #(B,1,64,48)

    def forward(self,x):
        # 只输入LR

        x = self.pixel_unshuffle2(x)  #(B,4*4*C,H/4,W/4)  (B,48,64,48)  只输入HR  (4scale)

        fea = self.E(x).squeeze(-1).squeeze(-1)

        fea = torch.mean(fea, dim=1, keepdim=True) #(1,H,W)

        prior = self.mlp(fea)
        return prior

@MODELS.register_module()
class LR_Prior_TopdownPoseEstimator(BaseModel, metaclass=ABCMeta):
    _version = 2

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.metainfo = self._load_metainfo(metainfo)


        self.backbone = MODELS.build(backbone)

        # the PR #2108 and #2126 modified the interface of neck and head.
        # The following function automatically detects outdated
        # configurations and updates them accordingly, while also providing
        # clear and concise information on the changes made.
        neck, head = check_and_update_config(neck, head)
        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            self.head = MODELS.build(head)

        self.train_cfg = train_cfg if train_cfg else {}
        self.test_cfg = test_cfg if test_cfg else {}

        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        self._enable_normalize = True
        self.pad_size_divisor = 1
        self.pad_value = 0
        self.register_buffer('mean',
                             torch.tensor(mean).view(-1, 1, 1), False)
        self.register_buffer('std',
                             torch.tensor(std).view(-1, 1, 1), False)

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

        self.E = CPEN()

    @property
    def with_neck(self) -> bool:
        """bool: whether the pose estimator has a neck."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """bool: whether the pose estimator has a head."""
        return hasattr(self, 'head') and self.head is not None

    @staticmethod
    def _load_metainfo(metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """

        if metainfo is None:
            return None

        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList,
                mode: str = 'tensor') -> ForwardResults:

        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # use customed metainfo to override the default metainfo
            if self.metainfo is not None:
                for data_sample in data_samples:
                    data_sample.set_metainfo(self.metainfo)
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode.')

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        # # torch.Size([B,H,W,C])  BGR
        # hr_img = torch.stack(  # HR图片
        #     [d.gt_fields.img for d in data_samples])
        # # cv2.imwrite('/mnt/private/mmpose/input2.png', to_numpy(hr_img[0]))
        #
        # hr_img = rearrange(hr_img, "n h w c -> n c h w").contiguous()
        # ##torch.Size([B,C,H,W]) RGB
        # hr_img = hr_img[:, [2, 1, 0], ...]
        #
        # # 输入网络之前的数据normalization、padding
        # hr_inputs = self.DataPreprocessor(hr_img)
        #
        # prior = self.E(hr_inputs)
        #
        #prior = self.E(hr_inputs,inputs)

        prior = self.E(inputs)

        # gt_heatmaps = torch.stack(
        #     [d.gt_fields.heatmaps for d in data_samples])
        #
        # prior = self.E(gt_heatmaps,inputs)

        fea = self.backbone(inputs, prior)

        # 这段代码对各种head兼容性更强
        losses = dict()
        losses.update(
            self.head.loss(fea, data_samples, train_cfg=self.train_cfg))

        return losses



    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""

        # # torch.Size([B,H,W,C])  BGR
        # hr_img = torch.stack(  # HR图片
        #     [d.gt_fields.img for d in data_samples])
        # # cv2.imwrite('/mnt/private/mmpose/input.png', to_numpy(hr_img[0]))
        # # for i, d in enumerate(data_samples):
        # #     print(f"Attributes of data_samples[{i}]:")
        # #     print(vars(d))  # 或者使用 print(d.__dict__)
        # hr_img = rearrange(hr_img, "n h w c -> n c h w").contiguous()
        # ##torch.Size([B,C,H,W]) RGB
        # hr_img = hr_img[:, [2, 1, 0], ...]
        #
        # # 输入网络之前的数据normalization、padding
        # hr_inputs = self.DataPreprocessor(hr_img)
        #
        # prior = self.E(hr_inputs)
        # prior = self.E(hr_inputs,inputs)
        prior = self.E(inputs)

        # gt_heatmaps = torch.stack(
        #     [d.gt_fields.heatmaps for d in data_samples])
        #
        # prior = self.E(gt_heatmaps,inputs)

        #
        # vis=prior[0].squeeze(-1)
        # vis = rearrange(vis, " c h w ->h w c").contiguous()
        # vis=to_numpy(vis)*255
        # print("111",vis.shape)
        # cv2.imwrite('/mnt/private/mmpose/mask.png', vis)
        #
        #
        # viss=inputs[0].squeeze(-1)
        # viss = viss[[2, 1, 0], ...]
        # viss = viss * self.std + self.mean
        # viss = rearrange(viss, " c h w ->h w c").contiguous()
        # viss=to_numpy(viss)
        # print("222", viss.shape)
        # cv2.imwrite('/mnt/private/mmpose/input2.png', viss)

        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            # prior_flip = self.E(hr_inputs.flip(-1))
            prior_flip = self.E(inputs.flip(-1))
            _feats = self.extract_feat(inputs,prior)
            _feats_flip = self.extract_feat(inputs.flip(-1),prior_flip)
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs,prior)

        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        return results

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None
                 ) -> Union[Tensor, Tuple[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            Union[Tensor | Tuple[Tensor]]: forward output of the network.
        """
        # # torch.Size([B,H,W,C])  BGR
        # hr_img = torch.stack(  # HR图片
        #     [d.gt_fields.img for d in data_samples])
        #
        # hr_img = rearrange(hr_img, "n h w c -> n c h w").contiguous()
        # ##torch.Size([B,C,H,W]) RGB
        # hr_img = hr_img[:, [2, 1, 0], ...]
        #
        # # 输入网络之前的数据normalization、padding
        # hr_inputs = self.DataPreprocessor(hr_img)
        #
        # prior = self.E(hr_inputs)
        # prior = self.E(hr_inputs,inputs)
        prior = self.E(inputs)

        # gt_heatmaps = torch.stack(
        #     [d.gt_fields.heatmaps for d in data_samples])
        #
        # prior = self.E(gt_heatmaps,inputs)

        x = self.backbone(inputs,prior)
        if self.with_neck:
            x = self.neck(x)

        return x

    def extract_feat(self, inputs: Tensor,prior) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """

        x = self.backbone(inputs, prior)
        if self.with_neck:
            x = self.neck(x)

        return x

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for k in keys:
            if 'keypoint_head' in k:
                v = state_dict.pop(k)
                k = k.replace('keypoint_head', 'head')
                state_dict[k] = v

    def DataPreprocessor(self, x):

        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.
        """
        _batch_inputs = x
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim(
                        ) == 3 and _batch_input.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input.shape}')
                    _batch_input = (_batch_input - self.mean) / self.std
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}： {data}')
        return batch_inputs

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            bbox_centers = gt_instances.bbox_centers
            bbox_scales = gt_instances.bbox_scales
            input_size = data_sample.metainfo['input_size']
            w, h = input_size
            pred_instances.keypoints = pred_instances.keypoints / np.array([w, h]) \
                                       * bbox_scales + bbox_centers - 0.5 * bbox_scales

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
