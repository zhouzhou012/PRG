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
import matplotlib.pyplot as plt
from einops import rearrange
from mmpose.utils.tensor_utils import to_numpy
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
# from ..utils.DDPM import DDPM
from ..utils.ddpm import DDPM
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


        E1=[nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True)]

        E2=[
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        #plan1
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            # fea = torch.mean(fea, dim=1, keepdim=True),
        ]

        # # plan3
        # E3 = [
        #     nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1, True),
        # ]

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
        self.mlp = nn.Identity()


    def forward(self,x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea = torch.mean(fea, dim=1, keepdim=True)
        cond = self.mlp(fea)
        return cond


# class ResMLP(nn.Module):
#     def __init__(self,n_feats=256):
#         super(ResMLP, self).__init__()
#         self.resmlp = nn.Sequential(
#             nn.Linear(n_feats , n_feats),
#             nn.LeakyReLU(0.1, True),
#         )
#     def forward(self, x):
#         res=self.resmlp(x)
#         # res=res+x
#         return res

#通道注意的denoise
# class denoise(nn.Module):
#     def __init__(self, n_feats=64, n_denoise_res=5, timesteps=5):
#         super(denoise, self).__init__()
#         self.max_period = timesteps * 10
#         n_featsx4 = 4 * n_feats
#         resmlp = [
#             nn.Linear(n_featsx4 * 2 + 1, n_featsx4),
#             nn.LeakyReLU(0.1, True),
#         ]
#         for _ in range(n_denoise_res):
#             resmlp.append(ResMLP(n_featsx4))
#         self.resmlp = nn.Sequential(*resmlp)
#
#     def forward(self, x,c,t):
#         t = t.float()
#         t = t / self.max_period
#         t = t.view(-1, 1)
#         c = torch.cat([c, t, x], dim=1)
#
#         fea = self.resmlp(c)
#
#         return fea

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Block(nn.Module):
    def __init__(self,dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding = 1)
        self.norm = nn.GroupNorm(1,dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


#
# #空间注意的denoise
# class denoise(nn.Module):
#     def __init__(self,):
#         super(denoise, self).__init__()
#         sinu_pos_emb = SinusoidalPosEmb(dim=32)
#         fourier_dim = 32
#         time_dim=16
#         out_dim=2
#         self.time_mlp = nn.Sequential(
#             sinu_pos_emb,
#             nn.Linear(fourier_dim, time_dim),
#             nn.GELU(),
#             nn.Linear(time_dim, time_dim),
#             nn.SiLU(),
#             nn.Linear(time_dim, out_dim),
#         )
#         self.block1 = Block(dim=2,dim_out=1)
#         self.block2 = Block(dim=1,dim_out=1)
#
#     def forward(self, x,c,t):
#         time_emb = self.time_mlp(t)  #(B,2)
#         time_emb = rearrange(time_emb, 'b c -> b c 1 1')
#         scale_shift = time_emb.chunk(2, dim=1)
#         x = torch.cat([c, x], dim=1)
#         x = self.block1(x, scale_shift=scale_shift)
#         x = self.block2(x)
#         return x


#空间注意的denoise
class denoise(nn.Module):
    def __init__(self,):
        super(denoise, self).__init__()
        sinu_pos_emb = SinusoidalPosEmb(dim=32)
        fourier_dim = 32
        time_dim=16
        out_dim=2
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, out_dim),
        )
        Conv=[
           nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
           # nn.BatchNorm2d(1),  # 添加 BatchNorm2d 层
           nn.LeakyReLU(0.1, True),
           nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
           # nn.BatchNorm2d(1),  # 添加 BatchNorm2d 层
           nn.LeakyReLU(0.1, True),
        ]
        # Conv = [
        #     nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1, True),
        # ]
        # Conv = [
        #     nn.Conv2d(in_channels=256+1, out_channels=64, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1, True),
        #
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1, True),
        # ]
        # Conv = [
        #     nn.Conv2d(in_channels=256 + 1, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),  # 添加 BatchNorm2d 层
        #     nn.LeakyReLU(0.1, True),
        #
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),  # 添加 BatchNorm2d 层
        #     nn.LeakyReLU(0.1, True),
        #
        #     nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(1),  # 添加 BatchNorm2d 层
        #     nn.LeakyReLU(0.1, True),
        # ]
        self.Conv = nn.Sequential(*Conv)

        # self.act = nn.SiLU()
        # self.Conv=BasicUNet()

    def forward(self, x, c,t):
        time_emb = self.time_mlp(t)  #(B,2)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale_shift = time_emb.chunk(2, dim=1)

        c = torch.cat([c,x], dim=1)

        scale, shift = scale_shift
        c = c * (scale + 1) + shift

        fea = self.Conv(c)

        return fea



@MODELS.register_module()
class Prior_TopdownPoseEstimator_Stage2(BaseModel, metaclass=ABCMeta):
    _version = 2

    def __init__(self,
                 s1_pretrained:None,
                 backbone,
                 head,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None,):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.metainfo = self._load_metainfo(metainfo)
        self.s1_pretrained=s1_pretrained
        self.backbone=backbone
        self.head=head
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
        pre_s1 = dict(
            type='Prior_TopdownPoseEstimator_Stage1',
            backbone=self.backbone,
            head=self.head,
            )
        self.net = MODELS.build(pre_s1)
        for param in self.net.E.parameters():
            param.requires_grad = False

        self.E2 = CPEN()
        # self.denoise = denoise(n_feats=64, n_denoise_res=n_denoise_res, timesteps=timesteps)
        self.denoise = denoise()
        # self.diffusion = DDPM(denoise=self.denoise)
        self.diffusion = DDPM(denoise=self.denoise,linear_start= 0.1,linear_end= 0.99, timesteps =4)

    def init_weights(self):
        if self.s1_pretrained is not None:
          load_checkpoint(self.net, self.s1_pretrained, map_location='cpu')

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
        losses = dict()
        # torch.Size([B,H,W,C])  BGR
        hr_img = torch.stack(  # HR图片
            [d.gt_fields.img for d in data_samples])

        hr_img = rearrange(hr_img, "n h w c -> n c h w").contiguous()
        ##torch.Size([B,C,H,W]) RGB
        hr_img = hr_img[:, [2, 1, 0], ...]

        # 输入网络之前的数据normalization、padding
        hr_inputs = self.DataPreprocessor(hr_img)
        with torch.no_grad():
           prior1 = self.net.E(hr_inputs)

           # prior1 = self.net.E(hr_inputs,inputs)

           # gt_heatmaps = torch.stack(
           #     [d.gt_fields.heatmaps for d in data_samples])
           # prior1 = self.net.E(gt_heatmaps,inputs)

        cond = self.E2(inputs)
        prior2 = self.diffusion(cond,prior1)

        # # 直接用cond 作为prior
        # prior2 = self.E2(inputs)

        loss_name='diff_loss'

        # global num
        # num+=1# # torch.Size([B,H,W,C])  BGR
        # hr_img = torch.stack(  # HR图片
        #     [d.gt_fields.img for d in data_samples])
        #
        # hr_img = rearrange(hr_img, "n h w c -> n c h w").contiguous()
        # ##torch.Size([B,C,H,W]) RGB
        # hr_img = hr_img[:, [2, 1, 0], ...]
        #
        # # 输入网络之前的数据normalization、padding
        # hr_inputs = self.DataPreprocessor(hr_img)
        # prior1 = self.net.E(hr_inputs)
        # vis = prior1[0].squeeze(-1)
        # vis = rearrange(vis, " c h w ->h w c").contiguous()
        # vis = to_numpy(vis) * 255
        # cv2.imwrite('/mnt/private/mmpose/mask3.png', vis)

        # if num==1:
        #     print("111",prior1.shape)
        #     print("222",prior2.shape)
        #     vis = prior1[0].squeeze(-1)
        #     vis = rearrange(vis, " c h w ->h w c").contiguous()
        #     vis = to_numpy(vis) * 255
        #     cv2.imwrite('/mnt/private/mmpose/mask.png', vis)
        #     vis = prior2[0].squeeze(-1)
        #     vis = rearrange(vis, " c h w ->h w c").contiguous()
        #     vis = to_numpy(vis) * 255
        #     cv2.imwrite('/mnt/private/mmpose/mask1.png', vis)

        losses[loss_name] = F.l1_loss(prior2,prior1)

        # losses[loss_name] = F.mse_loss(prior2, prior1)*0.1

        fea = self.net.backbone(inputs, prior2)

        # 这段代码对各种head兼容性更强
        losses.update(
            self.net.head.loss(fea, data_samples, train_cfg=self.train_cfg))

        # ori_loss, pred, _, _ = self.head_loss(fea, data_samples, train_cfg=self.train_cfg)
        # losses.update(ori_loss)

        return losses


    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        # # 创建一个副本，避免修改原始图像
        # ori_inputs = inputs.clone()
        # inputs = self.DataPreprocessor(inputs)

        cond = self.E2(inputs)
        prior2 = self.diffusion(cond)

        # global num
        # num += 1
        # if num == 6:
        #     vis = prior2[0].squeeze(-1)
        #     print("111", inputs.shape)
        #     print("111", vis.shape)
        #     vis = rearrange(vis, " c h w ->h w c").contiguous()
        #     vis1 = to_numpy((torch.sigmoid(vis * vis)))
        #     plt.imshow(vis1.squeeze(-1), cmap='viridis')
        #     plt.axis('off')
        #     output_file = '/mnt/private/mmpose/mask2.png'
        #     plt.savefig(output_file)

        # vis = cond[0].squeeze(-1)
        # vis = rearrange(vis, " c h w ->h w c").contiguous()
        # vis = to_numpy(vis) * 255
        # cv2.imwrite('/mnt/private/mmpose/mask.png', vis)

        # vis=prior2[0].squeeze(-1)
        # vis = rearrange(vis, " c h w ->h w c").contiguous()
        # vis=to_numpy(vis)*255
        # cv2.imwrite('/mnt/private/mmpose/mask.png', vis)

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
        # prior1 = self.net.E(hr_inputs)
        # vis = prior1[0].squeeze(-1)
        # vis = rearrange(vis, " c h w ->h w c").contiguous()
        # vis = to_numpy(vis) * 255
        # cv2.imwrite('/mnt/private/mmpose/mask3.png', vis)

        # # 直接用cond 作为prior
        # prior2 = self.E2(inputs)
        #
        # vis=prior2[0].squeeze(-1)
        # vis = rearrange(vis, " c h w ->h w c").contiguous()
        # vis=to_numpy(vis)*255
        # cv2.imwrite('/mnt/private/mmpose/mask2.png', vis)

        assert self.net.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            cond_flip = self.E2(inputs.flip(-1))
            prior2_flip = self.diffusion(cond_flip)




            # vis = cond_flip[0].squeeze(-1)
            # vis = rearrange(vis, " c h w ->h w c").contiguous()
            # vis = to_numpy(vis) * 255
            # cv2.imwrite('/mnt/private/mmpose/mask1.png', vis)
            
            # vis = prior2_flip[0].squeeze(-1)
            # vis = rearrange(vis, " c h w ->h w c").contiguous()
            # vis = to_numpy(vis) * 255
            # cv2.imwrite('/mnt/private/mmpose/mask1.png', vis)

            # # 直接用cond 作为prior
            # prior2_flip = self.E2(inputs.flip(-1))


            _feats = self.extract_feat(inputs,prior2)
            _feats_flip = self.extract_feat(inputs.flip(-1),prior2_flip)
            feats = [_feats, _feats_flip]
        else:
            feats= self.extract_feat(inputs,prior2)

        preds = self.net.head.predict(feats,data_samples, test_cfg=self.test_cfg)
        # preds = self.net.head.predict(feats,ori_inputs, data_samples, test_cfg=self.test_cfg)

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
        cond = self.E2(inputs)
        prior2 = self.diffusion(cond)
        #
        # # 直接用cond 作为prior
        # prior2 = self.E2(inputs)

        x = self.net.backbone(inputs,prior2)
        if self.net.with_neck:
            x = self.net.neck(x)

        return x

    def extract_feat(self, inputs: Tensor,prior) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.net.backbone(inputs, prior)
        if self.net.with_neck:
            x = self.net.neck(x)

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
