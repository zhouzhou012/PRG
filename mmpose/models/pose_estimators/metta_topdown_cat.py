from abc import ABCMeta, abstractmethod
from typing import Tuple, Union
from itertools import zip_longest
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import random
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
import seaborn as sns
import matplotlib.pyplot as plt
from einops import rearrange
from mmpose.utils.tensor_utils import to_numpy
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
import torch.nn.functional as F
num=0
n1=0
n2=0
class PoseEncoder(nn.Module):
    def __init__(self, in_channels=14, out_channels=64):
        super(PoseEncoder, self).__init__()
        self.layer1 = self._make_layer(in_channels, 32)  # 14 → 32
        self.layer2 = self._make_layer(32, 64)  # 32 → 64

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),  # 3×3 卷积
            #nn.BatchNorm2d(out_channels),
            # 替换为 GroupNorm（推荐分组数=32）
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),  # 3×3 卷积
            #nn.BatchNorm2d(out_channels),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)  # (B, 17, H, W) → (B, 32, H, W)
        x = self.layer2(x)  # (B, 32, H, W) → (B, 64, H, W)
        return x

class DecomNet(nn.Module):
    def __init__(self, mid_channels=64, kernel_size=3):
        super(DecomNet, self).__init__()

        #cat 拼接特征
        self.pose_encoder=PoseEncoder()
        self.conv = nn.Conv2d(4, mid_channels, kernel_size * 3, padding='same')
        self.conv0 = nn.Conv2d(4, mid_channels // 2, kernel_size, padding='same')
        self.conv1 = nn.Sequential(
        nn.Conv2d(mid_channels, mid_channels, kernel_size, padding='same'),
         nn.ReLU()
         # nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2_3 = nn.Sequential(  # 下采样
            nn.Conv2d(mid_channels, mid_channels*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels*2, mid_channels*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
            # nn.LeakyReLU(0.2, inplace=True)
        )

        #融合模块，1×1卷积进行通道压缩
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),  # 从128+64 -> 128
            #nn.BatchNorm2d(128),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(0.2, inplace=True)
        )

        self.deconv4 = nn.Sequential(
            #nn.ConvTranspose2d(mid_channels*3,  # 输入通道数
            nn.ConvTranspose2d(mid_channels * 2,  # 输入通道数
                               mid_channels,  # 输出通道数
                               kernel_size=3,  # 卷积核大小
                               stride=2,  # 步长
                               padding=1,
                               output_padding=1,  # 默认输出无额外填充
                               bias=False),
            # nn.BatchNorm2d(num_features=mid_channels),
            nn.GroupNorm(num_groups=32, num_channels=mid_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(mid_channels,  # 输入通道数
                               mid_channels,  # 输出通道数
                               kernel_size=3,  # 卷积核大小
                               stride=2,  # 步长
                               padding=1,
                               output_padding=1,  # 默认输出无额外填充
                               bias=False),
            #nn.BatchNorm2d(num_features=mid_channels),
            nn.GroupNorm(num_groups=32, num_channels=mid_channels),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d( mid_channels*2, mid_channels, kernel_size, padding='same'),
            nn.ReLU()
            # nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv6 = nn.Conv2d(mid_channels + mid_channels // 2, mid_channels, kernel_size, padding=1)
        self.conv7 = nn.Conv2d(mid_channels, 4, kernel_size, padding=1)

    def forward(self, input_im,heatmap=None):

        #cat 拼接热图特征
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]  # 在颜色通道（dim=1）上取最大值
        input_im = torch.cat([input_max, input_im], dim=1)  # 在通道维度上拼接
        fea = self.conv(input_im)  # (B,64,256,192)

        fea_0 = self.conv0(input_im)  # (B,32,256,192)

        fea1 = self.conv1(fea)  # (B,64,256,192)

        fea = self.conv2_3(fea1)  # (B,128,64,48)

        pose_fea=self.pose_encoder(heatmap)
        fea = torch.cat((fea, pose_fea), dim=1)  # (B,128+64,64,48)

        fea=self.fusion_conv(fea)

        fea = self.deconv4(fea)  # (B,64,256,192)


        fea = torch.cat((fea, fea1), dim=1)  # (B,64+64,256,192)

        fea = self.conv5(fea)  # (B,64,256,192)
        fea = torch.cat((fea, fea_0), dim=1)  # (B,64+32,256,192)

        fea = self.conv6(fea)  # (B,64,256,192)
        fea = self.conv7(fea)  # (B,4,256,192)

        R = torch.sigmoid(fea[:, :3, :, :])
        L = torch.sigmoid(fea[:, 3:4, :, :])
        return R, L

class LSID(nn.Module):
    def __init__(self, inchannel=4, block_size=2):
        super(LSID, self).__init__()
        self.block_size = block_size
        # cat 拼接特征
        self.pose_encoder = PoseEncoder()

        self.conv1_1 = nn.Conv2d(inchannel, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
        self.conv7_1 = nn.Conv2d(256+64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        out_channel = 4
        self.conv10 = nn.Conv2d(32, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                  m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input_im,heatmap=None):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]  # 在颜色通道（dim=1）上取最大值
        input_im = torch.cat([input_max, input_im], dim=1)  # 在通道维度上拼接
        x = self.conv1_1(input_im)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        x = self.maxpool(x)

        x = self.conv5_1(x)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.lrelu(x)


        x = self.up6(x)
        x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4), 1)
        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)


        x = self.up7(x)
        x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3), 1)  #777 torch.Size([2, 256, 64, 48])
        pose_fea = self.pose_encoder(heatmap)
        x = torch.cat((x, pose_fea), dim=1)  # (B,256+64,64,48)
        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)


        x = self.up8(x)
        x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2), 1)

        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)


        x = self.up9(x)
        x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)
        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)

        x = self.conv10(x)


        R = torch.sigmoid(x[:, :3, :, :])
        L = torch.sigmoid(x[:, 3:4, :, :])
        return R, L

@MODELS.register_module()
class MeTTA_TopdownPoseEstimator_CAT(BaseModel, metaclass=ABCMeta):
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
        self.auxnet=DecomNet()
        # self.auxnet = LSID()

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


    def region_l1_loss(self,R_low_max, S_low_max, masks):
        """
        R_low_max: Tensor of shape (B, 1, H, W)
        S_low_max: Tensor of shape (B, 1, H, W)
        masks:     Tensor of shape (B, R, H, W)  # R = 区域数，如 5

        Returns:
            final_loss: scalar
        """
        B, _, H, W = R_low_max.shape
        _, R, _, _ = masks.shape

        total_loss = 0.0

        for b in range(B):
            batch_loss = 0.0
            for r in range(R):
                mask = masks[b, r].unsqueeze(0)  # (1, H, W)
                mask_sum = mask.sum()

                if mask_sum > 0:
                    diff = torch.abs(R_low_max[b] - S_low_max[b])  # (1, H, W)
                    masked_diff = diff * mask
                    per_loss = masked_diff.sum() / (mask_sum)
                    batch_loss += per_loss

            total_loss += batch_loss

        final_loss = total_loss / B  # 均分到每个 batch 样本
        return final_loss

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        # augmenter = ImageAugmenter()
        # augmented_images = []
        # for img in inputs:  # 遍历批次中的每张图像
        #     aug_img=augmenter.augment(img)
        #     augmented_images.append(aug_img)
        # inputs=torch.stack([item for item in augmented_images])  # 合并为批次


        # # 创建一个副本，避免修改原始图像
        # ori_inputs = inputs.clone()
        # augmenter = ImageAugmenter()
        # augmented_images = []
        # #weak_aug
        # for img in inputs:  # 遍历批次中的每张图像
        #    if self.is_wl_light(img):
        #       # global n1
        #       # n1 += 1
        #       aug_img=augmenter.weak_augment_well_light(img)
        #       augmented_images.append(aug_img)
        #    else:
        #       # global n2
        #       # n2 += 1
        #       augmented_images.append(img)
        # inputs=torch.stack([item for item in augmented_images])  # 合并为批次

        inputs = inputs / 255.0

        fea = self.backbone(inputs)
        losses = dict()
        primary_loss,heatmaps,self_supervised_heatmap=self.head.loss(fea, data_samples, train_cfg=self.train_cfg)
        losses.update(primary_loss)
        #
        # 
        #
        R_low, I_low = self.auxnet(inputs, self_supervised_heatmap)
        # R_low, I_low = self.auxnet(aux_inputs, self_supervised_heatmap)
        # R_low,I_low= self.auxnet(inputs,heatmaps)

        I_low_3= torch.cat([I_low, I_low, I_low], dim=1)
        train_low_data_eq = []
        for i in range(inputs.shape[0]):
            train_low_data_max_chan = torch.max(inputs[i],dim=0, keepdim=True).values
            train_low_data_max_channel = self.histeq(train_low_data_max_chan)
            train_low_data_eq.append(train_low_data_max_channel)

        # 如果 train_low_data_eq 中的元素形状相同，可以使用 torch.stack() 来创建一个新的 tensor
        low_data_eq_tensor = torch.stack([item for item in train_low_data_eq])
        R_low_max = torch.max(R_low, dim=1, keepdim=True).values


        weight=0.1
        loss_name1 = 'recon_loss'
        losses[loss_name1] = F.l1_loss(R_low * I_low_3, inputs)*weight

        loss_name2 = 'recon_loss_eq'
        losses[loss_name2] = F.l1_loss(R_low_max ,low_data_eq_tensor)*weight*0.1
        #
        # # 平滑损失
        # loss_name3 = 'R_loss_smooth'
        # R_low_gray =self.rgb_to_gray(R_low)
        # # losses[loss_name3]= torch.mean(
        # #     torch.abs(self.gradient(R_low_gray, "x")) + torch.abs(self.gradient(R_low_gray, "y")))*weight*0.01
        # losses[loss_name3] = torch.mean(
        #     torch.abs(self.ave_gradient(R_low_gray, "x")) + torch.abs(self.ave_gradient(R_low_gray, "y"))) * weight * 0.01


        #光照loss
        loss_name4 = 'I_loss_smooth'
        losses[loss_name4] = self.smooth(I_low,R_low)*weight*0.1

        # #区域损失
        # loss_name5= 'region_recon_loss_eq'
        # losses[loss_name5] = self.region_l1_loss(R_low_max ,low_data_eq_tensor, binary_masks) * weight * 0.1

        # #增加一个增强后图像的姿态估计输出
        # fea = self.backbone(R_low)
        # enhance_primary_loss, heatmaps, self_supervised_heatmap, binary_masks = self.head.loss(fea, data_samples,
        #                                                                               train_cfg=self.train_cfg)
        #
        # # losses.update(enhance_primary_loss)
        # # 提取损失
        # loss_name6 = 'enhance_global_loss'
        # loss_name7 = 'enhance_refine_loss'
        # loss_name8 = 'enhance_acc_pose'
        # losses[loss_name6] = enhance_primary_loss['global_loss']
        # losses[loss_name7] = enhance_primary_loss['refine_loss']
        # losses[loss_name8]= enhance_primary_loss['acc_pose']


        return losses



    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        # 创建一个副本，避免修改原始图像
        ori_inputs = inputs.clone()

        # augmenter = ImageAugmenter()
        # augmented_images = []
        # for img in inputs:  # 遍历批次中的每张图像
        #     aug_img=augmenter.augment(img)
        #     augmented_images.append(aug_img)
        # inputs=torch.stack([item for item in augmented_images])  # 合并为批次
        # viss = inputs[0].squeeze(-1)
        # viss = viss[[2, 1, 0], ...]
        # viss = rearrange(viss, " c h w ->h w c").contiguous()
        # cv2.imwrite('/mnt/private/mmpose/clmb33.png', to_numpy(viss))


        inputs = inputs / 255.0

        # global num
        # num += 1
        # if num == 1:
        #     # viss = torch.max(ori_inputs[1], dim=0, keepdim=True).values
        #     # viss= self.histeq(viss)*255.0
        #     viss = ori_inputs[1].squeeze(-1)*255.0
        #     # print("111",viss.tolist())
        #     viss = viss[[2, 1, 0], ...]
        #     viss = rearrange(viss, " c h w ->h w c").contiguous()
        #     cv2.imwrite('/mnt/private/mmpose/1.png', to_numpy(viss))

        # viss = ori_inputs[0].squeeze(-1)
        # viss = viss[[2, 1, 0], ...]
        # viss = rearrange(viss, " c h w ->h w c").contiguous()
        # cv2.imwrite('/mnt/private/mmpose/101.png', to_numpy(viss))
        #
        # cv2.imwrite('/mnt/private/mmpose/202.png', to_numpy(viss*50.0))




        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)


        preds,heatmaps,self_supervised_heatmap= self.head.predict(feats, data_samples, test_cfg=self.test_cfg)
        # preds, heatmaps= self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        # 增加一个增强后图像的姿态估计输出
        # R_low, I_low = self.auxnet(inputs, self_supervised_heatmap)
        # if self.test_cfg.get('flip_test', False):
        #     _feats = self.backbone(R_low)
        #     _feats_flip = self.backbone(R_low.flip(-1))
        #     feats = [_feats, _feats_flip]
        # else:
        #     feats = self.backbone(R_low)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        # batch_pred_fields = heatmaps

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)



        # viss = R_low[0].squeeze(-1)*255.0
        # # viss = torch.max(R_low[1], dim=0, keepdim=True).values*255.0
        # # print("222", viss[2,...].tolist())
        # viss = viss[[2, 1, 0], ...]
        # viss = rearrange(viss, " c h w ->h w c").contiguous()
        # cv2.imwrite('/mnt/private/mmpose/1cat.png', to_numpy(viss))

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
        inputs = inputs / 255.0
        x = self.extract_feat(inputs)
        if self.with_head:
            # x,self_supervised_heatmap = self.head.forward(x)
            global_outs, x, self_supervised_heatmap= self.head.forward(x)
        R_low, I_low = self.auxnet(inputs, self_supervised_heatmap)
       #  R_low, I_low = self.auxnet(inputs, x)

        return (x,R_low, I_low)

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.backbone(inputs)
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

    def histeq(self,im, nbr_bins=256):
        """对一幅灰度图像进行直方图均衡化，支持形状为 (1, H, W) 的张量输入"""

        # 1. 计算图像的直方图
        # Flatten the image to a 1D vector
        im_flat = im.reshape(-1)

        # # 计算图像的直方图
        # range_min = im.min().item()
        # range_max = im.max().item()
        hist = torch.histc(im_flat, bins=nbr_bins, min=0., max=1.0)
        hist = hist / hist.sum()  # 归一化  #torch.Size([256])

        # 2. 计算累积分布函数 (CDF)
        cdf = hist.cumsum(0)
        cdf = cdf / cdf[-1]  # 归一化

        # 3. 使用CDF对图像进行线性插值
        # 创建一个用于查找的线性空间 [0, 1]，并将每个像素的值映射到CDF的结果
        cdf_mapping = torch.bucketize(im_flat, torch.linspace(0., 1., nbr_bins).to(im_flat.device))

        # 将每个像素映射到CDF中对应的值
        im2_flat = cdf[cdf_mapping - 1]  # `bucketize`返回的是 1-based index，需要减去1

        # 4. 重塑为原始图像的形状
        im2 = im2_flat.view(im.shape)

        return im2

    def gradient(self, input_tensor, direction):
        # 定义平滑核 (Kernel)
        smooth_kernel_x = torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2)
        smooth_kernel_y = smooth_kernel_x.transpose(2, 3)  # 转置核

        # 根据方向选择核
        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        else:
            raise ValueError("Direction must be 'x' or 'y'")

        # 使用 F.conv2d 进行卷积操作
        # input_tensor 的形状为 (batch_size, channels, height, width)
        # 在 PyTorch 中，padding='same' 类似的效果可以通过计算 padding 来获得
        # 我们在这里设置 padding = 1 来模仿 SAME padding（适用于 2x2 kernel）
        # 将 kernel 移动到和 input_tensor 相同的设备
        kernel = kernel.to(input_tensor.device)
        # 进行卷积操作
        output_tensor = F.conv2d(input_tensor, kernel, stride=1, padding=1)
        return torch.abs(output_tensor)

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def rgb_to_gray(self,rgb_image):
        # 使用权重将 RGB 转换为灰度图像
        # 权重分别为 R: 0.2989, G: 0.5870, B: 0.1140
        gray_image = 0.2989 * rgb_image[:, 0, :, :] + 0.5870 * rgb_image[:, 1, :, :] + 0.1140 * rgb_image[:, 2, :, :]
        return gray_image.unsqueeze(1)  # 保持通道维度，形状为 (batch_size, 1, height, width)

    def smooth(self, input_I, input_R):
        # 将RGB图像转换为灰度图像
        input_R_gray = self.rgb_to_gray(input_R)

        # 计算梯度
        grad_I_x = self.gradient(input_I, "x")
        grad_I_y = self.gradient(input_I, "y")
        # grad_R_x = self.gradient(input_R_gray, "x")
        # grad_R_y = self.gradient(input_R_gray, "y")
        grad_R_x = self.ave_gradient(input_R_gray, "x")
        grad_R_y = self.ave_gradient(input_R_gray, "y")

        # 计算平滑损失
        smooth_loss = torch.mean(grad_I_x * torch.exp(-10 * grad_R_x) + grad_I_y * torch.exp(-10 * grad_R_y))

        return smooth_loss



    def is_wl_light( self,img_tensor, threshold=0.1):
        """
        判断一个(C, H, W)格式的图像Tensor是否为正常光图像
        参数：
            img_tensor: torch.Tensor, 范围为[0,1]，shape为(C, H, W)
            threshold: float, 判断是否低光的亮度阈值（0~1之间）
        返回：
            bool: True 表示低光图像
        """
        assert img_tensor.ndim == 3, "输入必须是 (C, H, W)"
        assert img_tensor.shape[0] == 3, "必须是RGB图像"

        # 将 RGB 转为灰度（加权平均）
        gray = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
        gray=gray/255.0
        # 计算亮度平均值
        brightness = gray.mean().item()

        return brightness > threshold


class ImageAugmenter:
    def __init__(self):
        pass

    def adjust_gamma(self, img, gamma):
        """伽马校正 (公式3)"""
        return 255 * ((img / 255) ** gamma)

    def adjust_brightness(self, img, b):
        """亮度调整 (公式4)"""
        return b * img

    def adjust_contrast(self, img, c):
        """对比度调整 (公式5)"""
        grey = torch.mean(img, dim=(-3, -2, -1), keepdim=True)  # 计算灰度均值
        return c * img + (1 - c) * grey

    def add_gaussian_noise(self, img, var):
        """高斯噪声 (公式6)"""
        noise = torch.randn_like(img) * (var ** 0.5)  # 标准差=sqrt(var)
        noise = torch.clamp(noise, min=0)  # 之前没加
        return img + noise

    # (最多执行2种增强)
    # def augment(self, img):
    #     """
    #     输入: img (Tensor [C,H,W], uint8, [0,255])
    #     输出: 增强后的图像 (最多执行2种增强)
    #     """
    #     img = img.float()
    #
    #     # 定义增强操作（使用lambda延迟执行）
    #     ops = []
    #
    #     # 每个操作都有一定执行概率，避免每次都进列表
    #     if random.random() < 0.5:
    #         gamma = random.uniform(2,5)
    #         ops.append(lambda x: self.adjust_gamma(x, gamma))
    #
    #     if random.random() < 0.5:
    #         b = random.uniform(0.01, 0.05)
    #         ops.append(lambda x: self.adjust_brightness(x, b))
    #
    #     if random.random() < 0.5:
    #         c = random.uniform(0.2, 1.0)
    #         ops.append(lambda x: self.adjust_contrast(x, c))
    #
    #     if random.random() < 0.5:
    #         var = random.uniform(0, 40)
    #         ops.append(lambda x: self.add_gaussian_noise(x, var))
    #
    #     # 随机打乱，最多选择2/3个
    #     random.shuffle(ops)
    #     selected_ops = ops[:3]
    #
    #     # 顺序执行选择的增强
    #     for op in selected_ops:
    #         img = op(img)
    #
    #     img = torch.clamp(img, 0, 255).type(torch.uint8)
    #     return img

    def augment(self, img):
        """
        输入: img (Tensor [C,H,W], 范围[0,255], dtype=uint8)
        输出: 增强后的图像 (同输入格式)
        """
        img = img.float()  # 转为浮点计算

        # 1. 伽马校正 (gamma ∈ [2, 5])
        if random.random() > 0.5:
            gamma = random.uniform(2, 5)
            img = self.adjust_gamma(img, gamma)

        # 2. 亮度调整 (b ∈ [0.01, 0.05])
        if random.random() > 0.5:
            b = random.uniform(0.01, 0.05)
            img = self.adjust_brightness(img, b)

        # 3. 对比度调整 (c ∈ [0.2, 1.0])
        if random.random() > 0.5:
            c = random.uniform(0.2, 1.0)
            img = self.adjust_contrast(img, c)

        # 4. 高斯噪声 (var ∈ [0, 40])
        if random.random() > 0.5:
            var = random.uniform(0, 40)
            img = self.add_gaussian_noise(img, var)

        # 截断到[0,255]并转回uint8
        img = torch.clamp(img, 0, 255).type(torch.uint8)
        return img

    # weak_aug
    def weak_augment_well_light(self, img):
        """
        输入: img (Tensor [C,H,W], 范围[0,255], dtype=uint8)
        输出: 增强后的图像 (同输入格式)
        """
        img = img.float()  # 转为浮点计算

        # 1. 伽马校正 (gamma ∈ [2, 5])
        if random.random() > 0.5:
            gamma = random.uniform(2, 4)
            img = self.adjust_gamma(img, gamma)

        # 2. 亮度调整 (b ∈ [0.01, 0.05])
        if random.random() > 0.5:
            b = random.uniform(0.2, 0.6)
            # b = random.uniform(0.4, 0.6)
            img = self.adjust_brightness(img, b)

        # 3. 对比度调整 (c ∈ [0.2, 1.0])
        if random.random() > 0.5:
            c = random.uniform(0.4, 1.0)
            img = self.adjust_contrast(img, c)

        # 4. 高斯噪声 (var ∈ [0, 40])
        if random.random() > 0.5:
            var = random.uniform(0, 20)
            img = self.add_gaussian_noise(img, var)

        # 截断到[0,255]并转回uint8
        img = torch.clamp(img, 0, 255).type(torch.uint8)
        return img


    # def slight_augment_light(self, img):
    #     """
    #     输入: 极端低光照图像 (Tensor [C,H,W], 范围[0,255], dtype=uint8)
    #     输出: 微扰后的图像 (同输入格式)
    #     """
    #     img = img.float()
    #
    #     # 1. 伽马校正（微扰：轻微提亮或压暗）
    #     if random.random() > 0.5:
    #         # gamma < 1 提亮, gamma > 1 压暗
    #         gamma = random.uniform(0.8, 1.2)
    #         img = self.adjust_gamma(img, gamma)
    #
    #     # 2. 亮度乘性调整（微扰：轻微增加或减少亮度）
    #     if random.random() > 0.5:
    #         # 乘性因子 beta 接近 1.0
    #         beta = random.uniform(0.9, 1.1)
    #         # 注意：你需要确保 self.adjust_brightness 实现的是乘性调整
    #         img = self.adjust_brightness(img, beta)
    #
    #         # 3. 对比度调整（微扰：轻微增加或减少对比度）
    #     if random.random() > 0.5:
    #         # 因子接近 1.0
    #         c = random.uniform(0.8, 1.2)
    #         img = self.adjust_contrast(img, c)
    #
    #     # 4. 高斯噪声（微扰：少量新增噪声）
    #     if random.random() > 0.5:
    #         # 方差 var 范围减小
    #         var = random.uniform(0, 20)
    #         img = self.add_gaussian_noise(img, var)
    #
    #     # 截断到[0,1]再转回[0,255]
    #     img = torch.clamp(img, 0, 255).type(torch.uint8)
    #     return img




