# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple, Union
import torch.nn.functional as F
import torch
import torchvision
import math
import cv2
import os
import numpy as np
from mmcv.cnn import build_conv_layer
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn
from einops import rearrange
from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy,simcc_soft_argmax_pck_accuracy
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead
from mmpose.codecs.utils.refinement import refine_simcc_dark
OptIntSeq = Optional[Sequence[int]]
num=0

# def save_batch_image_with_joints_s(batch_image, batch_joints, batch_joints_vis, file_name):
#
#     print("=== USING NEW VISUALIZATION FUNCTION ===")
#
#     # -----------------------------------------
#     # 区域配色 (RGB)  → 转 BGR 供 cv2 使用
#     # -----------------------------------------
#
#     # -----------------------------------------
#     # RGB 颜色定义
#     # -----------------------------------------
#     # COLOR_FACE = (230, 140, 60)  # 橙色（面部）
#     # COLOR_TORSO = (220, 190, 40)  # 黄色（身体）
#     # COLOR_LEFT = (60, 100, 180)  # 蓝色（左侧肢体）
#     # COLOR_RIGHT = (70, 160, 90)  # 绿色（右侧肢体）
#
#     COLOR_FACE = (255, 109, 106)  # 珊瑚红/橙色
#     COLOR_TORSO = (255, 194, 70)  # 琥珀黄
#     COLOR_LEFT = (64, 154, 244)  # 天蓝色
#     COLOR_RIGHT = (67, 220, 147)  # 翡翠绿
#
#     # 转 cv2 BGR 颜色
#     FACE = COLOR_FACE[::-1]
#     TORSO = COLOR_TORSO[::-1]
#     LEFT = COLOR_LEFT[::-1]
#     RIGHT = COLOR_RIGHT[::-1]
#
#     skeleton_groups = {
#         FACE: [(0, 1), (1, 3), (0, 2), (2, 4)],  # 面部
#         TORSO: [(5, 6), (11, 12), (5, 11), (6, 12)],  # 身体（躯干）
#         LEFT: [(5, 7), (7, 9), (11, 13), (13, 15)],  # 左臂 + 左腿
#         RIGHT: [(6, 8), (8, 10), (12, 14), (14, 16)]  # 右臂 + 右腿
#     }
#
#     # -----------------------------------------
#     # 专业关键点（白细描边 + 深蓝实心）
#     # -----------------------------------------
#     # KP_BORDER = (255,255,255)    # 白
#     KP_FILL   = (220,60,30)
#     KP_FILL   = KP_FILL[::-1]
#
#     os.makedirs(file_name, exist_ok=True)
#
#     for k in range(batch_image.size(0)):
#
#         # 原图 RGB → BGR（修复偏蓝问题）
#         img = batch_image[k].permute(1,2,0).cpu().numpy()
#         img = np.clip(img,0,255).astype(np.uint8)
#         canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#         joints = batch_joints[k]
#         vis = batch_joints_vis[k]
#         if vis.ndim == 2:
#             vis = vis[:,0]
#
#         # 绘制骨架
#         for color, pairs in skeleton_groups.items():
#             for (s,e) in pairs:
#                 if vis[s] > 0.5 and vis[e] > 0.5:
#                     x1, y1 = joints[s]
#                     x2, y2 = joints[e]
#                     cv2.line(canvas, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
#
#         # 绘制关键点（白描边 + 深蓝点）
#         for j in range(17):
#             if vis[j] > 0.5:
#                 x, y = int(joints[j][0]), int(joints[j][1])
#                 # cv2.circle(canvas, (x,y), 5, KP_BORDER, 2)
#                 cv2.circle(canvas, (x,y), 3, KP_FILL, -1)
#
#         cv2.imwrite(os.path.join(file_name, f"imge_{k}.png"), canvas)

def save_batch_image_with_joints_s(batch_image, batch_joints, batch_joints_vis, file_name):
    """
    Visualize COCO-17 skeleton on batch images with region-based coloring:
      - Face  (0-4)               : COLOR_FACE
      - Torso (shoulders+hips)    : COLOR_TORSO -> (5,6,11,12)
      - Left  (l-elbow,wrist,knee,ankle)  : (7,9,13,15)
      - Right (r-elbow,wrist,knee,ankle)  : (8,10,14,16)

    Args:
        batch_image: Tensor [B, C, H, W], RGB, value range [0,255] or [0,1] (will be clipped).
        batch_joints: Tensor/ndarray [B, 17, 2], joints in (x,y) on the image.
        batch_joints_vis: Tensor/ndarray [B, 17] or [B, 17, 1], visibility scores in [0,1].
        file_name: output directory path.
    """
    # ---------------------------
    # RGB colors (as you defined)
    # ---------------------------
    COLOR_FACE = (255, 109, 106)   # coral / orange (face)
    COLOR_TORSO = (255, 194, 70)   # amber (torso)
    COLOR_LEFT = (64, 154, 244)    # sky blue (left limbs)
    COLOR_RIGHT = (67, 220, 147)   # emerald green (right limbs)

    # Convert to OpenCV BGR
    FACE = COLOR_FACE[::-1]
    TORSO = COLOR_TORSO[::-1]
    LEFT = COLOR_LEFT[::-1]
    RIGHT = COLOR_RIGHT[::-1]

    # ---------------------------
    # Skeleton groups (line color)
    # ---------------------------
    skeleton_groups = {
        FACE:  [(0, 1), (1, 3), (0, 2), (2, 4)],                 # face
        TORSO: [(5, 6), (11, 12), (5, 11), (6, 12)],             # torso
        LEFT:  [(5, 7), (7, 9), (11, 13), (13, 15)],             # left arm + left leg
        RIGHT: [(6, 8), (8, 10), (12, 14), (14, 16)]             # right arm + right leg
    }

    # ---------------------------
    # Keypoint region mapping
    # ---------------------------
    FACE_KPS = {0, 1, 2, 3, 4}
    TORSO_KPS = {5, 6, 11, 12}      # shoulders + hips (fixed torso color)
    LEFT_KPS = {7, 9, 13, 15}       # left elbow, wrist, knee, ankle
    RIGHT_KPS = {8, 10, 14, 16}     # right elbow, wrist, knee, ankle

    def kp_color_by_index(j: int):
        if j in FACE_KPS:
            return FACE
        if j in TORSO_KPS:
            return TORSO
        if j in LEFT_KPS:
            return LEFT
        if j in RIGHT_KPS:
            return RIGHT
        # fallback (should not happen for COCO-17)
        return (0, 0, 255)

    os.makedirs(file_name, exist_ok=True)

    # Ensure numpy for joints[表情]is (supports torch tensors too)
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    imgs = batch_image
    joints_all = to_numpy(batch_joints)
    vis_all = to_numpy(batch_joints_vis)

    # vis shape normalize: [B,17]
    if vis_all.ndim == 3 and vis_all.shape[-1] == 1:
        vis_all = vis_all[..., 0]

    B = imgs.size(0) if hasattr(imgs, "size") else imgs.shape[0]

    for k in range(B):
        # -------- image: tensor RGB -> uint8 BGR --------
        if hasattr(imgs, "detach"):
            img = imgs[k].permute(1, 2, 0).detach().cpu().numpy()
        else:
            img = imgs[k]
            if img.shape[0] in (1, 3):  # maybe CHW
                img = np.transpose(img, (1, 2, 0))

        img = np.clip(img, 0, 255).astype(np.uint8)
        canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        joints = joints_all[k]  # [17,2]
        vis = vis_all[k]  # [17]

        # -------- draw skeleton lines --------
        for color, pairs in skeleton_groups.items():
            for (s, e) in pairs:
                if vis[s] > 0.5 and vis[e] > 0.5:
                    x1, y1 = joints[s]
                    x2, y2 = joints[e]
                    cv2.line(
                        canvas,
                        (int(round(x1)), int(round(y1))),
                        (int(round(x2)), int(round(y2))),
                        color,
                        2
                    )

        # -------- draw keypoints (color by region) --------
        for j in range(17):
            if vis[j] > 0.5:
                x, y = joints[j]
                x_i, y_i = int(round(x)), int(round(y))
                c = kp_color_by_index(j)
                # point fill
                cv2.circle(canvas, (x_i, y_i), 3, c, -1)
                # optional thin white border for clarity
                # cv2.circle(canvas, (x_i, y_i), 4, (255, 255, 255), 1)

        out_path = os.path.join(file_name, f"imge_{k}.png")
        cv2.imwrite(out_path, canvas)





@MODELS.register_module()
class SimCCHead(BaseHead):
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
        loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
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
        self.loss_module = MODELS.build(loss)
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
        flatten_dims = self.heatmap_size[0] * self.heatmap_size[1]  #64*48=3072
        # flatten_dims = 192
        W = int(self.input_size[0] * self.simcc_split_ratio)  #192*2=384
        H = int(self.input_size[1] * self.simcc_split_ratio)  #256*2=512


        self.mlp_head_x = nn.Linear(flatten_dims, W)
        self.mlp_head_y = nn.Linear(flatten_dims, H)



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
        # print("111", feats[-1].shape)
        if self.deconv_head is None:  #hrnet 选这个
            feats = feats[-1]
            if self.final_layer is not None:
                feats = self.final_layer(feats)
        else:  #res50 选这个
            feats = self.deconv_head(feats)
        # flatten the output heatmap

        # print("222",feats.shape)


        x = torch.flatten(feats, 2)
        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        return pred_x, pred_y





    def predict(
        self,
        feats: Tuple[Tensor],
        # inputs,
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

            _batch_pred_x, _batch_pred_y = self.forward(_feats)

            _batch_pred_x_flip, _batch_pred_y_flip = self.forward(_feats_flip)
            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                _batch_pred_x_flip,
                _batch_pred_y_flip,
                flip_indices=flip_indices)

            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
            # batch_pred_x = _batch_pred_x_flip
            # batch_pred_y = _batch_pred_y_flip
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)
        preds= self.decode((batch_pred_x, batch_pred_y))

        # keypoint_weights = [
        #     d.gt_instances.keypoints_visible for d in batch_data_samples
        # ]
        # # 将 Python 列表转换为 NumPy 数组
        # keypoint_weights = np.array(keypoint_weights).transpose(0, 2, 1)

        # 获取批次大小
        batch_size = len(batch_data_samples)
        # 直接创建形状为 (B, 17, 1) 的全1张量
        keypoint_weights = torch.ones((batch_size, 17, 1))
        # 如果需要转换为列表（保持与原始代码兼容）
        keypoint_weights = [keypoint_weights[i] for i in range(batch_size)]
        keypoint_weights = np.array(keypoint_weights)

        init_coords = [data.keypoints for data in preds]
        # 将 Python 列表转换为 NumPy 数组
        init_coords = np.array(init_coords)
        batch = init_coords.shape[0]
        init_coords = init_coords.reshape(batch, 17, 2)

        # global num
        # num+=1
        # if num>20 and num<=40:
           # #1. 上采样输入图像到 256x256
           # upsampled_inputs = F.interpolate(inputs, size=(256, 256), mode='bicubic', align_corners=False)
           #
           #  # 2. 关键点缩放
           # scaled_coords = init_coords * 8.0  # 因为 256/64 = 4
           # save_batch_image_with_joints_s(upsampled_inputs, scaled_coords, keypoint_weights, "/mnt/private/mmpose/vis_results3")
           # save_batch_image_with_joints_s(inputs, init_coords,keypoint_weights,"/mnt/private/mmpose/vis_results4")

        # # 1. 上采样输入图像到 256x256
        # upsampled_inputs = F.interpolate(inputs, size=(256, 256), mode='bicubic', align_corners=False)
        #
        # # 2. 关键点缩放
        # scaled_coords = init_coords * 4.0  # 因为 256/64 = 4
        # save_batch_image_with_joints_s(upsampled_inputs, scaled_coords, keypoint_weights,
        #                                "/mnt/private/mmpose/vis_results")
        #
        # # 直接转换CUDA张量
        # arr = upsampled_inputs[0].detach().cpu().numpy()
        # img = np.transpose(arr, (1, 2, 0))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #
        # cv2.imwrite('/mnt/private/mmpose/vis_results/image2.png', img)

        # 　可视化部分关键点热图
        # (14,) 的 mask
        keep_mask = torch.zeros(17, device=batch_pred_x.device)
        keep_mask[10] = 1.0  # left knee
        # keep_mask[11] = 1.0  # right knee

        # broadcast 到 (1,14,1,1)
        keep_mask = keep_mask.view(1, 17, 1)



        selected_heatmaps_x = batch_pred_x * keep_mask
        selected_heatmaps_y = batch_pred_y * keep_mask

        if test_cfg.get('output_heatmaps', False):
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('The predicted simcc values are normalized for '
                              'visualization. This may cause discrepancy '
                              'between the keypoint scores and the 1D heatmaps'
                              '.')

            # normalize the predicted 1d distribution
            sigma = self.decoder.sigma
            # batch_pred_x = get_simcc_normalized(batch_pred_x, sigma[0])
            # batch_pred_y = get_simcc_normalized(batch_pred_y, sigma[1])

            batch_pred_x = get_simcc_normalized(selected_heatmaps_x, sigma[0])
            batch_pred_y = get_simcc_normalized(selected_heatmaps_y, sigma[1])

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

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.forward(feats)
        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
                         dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)


        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        # _, avg_acc, _ = simcc_soft_argmax_pck_accuracy(
        #     output=to_numpy(pred_simcc),
        #     target=to_numpy(gt_simcc),
        #     simcc_split_ratio=self.simcc_split_ratio,
        #     mask=to_numpy(keypoint_weights) > 0,
        #     normalize=np.ones((pred_x.size(0), 2), dtype=np.float32)
        # )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
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
