# Copyright (c) OpenMMLab. All rights reserved.
import json
import copy
from typing import Optional, Sequence, Tuple, Union
import math
import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from functools import partial
from torch import Tensor, nn
import torch.nn.functional as F
from collections import namedtuple
from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.models.utils.tta import flip_coordinates,flip_coordinates1
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from mmengine.model import BaseModule, ModuleDict, Sequential
from ..base_head import BaseHead

import numpy as np
from mmpose.evaluation.functional.keypoint_eval import _calc_distances
from mmengine.runner.checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                         find_latest_checkpoint, save_checkpoint,
                         weights_to_cpu)
OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class HRNet_Model_Head(BaseHead):

    _version = 2

    def __init__(self,decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        if init_cfg is None:
            init_cfg = self.default_init_cfg
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        super().__init__(init_cfg)
        network = dict(
            type='TopdownPoseEstimator',
            backbone=dict(
                type='HRNet',
                frozen_stages=4, #冻模型
                in_channels=3,
                extra=dict(
                    stage1=dict(
                        num_modules=1,
                        num_branches=1,
                        block='BOTTLENECK',
                        num_blocks=(4,),
                        num_channels=(64,)),
                    stage2=dict(
                        num_modules=1,
                        num_branches=2,
                        block='BASIC',
                        num_blocks=(4, 4),
                        num_channels=(32, 64)),
                    stage3=dict(
                        num_modules=4,
                        num_branches=3,
                        block='BASIC',
                        num_blocks=(4, 4, 4),
                        num_channels=(32, 64, 128)),
                    stage4=dict(
                        num_modules=3,
                        num_branches=4,
                        block='BASIC',
                        num_blocks=(4, 4, 4, 4),
                        num_channels=(32, 64, 128, 256)))
            ),
            head=dict(
                type='HeatmapHead',
                in_channels=32,
                out_channels=17,
                deconv_out_channels=None
            ),
        )
        self.net = MODELS.build(network)
        self.is_load = 0

    @property
    def default_init_cfg(self):
        init_cfg = [dict(type='Normal', layer=['Linear'], std=0.01, bias=0)]
        return init_cfg


    def forward(self,feats: Tuple[Tensor]) -> Tensor:
        if self.is_load == 0:
            self.load_checkpoint('cpu', False)
            self.is_load = 1

        image = feats
        with torch.no_grad():
            feature= self.net.backbone(image)  #B,32,64,48
            init_x = self.net.head(feature)  #B,17,64,48
        return init_x




    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_heatmaps = self.forward(_feats)
            _batch_heatmaps_flip = flip_heatmaps(
                self.forward(_feats_flip),
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_heatmaps = self.forward(feats)

        preds = self.decode(batch_heatmaps)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds


    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:

        pred_fields = self.forward(feats)
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

        return losses

    def load_checkpoint(self,
                        map_location,
                        strict,
                        revise_keys: list = [(r'^module.', '')]):
        """Load checkpoint from given ``filename``.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            map_location (str or callable): A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'cpu'.
            strict (bool): strict (bool): Whether to allow different params for
                the model and checkpoint.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Defaults to strip
                the prefix 'module.' by [(r'^module\\.', '')].
        """
        checkpoint = _load_checkpoint(
            'td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth',
                                       map_location=map_location)

        checkpoint = _load_checkpoint_to_model(self.net, checkpoint, strict, revise_keys=revise_keys)
        print("Loaded HRNet checkpoint")


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



