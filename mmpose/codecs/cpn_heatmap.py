# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numpy as np
from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils.gaussian_heatmap import (generate_gaussian_heatmaps,
                                     generate_unbiased_gaussian_heatmaps)
from .utils.post_processing import get_heatmap_maximum
from .utils.refinement import refine_keypoints, refine_keypoints_dark


@KEYPOINT_CODECS.register_module()
class CPNHeatmap(BaseKeypointCodec):

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: list,
                 unbiased: bool = False,
                 blur_kernel_size: int = 11) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma  # Now sigma is a list of values
        self.unbiased = unbiased

        # The Gaussian blur kernel size of the heatmap modulation
        # in DarkPose and the sigma value follows the empirical
        # formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`
        # which gives:
        #   sigma~=3 if ks=17
        #   sigma=2 if ks=11;
        #   sigma~=1.5 if ks=7;
        #   sigma~=1 if ks=3;
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (np.array(input_size) /
                             heatmap_size).astype(np.float32)

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints into heatmaps for multiple sigma values. Note that the original keypoint
        coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - heatmaps (np.ndarray): The generated heatmap in shape
                (S,K, H, W) where [W, H] is the `heatmap_size` and `S` is the number of sigmas
            - keypoint_weights (np.ndarray): The target weights in shape
                (S,N, K)
        """

        assert keypoints.shape[0] == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        heatmaps_list = []

        # Iterate over all sigma values
        for sigma_value in self.sigma:
            if self.unbiased:
                heatmaps, keypoint_weights = generate_unbiased_gaussian_heatmaps(
                    heatmap_size=self.heatmap_size,
                    keypoints=keypoints / self.scale_factor,
                    keypoints_visible=keypoints_visible,
                    sigma=sigma_value)
            else:
                heatmaps, keypoint_weights = generate_gaussian_heatmaps(
                    heatmap_size=self.heatmap_size,
                    keypoints=keypoints / self.scale_factor,
                    keypoints_visible=keypoints_visible,
                    sigma=sigma_value)

            heatmaps_list.append(heatmaps)

        # Stack the heatmaps and weights along the new axis for sigma
        heatmaps_stack = np.stack(heatmaps_list, axis=0)

        B, S, H, W = heatmaps_stack.shape
        heatmaps= heatmaps_stack.reshape(B * S, H, W)  # (56, 64, 48)

        encoded = dict(heatmaps=heatmaps,
                       keypoint_weights=keypoint_weights,
                       keypoints=keypoints,
                       keypoints_visible=keypoints_visible)

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        heatmaps = encoded.copy()
        K, H, W = heatmaps.shape

        keypoints, scores = get_heatmap_maximum(heatmaps)

        # Unsqueeze the instance dimension for single-instance results
        keypoints, scores = keypoints[None], scores[None]

        if self.unbiased:
            # Alleviate biased coordinate
            keypoints = refine_keypoints_dark(
                keypoints, heatmaps, blur_kernel_size=self.blur_kernel_size)

        else:
            keypoints = refine_keypoints(keypoints, heatmaps)

        # Restore the keypoint scale
        keypoints = keypoints * self.scale_factor

        return keypoints, scores
