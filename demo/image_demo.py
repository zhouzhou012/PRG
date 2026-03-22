# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from mmcv.image import imread
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
import json

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # init visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style=args.skeleton_style)

    # inference a single image
    batch_results = inference_topdown(model, args.img)
    results = merge_data_samples(batch_results)

    # show the results
    img = imread(args.img, channel_order='rgb')


    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=args.kpt_thr,
        draw_heatmap=args.draw_heatmap,
        show_kpt_idx=args.show_kpt_idx,
        skeleton_style=args.skeleton_style,
        show=args.show,
        out_file=args.out_file)


if __name__ == '__main__':
    main()

# # Copyright (c) OpenMMLab. All rights reserved.
# import os
# import json
# from argparse import ArgumentParser
#
# import numpy as np
# import torch
# from mmcv.image import imread
#
# from mmengine.structures import InstanceData, PixelData
# from mmpose.apis import inference_topdown, init_model
# from mmpose.registry import VISUALIZERS
# from mmpose.structures import merge_data_samples
# import torchvision.transforms.functional as TF
#
#
# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument('img', help='Image file')
#     parser.add_argument('config', help='Config file')
#     parser.add_argument('checkpoint', help='Checkpoint file')
#
#     parser.add_argument('--out-file', default=None, help='Path to output file')
#     parser.add_argument('--device', default='cuda:0', help='Device used')
#
#     parser.add_argument('--draw-heatmap', action='store_true',
#                         help='Visualize heatmap (GT & Pred)')
#     parser.add_argument('--show-kpt-idx', action='store_true')
#     parser.add_argument('--skeleton-style', default='mmpose',
#                         choices=['mmpose', 'openpose'])
#     parser.add_argument('--kpt-thr', type=float, default=0.3)
#
#     parser.add_argument('--radius', type=int, default=3)
#     parser.add_argument('--thickness', type=int, default=1)
#     parser.add_argument('--alpha', type=float, default=0.8)
#
#     parser.add_argument('--show', action='store_true')
#
#     # 🔴 新增：GT joints json
#     parser.add_argument(
#         '--gt-joints-file',
#         type=str,
#         default=None,
#         help='Path to GT joints json file (COCO/CrowdPose format)'
#     )
#     return parser.parse_args()
#
#
# def main():
#     args = parse_args()
#
#     # -------------------------------------------------
#     # build model
#     # -------------------------------------------------
#     if args.draw_heatmap:
#         cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
#     else:
#         cfg_options = None
#
#     model = init_model(
#         args.config,
#         args.checkpoint,
#         device=args.device,
#         cfg_options=cfg_options
#     )
#
#     # -------------------------------------------------
#     # init visualizer
#     # -------------------------------------------------
#     model.cfg.visualizer.radius = args.radius
#     model.cfg.visualizer.alpha = args.alpha
#     model.cfg.visualizer.line_width = args.thickness
#
#     visualizer = VISUALIZERS.build(model.cfg.visualizer)
#     visualizer.set_dataset_meta(
#         model.dataset_meta, skeleton_style=args.skeleton_style
#     )
#
#     # -------------------------------------------------
#     # inference
#     # -------------------------------------------------
#     batch_results = inference_topdown(model, args.img)
#     results = merge_data_samples(batch_results)
#
#     img = imread(args.img, channel_order='rgb')
#
#     # -------------------------------------------------
#     # load GT json (optional)
#     # -------------------------------------------------
#     gt_anns = None
#     if args.gt_joints_file is not None:
#         with open(args.gt_joints_file, 'r') as f:
#             gt_anns = json.load(f)
#
#     # -------------------------------------------------
#     # 🔴 construct GT instances / heatmaps
#     # -------------------------------------------------
#     if gt_anns is not None:
#         image_name = os.path.basename(args.img)
#
#         # find image_id
#         image_id = None
#         for img_info in gt_anns.get('images', []):
#             if img_info['file_name'] == image_name:
#                 image_id = img_info['id']
#                 break
#
#         if image_id is not None:
#             gt_kpts = []
#             gt_scores = []
#
#             for ann in gt_anns.get('annotations', []):
#                 if ann['image_id'] != image_id:
#                     continue
#
#                 kpts = np.array(ann['keypoints']).reshape(-1, 3)
#                 gt_kpts.append(kpts[:, :2])
#                 gt_scores.append((kpts[:, 2] > 0).astype(np.float32))
#
#             if len(gt_kpts) > 0:
#                 # ---- GT keypoints ----
#                 gt_instances = InstanceData()
#                 gt_instances.keypoints = np.stack(gt_kpts)  # (N,K,2)
#                 gt_instances.keypoint_scores = np.stack(gt_scores)  # (N,K)
#
#                 results.gt_instances = gt_instances
#
#                 # ---- GT heatmaps (optional) ----
#                 if args.draw_heatmap:
#                     # heatmap size: use pred heatmap size if available
#                     if hasattr(results, 'pred_fields') and \
#                             hasattr(results.pred_fields, 'heatmaps'):
#                         H, W = results.pred_fields.heatmaps.shape[-2:]
#                     else:
#                         H, W = 64, 48  # fallback
#
#                     yy = torch.arange(H).view(H, 1).float()
#                     xx = torch.arange(W).view(1, W).float()
#
#                     sigma = 20.0
#                     gt_hms = []
#
#                     for person_kpts in gt_instances.keypoints:
#                         K = person_kpts.shape[0]
#                         hm = torch.zeros((K, H, W))
#                         for k in range(K):
#                             x, y = person_kpts[k]
#                             if x < 0 or y < 0:
#                                 continue
#                             hm[k] = torch.exp(
#                                 -((xx - x) ** 2 + (yy - y) ** 2) /
#                                 (2 * sigma ** 2)
#                             )
#
#                             hm[k] = TF.gaussian_blur(
#                                 hm[k].unsqueeze(0),  # (1, H, W)
#                                 kernel_size=25,
#                                 sigma=20.0
#                             ).squeeze(0)
#
#                         gt_hms.append(hm)
#
#                     gt_hms = torch.stack(gt_hms).max(dim=0)[0]
#
#                     gt_fields = PixelData()
#                     gt_fields.heatmaps = gt_hms
#                     results.gt_fields = gt_fields
#
#                 # # ---- GT heatmaps (optional) ----
#                 # if args.draw_heatmap:
#                 #     if hasattr(results, 'pred_fields') and \
#                 #             hasattr(results.pred_fields, 'heatmaps'):
#                 #         H, W = results.pred_fields.heatmaps.shape[-2:]
#                 #     else:
#                 #         H, W = 64, 48
#                 #
#                 #     yy = torch.arange(H).view(H, 1).float()
#                 #     xx = torch.arange(W).view(1, W).float()
#                 #
#                 #     sigma = 20.0  # ⭐ 关键：平缓而不尖
#                 #     gt_hms = []
#                 #
#                 #     img_h, img_w = img.shape[:2]
#                 #
#                 #     for person_kpts in gt_instances.keypoints:
#                 #         K = person_kpts.shape[0]
#                 #         hm = torch.zeros((K, H, W))
#                 #         for k in range(K):
#                 #             x_img, y_img = person_kpts[k]
#                 #             if x_img < 0 or y_img < 0:
#                 #                 continue
#                 #
#                 #             # ⭐ image → heatmap 坐标映射（必须）
#                 #             x = x_img * W / img_w
#                 #             y = y_img * H / img_h
#                 #
#                 #             hm[k] = torch.exp(
#                 #                 -((xx - x) ** 2 + (yy - y) ** 2) /
#                 #                 (2 * sigma ** 2)
#                 #             )
#                 #
#                 #             # ⭐ 轻度 blur，模拟 pred 的平滑感
#                 #             hm[k] = TF.gaussian_blur(
#                 #                 hm[k].unsqueeze(0),
#                 #                 kernel_size=11,
#                 #                 sigma=2.0
#                 #             ).squeeze(0)
#                 #
#                 #         gt_hms.append(hm)
#                 #
#                 #     gt_hms = torch.stack(gt_hms).max(dim=0)[0]
#                 #
#                 #     gt_fields = PixelData()
#                 #     gt_fields.heatmaps = gt_hms
#                 #     results.gt_fields = gt_fields
#
#     # -------------------------------------------------
#     # visualize
#     # -------------------------------------------------
#     visualizer.add_datasample(
#         name='result',
#         image=img,
#         data_sample=results,
#         draw_gt=True,
#         draw_pred=True,
#         draw_heatmap=args.draw_heatmap,
#         show_kpt_idx=args.show_kpt_idx,
#         skeleton_style=args.skeleton_style,
#         kpt_thr=args.kpt_thr,
#         show=args.show,
#         out_file=args.out_file
#     )
#
#
# if __name__ == '__main__':
#     main()

