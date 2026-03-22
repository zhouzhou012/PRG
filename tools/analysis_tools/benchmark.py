from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown
from mmpose.evaluation.metrics import CocoMetric
from mmpose.registry import METRICS
from mmpose.datasets import build_dataset
from mmengine.registry import FUNCTIONS


from mmengine.config import Config
from mmpose.models.data_preprocessors import PoseDataPreprocessor

import torch
from torch.utils.data import DataLoader
import tqdm
import time

if __name__ == "__main__":
    device = 'cuda'
    # mmpose_cfg = "/mnt/private/mmpose/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py"
    # mmpose_ckpt = '/mnt/private/mmpose/td-reg_res50_rle-8xb64-210e_coco-256x192-d37efd64_20220913.pth'

    # mmpose_cfg="/mnt/private/mmpose/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res101_rle-8xb64-210e_coco-256x192.py"
    # mmpose_ckpt='/mnt/private/mmpose/deeppose_res101_coco_256x192_rle-16c3d461_20220615.pth'

    # mmpose_cfg = "/mnt/private/mmpose/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res152_rle-8xb64-210e_coco-256x192.py"
    # mmpose_ckpt = "/mnt/private/mmpose/deeppose_res152_coco_256x192_rle-c05bdccf_20220615.pth"


    # mmpose_cfg = "/mnt/private/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
    # mmpose_ckpt = "/mnt/private/mmpose/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"

    # mmpose_cfg = "/mnt/private/mmpose/configs/body_2d_keypoint/topdown_regression/coco/td-reg_hrnet-w32_rle-8xb20-210e_coco-256x192.py"
    # mmpose_ckpt="/mnt/private/mmpose/work_dirs/td-reg_hrnet-w32_rle-8xb20-210e_coco-256x192/best_coco_AP_epoch_210.pth"
    #
    # mmpose_cfg="/mnt/private/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_Tokenpose-base-210e_coco_256x192.py"
    # mmpose_ckpt="/mnt/private/mmpose/work_dirs/td-hm_Tokenpose-base-210e_coco_256x192/best_coco_AP_epoch_210.pth"
    mmpose_cfg="/mnt/private/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
    mmpose_ckpt="/mnt/private/mmpose/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth"

    # mmpose_cfg="/mnt/private/mmpose/configs/body_2d_keypoint/simcc/coco/simcc_hrnet-w32_8xb20-210e_coco-256x192-256.py"
    # mmpose_ckpt="/mnt/private/mmpose/work_dirs/simcc_hrnet-w32_8xb20-210e_coco-256x192/best_coco_AP_epoch_210.pth"

    # mmpose_cfg="/mnt/private/mmpose/configs/body_2d_keypoint/simcc/coco/simcc_hrnet-w32_8xb20-210e_coco-256x192-256.py"
    # mmpose_ckpt="/mnt/private/mmpose/work_dirs/simcc_hrnet-w32_8xb20-210e_coco-256x192-256/best_coco_AP_epoch_210.pth"

    # mmpose_cfg="/mnt/private/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"
    # mmpose_ckpt="/mnt/private/mmpose/work_dirs/td-hm_ViTPose-base_8xb64-210e_coco-256x192/best_coco_AP_epoch_210.pth"

    warm_up_iters = 5
    batch_size =128
    num_workers = 8

    config = Config.fromfile(mmpose_cfg)
    pose_estimator = None
    pose_estimator = init_pose_estimator(
        config,
        mmpose_ckpt,
        device=device,
        cfg_options=dict(
        model=dict(test_cfg=dict(output_heatmaps=False)))
    )
    pose_estimator.eval()

    config.val_dataloader.dataset.bbox_file = None
    config.val_dataloader.dataset.test_mode = False
    
    dataset = build_dataset(config.val_dataloader.dataset)
    # print(len(dataset))

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=FUNCTIONS.get('pseudo_collate'),
        persistent_workers=True,
        drop_last=False,
    )
    
    evaluator: CocoMetric = METRICS.build(config.val_evaluator)
    evaluator.dataset_meta = dataset.metainfo

    accumulate_time = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            start = time.time()
            pose_estimator.val_step(data)
            if i > warm_up_iters:
                accumulate_time += time.time() - start
    print(f"BenchMark on validation: {(len(dataloader) - warm_up_iters) * batch_size / (accumulate_time)}")
    
    # start = time.time()
    # with torch.no_grad():
    #     for i, data in enumerate(tqdm.tqdm(dataloader)):
    #         if i == warm_up_iters - 1:
    #             start = time.time()
    #         pose_estimator.val_step(data)
    #
    # end = time.time()
    # # print(len(dataloader))
    # print(f"BenchMark on validation: {(len(dataloader) - warm_up_iters) * batch_size / (end - start)}")