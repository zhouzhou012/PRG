_base_ = ['../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=64)
# codec settings
codec = dict(
    type='Img_SimCCLabel', input_size=(64,64), sigma=1.0, simcc_split_ratio=3.0)

# model settings
model = dict(
    type='Prior_TopdownPoseEstimator_Stage2',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    s1_pretrained='/mnt/private/mmpose/work_dirs/PRG/Prior_stage1_simcc_res50-1xb64_210e_coco_64x64/best_coco_AP_epoch_210.pth',
    backbone=dict(
        type='Dynamic_ResNet',
        depth=50,
            ),
    head=dict(
        type='SimCCHead',
        in_channels=2048,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        loss=dict(type='KLDiscretLoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='Img_TopdownAffine', hr_input_size=(256,256),input_size=codec['input_size']),
    dict(type='Generate_Img_Target', encoder=codec),
    dict(type='PackImg_PoseInputs')
]
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine',input_size=codec['input_size']),
    dict(type='PackPoseInputs')
    # dict(type='Img_TopdownAffine', hr_input_size=(256,256),input_size=codec['input_size']),
    # dict(type='PackImg_PoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        # ann_file='annotations/person_keypoints_val2017.json',
        # # bbox_file=f'{data_root}person_detection_results/'
        # #           'COCO_val2017_detections_AP_H_56_person.json',
        # data_prefix=dict(img='val2017/'),
        # ann_file='/mnt/private/Poseur/data/person_keypoints_785demo.json',
        # data_prefix=dict(img='val2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        # ann_file='annotations/person_keypoints_val2017.json',
        # bbox_file=f'{data_root}person_detection_results/'
        #           'COCO_val2017_detections_AP_H_56_person.json',
        # data_prefix=dict(img='val2017/'),
        ann_file='/mnt/private/Poseur/data/person_keypoints_785demo.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater',max_keep_ckpts=1))

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator

find_unused_parameters=True
