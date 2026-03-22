_base_ = ['../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=5)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=train_cfg['max_epochs'],
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
    type='Prior_TopdownPoseEstimator_Stage1',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='Dynamic_HRNet',
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
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
                       'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    head=dict(
        type='SimCCHead',
        in_channels=32,
        out_channels=16,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 4 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        deconv_out_channels=None,
        loss=dict(type='KLDiscretLoss', use_target_weight=True),
        decoder=codec),
    # test_cfg=dict(flip_test=False))
    test_cfg=dict(
        flip_test=True,
        shift_coords=True,)
    )

# base dataset settings
dataset_type = 'MpiiDataset'
data_mode = 'topdown'
data_root = 'data/mpii/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomBBoxTransform', shift_prob=0),
    dict(type='RandomBBoxTransform'),
    dict(type='Img_TopdownAffine', hr_input_size=(256, 256),input_size=codec['input_size']),
    dict(type='Generate_Img_Target', encoder=codec),
    dict(type='PackImg_PoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='Img_TopdownAffine', hr_input_size=(256, 256),input_size=codec['input_size']),
    dict(type='PackImg_PoseInputs')
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
        ann_file='annotations/mpii_train.json',
        data_prefix=dict(img='images/'),
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
        ann_file='annotations/mpii_val.json',
        headbox_file=f'{data_root}/annotations/mpii_gt_val.mat',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(checkpoint=dict(save_best='PCK', rule='greater',max_keep_ckpts=1))

# evaluators
val_evaluator = dict(type='MpiiPCKAccuracy')
test_evaluator = val_evaluator
