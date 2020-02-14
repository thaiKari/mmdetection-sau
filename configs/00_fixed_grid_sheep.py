# model settings
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=None,
    bbox_head=dict(
        type='FixedGridSheepHead2',
        num_classes=2,
        in_channels=256,
        grid_size = (6,8)
    ))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    sampler=dict(#copied from libra rcnn config
            type='RandomSampler',
            num=48,#number of grid cells..
            pos_fraction=0.5,
            neg_pos_ub=5,
            add_gt_as_proposals=False),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'SauDataset'
data_root = 'data/data_external/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True,rgb_only=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(40, 30), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True,rgb_only=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(40, 30), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train2020_fixed_grid_bbox_coco.json',
        img_prefix=data_root + 'train2020_4d/',
        #ann_file=data_root + 'annotations/train2020_4d_crop1024_bgtosheep_1to2.json',
        #img_prefix=data_root + 'train2020_4d_crop1024/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val2020_fixed_grid_bbox_coco.json',
        img_prefix=data_root + 'val2020_4d/',
        #ann_file=data_root + 'annotations/val2020_4d_crop1024_bgtosheep_1to1.json',
        #img_prefix=data_root + 'val2020_4d_crop1024/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val2020_fixed_grid_bbox_coco.json',
        img_prefix=data_root + 'val2020_4d/',
        #ann_file=data_root + 'annotations/val2020_4d_crop1024_bgtosheep_1to1.json',
        #img_prefix=data_root + 'val2020_4d_crop1024/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/retinanet_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
