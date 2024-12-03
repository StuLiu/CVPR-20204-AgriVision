_base_ = [
    '../_base_/models/upernet_mit-b0.py', '../_base_/datasets/agriculture_vision.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(
    size=crop_size,
    mean=[111.46, 113.90, 112.23, 118.30],
    std=[43.75, 41.29, 41.72, 46.56],
    bgr_to_rgb=False,
)
data_root = '/home/liuwang/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN'
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa

train_pipeline = [
    dict(type='LoadTifImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.75, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal']),
    dict(type='RandomFlip', prob=0.5, direction=['vertical']),
    dict(type='RandomRotate90', prob=0.5, degree=90),
    dict(type='PhotoMetricDistortionTif'),
    dict(type='BioMedicalGaussianNoise'),
    dict(type='BioMedicalGaussianBlur'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        pipeline=train_pipeline,
        data_prefix=dict(
            img_path='img_dir/train_val_mosaic',
            seg_map_path='ann_dir/train_val_mosaic'),
    )
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        data_root=data_root
    )
)
test_dataloader = val_dataloader

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5]
tta_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadTifImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ]
    )
]

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=4,
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3],
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=9,
        loss_decode=dict(
            type='ACWLoss',
            loss_weight=1.0
        )
    ),
    auxiliary_head=dict(
        in_channels=320,
        num_classes=9,
        loss_decode=dict(
            type='ACWLoss',
        )
    ),
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=80000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, max_keep_ckpts=1,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))
