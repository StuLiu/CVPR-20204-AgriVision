_base_ = ['./05_16_upernet_mit-b5_4xb4-80k_agrivisoin-512x512-mosaic4x.py']


data_preprocessor = dict(
    mean=[111.46, 113.90, 112.23, 118.30, 0],
    std=[43.75, 41.29, 41.72, 46.56, 1],
)
data_root = 'data/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBNM'
train_pipeline = [
    dict(type='LoadTifImageFromFileV2'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True
    ),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal']),
    dict(type='RandomFlip', prob=0.5, direction=['vertical']),
    dict(type='RandomRotate90', prob=0.5, degree=90),
    dict(type='PhotoMetricDistortionTif'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadTifImageFromFileV2'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        pipeline=train_pipeline,
        data_prefix=dict(
            img_path='img_dir/train_val_mosaic',
            seg_map_path='ann_dir/train_val_mosaic'
        ),
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    #dict(type='LoadImageFromFileV2'),
    dict(type='LoadTifImageFromFileV2'),
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
        ])
]

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        in_channels=5,
    )
)

# train_cfg = dict(max_iters=80000, type='IterBasedTrainLoop', val_interval=50)
