_base_ = ['./05_16_deeplabv3plus_mit-b3_2xb8-80k_agrivision-512x512-mosaic4x.py']

train_dataloader = dict(
    batch_size=4
)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3],
    )
)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=8000,
        max_keep_ckpts=2,
        save_last=True,
        save_best=['mIoU'],
        type='CheckpointHook'
    )
)
