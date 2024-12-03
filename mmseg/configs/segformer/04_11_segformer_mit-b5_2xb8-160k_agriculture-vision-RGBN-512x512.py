_base_ = ['./segformer_mit-b0_8xb2-160k_agriculture-vision-512x512.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa

data_preprocessor = dict(
    mean=[111.46, 113.90, 112.23, 118.30],
    std=[43.75, 41.29, 41.72, 46.56],
    bgr_to_rgb=False,
)

# model settings
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=4,
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

train_dataloader = dict(batch_size=8, num_workers=4)