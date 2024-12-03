_base_ = ['./05_16_upernet_mit-b5_4xb4-80k_agrivisoin-512x512-mosaic4x.py']

model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='ACWLossV2'
        )
    ),
    auxiliary_head=dict(
        loss_decode=dict(
            type='ACWLossV2'
        )
    )
)