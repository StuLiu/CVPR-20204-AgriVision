'''
@Project : WRL-Agriculture-Vision 
@File    : segformer.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2024/5/8 上午1:35
@e-mail  : 1183862787@qq.com
'''
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional

from segmentation_models_pytorch.base import (
    SegmentationModel,
)
from segmentation_models_pytorch.encoders.mix_transformer import (
    mix_transformer_encoders,
    MixVisionTransformerEncoder
)
from segmentation_models_pytorch.encoders import get_encoder


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class SegFormerNeck(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        # for i, dim in enumerate(dims):
        #     self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))
        self.linear_c1 = MLP(dims[0], embed_dim)
        self.linear_c2 = MLP(dims[1], embed_dim)
        self.linear_c3 = MLP(dims[2], embed_dim)
        self.linear_c4 = MLP(dims[3], embed_dim)

        self.linear_fuse = ConvModule(embed_dim*4, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, *features) -> Tensor:
        features = features[-4:]
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:]).contiguous()]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i+2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:]).contiguous()
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        return seg

class SegmentationHead(nn.Sequential):
    def __init__(self, upsampling=1):
        upmodule = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(upmodule)


class SegFormer(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "mit_b3",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        # image_size=512,
        in_channels: int = 3,
        classes: int = 1,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            # image_size=image_size
        )

        self.decoder = SegFormerNeck(
            dims=self.encoder.out_channels[-4:],
            embed_dim=256 if int(encoder_name[-1:]) <= 1 else 768,
            num_classes=classes
        )

        self.segmentation_head = SegmentationHead(
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "segformer-{}".format(encoder_name)
        self.initialize()


if __name__ == '__main__':
    model = SegFormer('mit_b3', classes=19, in_channels=5)
    model.train()
    _x = torch.zeros(2, 5, 512, 512)
    _y = model(_x)
    print(_y.shape)
