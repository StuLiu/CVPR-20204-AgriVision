'''
@Project : WRL-Agriculture-Vision 
@File    : segformer_semseg.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2024/5/8 下午12:12
@e-mail  : 1183862787@qq.com
'''
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from semseg.models import SegFormer


class SegFormerAgr(SegFormer):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19, in_channel=3) -> None:
        super().__init__(backbone, num_classes)
        self.in_channel = in_channel

    def adjust_inchannels(self):
        # for n, v in self.backbone.named_parameters():
        #     print(n)
        if self.in_channel == 3:
            return
        if self.in_channel > 3: #  <= 6
            backbone_modules = dict(self.backbone.named_children())
            newconv = nn.Conv2d(self.in_channel, self.backbone.channels[0], kernel_size=7, stride=4, padding=3)
            oldconv = backbone_modules['patch_embed1'].proj
            newconv.weight.data[:, -3:, :, :].copy_(oldconv.weight.data[:, :3, :, :])
            newconv.weight.data[:, :self.in_channel - 3, :, :].copy_(
                oldconv.weight.data[:, :self.in_channel - 3, :, :])
            backbone_modules['patch_embed1'].proj = newconv

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x[:, -self.in_channel:, :, :])
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


if __name__ == '__main__':
    model = SegFormerAgr('MiT-B3', in_channel=5)
    model.init_pretrained('/home/liuwang/liuwang_data/documents/projects/waterSeg/checkpoints/backbones/mit/mit_b3.pth')
    model.adjust_inchannels()
    x = torch.zeros(2, 5, 512, 512)
    y = model(x)

    print(y.shape)