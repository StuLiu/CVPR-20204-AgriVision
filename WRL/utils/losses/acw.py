"""
@Project : WRL-Agriculture-Vision
@File    : acw.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/5/6 上午8:45
@e-mail  : 1183862787@qq.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from utils.losses.jaccard import CustomiseJaccardLoss
from utils.losses.dice_loss import CustomizeDiceLoss


class MaskBinaryCrossEntropy(nn.Module):
    def __init__(self, weight: Optional[Tensor] = None, reduction='none'):
        super(MaskBinaryCrossEntropy, self).__init__()
        self.weight = weight
        self.reduction = reduction
        if self.weight is not None:
            self.weight = torch.tensor(self.weight, dtype=torch.float)
            self.weight = self.weight.reshape(1, -1, 1, 1)
            self.weight = self.weight.cuda()

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        # pred N,C,W,H
        # target N,C,W,H
        # mask:  N,W,H
        loss = mask.unsqueeze(dim=1) * F.binary_cross_entropy_with_logits(pred, target, weight=self.weight)
        if self.reduction == 'none':
            return loss
        return loss.sum(dim=1).mean()


class MLACWBCELoss(nn.Module):
    def __init__(self, l1: float = 1.0, weight: Optional = None, **kwargs):
        super(MLACWBCELoss, self).__init__()
        self.bce = MaskBinaryCrossEntropy(weight=weight, reduction='none')
        self.eps = 1e-7
        self.itr = 0
        self.weight = 0

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        acw_class, acw_pixel = self.adaptive_class_weight(F.logsigmoid(pred).exp(), target, mask)
        loss_bce = acw_pixel * self.bce(pred, target, mask)
        loss = torch.sum(loss_bce, dim=1).mean()
        return loss

    def adaptive_class_weight(self, pred, target, mask):
        self.itr += 1
        sum_class = torch.sum(target * mask.unsqueeze(dim=1), dim=(0, 2, 3))
        weight_curr = sum_class / (sum_class.sum() + self.eps)

        self.weight = (self.weight * (self.itr - 1) + weight_curr) / self.itr
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / (mfb.sum() + self.eps)

        acw_class = mfb
        acw_pixel = (1. + pred.detach() + target) * mfb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return acw_class, acw_pixel

class MLACWJaccardLoss(MLACWBCELoss):
    def __init__(self, l1: float = 1.0, weight: Optional = None, **kwargs):
        super().__init__(l1, weight, **kwargs)
        self.l1 = l1
        self.jaccard = CustomiseJaccardLoss(**kwargs)

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        acw_class, acw_pixel = self.adaptive_class_weight(F.logsigmoid(pred).exp(), target, mask)
        loss_bce = acw_pixel * self.bce(pred, target, mask)

        loss_jcd = self.jaccard(pred, target, mask)
        # if self.itr % 20 == 0:
        #     print(f'\n{torch.unique(target)}')
        #     print(loss_bce.mean(), loss_jcd, acw_class)
        loss = (torch.sum(loss_bce, dim=1).mean() + self.l1 * loss_jcd) / (1 + self.l1)
        return loss


class MLACWJaccardLoss2(MLACWBCELoss):
    def __init__(self, l1: float = 1.0, weight: Optional = None, **kwargs):
        super().__init__(l1, weight, **kwargs)
        self.l1 = l1
        self.jaccard = CustomiseJaccardLoss(reduction='none', **kwargs)

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        acw_class, acw_pixel = self.adaptive_class_weight(F.logsigmoid(pred).exp(), target, mask)
        loss_bce = acw_pixel * self.bce(pred, target, mask)
        # print(acw_class)
        loss_jcd = acw_class * self.jaccard(pred, target, mask)
        # if self.itr % 20 == 0:
        #     print(f'\n{torch.unique(target)}')
        #     print(loss_bce.mean(), loss_jcd, acw_class)
        loss = (torch.sum(loss_bce, dim=1).mean() + self.l1 * loss_jcd.mean()) / (1 + self.l1)
        return loss


class MLFCWLoss(MLACWBCELoss):
    def __init__(self, l1: float = 1.0, weight: Optional = None, **kwargs):
        super().__init__(l1, weight, **kwargs)
        self.l1 = l1
        self.jaccard = CustomiseJaccardLoss(reduction='none', **kwargs)
        self.weight = [0.01, 0.9, 0.08, 1.0, 0.5, 1.0, 0.1, 0.9, 1.2]
        self.class_weight = None

    def fixed_class_weight(self, pred, target, mask):
        if self.class_weight is None:
            self.class_weight = torch.tensor(self.weight).float().to(pred.device)
        self.itr += 1
        fcw_class = self.class_weight
        fcw_pixel = (1. + pred.detach() + target) * fcw_class.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return fcw_class, fcw_pixel

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        fcw_class, fcw_pixel = self.fixed_class_weight(F.logsigmoid(pred).exp(), target, mask)
        loss_bce = fcw_pixel * self.bce(pred, target, mask)
        # print(fcw_class)
        loss_jcd = fcw_class * self.jaccard(pred, target, mask)
        # if self.itr % 20 == 0:
        #     print(f'\n{torch.unique(target)}')
        #     print(loss_bce.mean(), loss_jcd, acw_class)
        loss = (torch.sum(loss_bce, dim=1).mean() + self.l1 * loss_jcd.mean()) / (1 + self.l1)
        return loss


class MLACWDiceLoss(MLACWBCELoss):
    def __init__(self, l1: float = 1.0, weight: Optional = None, **kwargs):
        super().__init__(l1, weight, **kwargs)
        self.l1 = l1
        self.dice = CustomizeDiceLoss(**kwargs)

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        acw_class, acw_pixel = self.adaptive_class_weight(F.logsigmoid(pred).exp(), target, mask)
        loss_bce = acw_pixel * self.bce(pred, target, mask)

        loss_dice = self.dice(pred, target, mask)
        # if self.itr % 20 == 0:
        #     print(f'\n{torch.unique(target)}')
        #     print(loss_bce.mean(), loss_jcd, acw_class)
        loss = (torch.sum(loss_bce, dim=1).mean() + self.l1 * loss_dice) / (1 + self.l1)
        return loss


class MLACWLoss(MLACWBCELoss):
    def __init__(self, l1: float = 1.0, weight: Optional = None, **kwargs):
        super().__init__(l1, weight, **kwargs)


    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        pred = F.logsigmoid(pred).exp()

        _, acw = self.adaptive_class_weight(pred, target, mask)

        err = torch.pow((target - pred), 2)

        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(acw * pnc, 1)

        intersection = 2 * torch.sum(pred * target * mask.unsqueeze(dim=1), dim=(0, 2, 3)) + self.eps
        union = (pred + target) * mask.unsqueeze(dim=1)
        union = torch.sum(union, dim=(0, 2, 3)) + self.eps
        dice = intersection / union

        return loss_pnc.mean() - dice.mean().log()



if __name__ == '__main__':
    # loss_mlbce = MultiLabelBCELoss()
    # loss_mljcd = MultiLabelJaccardLoss(mean=False)
    loss_mlacw = MLACWLoss()
    loss_mlacwbce = MLACWBCELoss()

    a = torch.randint(-10, 10, (2, 2, 512, 512)).float().cuda()
    a[:, 0, :, :] = -10000
    a[:, 1, :, :] = 10000
    b = torch.randint(0, 2, (2, 2, 512, 512)).float().cuda()
    b[:, 0, :, :] = 0
    b[:, 1, :, :] = 1

    m = torch.ones((2, 512, 512)).float().cuda()
    # print(loss_mlbce(a, b))
    # print(loss_mljcd(a, b))
    print(loss_mlacw(a, b, m))
    print(loss_mlacwbce(a, b, m))
