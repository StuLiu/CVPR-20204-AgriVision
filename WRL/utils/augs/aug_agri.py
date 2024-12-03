"""
@Project : WRL-Agriculture-Vision
@File    : aug_agri.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/5/10 下午6:17
@e-mail  : 1183862787@qq.com
"""
import torchvision.transforms.functional as TF
import random
import math
import torch
from torch import Tensor
from typing import *
import numpy as np

from utils.augs.colorjitter import PhotoMetricDistortionTif


class ToTensor:
    def __init__(self) -> None:
        pass

    def __call__(self, image: np.ndarray, segmentation_maps: np.ndarray) -> Tuple[Tensor, Tensor]:
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous() / 255.0
        segmentation_maps = torch.from_numpy(segmentation_maps[0, :, :, :]).permute(2, 0, 1).contiguous()

        return image, segmentation_maps


class ToNumpy:
    def __init__(self) -> None:
        pass

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        image = image.permute(1, 2, 0).contiguous().numpy()
        segmentation_maps = segmentation_maps.permute(1, 2, 0).contiguous().unsqueeze(dim=0).numpy()

        return image, segmentation_maps


class RandomApply(torch.nn.Module):
    """Apply randomly a list of transformations with a given probability.

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        transforms = transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(),
        ]), p=0.3)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, image, segmentation_maps):
        if self.p < torch.rand(1):
            return image, segmentation_maps
        if self.transforms is not Iterable:
            return self.transforms(image, segmentation_maps)
        else:
            for t in self.transforms:
                image, segmentation_maps = t(image, segmentation_maps)
        return image, segmentation_maps


class OneOf(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        if not isinstance(transforms, Iterable):
            self.transforms = [transforms]
        else:
            self.transforms = transforms

        self.candi = list(range(len(transforms)))

    def forward(self, image, segmentation_maps):
        idx = random.choice(self.candi)
        image, segmentation_maps = self.transforms[idx](image, segmentation_maps)
        return image, segmentation_maps


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        for transform in self.transforms:
            image, segmentation_maps = transform(image, segmentation_maps)

        return image, segmentation_maps


class Normalize:
    def __init__(self, mean: Union[list, tuple] = (0.485, 0.456, 0.406),
                 std: Union[list, tuple] = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        image = image.float()
        image[-3:, :, :] /= 255
        image[-3:, :, :] = TF.normalize(image[-3:, :, :], self.mean, self.std)
        return image, segmentation_maps


class RandomAdjustSharpness:
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        self.sharpness = sharpness_factor
        self.p = p

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            image[-3:, :, :] = TF.adjust_sharpness(image[-3:, :, :], self.sharpness)
        return image, segmentation_maps


class RandomGaussianBlur:
    def __init__(self, kernel_size: Union[int, list] = 3, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            image[-3:, :, :] = TF.gaussian_blur(image[-3:, :, :], self.kernel_size)
        return image, segmentation_maps


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(segmentation_maps)
        return image, segmentation_maps


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(segmentation_maps)
        return image, segmentation_maps


class RandomGrayscale:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            image[-3:, :, :] = TF.rgb_to_grayscale(image[-3:, :, :], 3)
        return image, segmentation_maps


class Equalize:
    def __call__(self, image, segmentation_maps):
        return TF.equalize(image), segmentation_maps


class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, seg_fill: int = 255, expand: bool = False) -> None:
        """Rotate the image by a random angle between -angle and angle with probability p

        Args:
            p: probability
            angle: rotation angle value in degrees, counter-clockwise.
            expand: Optional expansion flag.
                    If true, expands the output image to make it large enough to hold the entire rotated image.
                    If false or omitted, make the output image the same size as the input image.
                    Note that the expand flag assumes rotation around the center and no translation.
        """
        self.p = p
        self.angle = degrees
        self.expand = expand
        self.seg_fill = seg_fill

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        random_angle = random.random() * 2 * self.angle - self.angle
        nH, nW = image.shape[-2:]
        if random.random() < self.p:
            image = TF.rotate(image, random_angle, TF.InterpolationMode.BILINEAR, self.expand, fill=[0])
            segmentation_maps = TF.rotate(segmentation_maps, random_angle, TF.InterpolationMode.NEAREST, self.expand,
                                          fill=[self.seg_fill])
            # resize to original size
            if self.expand:
                image = TF.resize(image, [nH, nW], TF.InterpolationMode.BILINEAR)
                segmentation_maps = TF.resize(segmentation_maps, [nH, nW], TF.InterpolationMode.NEAREST)
        return image, segmentation_maps


class RandomRotation90:
    def __init__(self, degrees: int = 90, p: float = 0.5, seg_fill: int = 255, expand: bool = False) -> None:
        """Rotate the image by a random angle between -angle and angle with probability p

        Args:
            p: probability
            angle: rotation angle value in degrees, counter-clockwise.
            expand: Optional expansion flag.
                    If true, expands the output image to make it large enough to hold the entire rotated image.
                    If false or omitted, make the output image the same size as the input image.
                    Note that the expand flag assumes rotation around the center and no translation.
        """
        assert degrees in [0, 90, 180, 270]
        self.p = p
        self.angle = degrees
        self.expand = expand
        self.seg_fill = seg_fill

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        nH, nW = image.shape[-2:]
        if random.random() < self.p:
            image = TF.rotate(image, self.angle, TF.InterpolationMode.BILINEAR, self.expand, fill=[0])
            segmentation_maps = TF.rotate(segmentation_maps, self.angle, TF.InterpolationMode.NEAREST, self.expand,
                                          fill=[self.seg_fill])
            # resize to original size
            if self.expand:
                image = TF.resize(image, [nH, nW], TF.InterpolationMode.BILINEAR)
                segmentation_maps = TF.resize(segmentation_maps, [nH, nW], TF.InterpolationMode.NEAREST)
        return image, segmentation_maps


class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]]) -> None:
        """Crops the image at the center

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.center_crop(image, self.size), TF.center_crop(segmentation_maps, self.size)


class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.5) -> None:
        """Randomly Crops the image.

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = image.shape[1:]
        tH, tW = self.size

        if random.random() < self.p:
            margin_h = max(H - tH, 0)
            margin_w = max(W - tW, 0)
            y1 = random.randint(0, margin_h + 1)
            x1 = random.randint(0, margin_w + 1)
            y2 = y1 + tH
            x2 = x1 + tW
            image = image[:, y1:y2, x1:x2]
            segmentation_maps = segmentation_maps[:, y1:y2, x1:x2]
        return image, segmentation_maps


class RandomCutOut:
    def __init__(self, p: float = 0.5, alpha=1.0, seg_fill=255) -> None:
        """Randomly Crops the image.

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.p = p
        self.alpha = alpha
        self.seg_fill = seg_fill

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        image_h, image_w = image.shape[-2:]

        if random.random() < self.p:
            cx = np.random.uniform(0, image_w)
            cy = np.random.uniform(0, image_h)

            lam = np.random.beta(self.alpha, self.alpha)
            w = image_w * np.sqrt(1 - lam) * 0.9
            h = image_h * np.sqrt(1 - lam) * 0.9
            # w = image_w * np.random.uniform(0, 1)
            # h = image_h * np.random.uniform(0, 1)

            x0 = int(np.round(max(cx - w / 2, 0)))
            x1 = int(np.round(min(cx + w / 2, image_w)))
            y0 = int(np.round(max(cy - h / 2, 0)))
            y1 = int(np.round(min(cy + h / 2, image_h)))
            image[:, y0:y1, x0:x1] = 0
            segmentation_maps[:, y0:y1, x0:x1] = self.seg_fill
        return image, segmentation_maps


class Pad:
    def __init__(self, size: Union[List[int], Tuple[int], int], seg_fill: int = 255) -> None:
        """Pad the given image on all sides with the given "pad" value. Args: size: expected output image size (h,
        w) fill: Pixel fill value for constant fill. Default is 0. This value is only used when the padding mode is
        constant.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        padding = [0, 0, self.size[1] - image.shape[2], self.size[0] - image.shape[1]]
        return TF.pad(image, padding), TF.pad(segmentation_maps, padding, self.seg_fill)


class ResizePad:
    def __init__(self, size: Union[Tuple[int], List[int]], seg_fill: int = 255) -> None:
        """Resize the input image to the given size. Args: size: Desired output size. If size is a sequence,
        the output size will be matched to this. If size is an int, the smaller edge of the image will be matched to
        this number maintaining the aspect ratio.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = image.shape[1:]
        tH, tW = self.size

        # scale the image
        scale_factor = min(tH / H, tW / W) if W > H else max(tH / H, tW / W)
        # nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        nH, nW = round(H * scale_factor), round(W * scale_factor)
        image = TF.resize(image, [nH, nW], TF.InterpolationMode.BILINEAR)
        segmentation_maps = TF.resize(segmentation_maps, [nH, nW], TF.InterpolationMode.NEAREST)

        # pad the image
        padding = [0, 0, tW - nW, tH - nH]
        image = TF.pad(image, padding, fill=0)
        segmentation_maps = TF.pad(segmentation_maps, padding, fill=self.seg_fill)
        return image, segmentation_maps


class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        """Resize the input image to the given size. Args: size: Desired output size. If size is a sequence,
        the output size will be matched to this. If size is an int, the smaller edge of the image will be matched to
        this number maintaining the aspect ratio.
        """
        self.size = size

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = image.shape[1:]

        if isinstance(self.size, int):
            scale_factor = self.size[0] / min(H, W)

            nH, nW = round(H * scale_factor), round(W * scale_factor)
        else:
            nH, nW = self.size[: 2]
        # make the image divisible by stride
        alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        image = TF.resize(image, [alignH, alignW], TF.InterpolationMode.BILINEAR)
        segmentation_maps = TF.resize(segmentation_maps, [alignH, alignW], TF.InterpolationMode.NEAREST)
        return image, segmentation_maps


class RandomResizedCrop:
    def __init__(self, size, scale: Tuple[float, float] = (0.5, 2.0),
                 seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        """
        self.size = size
        self.scale = scale
        self.seg_fill = seg_fill

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = image.shape[1:]
        tH, tW = self.size

        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))
        scale = int(tH * ratio), int(tW * 4 * ratio)

        # scale the image
        scale_factor = min(max(scale) / max(H, W), min(scale) / min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        image = TF.resize(image, [nH, nW], TF.InterpolationMode.BILINEAR)
        segmentation_maps = TF.resize(segmentation_maps, [nH, nW], TF.InterpolationMode.NEAREST)

        # random crop
        margin_h = max(image.shape[1] - tH, 0)
        margin_w = max(image.shape[2] - tW, 0)
        y1 = random.randint(0, margin_h + 1)
        x1 = random.randint(0, margin_w + 1)
        y2 = y1 + tH
        x2 = x1 + tW
        image = image[:, y1:y2, x1:x2]
        segmentation_maps = segmentation_maps[:, y1:y2, x1:x2]

        # pad the image
        if image.shape[1:] != self.size:
            left = (tW - image.shape[2]) // 2
            right = tW - image.shape[2] - left
            top = (tH - image.shape[1]) // 2
            bottom = tH - image.shape[1] - top
            padding = [left, top, right, bottom]
            image = TF.pad(image, padding, fill=0)
            segmentation_maps = TF.pad(segmentation_maps, padding, fill=self.seg_fill)
        return image, segmentation_maps


class ColorJitterxxx:

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18):
        self.trans = PhotoMetricDistortionTif(brightness_delta, contrast_range, saturation_range, hue_delta)

    def __call__(self, image: Tensor, segmentation_maps: Tensor) -> Tuple[Tensor, Tensor]:
        image = image.permute(1, 2, 0).contiguous().numpy()
        # import cv2
        # cv2.imshow('before', cv2.cvtColor(image[:,:,-3:].astype(np.uint8), cv2.COLOR_RGB2BGR))
        # cv2.imshow('after',  cv2.cvtColor(self.trans.transform(image)[:,:,-3:].astype(np.uint8), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        image = torch.from_numpy(self.trans.transform(image)).permute(2, 0, 1).contiguous()
        return image, segmentation_maps


class ColorJitter:
    def __init__(self,
                 brightness=0.2,
                 contrast=0.5,
                 saturation=0.5,
                 hue=0.2,
                 p=0.5):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.p = p

    def __call__(self, image: Tensor, segmentation_maps: Tensor):
        # import cv2
        # cv2.imshow('before', cv2.cvtColor(image.permute(1, 2, 0).numpy()[:, :, -3:].astype(np.uint8), cv2.COLOR_RGB2BGR))

        image[-3:, :, :] = self._do_color_jitter(image[-3:, :, :])

        # cv2.imshow('after', cv2.cvtColor(
        #     image[-3:, :, :].permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        return image, segmentation_maps

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, (float, int)):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness: Optional[List[float]],
                   contrast: Optional[List[float]],
                   saturation: Optional[List[float]],
                   hue: Optional[List[float]]
                   ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """

        if np.random.rand() < 0.5:
            fn_idx = [0, 1, 2, 3]
        else:
            fn_idx = [0, 2, 3, 1]

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return torch.from_numpy(np.array(fn_idx)), b, c, s, h

    def _do_color_jitter(self, img):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if np.random.rand() < self.p:
                if fn_id == 0 and brightness_factor is not None:
                    img = TF.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = TF.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = TF.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = TF.adjust_hue(img, hue_factor)

        return img


def get_train_augmentation(size=(512, 512), seg_fill: int = 255,
                           pre_resize=False, resize_prob=1.0, resize_scale=(0.5, 2.0),
                           flip_v=0.5, flip_h=0.5, rotate_prob=0.5, rotate_angle=90,
                           color_jitter=0.8, sharpness=0.1, gaussian_blur=0.1,
                           mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                           cutout=0.5):
    print(mean, std)
    augs = list()
    if pre_resize:
        augs.append(Resize(size))
    if resize_prob > 0:
        augs.append(RandomApply(RandomResizedCrop(size, scale=resize_scale, seg_fill=seg_fill), p=resize_prob))
    else:
        augs.append(Resize(size))
    if flip_v > 0:
        augs.append(RandomVerticalFlip(p=flip_v))
    if flip_h > 0:
        augs.append(RandomHorizontalFlip(p=flip_h))
    if rotate_prob > 0:
        augs.append(RandomRotation90(degrees=rotate_angle, p=rotate_prob, seg_fill=seg_fill, expand=False))
    if color_jitter > 0:
        augs.append(ColorJitter())
    if sharpness > 0:
        augs.append(RandomAdjustSharpness(sharpness_factor=2, p=sharpness))
    if gaussian_blur > 0:
        augs.append(RandomGaussianBlur(3, p=gaussian_blur))
    if cutout > 0:
        augs.append(RandomCutOut(p=cutout, alpha=1.0, seg_fill=seg_fill))

    augs.append(Normalize(mean, std))

    return Compose(augs)


def get_train_augmentation2(size=(512, 512), seg_fill: int = 255, resize_scale=(0.5, 2.0),
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    print(mean, std)

    augs = [
        RandomResizedCrop(size, scale=resize_scale, seg_fill=seg_fill),
        OneOf([
            RandomVerticalFlip(p=1.0),
            RandomHorizontalFlip(p=1.0),
            RandomRotation90(0, p=1.0),
            RandomRotation90(90, p=1.0),
            RandomRotation90(180, p=1.0),
            RandomRotation90(270, p=1.0),
        ]),
        ColorJitter()
    ]
    augs.append(Normalize(mean, std))
    return Compose(augs)


def get_val_augmentation(size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return Compose([
        Resize(size),
        Normalize(mean, std)
    ])


def show_img_tensor(img_tensor, window_name, mean_rgb, std_rgb, block=False):
    _mean = torch.Tensor(mean_rgb).unsqueeze(dim=-1).unsqueeze(dim=-1)
    _std = torch.Tensor(std_rgb).unsqueeze(dim=-1).unsqueeze(dim=-1)
    img_tensor = (img_tensor * _std + _mean) * 255
    img_tensor = img_tensor.permute(1, 2, 0)
    _img_bgr = cv2.cvtColor(img_tensor.numpy(), cv2.COLOR_RGB2BGR).astype(np.uint8)
    cv2.imshow(window_name, _img_bgr)
    if block:
        return cv2.waitKey(0)
    else:
        return ord(' ')


if __name__ == '__main__':
    from torchvision import io
    import cv2
    import numpy as np

    aug = get_train_augmentation2()
    for _ in range(100):
        _img = io.read_image('/home/liuwang/liuwang_data/documents/projects/mmseg-agri/data/'
                             '2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021/test/images/'
                             'rgb/1WULXE3BJ_761-8609-1273-9121.jpg')
        cv2.imshow('origin', cv2.cvtColor(_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
        _mask = _img
        _img, _mask = aug(_img, _mask)

        lbl_src = _mask[0, :, :].numpy().astype(np.uint8)
        cv2.imshow("lbl", lbl_src)

        key = show_img_tensor(_img,
                              "img",
                              (0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225),
                              block=True)
        if key == ord('q'):
            break
        print(_img.shape, _mask.shape)
