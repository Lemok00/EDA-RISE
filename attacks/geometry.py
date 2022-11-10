# https://pytorch.org/vision/0.8/transforms.html#module-torchvision.transforms.functional
import torch
import torchvision
from torchvision.transforms.functional import rotate, resize, center_crop, five_crop, affine
from attacks import BaseAttack
import random


class RotationAttacker(BaseAttack):
    def __init__(self):
        super(RotationAttacker, self).__init__()
        self.name = 'Rotation'

    def get_intensity_name(self, intensity=None):
        return f'deg. = {intensity}'

    def attack_image(self, image, intensity):
        return rotate(image, intensity * random.choice([-1, 1]))


class EdgeCroppingAttacker(BaseAttack):
    def __init__(self):
        super(EdgeCroppingAttacker, self).__init__()
        self.name = 'Edge Cropping'

    def get_intensity_name(self, intensity=None):
        return f'r = {intensity}'

    def attack_image(self, image, intensity):
        ori_h, ori_w = image.shape[-2], image.shape[-1]
        tar_h, tar_w = int((1 - intensity) * ori_h), int((1 - intensity) * ori_w)
        result_image = torch.zeros_like(image)
        result_image[:, :, ((ori_h - tar_h) // 2):((ori_h + tar_h) // 2), (ori_w - tar_w) // 2: (ori_w + tar_w) // 2] \
            = image[:, :, ((ori_h - tar_h) // 2):((ori_h + tar_h) // 2), (ori_w - tar_w) // 2: (ori_w + tar_w) // 2]
        return result_image


class CenterCroppingAttacker(BaseAttack):
    def __init__(self):
        super(CenterCroppingAttacker, self).__init__()
        self.name = 'Center Cropping'

    def get_intensity_name(self, intensity=None):
        return f'r = {intensity}'

    def attack_image(self, image, intensity):
        ori_h, ori_w = image.shape[-2], image.shape[-1]
        tar_h, tar_w = int(intensity * ori_h), int(intensity * ori_w)
        cropped_image = torch.zeros(size=(image.shape[0], image.shape[1], tar_h, tar_w))
        result_image = image.clone()
        result_image[:, :, ((ori_h - tar_h) // 2):((ori_h + tar_h) // 2), (ori_w - tar_w) // 2: (ori_w + tar_w) // 2] \
            = cropped_image
        return result_image


class ResizingAttacker(BaseAttack):
    def __init__(self):
        super(ResizingAttacker, self).__init__()
        self.name = 'Resizing'

    def get_intensity_name(self, intensity=None):
        return f'r = {intensity}'

    def attack_image(self, image, intensity):
        ori_h, ori_w = image.shape[-2], image.shape[-1]
        tar_h, tar_w = int(ori_h * (1 - intensity)), int(ori_w * (1 - intensity))
        resized_image = resize(image, size=(tar_h, tar_w))
        resized_image = resize(resized_image, size=(ori_h, ori_w))
        return resized_image


class ShearingAttacker(BaseAttack):
    def __init__(self):
        super(ShearingAttacker, self).__init__()
        self.name = 'Shearing'

    def get_intensity_name(self, intensity=None):
        return f'r = {intensity}'

    def attack_image(self, image, intensity):
        intensity = (intensity * random.choice([-1, 1]), intensity * random.choice([-1, 1]))
        return affine(image, angle=0, translate=(0, 0), scale=1, shear=intensity)
