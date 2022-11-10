# https://pytorch.org/vision/0.8/transforms.html#module-torchvision.transforms.functional
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from attacks import BaseAttack
import random


class BrightnessAdjustAttacker(BaseAttack):
    def __init__(self):
        super(BrightnessAdjustAttacker, self).__init__()
        self.name = 'Brightness Adjust'

    def get_intensity_name(self, intensity=None):
        return f'r = {intensity}'

    def attack_image(self, image, intensity):
        return adjust_brightness(image, 1 + intensity * random.choice([-1, 1]))


class ContrastAdjustAttacker(BaseAttack):
    def __init__(self):
        super(ContrastAdjustAttacker, self).__init__()
        self.name = 'Contrast Adjust'

    def get_intensity_name(self, intensity=None):
        return f'r = {intensity}'

    def attack_image(self, image, intensity):
        return adjust_contrast(image, 1 + intensity * random.choice([-1, 1]))
