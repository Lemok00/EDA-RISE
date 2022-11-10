import kornia
from attacks import BaseAttack


class GaussianBlurAttacker(BaseAttack):
    def __init__(self):
        super(GaussianBlurAttacker, self).__init__()
        self.name = 'Gaussian Blur'

    def get_intensity_name(self, intensity=None):
        return f'K.S. = {intensity}'

    def attack_image(self, image, intensity):
        return kornia.filters.gaussian_blur2d(image, kernel_size=(intensity, intensity), sigma=(1, 1))


class AveragingBlurAttacker(BaseAttack):
    def __init__(self):
        super(AveragingBlurAttacker, self).__init__()
        self.name = 'Averaging Blur'

    def get_intensity_name(self, intensity=None):
        return f'K.S. = {intensity}'

    def attack_image(self, image, intensity):
        return kornia.filters.box_blur(image, kernel_size=(intensity, intensity))
