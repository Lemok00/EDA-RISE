import torch
from attacks import BaseAttack


class GaussianNoiseAttacker(BaseAttack):
    def __init__(self):
        super(GaussianNoiseAttacker, self).__init__()
        self.name = 'Gaussian Noise'

    def get_intensity_name(self, intensity=None):
        return f'sig. = {intensity}'

    def attack_image(self, image, intensity):
        gaussian_noise = torch.randn_like(image) * intensity
        image += gaussian_noise

        return image


class SaltAndPepperNoiseAttacker(BaseAttack):
    def __init__(self):
        super(SaltAndPepperNoiseAttacker, self).__init__()
        self.name = 'Salt & Pepper Noise'

    def get_intensity_name(self, intensity=None):
        return f'p = {intensity}'

    def attack_image(self, image, intensity):
        pepper_and_salt_noise = torch.rand_like(image)
        salt = 255 if image.max() > 1 else 1
        pepper = -1 if image.min() < 1 else 0
        image[pepper_and_salt_noise < intensity / 2] = salt
        image[pepper_and_salt_noise > 1 - intensity / 2] = pepper

        return image