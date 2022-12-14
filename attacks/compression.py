import cv2 as cv
import numpy as np
import torch
import os

from attacks import BaseAttack

class BaseCompressionAttacker(BaseAttack):
    def __init__(self):
        super(BaseCompressionAttacker, self).__init__()
        self.name = 'Compression'
        self.temp_path = None
        self.quality_controller = None

    def get_intensity_name(self, intensity=None):
        return f'qua. = {intensity}'

    def attack(self, image, intensity):
        device = image.device
        image = self.normalize(image)
        for i in range(image.shape[0]):
            compressed_image = image[i].cpu().clone() * 255
            compressed_image = np.asarray(compressed_image).transpose((1, 2, 0))
            cv.imwrite(self.temp_path, cv.cvtColor(compressed_image, cv.COLOR_RGB2BGR),
                       [int(self.quality_controller), intensity])
            compressed_image = cv.cvtColor(cv.imread(self.temp_path), cv.COLOR_BGR2RGB).transpose((2, 0, 1))
            image[i] = torch.from_numpy(compressed_image).float().to(device) / 255
        image = self.denormalize(image)
        return image

class JPEGCompressionAttacker(BaseCompressionAttacker):
    def __init__(self):
        super(JPEGCompressionAttacker, self).__init__()
        self.name = 'JPEG Compression'
        self.temp_path = 'tmp.jpg'
        self.quality_controller = cv.IMWRITE_JPEG_QUALITY


class WebPCompressionAttacker(BaseCompressionAttacker):
    def __init__(self):
        super(WebPCompressionAttacker, self).__init__()
        self.name = 'WebP Compression'
        self.temp_path = 'tmp.webp'
        self.quality_controller = cv.IMWRITE_WEBP_QUALITY
