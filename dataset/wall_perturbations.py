import random

from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


class WallPerturbation:
    def __init__(self, crop_size=224):
        self.crop_size = crop_size
        self.image_jitters = [
            transforms.ColorJitter(0.2, 0.1, 0.08, 0.01),
            transforms.ColorJitter(0.3, 0.2, 0.15, 0.02),
            transforms.Compose([
                transforms.ColorJitter(0.25, 0.25, 0.12, 0.015),
                transforms.RandomGrayscale(p=0.15),
            ]),
        ]

    def __call__(self, image, raw_mask, gt_mask):
        choice = random.randrange(3)

        if choice == 0:
            image, raw_mask, gt_mask = self._crop_flip(image, raw_mask, gt_mask)
            image = self.image_jitters[0](image)
        elif choice == 1:
            image, raw_mask, gt_mask = self._rotate_crop_flip(image, raw_mask, gt_mask)
            image = self.image_jitters[1](image)
        else:
            image, raw_mask, gt_mask = self._affine_crop_flip(image, raw_mask, gt_mask)
            image = self.image_jitters[2](image)

        return image, raw_mask, gt_mask

    def _crop_flip(self, image, raw_mask, gt_mask):
        image, raw_mask, gt_mask = self._random_crop(image, raw_mask, gt_mask)
        return self._maybe_hflip(image, raw_mask, gt_mask)

    def _rotate_crop_flip(self, image, raw_mask, gt_mask):
        angle = random.uniform(-7, 7)
        image = TF.rotate(image, angle, interpolation=Image.BILINEAR, fill=0)
        raw_mask = TF.rotate(raw_mask, angle, interpolation=Image.NEAREST, fill=0)
        gt_mask = TF.rotate(gt_mask, angle, interpolation=Image.NEAREST, fill=0)
        image, raw_mask, gt_mask = self._random_crop(image, raw_mask, gt_mask)
        return self._maybe_hflip(image, raw_mask, gt_mask)

    def _affine_crop_flip(self, image, raw_mask, gt_mask):
        angle = random.uniform(-4, 4)
        translate = [random.randint(-12, 12), random.randint(-12, 12)]
        scale = random.uniform(0.9, 1.15)
        shear = random.uniform(-4, 4)

        image = TF.affine(image, angle, translate, scale, shear, interpolation=Image.BILINEAR, fill=0)
        raw_mask = TF.affine(raw_mask, angle, translate, scale, shear, interpolation=Image.NEAREST, fill=0)
        gt_mask = TF.affine(gt_mask, angle, translate, scale, shear, interpolation=Image.NEAREST, fill=0)
        image, raw_mask, gt_mask = self._random_crop(image, raw_mask, gt_mask)
        return self._maybe_hflip(image, raw_mask, gt_mask)

    def _random_crop(self, image, raw_mask, gt_mask):
        image = TF.pad(image, self._padding(image), fill=0)
        raw_mask = TF.pad(raw_mask, self._padding(raw_mask), fill=0)
        gt_mask = TF.pad(gt_mask, self._padding(gt_mask), fill=0)

        i, j, h, w = transforms.RandomCrop.get_params(image, (self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        raw_mask = TF.crop(raw_mask, i, j, h, w)
        gt_mask = TF.crop(gt_mask, i, j, h, w)
        return image, raw_mask, gt_mask

    def _maybe_hflip(self, image, raw_mask, gt_mask):
        if random.random() < 0.5:
            image = TF.hflip(image)
            raw_mask = TF.hflip(raw_mask)
            gt_mask = TF.hflip(gt_mask)
        return image, raw_mask, gt_mask

    def _padding(self, image):
        width, height = image.size
        pad_w = max(self.crop_size - width, 0)
        pad_h = max(self.crop_size - height, 0)
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        return [left, top, right, bottom]
