import os
from os import path

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from dataset.wall_perturbations import WallPerturbation


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg")


seg_normalization = transforms.Normalize(
    mean=[0.5],
    std=[0.5],
)


class WallRefinementDataset(Dataset):
    def __init__(self, root, image_dir="images", raw_mask_dir="raw_masks", gt_mask_dir="gt_masks", perturb=True):
        self.root = root
        self.image_dir = self._resolve_dir(image_dir)
        self.raw_mask_dir = self._resolve_required_dir(raw_mask_dir, "RAW MASK")
        self.gt_mask_dir = self._resolve_required_dir(gt_mask_dir, "GROUND TRUTH MASK")
        self.perturb = perturb

        self.samples = self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError("No wall-refinement samples found in %s." % root)

        if perturb:
            self.perturbation = WallPerturbation(crop_size=224)
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.geom_transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
            ])
            self.mask_geom_transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.NEAREST),
                transforms.CenterCrop(224),
            ])
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        self.gt_transform = transforms.ToTensor()
        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            seg_normalization,
        ])

    def _resolve_dir(self, directory):
        if path.isabs(directory):
            return directory
        return path.join(self.root, directory)

    def _resolve_required_dir(self, directory, label):
        resolved = self._resolve_dir(directory)
        if not path.isdir(resolved):
            raise FileNotFoundError(
                "%s FOLDER NOT FOUND: %s. CascadePSP fine-tuning requires real raw masks; "
                "it will not synthesize masks from GT." % (label, resolved)
            )
        return resolved

    def _build_samples(self):
        if not path.isdir(self.image_dir):
            raise FileNotFoundError("IMAGE FOLDER NOT FOUND: %s." % self.image_dir)

        raw_masks = self._index_files(self.raw_mask_dir, MASK_EXTENSIONS)
        gt_masks = self._index_files(self.gt_mask_dir, MASK_EXTENSIONS)
        samples = []

        for image_name in sorted(os.listdir(self.image_dir)):
            if not image_name.lower().endswith(IMAGE_EXTENSIONS):
                continue

            stem = path.splitext(image_name)[0]
            if stem not in raw_masks:
                raise FileNotFoundError("RAW MASK MISSING for image '%s' in %s." % (image_name, self.raw_mask_dir))
            if stem not in gt_masks:
                raise FileNotFoundError("GT MASK MISSING for image '%s' in %s." % (image_name, self.gt_mask_dir))

            samples.append((path.join(self.image_dir, image_name), raw_masks[stem], gt_masks[stem]))

        return samples

    def _index_files(self, directory, extensions):
        files = {}
        for file_name in os.listdir(directory):
            if file_name.lower().endswith(extensions):
                files[path.splitext(file_name)[0]] = path.join(directory, file_name)
        return files

    def __getitem__(self, idx):
        im_path, raw_mask_path, gt_mask_path = self.samples[idx]
        im = Image.open(im_path).convert("RGB")
        raw_mask = Image.open(raw_mask_path).convert("L")
        gt = Image.open(gt_mask_path).convert("L")

        if self.perturb:
            im, raw_mask, gt = self.perturbation(im, raw_mask, gt)
        else:
            im = self.geom_transform(im)
            raw_mask = self.mask_geom_transform(raw_mask)
            gt = self.mask_geom_transform(gt)

        im = self.im_transform(im)
        raw_mask = self.seg_transform(raw_mask)
        gt = self.gt_transform(gt)

        return im, raw_mask, gt

    def __len__(self):
        return len(self.samples)
