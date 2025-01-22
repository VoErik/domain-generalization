from typing import Tuple
import torchvision
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Compose
import numpy as np
import torch



class Transforms:
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, img, *args, **kwargs):
        return self.transform(image=np.array(img))

class TransformsWrapper:
    def __init__(self, albumentations_transform=None, torchvision_transform=None):
        self.albumentations_transform = albumentations_transform
        self.torchvision_transform = torchvision_transform

    def __call__(self, img):
        # If Torchvision transform is provided, apply it
        if self.torchvision_transform:
            # Ensure the input is a PIL image (required by Torchvision)
            if isinstance(img, np.ndarray):  # Convert NumPy to PIL
                img = Image.fromarray(img)
            img = self.torchvision_transform(img)  # Apply Torchvision transforms

        # If Albumentations transform is provided, apply it
        if self.albumentations_transform:
            # Ensure the input is a NumPy array (required by Albumentations)
            if isinstance(img, torch.Tensor):  # Convert PyTorch tensor to NumPy
                img = img.permute(1, 2, 0).numpy()
            elif isinstance(img, Image.Image):  # Convert PIL to NumPy
                img = np.array(img)
            augmented = self.albumentations_transform(image=img)
            img = augmented['image']  # Extract the transformed image
            # Convert back to PyTorch tensor if necessary

        return img


def imagenet_transform(
        input_size: Tuple[int, int]
) -> TransformsWrapper:
    transform = A.Compose([
        A.Resize(input_size[0], input_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    return TransformsWrapper(albumentations_transform=transform)


def create_augmentation_pipeline(augmentations: dict) -> TransformsWrapper:
    pipeline = []
    for key, aug in augmentations.items():
        pipeline.append(A.OneOrOther(A.NoOp(), aug))
    pipeline.append(A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    pipeline.append(ToTensorV2())

    augment = TransformsWrapper(albumentations_transform=A.Compose(pipeline))

    return augment


def combine_augmentations(augment: list) -> Compose:
    full_pipeline = []
    for aug_dict in augment:
        pipeline = create_augmentation_pipeline(aug_dict)
        full_pipeline.extend(pipeline.transform)

    return Compose(full_pipeline)


all_augmentations = {
    "blur": A.Blur((3, 7), 1),
    "channel_dropout": A.ChannelDropout((1, 2), 128, p=1),
    "clahe": A.CLAHE((1, 4), (8, 8), False, 0.5),
    "color_jitter": A.ColorJitter(0.2, 0.2, 0.2, 0.1, 1),
    "defocus": A.Defocus((4, 8), (0.2, 0.4), True),
    "fancyPCA": A.FancyPCA(0.1, 1, True),
    "glass_blur": A.GlassBlur(0.7, 4, 3, "fast", False, 1),
    "gaussian_noise": A.GaussNoise(std_range=(0.1, 0.2), p=1),
    "grid_distortion": A.GridDistortion(5, (-0.3, 0.3), p=0.5),
    "grid_dropout": A.GridDropout(ratio=0.3, unit_size_range=(10, 20), random_offset=True, p=1.0),
    "grid_elastic_deform": A.GridElasticDeform(num_grid_xy=(4, 4), magnitude=10, p=1),
    "horizontal_flip": A.HorizontalFlip(p=0.5),
    "hue_saturation_value": A.HueSaturationValue((-20, 20), (-30, 30), (-20, 20), p=0.5),
    "iso_noise": A.ISONoise((0.01, 0.05), (0.1, 0.5), p=0.2),
    "median_blur": A.MedianBlur(7, 1),
    "noop": A.NoOp(),
    "pixel_dropout": A.PixelDropout(0.01, False, 0, None, True),
    "random_brightness_contrast": A.RandomBrightnessContrast((-0.2, 0.2), (-0.2, 0.2), True, False, p=0.5),
    "random_gamma": A.RandomGamma((80, 120), p=0.5),
    "random_resized_crop": A.RandomResizedCrop((512, 320), scale=(0.08, 1), ratio=(0.75, 1.3333333333333333), p=1),
    "random_tone_curve": A.RandomToneCurve(0.1, False, p=0.1),
    "rotate": A.Rotate((-90, 90), interpolation=1, border_mode=4, rotate_method='largest_box',
                       crop_border=False, mask_interpolation=0, fill=False, fill_mask=0, p=0.5),
    "sharpen": A.Sharpen((0.2, 0.5), (0.5, 1), 'kernel', 5, 1, 0.5),
    "shift_scale_rotate": A.ShiftScaleRotate((-0.0625, 0.0625), (-0.1, 0.1), (-90, 90), 1, 4, shift_limit_x=None,
                                             shift_limit_y=None, mask_interpolation=0, fill=0, fill_mask=0, p=0.5),
    "solarize": A.Solarize(threshold=(0.5, 0.5), p=0.5),
    "spatter": A.Spatter((0.65, 0.65), (0.3, 0.3), (2, 2), (0.68, 0.68), (0.6, 0.6), 'rain', color=None, p=0.5),
    "transpose": A.Transpose(p=0.5),
    "xy_masking": A.XYMasking((1, 3), (1, 3), (10, 100), (10, 100), fill=0, fill_mask=0, p=0.5)
}

pacs_aug = {
    "channel_dropout": A.ChannelDropout((1, 2), 128, p=1),
    "clahe": A.CLAHE((1, 4), (8, 8), False, 1),
    "color_jitter": A.ColorJitter(0.2, 0.2, 0.2, 0.1, 1),
    "grid_elastic_deform": A.GridElasticDeform(num_grid_xy=(4, 4), magnitude=10, p=1),
    "hue_saturation_value": A.HueSaturationValue((-20, 20), (-30, 30), (-20, 20), p=1),
    "noop": A.NoOp(),
    "solarize": A.Solarize(threshold_range=(0.5, 0.5), p=1),
    "xy_masking": A.XYMasking((1, 3), (1, 3), (10, 20), (10, 20), fill=0, fill_mask=0, p=1)
}

camelyon17_aug = {
    "fancyPCA": A.FancyPCA(0.1, 1, True),
    "glass_blur": A.GlassBlur(0.7, 4, 3, "fast", False, 1),
    "iso_noise": A.ISONoise((0.01, 0.05), (0.1, 0.5), p=0.2),
    "median_blur": A.MedianBlur(7, 1),
    "noop": A.NoOp(),
    "random_brightness_contrast": A.RandomBrightnessContrast((-0.2, 0.2), (-0.2, 0.2), True, False, p=0.5),
    "random_gamma": A.RandomGamma((80, 120), p=0.5),
    "random_tone_curve": A.RandomToneCurve(0.1, False, p=0.1),
    "sharpen": A.Sharpen((0.2, 0.5), (0.5, 1), 'kernel', 5, 1, 0.5),
    "shift_scale_rotate": A.ShiftScaleRotate((-0.0625, 0.0625), (-0.1, 0.1), (-90, 90), 1, 4, shift_limit_x=None,
                                             shift_limit_y=None, mask_interpolation=0, fill=0, fill_mask=0, p=0.5),
    "spatter": A.Spatter((0.65, 0.65), (0.3, 0.3), (2, 2), (0.68, 0.68), (0.6, 0.6), 'rain', color=None, p=0.5)
}

shared_aug = {
    "blur": A.Blur((3, 7), 1),
    "defocus": A.Defocus((4, 8), (0.2, 0.4), True),
    "gaussian_noise": A.GaussNoise(std_range=(0.1, 0.2), p=1),
    "grid_distortion": A.GridDistortion(5, (-0.3, 0.3), p=0.5),
    "grid_dropout": A.GridDropout(ratio=0.3, unit_size_range=(10, 20), random_offset=True, p=1.0),
    "horizontal_flip": A.HorizontalFlip(p=0.5),
    "noop": A.NoOp(),
    "pixel_dropout": A.PixelDropout(0.01, False, 0, None, 0.5),
    "random_resized_crop": A.RandomResizedCrop((512, 320), scale=(0.08, 1), ratio=(0.75, 1.3333333333333333), p=1),
    "rotate": A.Rotate((-90, 90), interpolation=1, border_mode=4, rotate_method='largest_box',
                       crop_border=False, mask_interpolation=0, fill=0, fill_mask=0, p=0.5),
    "transpose": A.Transpose(p=0.5)
}

"""Custom Augmentations for the PACS dataset"""

color_geometric_aug = {
    "color_jitter": A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1),
    "rotate": A.Rotate((-45, 45), interpolation=1, border_mode=4, p=0.5),
    "horizontal_flip": A.HorizontalFlip(p=0.5)
}

color_distortion_aug = {
    "hue_saturation_value": A.HueSaturationValue((-20, 20), (-30, 30), (-20, 20), p=1),
    "grid_distortion": A.GridDistortion(5, (-0.3, 0.3), p=0.5),
    "transpose": A.Transpose(p=0.5)
}

contrast_geometric_aug = {
    "clahe": A.CLAHE((1, 4), (8, 8), always_apply=False, p=1),
    "random_resized_crop": A.RandomResizedCrop((512, 320), scale=(0.08, 1), ratio=(0.75, 1.33), p=1),
    "rotate": A.Rotate((-90, 90), interpolation=1, border_mode=4, p=0.5)
}

noise_geometric_aug = {
    "gauss_noise": A.GaussNoise(std_range=(0.1, 0.2), p=1),
    "rotate": A.Rotate((-45, 45), interpolation=1, border_mode=4, p=0.5),
    "horizontal_flip": A.HorizontalFlip(p=0.5)
}

noise_color_geometric_aug = {
    "defocus": A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4), p=1),
    "color_jitter": A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1),
    "horizontal_flip": A.HorizontalFlip(p=0.5)
}

masking_color_aug = {
    "solarize": A.Solarize(threshold=0.5, p=1),
    "grid_dropout": A.GridDropout(ratio=0.3, unit_size_min=10, unit_size_max=20, random_offset=True, p=1.0),
    "hue_saturation_value": A.HueSaturationValue((-20, 20), (-30, 30), (-20, 20), p=1)
}

masking_noise_aug = {
    "xy_masking": A.XYMasking((1, 3), (1, 3), (10, 20), (10, 20), fill=0, fill_mask=0, p=1),
    "gauss_noise": A.GaussNoise(std_range=(0.1, 0.3), p=1),
    "rotate": A.Rotate((-30, 30), p=0.5)
}

distortion_noise_aug = {
    "grid_distortion": A.GridDistortion(num_steps=5, distort_limit=(-0.4, 0.4), p=1),
    "defocus": A.Defocus(radius=(3, 6), alias_blur=(0.1, 0.3), p=1),
    "transpose": A.Transpose(p=0.5)
}

color_mask_geometric_aug = {
    "solarize": A.Solarize(threshold=0.5, p=1),
    "xy_masking": A.XYMasking((1, 2), (1, 3), (15, 25), (15, 25), fill=0, fill_mask=0, p=1),
    "rotate": A.Rotate((-90, 90), p=0.5)
}

PACS_CUSTOM = {
    "color_geometric": color_geometric_aug,
    "color_distortion": color_distortion_aug,
    "contrast_geometric": contrast_geometric_aug,
    "noise_geometric": noise_geometric_aug,
    "noise_color_geometric": noise_color_geometric_aug,
    "masking_color": masking_color_aug,
    "masking_noise": masking_noise_aug,
    "distortion_noise": distortion_noise_aug,
    "color_mask_geometric": color_mask_geometric_aug,
}