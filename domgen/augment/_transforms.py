from typing import Tuple
import torchvision
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Compose
import numpy as np


class Transforms:
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, img, *args, **kwargs):
        return self.transform(image=np.array(img))


def imagenet_transform(
        input_size: Tuple[int, int]
) -> torchvision.transforms.Compose:
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    return transform


def create_augmentation_pipeline(augmentations: dict) -> Transforms:
    pipeline = []
    for key, aug in augmentations.items():
        pipeline.append(A.OneOrOther(A.NoOp(), aug))
    pipeline.append(A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    pipeline.append(ToTensorV2())

    augment = Transforms(A.Compose(pipeline))

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

"""
Carlucci, F. M., D'Innocente, A., Bucci, S., Caputo, B., & Tommasi, T. (2019). 
Domain generalization by solving jigsaw puzzles. 
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 2229-2238).
"""

carlucci_1 = A.Compose([
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5)
    # ToTensorV2()
])

carlucci_2 = A.Compose([
    A.ToGray(p=0.1)
    # ToTensorV2()
])

carlucci_3 = A.Compose([
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ToGray(p=0.1)
    # ToTensorV2()
])

"""
Wang, S., Yu, L., Li, C., Fu, C. W., & Heng, P. A. (2020). 
Learning from extrinsic and intrinsic supervisions for domain generalization. 
In European Conference on Computer Vision (pp. 159-176). Cham: Springer International Publishing.
"""

wang_1 = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
    # ToTensorV2()
])

wang_2 = A.Compose([
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
    # ToTensorV2()
])

"""
Zhou, K., Yang, Y., Hospedales, T., & Xiang, T. (2020). 
Deep domain-adversarial image generation for domain generalisation. 
In Proceedings of the AAAI conference on artificial intelligence (Vol. 34, No. 07, pp. 13025-13032).
"""

zhou = A.Compose([
    A.Resize(height=int(256 * 1.25), width=int(256 * 1.25)),
    A.RandomCrop(height=256, width=256),
    A.HorizontalFlip(p=0.5)
    # ToTensorV2()
])

"""
Specifically for Camelyon17:

Di Salvo, F., Doerrich, S., & Ledig, C. (2024). 
MedMNIST-C: Comprehensive benchmark and improved classifier robustness by simulating realistic image corruptions. 
arXiv preprint arXiv:2406.17536.
"""

diSalvo_blur = A.Compose([
    A.Defocus(radius=(3, 8), alias_blur=(0.2, 0.5), p=0.8),
    A.MotionBlur(blur_limit=(3, 10), p=0.8)
])

# increase brightness, increase contrast, increase saturate
diSalvo_color_1 = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(0.2, 0.5), contrast_limit=(0.2, 0.5), p=1.0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(20, 40), val_shift_limit=0, p=1.0),
])

# increase brightness, decrease contrast, decrease saturate
diSalvo_color_2 = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(0.2, 0.5), contrast_limit=(-0.5, -0.2), p=1.0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-40, -20), val_shift_limit=0, p=1.0),
])

# decrease brightness, increase contrast, increase saturate
diSalvo_color_3 = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.2), contrast_limit=(0.2, 0.5), p=1.0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(20, 40), val_shift_limit=0, p=1.0),
])

# decrease brightness, decrease contrast, decrease saturate
diSalvo_color_4 = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.2), contrast_limit=(-0.5, -0.2), p=1.0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-40, -20), val_shift_limit=0, p=1.0),
])

# increase brightness, increase contrast, decrease saturate
diSalvo_color_5 = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(0.2, 0.5), contrast_limit=(0.2, 0.5), p=1.0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-40, -20), val_shift_limit=0, p=1.0),
])

# increase brightness, decrease contrast, increase saturate
diSalvo_color_6 = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(0.2, 0.5), contrast_limit=(-0.5, -0.2), p=1.0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(20, 40), val_shift_limit=0, p=1.0),
])

# decrease brightness, increase contrast, decrease saturate
diSalvo_color_7 = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.2), contrast_limit=(0.2, 0.5), p=1.0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-40, -20), val_shift_limit=0, p=1.0),
])

# decrease brightness, decrease contrast, increase saturate
diSalvo_color_8 = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.2), contrast_limit=(-0.5, -0.2), p=1.0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(20, 40), val_shift_limit=0, p=1.0),
])

"""
Custom augmentations
"""

color_geometric = A.Compose([
    A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1),
    A.Rotate((-45, 45), interpolation=1, border_mode=4, p=0.5),
    A.HorizontalFlip(p=0.5)
])

color_distortion = A.Compose([
    A.HueSaturationValue((-20, 20), (-30, 30), (-20, 20), p=1),
    A.GridDistortion(5, (-0.3, 0.3), p=0.5),
    A.Transpose(p=0.5)
])

contrast_geometric = A.Compose([
    A.CLAHE((1, 4), (8, 8), always_apply=False, p=1),
    A.RandomResizedCrop((512, 320), scale=(0.08, 1), ratio=(0.75, 1.33), p=1),
    A.Rotate((-90, 90), interpolation=1, border_mode=4, p=0.5)
])

noise_geometric = A.Compose([
    A.GaussNoise(std_range=(0.1, 0.2), p=1),
    A.Rotate((-45, 45), interpolation=1, border_mode=4, p=0.5),
    A.HorizontalFlip(p=0.5)
])

noise_color_geometric = A.Compose([
    A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4), p=1),
    A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1),
    A.HorizontalFlip(p=0.5)
])

masking_color = A.Compose([
    A.Solarize(threshold=0.5, p=1),
    A.GridDropout(ratio=0.3, unit_size_min=10, unit_size_max=20, random_offset=True, p=1.0),
    A.HueSaturationValue((-20, 20), (-30, 30), (-20, 20), p=1)
])

masking_noise = A.Compose([
    A.XYMasking((1, 3), (1, 3), (10, 20), (10, 20), fill=0, fill_mask=0, p=1),
    A.GaussNoise(std_range=(0.1, 0.3), p=1),
    A.Rotate((-30, 30), p=0.5)
])

distortion_noise = A.Compose([
    A.GridDistortion(num_steps=5, distort_limit=(-0.4, 0.4), p=1),
    A.Defocus(radius=(3, 6), alias_blur=(0.1, 0.3), p=1),
    A.Transpose(p=0.5)
])

# particular for sketches
# should emphasize edges, mask & rotate for variability
sketches = A.Compose([
    A.Solarize(threshold=0.5, p=1),
    A.XYMasking((1, 2), (1, 3), (15, 25), (15, 25), fill=0, fill_mask=0, p=1),
    A.Rotate((-90, 90), p=0.5)
])
