import albumentations as A
from albumentations.pytorch import ToTensorV2



# custom
# Todo: include link to paper:
carlucci_1 = A.Compose([
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=1),
    ToTensorV2()
])

carlucci_2 = A.Compose([
    A.ToGray(p=1),
    ToTensorV2()
])

carlucci_3 = A.Compose([
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=1), A.ToGray(p=1),
    ToTensorV2()
])

# Todo: include link to paper:
wang_1 = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
    ToTensorV2()
])

wang_2 = A.Compose([
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=1),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
    ToTensorV2()
])

# Todo: include link to paper:
zhou = A.Compose([
    A.Resize(height=int(256 * 1.25), width=int(256 * 1.25)),
    A.RandomCrop(height=256, width=256),
    A.HorizontalFlip(p=1),
    ToTensorV2()
])

color_geometric = A.Compose([
        A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1),
        A.Rotate((-45, 45), interpolation=1, border_mode=4, p=1),
        A.HorizontalFlip(p=1),
        ToTensorV2()
])

color_distortion = A.Compose([
    A.HueSaturationValue((-20, 20), (-30, 30), (-20, 20), p=1),
    A.GridDistortion(5, (-0.3, 0.3), p=1),
    A.Transpose(p=1),
    ToTensorV2()
])

contrast_geometric = A.Compose([
    A.CLAHE((1, 4), (8, 8), p=1),
    A.RandomResizedCrop((512, 320), scale=(0.08, 1), ratio=(0.75, 1.33), p=1),
    A.Rotate((-90, 90), interpolation=1, border_mode=4, p=1),
    ToTensorV2()
])

noise_geometric = A.Compose([
    A.GaussNoise(std_range=(0.1, 0.2), p=1),
    A.Rotate((-45, 45), interpolation=1, border_mode=4, p=1),
    A.HorizontalFlip(p=1),
    ToTensorV2()
])

noise_color_geometric = A.Compose([
    A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4), p=1),
    A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1),
    A.HorizontalFlip(p=1),
    ToTensorV2()
])

masking_color = A.Compose([
    A.Solarize(threshold=0.5, p=1),
    A.GridDropout(ratio=0.3, unit_size_min=10, unit_size_max=20, random_offset=True, p=1.0),
    A.HueSaturationValue((-20, 20), (-30, 30), (-20, 20), p=1),
    ToTensorV2()
])

masking_noise = A.Compose([
    A.XYMasking((1, 3), (1, 3),
                (10, 20), (10, 20),
                fill=0, fill_mask=0, p=1),
    A.GaussNoise(std_range=(0.1, 0.3), p=1),
    A.Rotate((-30, 30), p=1),
    ToTensorV2()
])

distortion_noise = A.Compose([
    A.GridDistortion(num_steps=5, distort_limit=(-0.4, 0.4), p=1),
    A.Defocus(radius=(3, 6), alias_blur=(0.1, 0.3), p=1),
    A.Transpose(p=0.5),ToTensorV2()
])

color_mask_geometric = A.Compose([
    A.Solarize(threshold=0.5, p=1),
    A.XYMasking((1, 2), (1, 3),
                (15, 25), (15, 25),
                fill=0, fill_mask=0, p=1),
    A.Rotate((-90, 90), p=1),
    ToTensorV2()
])

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


all_custom_aug = {
    "Carlucci et al. (2019): 1": carlucci_1,
    "Carlucci et al. (2019): 2": carlucci_2,
    "Carlucci et al. (2019): 3": carlucci_3,
    "Wang et al. (2020): 1": wang_1,
    "Wang et al. (2020): 2": wang_2,
    "Zhou et al. (2020)": zhou,
    "ColorJitter + Rotate + H.Flip": color_geometric,
    "HueSaturation + GridDistortion + Transpose": color_distortion,
    "CLAHE + RandomResizedCrop + Transpose": contrast_geometric,
    "GaussNoise + Rotate + H.Flip": noise_geometric,
    "Defocus + ColorJitter + H.Flip": noise_color_geometric,
    "Solarize + GridDropout + HueSaturation": masking_color,
    "XY Mask + GaussNoise + Rotate": masking_noise,
    "GridDistortion + Defocus + Transpose": distortion_noise,
    "Solarize + XY Mask + Rotate": color_mask_geometric
}

custom_aug = {
    "ColorJitter + Rotate + H.Flip": color_geometric,
    "HueSaturation + GridDistortion + Transpose": color_distortion,
    "CLAHE + RandomResizedCrop + Transpose": contrast_geometric,
    "GaussNoise + Rotate + H.Flip": noise_geometric,
    "Defocus + ColorJitter + H.Flip": noise_color_geometric,
    "Solarize + GridDropout + HueSaturation": masking_color,
    "XY Mask + GaussNoise + Rotate": masking_noise,
    "GridDistortion + Defocus + Transpose": distortion_noise,
    "Solarize + XY Mask + Rotate": color_mask_geometric
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


AUGSETS = {
    "all_custom_aug": all_custom_aug,
    "custom": custom_aug,
    "carlucci1": carlucci_1,
    "carlucci2": carlucci_2,
    "carlucci3": carlucci_3,
    "wang_1": wang_1,
    "wang_2": wang_2,
    "zhou": zhou,
    "color_mask_geometric": color_mask_geometric,
    "noise_geometric": noise_geometric,
    "noise_color_geometric": noise_color_geometric,
    "masking_color": masking_color,
    "masking_noise": masking_noise,
    "distortion_noise": distortion_noise,
    "contrast_geometric": contrast_geometric,
    "color_geometric": color_geometric,
    "color_distortion": color_distortion,
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