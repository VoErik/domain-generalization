from typing import List, Dict
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path


def denormalize(
        tensor: torch.Tensor,
        mean: List[float],
        std: List[float],
) -> torch.Tensor:
    r"""
    Denormalize a tensor using a mean and standard deviation.
    The denormalization formula for each channel `c` is:

    .. math::
        \text{denormalized\_value}[c] = (\text{normalized\_value}[c] \times \text{std}[c]) + \text{mean}[c]

    :param tensor: The normalized tensor. Shape should be (C, H, W).
    :param mean: List of means for each channel.
    :param std: List of standard deviations for each channel.
    :return: The denormalized tensor.
    """
    # so the mean + std are broadcastable
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    denormalized_tensor = tensor * std + mean
    return denormalized_tensor


def plot_single_augmented(
        path: str = None,
        augment: A.Compose = None
) -> None:
    """
    Plot a single augmented image.
    :param path: Path to image file.
    :param augment: The compose object. Must return a torch.tensor.
    :return: None.
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed = augment(image=image)
    transformed_image = transformed['image']

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    original_img = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    ax[0].imshow(original_img.permute(1, 2, 0).numpy().astype('uint8'))
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # ax[1].imshow(transformed_image.astype('uint8'))
    ax[1].imshow(transformed_image.permute(1, 2, 0).numpy().astype('uint8'))
    ax[1].set_title("Transformed Image")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()



def plot_augmented_grid(
        path: str,
        augmentations: dict,
        grid_cols: int = 4
) -> None:
    """
    Plot a grid of augmented images.
    :param path: Path to image file.
    :param augmentations: Dictionary of compose objects, with {name: augmentation}. The augmentations must return a
    torch tensor.
    :param grid_cols: Number of columns. Defaults to 4.
    :return: None
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_img = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    augment_names = ['Original'] + list(augmentations.keys())
    augment_transforms = [None] + list(augmentations.values())

    grid_rows = (len(augment_transforms) + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 4 * grid_rows))
    axes = axes.flatten()

    for idx, (ax, name, augment) in enumerate(zip(axes, augment_names, augment_transforms)):
        if idx == 0:  # skip original
            display_image = original_img
        else:
            transformed = augment(image=image)
            display_image = transformed['image']

        ax.imshow(display_image.permute(1, 2, 0).numpy().astype('uint8'))
        ax.set_title(name)
        ax.axis("off")

    for ax in axes[len(augment_transforms):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_images_from_folder(
        folder_path: str,
        original_image_path: str,
        grid_cols: int = 4,
        file_extensions: tuple = (".png", ".jpg", ".jpeg")
):
    """
    Reads images from a folder and plots them in a grid.

    :param folder_path: Path to the folder containing images.
    :param original_image_path: Path to the original image to include as the first image.
    :param grid_cols: Number of columns in the grid layout. Defaults to 4.
    :param file_extensions: Tuple of valid image file extensions to consider.
    """
    folder = Path(folder_path)
    image_paths = [p for p in folder.iterdir() if p.suffix.lower() in file_extensions]

    if not image_paths:
        print("No images found in the folder.")
        return

    image_paths.insert(0, Path(original_image_path))

    grid_rows = (len(image_paths) + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 4 * grid_rows))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < len(image_paths):
            img = Image.open(image_paths[idx])
            ax.imshow(img)
            ax.set_title(image_paths[idx].name)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def get_examples() -> Dict:
    """
    Returns a dictionary of examples for the visualization functions.
    Probabilities are adapted to show up for every visualization attempt.
    """

    blur = A.Compose([A.Blur((3, 7), 1), ToTensorV2()])
    channel_dropout = A.Compose([A.ChannelDropout((1, 2), 128, p=1), ToTensorV2()])
    clahe = A.Compose([A.CLAHE((1, 4), (8, 8), False, 1), ToTensorV2()])
    color_jitter = A.Compose([A.ColorJitter(0.2, 0.2, 0.2, 0.1, 1), ToTensorV2()])
    defocus = A.Compose([A.Defocus((4, 8), (0.2, 0.4), True), ToTensorV2()])
    fancyPCA = A.Compose([A.FancyPCA(0.1, 1, True), ToTensorV2()])
    glass_blur = A.Compose([A.GlassBlur(0.7, 4, 3, "fast", False, 1), ToTensorV2()])
    gaussian_noise = A.Compose([A.GaussNoise(std_range=(0.1, 0.2), p=1), ToTensorV2()])
    grid_distortion = A.Compose([A.GridDistortion(5, (-0.3, 0.3), p=1), ToTensorV2()])
    grid_dropout = A.Compose([A.GridDropout(ratio=0.3, unit_size_range=(10, 20), random_offset=True, p=1), ToTensorV2()])
    grid_elastic_deform = A.Compose([A.GridElasticDeform(num_grid_xy=(4, 4), magnitude=10, p=1), ToTensorV2()])
    horizontal_flip = A.Compose([A.HorizontalFlip(p=1), ToTensorV2()])
    hue_saturation_value = A.Compose([A.HueSaturationValue((-20, 20), (-30, 30), (-20, 20), p=1), ToTensorV2()])
    iso_noise = A.Compose([A.ISONoise((0.01, 0.05), (0.1, 0.5), p=1), ToTensorV2()])
    median_blur = A.Compose([A.MedianBlur(7, 1), ToTensorV2()])
    noop = A.Compose([A.NoOp(), ToTensorV2()])
    pixel_dropout = A.Compose([A.PixelDropout(0.01, False, 0, None, True), ToTensorV2()])
    random_brightness_contrast = A.Compose([A.RandomBrightnessContrast((-0.2, 0.2), (-0.2, 0.2), True, False, p=1), ToTensorV2()])
    random_gamma = A.Compose([A.RandomGamma((80, 120), p=1), ToTensorV2()])
    random_resized_crop = A.Compose([A.RandomResizedCrop((512, 320), scale=(0.08, 1), ratio=(0.75, 1.3333333333333333), p=1), ToTensorV2()])
    random_tone_curve = A.Compose([A.RandomToneCurve(0.1, False, p=0.1), ToTensorV2()])
    rotate = A.Compose([A.Rotate((-90, 90), interpolation=1, border_mode=4, rotate_method='largest_box',
                       crop_border=False, mask_interpolation=0, fill=False, fill_mask=0, p=0.5), ToTensorV2()])
    sharpen = A.Compose([A.Sharpen((0.2, 0.5), (0.5, 1), 'kernel', 5, 1, 1), ToTensorV2()])
    shift_scale_rotate = A.Compose([A.ShiftScaleRotate((-0.0625, 0.0625), (-0.1, 0.1), (-90, 90), 1, 4, shift_limit_x=None,
                                             shift_limit_y=None, mask_interpolation=0, fill=0, fill_mask=0, p=1), ToTensorV2()])
    solarize = A.Compose([A.Solarize(threshold=(0.5, 0.5), p=1), ToTensorV2()])
    spatter = A.Compose([A.Spatter((0.65, 0.65), (0.3, 0.3), (2, 2), (0.68, 0.68), (0.6, 0.6), 'rain', color=None, p=1), ToTensorV2()])
    transpose = A.Compose([A.Transpose(p=1), ToTensorV2()])
    xy_masking = A.Compose([A.XYMasking((1, 3), (1, 3), (10, 20), (10, 20), fill=0, fill_mask=0, p=1), ToTensorV2()])

    # custom
    carlucci_1 = A.Compose([A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0)), A.HorizontalFlip(p=1), ToTensorV2()])
    carlucci_2 = A.Compose([A.ToGray(p=1), ToTensorV2()])
    carlucci_3 = A.Compose([A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0)), A.HorizontalFlip(p=1), A.ToGray(p=1), ToTensorV2()])
    wang_1 = A.Compose([A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1), ToTensorV2()])
    wang_2 = A.Compose([A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0)), A.HorizontalFlip(p=1), A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1), ToTensorV2()])
    zhou = A.Compose([A.Resize(height=int(256 * 1.25), width=int(256 * 1.25)), A.RandomCrop(height=256, width=256), A.HorizontalFlip(p=1), ToTensorV2()])
    color_geometric = A.Compose([A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1), A.Rotate((-45, 45), interpolation=1, border_mode=4, p=1), A.HorizontalFlip(p=1), ToTensorV2()])
    color_distortion = A.Compose([A.HueSaturationValue((-20, 20), (-30, 30), (-20, 20), p=1), A.GridDistortion(5, (-0.3, 0.3), p=1), A.Transpose(p=1), ToTensorV2()])
    contrast_geometric = A.Compose([A.CLAHE((1, 4), (8, 8), always_apply=False, p=1), A.RandomResizedCrop((512, 320), scale=(0.08, 1), ratio=(0.75, 1.33), p=1), A.Rotate((-90, 90), interpolation=1, border_mode=4, p=1), ToTensorV2()])
    noise_geometric = A.Compose([A.GaussNoise(std_range=(0.1, 0.2), p=1), A.Rotate((-45, 45), interpolation=1, border_mode=4, p=1), A.HorizontalFlip(p=1), ToTensorV2()])
    noise_color_geometric = A.Compose([A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4), p=1), A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1), A.HorizontalFlip(p=1), ToTensorV2()])
    masking_color = A.Compose([A.Solarize(threshold=0.5, p=1), A.GridDropout(ratio=0.3, unit_size_min=10, unit_size_max=20, random_offset=True, p=1.0), A.HueSaturationValue((-20, 20), (-30, 30), (-20, 20), p=1), ToTensorV2()])
    masking_noise = A.Compose([A.XYMasking((1, 3), (1, 3), (10, 20), (10, 20), fill=0, fill_mask=0, p=1),A.GaussNoise(std_range=(0.1, 0.3), p=1), A.Rotate((-30, 30), p=1), ToTensorV2()])
    distortion_noise = A.Compose([A.GridDistortion(num_steps=5, distort_limit=(-0.4, 0.4), p=1),A.Defocus(radius=(3, 6), alias_blur=(0.1, 0.3), p=1), A.Transpose(p=0.5),ToTensorV2()])
    color_mask_geometric = A.Compose([A.Solarize(threshold=0.5, p=1), A.XYMasking((1, 2), (1, 3), (15, 25), (15, 25), fill=0, fill_mask=0, p=1),A.Rotate((-90, 90), p=1), ToTensorV2()])

    pacs = {
        "Channel Dropout": channel_dropout,
        "CLAHE": clahe,
        "Color Jitter": color_jitter,
        "Grid Elastic Deform": grid_elastic_deform,
        "Hue Saturation": hue_saturation_value,
        "Solarize": solarize,
        "XY Masking": xy_masking
    }

    camelyon17 = {
        "Fancy PCA": fancyPCA,
        "Glass Blur": glass_blur,
        "ISO Noise": iso_noise,
        "Median Blur": median_blur,
        "Random Brightness/Contrast": random_brightness_contrast,
        "Random Gamma": random_gamma,
        "Random Tone-Curve": random_tone_curve,
        "Sharpen": sharpen,
        "Shift/Sale/Rotate": shift_scale_rotate,
        "Spatter": spatter
    }

    shared = {
        "Blur": blur,
        "Defocus": defocus,
        "Gaussian Noise": gaussian_noise,
        "Grid Distortion": grid_distortion,
        "Grid Dropout": grid_dropout,
        "Horizontal Flip": horizontal_flip,
        "Pixel Dropout": pixel_dropout,
        "Random Resized Crop": random_resized_crop,
        "Rotate": rotate,
        "Transpose": transpose
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

    presentation = {
        "Color + Masking + Geometric": color_mask_geometric,
        "Noise + Geometric": noise_geometric,
        "Mask + Color": masking_color
    }

    return presentation

