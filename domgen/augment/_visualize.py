from typing import List, Dict
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2


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


def get_examples() -> Dict:
    """Returns a dictionary of examples for the visualization functions."""
    vertical = A.Compose([A.VerticalFlip(p=1), ToTensorV2()])
    horizontal = A.Compose([A.HorizontalFlip(p=1), ToTensorV2()])
    colorjitter = A.Compose([A.ColorJitter(), ToTensorV2()])
    rotate = A.Compose([A.RandomRotate90(p=1), ToTensorV2()])
    brightness_contrast = A.Compose([A.RandomBrightnessContrast(p=1), ToTensorV2()])
    gaussian_blur = A.Compose([A.GaussianBlur(blur_limit=(13, 21), p=1), ToTensorV2()])
    clahe = A.Compose([A.CLAHE(p=1), ToTensorV2()])
    sharpen = A.Compose([A.Sharpen(p=1), ToTensorV2()])
    hue_saturation = A.Compose([A.HueSaturationValue(p=1), ToTensorV2()])
    solarize = A.Compose([A.Solarize(p=1), ToTensorV2()])
    equalize = A.Compose([A.Equalize(p=1), ToTensorV2()])

    augmentations = {
        "Vertical Flip": vertical,
        "Horizontal Flip": horizontal,
        "Color Jitter": colorjitter,
        "Rotate 90°": rotate,
        "Brightness/Contrast": brightness_contrast,
        "Gaussian Blur": gaussian_blur,
        "CLAHE": clahe,
        "Sharpen": sharpen,
        "Hue Saturation": hue_saturation,
        "Solarize": solarize,
        "Equalize": equalize,
    }
    return augmentations
