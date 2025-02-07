import cv2
import numpy as np
import torch
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image

from domgen.augment._augmentation_registry import AUGSETS
from typing import List, Dict


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
        grid_cols: int = 4,
        severity: int = 3
) -> None:
    """
    Plot a grid of augmented (corrupted) images with labels.
    :param path: Path to image file.
    :param augmentations: Dictionary of augmentation objects, with {name: augmentation}.
    :param grid_cols: Number of columns. Defaults to 4.
    :param severity: Severity level for corruptions. Defaults to 3.
    :return: None
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_img = torch.tensor(image, dtype=torch.float32)

    augment_names = ['Original'] + list(augmentations.keys())
    augment_transforms = [None] + list(augmentations.values())

    grid_rows = (len(augment_transforms) + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 4 * grid_rows))
    axes = axes.flatten()

    images = [original_img]

    for corruption_name, corruption_instance in augmentations.items():

        original_image_pil = Image.fromarray(image)

        if hasattr(corruption_instance, 'apply'):
            corrupted_image = corruption_instance.apply(original_image_pil, severity=severity)
            if isinstance(corrupted_image, np.ndarray):
                corrupted_image_pil = Image.fromarray(corrupted_image)
            else:
                corrupted_image_pil = corrupted_image
            corrupted_image_tensor = torch.tensor(np.array(corrupted_image_pil), dtype=torch.float32)
        else:
            transformed = corruption_instance(image=image)
            corrupted_image_pil = transformed['image']
            corrupted_image_tensor = torch.tensor(np.array(corrupted_image_pil), dtype=torch.float32).permute(1,2,0)


        images.append(corrupted_image_tensor)

    for idx, (ax, name, image_tensor) in enumerate(zip(axes, augment_names, images)):
        ax.imshow(image_tensor.numpy().astype('uint8'))
        ax.set_title(name)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def get_examples(name: str) -> Dict:
    """
    Returns a dictionary of examples for the visualization functions.
    Probabilities are adapted to show up for every visualization attempt.

    Args:
        name (str): The name of the desired augmentation dictionary. Options include:
                    - "custom"
                    - "all_custom_aug"

    Returns:
        Dict: The corresponding augmentation dictionary, or an empty dictionary if the name is invalid.
    """
    return AUGSETS.get(name.lower(), 'custom')




