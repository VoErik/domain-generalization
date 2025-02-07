import torch
import numpy as np
import albumentations as A

from typing import Tuple
from PIL import Image

from albumentations.pytorch import ToTensorV2

# TODO: move aug definitions to augmentation_registry.py

class TransformsWrapper:
    def __init__(
            self,
            albumentations_transform=None,
            torchvision_transform=None
    ) -> None:
        self.albumentations_transform = albumentations_transform
        self.torchvision_transform = torchvision_transform

    def __call__(self, img):
        """Returns transformed image based on which transforms library is used."""
        if self.torchvision_transform:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = self.torchvision_transform(img)

        if self.albumentations_transform:
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
            elif isinstance(img, Image.Image):
                img = np.array(img)
            augmented = self.albumentations_transform(image=img)
            img = augmented['image']

        return img


def imagenet_transform(
        input_size: Tuple[int, int]
) -> TransformsWrapper:
    """Base transforms for imagenet."""
    transform = A.Compose([
        A.Resize(input_size[0], input_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    return TransformsWrapper(albumentations_transform=transform)


def create_augmentation_pipeline(
        augmentations: dict
) -> TransformsWrapper:
    """Constructs an augmentation pipeline based on a predefined dictionary."""
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