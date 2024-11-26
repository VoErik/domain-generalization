from typing import Tuple

import torchvision
from torchvision import transforms


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
