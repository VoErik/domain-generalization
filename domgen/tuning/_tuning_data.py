import os
import random
from typing import Any, List, Tuple

import PIL
import torch
from torch.utils.data import Dataset
from domgen.augment import imagenet_transform


class TuningDataset(Dataset):
    def __init__(
            self,
            data: List[Tuple[str, str, str]],
            transform=None,
            cls2idx = None,
            dom2idx = None
    ):
        """
        Custom dataset for augmentation tuning.

        :param data: List of (file_path, class_label, domain_label) tuples.
        :param transform: Transformations to apply to the images.
        """
        self.data = data
        self.transform = transform
        self.cls2idx = cls2idx
        self.dom2idx = dom2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_label, domain_label = self.data[idx]
        image = PIL.Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            transform = imagenet_transform((227,227))
            image = transform(image)
        class_label = torch.tensor(self.cls2idx[class_label], dtype=torch.long)
        domain_label = torch.tensor(self.dom2idx[domain_label], dtype=torch.long)

        return image, class_label, domain_label

def create_datasets(
        dataset_path: str,
        class_name: str = None,
        leave_out_domain: str = None,
        subsample: int = None,
        transform: Any = None,
        cls2idx: Any = None,
        dom2idx: Any = None,
        val_split: float = 0.2  # Fraction of training data to use for validation
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Creates training, validation, and testing datasets.

    :param dataset_path: Path to the dataset.
    :param class_name: Name of the class to include (e.g., "dog"). If None, include all classes.
    :param leave_out_domain: Name of the domain to leave out for testing (e.g., "cartoon").
    :param subsample: Number of samples to randomly select per domain. If None, use all samples.
    :param val_split: Fraction of the training data to reserve for validation.
    :param transform: Transformation to apply on images.
    :param cls2idx: Class to index mapping (optional).
    :param dom2idx: Domain to index mapping (optional).

    :returns: Training, validation, and testing datasets.
    """
    train_data = []
    val_data = []
    test_data = []

    for domain in os.listdir(dataset_path):
        domain_path = os.path.join(dataset_path, domain)
        if not os.path.isdir(domain_path):
            continue

        for class_folder in os.listdir(domain_path):
            if os.path.isdir(os.path.join(domain_path, class_folder)):
                class_path = os.path.join(domain_path, class_folder)

                if class_name and class_folder != class_name:
                    continue
                if not os.path.exists(class_path):
                    continue

                images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith((".jpg", ".png", ".jpeg"))]

                if subsample is not None:
                    images = random.sample(images, min(subsample, len(images)))

                if domain == leave_out_domain:
                    test_data.extend([(img, class_folder, domain) for img in images])
                else:
                    # Split the images into training and validation sets
                    split_idx = int(len(images) * (1 - val_split))
                    train_data.extend([(img, class_folder, domain) for img in images[:split_idx]])
                    val_data.extend([(img, class_folder, domain) for img in images[split_idx:]])

    # Create the datasets
    train_dataset = TuningDataset(train_data, transform=transform, cls2idx=cls2idx, dom2idx=dom2idx)
    val_dataset = TuningDataset(val_data, transform=transform, cls2idx=cls2idx, dom2idx=dom2idx)
    test_dataset = TuningDataset(test_data, transform=transform)

    return train_dataset, val_dataset, test_dataset