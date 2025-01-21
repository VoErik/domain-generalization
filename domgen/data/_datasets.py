import os
from collections import defaultdict
import random
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
from domgen.augment import imagenet_transform
from domgen.augment._transforms import combine_augmentations, create_augmentation_pipeline
from domgen.augment._transforms import all_augmentations, pacs_aug, camelyon17_aug, shared_aug

"""To add a new dataset, just create a class that inherits from `DomainDataset`."""

DOMAIN_NAMES = {
    'PACS': ["art_painting", "cartoon", "photo", "sketch"],
    'camelyon17': ["center_0", "center_1", "center_2", "center_3", "center_4"],
}


class MultiDomainDataset:
    domains = None
    input_shape = None

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DomainDataset(MultiDomainDataset):
    def __init__(
            self,
            root: str,
            test_domain: int | None = None,
            augment: Any = None,
            subset: float = None,
    ) -> None:
        """
        Base dataset class for a multi-domain dataset. Expects folder structure to be compatible with ImageFolder.
        :param root: Dataset directory.
        :param test_domain: Leave out domain.
        :param augment: Augment that needs to be applied. Defaults to ImageNet transformation.
        :param subset: Fraction of dataset that ought to be used. Keeps class and target distribution true to original
         data. Defaults to None = use entire dataset.
        :return: None
        """
        super().__init__()
        self.domains = sorted([directory.name for directory in os.scandir(root) if directory.is_dir()])
        self.test_domain = test_domain
        self.subset = subset
        self.data = []

        # base augment = ImageNet
        input_size = self.input_shape[-2], self.input_shape[-1]
        transform =imagenet_transform(input_size=input_size)


        for i, domain in enumerate(self.domains):
            if augment and (i != self.test_domain):
                domain_transform = augment
            else:
                domain_transform = transform

            path = os.path.join(root, domain)
            domain_dataset = ImageFolder(path, transform=domain_transform)

            if self.subset is not None:
                # Ensures that target distribution remains true to original data
                num_samples_per_class = {}
                for target in set(domain_dataset.targets):
                    class_count = domain_dataset.targets.count(target)
                    num_samples_per_class[target] = int(class_count * self.subset)

                class_indices = defaultdict(list)
                for idx, target in enumerate(domain_dataset.targets):
                    class_indices[target].append(idx)

                subset_indices = []
                for target, indices in class_indices.items():
                    random.shuffle(indices)
                    subset_indices.extend(indices[:num_samples_per_class[target]])

                samples = [domain_dataset.samples[i] for i in subset_indices]
                targets = [domain_dataset.targets[i] for i in subset_indices]
                domain_dataset.samples = samples
                domain_dataset.targets = targets

            self.data.append(domain_dataset)

        self.num_classes = len(self.data[-1].classes)
        self.classes = list(self.data[-1].classes)
        self.idx_to_class = dict(zip(range(self.num_classes), self.classes))

    def get_domain_sizes(self) -> defaultdict:
        """Returns sizes of all domains."""
        size_dict = None
        domain_name_map = {i: name for i, name in enumerate(self.domains)}
        if self.data:
            size_dict = defaultdict()
            for i, domain_dataset in enumerate(self.data):
                size_dict[domain_name_map[i]] = len(domain_dataset.imgs)
        return size_dict

    def generate_loaders(
            self,
            batch_size: int = 32,
            test_size: float = 0.2,
            stratify: bool = True,
    ) -> (DataLoader, DataLoader, DataLoader):
        """
        Generates DataLoaders for training and testing domains.

        :param test_size: Size of the validation partition (default: 0.2).
        :param batch_size: Size of the batch. (default: 32)
        :param stratify: Whether to stratify class distribution (default: True).
        :return: A tuple of DataLoaders for training, validation and testing.
        """
        train_domains = [domain for i, domain in enumerate(self.data) if i != self.test_domain]
        train_subsets = []
        val_subsets = []
        for dom in train_domains:
            targets = dom.targets
            if stratify:
                train_idx, valid_idx = train_test_split(
                    np.arange(len(targets)), test_size=test_size, random_state=42, shuffle=True, stratify=targets
                )
            else:
                train_idx, valid_idx = train_test_split(
                    np.arange(len(targets)), test_size=test_size, random_state=42, shuffle=True
                )

            train_subsets.append(Subset(dom, train_idx))
            val_subsets.append(Subset(dom, valid_idx))

        train_split = ConcatDataset(train_subsets)
        val_split = ConcatDataset(val_subsets)

        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=True,drop_last=True)
        test_loader = None
        if self.test_domain is not None:
            test_loader = DataLoader(self.data[self.test_domain], batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


def get_dataset(
        name: str,
        root_dir: str,
        test_domain: int,
        **kwargs,
) -> DomainDataset:
    """
    Gets a domain dataset from a given name.
    :param name: Dataset name as string. Must be one of: PACS, camelyon17
    :param root_dir: Path to datasets directory.
    :param test_domain: Leave out domain.
    :return:
    """
    if name == 'PACS':
        return PACS(root_dir, test_domain=test_domain, **kwargs)
    if name == 'camelyon17':
        return Camelyon17(root_dir, test_domain=test_domain, **kwargs)


"""Insert new datasets below."""


class PACS(DomainDataset):
    domains = DOMAIN_NAMES['PACS']
    input_shape = (3, 244, 244)

    def __init__(self, root, test_domain, **kwargs):
        self.dir = os.path.join(root, "PACS/")
        self.aug = kwargs.get('augment', None)
        super().__init__(self.dir, test_domain, augment=self.aug)


class Camelyon17(DomainDataset):
    domains = DOMAIN_NAMES['camelyon17']
    input_shape = (3, 96, 96)

    def __init__(self, root, test_domain, **kwargs):
        self.dir = os.path.join(root, "camelyon17/")
        self.aug = kwargs.get('augment', None)
        super().__init__(self.dir, test_domain, augment=self.aug, subset=0.05)
