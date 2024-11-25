import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

"""To add a new dataset, just create a class that inherits from `DomainDataset`."""

DOMAIN_NAMES = {
    'PACS': ["art_painting", "cartoon", "photo", "sketch"],
}


class MultiDomainDataset:
    domains = None
    input_shape = None

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DomainDataset(MultiDomainDataset):
    def __init__(self, root, test_domain, augment):
        super().__init__()
        self.domains = sorted([directory.name for directory in os.scandir(root) if directory.is_dir()])
        self.test_domain = test_domain
        # resize to 224x224 and normalize (ImageNet)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data = []
        for i, domain in enumerate(self.domains):
            if augment and (i != self.test_domain):
                domain_transform = augment
            else:
                domain_transform = transform

            path = os.path.join(root, domain)
            domain_dataset = ImageFolder(path, transform=domain_transform)

            self.data.append(domain_dataset)

        self.input_shape = (3, 224, 224,)
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

    def generate_loaders(self,
                         batch_size: int = 32,
                         partition_size: float = 0.8) -> (DataLoader, DataLoader, DataLoader):
        """
        Generates DataLoaders for training and testing domains.
        Parameters
        :param partition_size: Size of the training partition (default: 0.8). Validation size is equal to 1-training.
        :param batch_size: Size of the batch. (default: 32)
        :return: A tuple of DataLoaders for training, validation and testing.
        """
        train_domains = [domain for i, domain in enumerate(self.data) if i != self.test_domain]
        train_partition = ConcatDataset(train_domains)
        train_set, val_set = torch.utils.data.random_split(train_partition,
                                                           [partition_size, 1 - partition_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.data[self.test_domain], batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, test_loader


def get_dataset(name: str, root_dir: str, test_domain: int) -> DomainDataset:
    if name == 'PACS':
        return PACS(root_dir, test_domain=test_domain)


"""Insert new datasets below."""


class PACS(DomainDataset):
    domains = DOMAIN_NAMES['PACS']
    input_shape = (3, 244, 244)

    def __init__(self, root, test_domain):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_domain, augment=None)
