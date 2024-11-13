import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image


class PACSData:
    def __init__(self,
                 path: str,
                 domains: tuple[str] = ('art_painting', 'cartoon', 'photo', 'sketch')):
        # TODO: make generic class when we integrate more datasets
        self.path = path
        self.domains = domains
        self.mean = None
        self.std = None
        self._get_data()
        self._set_classes()
        self._set_class_2_idx()
        self._set_idx_2_class()
        self._get_domain_sizes()
        self._create_partitions()

    def _get_data(self) -> None:
        """Creates PyTorch ImageFolder for each domain in the dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.data = {domain: {'dataset': ImageFolder(f'{self.path}/{domain}',
                                                     is_valid_file=validate_image,
                                                     transform=transform)} for domain in self.domains}

    def _get_domain_sizes(self) -> None:
        """Returns lengths of all domains plus total dataset size"""
        if self.data:
            self.domain_size = {domain: len(self.data[domain]['dataset'].samples) for domain in self.domains}
            self.domain_size['total'] = sum(value for _, value in self.domain_size.items())

    def _set_classes(self) -> None:
        """Sets class labels for all domains"""
        self.classes = self.data[list(self.data.keys())[0]]['dataset'].classes

    def _set_class_2_idx(self) -> None:
        """Sets up class to index mapping."""
        self.class_2_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _set_idx_2_class(self) -> None:
        """Sets up index to class mapping."""
        self.idx_2_class = {idx: cls for idx, cls in enumerate(self.classes)}

    def _create_partitions(self, partition_size: float = 0.8, seed: int = None) -> None:
        """
        Creates training and validation datasets.

        :param partition_size: Size of training partition in each domain. Default: 0.8
        :param seed: Random seed. Default: None
        :return: None
        """

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        for domain in self.domains:
            dataset = self.data[domain]['dataset']
            dataset_length = len(dataset)

            train_length = int(partition_size * dataset_length)
            val_length = dataset_length - train_length  # TODO: needed?

            indices = torch.randperm(dataset_length, generator=generator).tolist()
            train_indices = indices[:train_length]
            val_indices = indices[train_length:]

            self.data[domain]['train'] = FilteredDataset(dataset, train_indices)
            self.data[domain]['val'] = FilteredDataset(dataset, val_indices)

    def create_dataloaders(self,
                           test_domain: str,
                           train_batch_size: int = 16,
                           val_batch_size: int = 16,
                           shuffle_train: bool = True,
                           shuffle_val: bool = False,
                           shuffle_test: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:

        # TODO: assert the domains have the same labels

        train_sets = []
        val_sets = []

        for domain in self.data:
            if domain != test_domain:
                train_sets.append(self.data[domain]['train'])
                val_sets.append(self.data[domain]['val'])

        train_data = ConcatDataset(train_sets)
        val_data = ConcatDataset(val_sets)

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=train_batch_size,
            shuffle=shuffle_train)

        val_loader = DataLoader(
            dataset=val_data,
            batch_size=val_batch_size,
            shuffle=shuffle_val
        )

        test_loader = DataLoader(
            dataset=self.data[test_domain]['dataset'],
            batch_size=val_batch_size,
            shuffle=shuffle_test
        )

        return train_loader, val_loader, test_loader


class FilteredDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list):
        """
        A Dataset wrapper that only includes the samples specified by `indices`.

        :param dataset: Original dataset.
        :param indices: List of indices to include in the subset.
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def validate_image(path):
    """Generic image validation function for ImageFolder"""
    try:
        _ = Image.open(path)
        return True
    except IOError as e:
        print(e)
        return False
