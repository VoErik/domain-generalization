import os
import tempfile
from argparse import Namespace
from datetime import datetime
from typing import Tuple, Optional

import albumentations
import torch
from ray import train, tune
from ray.air import RunConfig, Result
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ruamel.yaml import YAML
from tqdm import tqdm

from domgen.model_training import get_device, get_model, get_criterion, get_optimizer
from domgen.tuning._base_tuner import BaseTuner

#################################################################
############ CURRENTLY BUGGY AND UNDER CONSTRUCTION #############
#################################################################

class AugmentationTuner(BaseTuner):
    """
    AugmentationTuner class for fine-tuning data augmentations.
    Integrates into the albumentations pipeline.
    """

    def __init__(self, base_config, tune_config, **kwargs):
        super().__init__(**kwargs)
        yaml = YAML(typ="safe")
        with open(base_config, "r") as f:
            self.base_config = yaml.load(f)
        self.tune_config = tune_config
        self.option = 'choose-compose'

    def get_trainable(self):
        """
        Returns a trainable function that Ray Tune can execute.
        """
        conf = self.tune_config
        self.base_config["datadir"] = os.path.abspath(self.base_config["datadir"])

        def trainable(conf):
            # Set up device, model, criterion, optimizer, and dataset
            device = get_device()
            model = get_model(
                model_name=self.base_config["model"], num_classes=self.base_config["num_classes"]
            ).to(device)
            # Todo: Add **kwargs to get_criterion function
            criterion = get_criterion(conf["criterion"])
            # Todo: Add **kwargs to get_optimizer function
            optimizer = get_optimizer(
                conf["optimizer"], model.parameters(), lr=conf["lr"]
            )
            # Todo: Here, the logic for creating the augmentation that is passed to the get_dataset function is located
            # It should work for the following scenarios:
            # 1. Pass in an entire torchvision.Compose object.
            if self.option == 'choose-compose':
                # search space: liste von Compose Objects -> nicht trivial
                # -> pro trial 1 Compose Object raus
                # augmentation = conf["candidate-compose"]
                pass
            # 2. Pass in list of possible augmentation steps, then create a Compose object from them
            if conf['option'] == "build-compose":
                # search space: dict von Augmentations (zB shared_augs)
                    # tune.choice fürs 1 Aug
                    # tune.choice fürs 2 Aug
                    # tune.choice fürs 3 Aug
                # augmentation = create_aug_pipeline()
                pass
            # 3. Pass in a Compose object but tune the parameters of the augmentation steps.
            if conf['option'] == "tune-compose":
                augmentation = albumentations.Compose(
                    [
                        albumentations.blur(conf["blur_limit"]),
                    ]
                )
            augmentation = None

            from domgen.data import get_dataset
            dataset = get_dataset(
                root_dir=self.base_config["datadir"], name=self.base_config["dataset"],
                test_domain=self.base_config["test_domain"], augmentation=augmentation,
            )
            train_loader, val_loader, _ = dataset.generate_loaders(batch_size=conf["batch_size"])

            # Call the tune_params method (training loop)
            self.tune_params(
                model=model, criterion=criterion, optimizer=optimizer,
                train_loader=train_loader, val_loader=val_loader, device=device,
            )

        return trainable

    def run(self) -> Result:

        trial_dir = f'{os.path.abspath(self.trial_dir)}'
        trial_name = f'{self.base_config["model"]}-{self.base_config["test_domain"]}-{datetime.now().strftime('%Y%m%d_%H%M%S')}'

        search_space = self.construct_search_space()
        trainable = self.get_trainable()

        trainable_with_gpu = tune.with_resources(trainable, {"gpu": self.num_gpu})
        tuner = tune.Tuner(
            trainable_with_gpu,
            tune_config=tune.TuneConfig(
                num_samples=self.num_trials,
                scheduler=ASHAScheduler(metric="mean_accuracy", mode=self.mode),
                max_concurrent_trials=self.max_concurrent
            ),
            run_config=RunConfig(
                storage_path=trial_dir,
                name=trial_name),
            param_space=search_space,
        )
        results = tuner.fit()

        df = results.get_dataframe()
        df.to_csv(f"{self.trial_dir}/{trial_name}/results.csv")
        return results.get_best_result(metric="mean_accuracy", mode=self.mode)

    def construct_search_space(self):
        # Todo: Create this function!
        # This function creates the search space dynamically based on the config. It should be able to handle the
        # three scenarios described in the get_trainable function.
        # It should return a suitable representation for the search space. Usually a dictionary-like object.
        pass

    def tune_params(self, model, criterion, optimizer, train_loader, val_loader, device):
        """
        Function to handle training and validation steps.
        """
        args = Namespace()
        args.use_scheduling = False
        args.silent = self.silent
        for i in range(self.base_config["num_epochs"]):
            _, _ = run_epoch(
                model=model, optimizer=optimizer, criterion=criterion,
                loader=train_loader, device=device, mode='train'
            )
            val_loss, acc = run_epoch(
                model=model, loader=val_loader, criterion=criterion, device=device, mode='val')

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                if (i + 1) % 5 == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(temp_checkpoint_dir, "model.pth")
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                train.report({"mean_accuracy": acc, "val_loss": val_loss}, checkpoint=checkpoint)


def run_epoch(
        mode: str,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
) -> Tuple[float, float]:
    """
        Generic method for training, validation, and testing.
        :param device: Device to train on.
        :param mode: 'train', 'val', or 'test'
        :param model: Model instance.
        :param loader: DataLoader instance for the respective set.
        :param criterion: Loss function.
        :param optimizer: Optimizer (only used for training).
        :return: Tuple of average loss and accuracy.
        """
    is_train = mode == 'train'
    model.train() if is_train else model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with tqdm(loader, desc=mode.capitalize(), unit="batch", disable=True) as batch:
        for inputs, labels in batch:

            inputs, labels = inputs['image'].to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if is_train:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    avg_loss = total_loss / total_predictions
    avg_accuracy = 100.0 * correct_predictions / total_predictions

    return avg_loss, avg_accuracy
