import json
import os
import pprint
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from ._utils import EarlyStopping
from ._model_config import get_device, get_model, get_criterion, get_optimizer
from ..data import DOMAIN_NAMES, get_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Experiment Logger')


def _checkpointing(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_accuracy: float,
        best_val_accuracy: float,
        checkpoint_dir: str,
        is_last: bool = False,
) -> float:
    """
    Save and manage checkpoints during training.

    :param model: Model instance.
    :param optimizer: Optimizer instance.
    :param epoch: Current epoch.
    :param val_accuracy: Validation accuracy for the current epoch.
    :param best_val_accuracy: Best validation accuracy so far.
    :param checkpoint_dir: Directory to save checkpoints.
    :param is_last: Whether this is the final checkpoint for the model.
    :return: Updated best validation accuracy.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_accuracy,
    }

    if is_last:
        last_model_path = f"{checkpoint_dir}/last.pth"
        torch.save(checkpoint, last_model_path)
        logger.info(f"Last model checkpoint saved at {last_model_path}")
    elif val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_path = f"{checkpoint_dir}/best.pth"
        torch.save(checkpoint, best_model_path)
        logger.info(f"Best model checkpoint saved with accuracy: {best_val_accuracy:.2f}%")

    return best_val_accuracy


class DomGenTrainer:
    def __init__(
            self,
            model: str,
            optimizer: str,
            criterion: str,
            dataset: str,
            strategy: str = 'leave one out',
            num_experiments: int = 1,
            epochs_per_experiment: int = 10,
            device: str = None,
            use_early_stopping: bool = True,
            use_scheduling: bool = True,
            patience: int = 5,
            delta: float = 0.1,
            log_dir: str = 'experiments',
            checkpoint_dir: str = 'checkpoints',
            silent: bool = False,
            **kwargs
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset = dataset
        self.strategy = strategy
        self.num_experiments = num_experiments
        self.epochs_per_experiment = epochs_per_experiment
        self.device = device if device else get_device()
        self.use_early_stopping = use_early_stopping
        self.use_scheduling = use_scheduling
        self.patience = patience
        self.delta = delta
        self.log_dir = log_dir
        self.silent = silent

        os.makedirs(f'{self.log_dir}/{checkpoint_dir}', exist_ok=True)
        self.checkpoint_dir = f'{self.log_dir}/{checkpoint_dir}'
        self.args = kwargs

        self.domains = DOMAIN_NAMES[self.dataset]
        self.metrics = {}
        self.train_metrics = {}
        self.test_metrics = {}

        self.current_experiment = -1
        if use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.patience,
                delta=self.delta,
                mode='min'
            )

    def fit(self):
        for run in range(self.num_experiments):
            self.current_experiment += 1
            logger.info(f"RUNNING EXPERIMENT {self.current_experiment + 1}/{self.num_experiments}")
            logger.info(f"TRAINING ON {self.device}.\n")

            for idx, domain in enumerate(self.domains):
                logger.info(f"LEAVE OUT DOMAIN: {domain}.")
                model, criterion, optimizer, train_loader, val_loader, test_loader = self._prepare_domain(idx)
                metrics, test_metrics = self.train_model(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                )
                self.train_metrics[domain] = metrics
                self.test_metrics[domain] = test_metrics
                logger.info(f"TEST LOSS: {test_metrics['Test Loss']}")
                logger.info(f"TEST ACCURACY: {test_metrics['Test Accuracy']}\n")
                self.early_stopping.reset()

            if run not in self.metrics:
                self.metrics[run] = {}
            self.metrics[run]['train'] = self.train_metrics
            self.metrics[run]['test'] = self.test_metrics

    def _run_epoch(
            self,
            mode: str,
            model: torch.nn.Module,
            loader: torch.utils.data.DataLoader,
            criterion: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            tb_writer: Optional[SummaryWriter] = None,
            epoch: Optional[int] = None,
            scheduler=None,
    ) -> Tuple[float, float]:
        """
            Generic method for training, validation, and testing.
            :param mode: 'train', 'val', or 'test'
            :param model: Model instance.
            :param loader: DataLoader instance for the respective set.
            :param criterion: Loss function.
            :param optimizer: Optimizer (only used for training).
            :param tb_writer: Tensorboard writer (optional).
            :param epoch: Current epoch (optional, used for training).
            :param scheduler: Learning rate scheduler (optional, used for validation).
            :return: Tuple of average loss and accuracy.
            """
        is_train = mode == 'train'
        model.train() if is_train else model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with tqdm(loader, desc=mode.capitalize(), unit="batch", disable=self.silent) as batch:
            for inputs, labels in batch:
                if mode == 'test':
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                else:
                    inputs, labels = inputs['image'].to(self.device), labels.to(self.device)

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

        if tb_writer:
            tb_writer.add_scalar(f'Loss/{mode}', avg_loss, epoch or 0)
            tb_writer.add_scalar(f'Accuracy/{mode}', avg_accuracy, epoch or 0)

        if mode == 'val' and scheduler:
            scheduler.step(avg_loss)

        return avg_loss, avg_accuracy

    def train_model(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
    ) -> Tuple[List[dict], dict]:
        train_writer, val_writer, test_writer = self._setup_tb_writers(self.log_dir)

        metrics_summary = []
        scheduler = ReduceLROnPlateau(optimizer, patience=self.patience) if self.use_scheduling else None
        best_val_accuracy = 0.0

        for epoch in range(self.epochs_per_experiment):
            train_loss, train_acc = self._run_epoch(
                mode="train",
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                tb_writer=train_writer,
                epoch=epoch,
            )
            val_loss, val_acc = self._run_epoch(
                mode="val",
                model=model,
                loader=val_loader,
                criterion=criterion,
                tb_writer=val_writer,
                epoch=epoch,
                scheduler=scheduler,
            )

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
            metrics_summary.append(metrics)
            pprint.pprint(metrics)

            last_epoch = True if epoch == self.epochs_per_experiment - 1 else False
            best_val_accuracy = _checkpointing(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_accuracy=val_acc,
                best_val_accuracy=best_val_accuracy,
                checkpoint_dir=self.checkpoint_dir,
                is_last=last_epoch
            )

            self.early_stopping(
                score=val_loss,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                checkpoint_path=f"{self.checkpoint_dir}/best.pth"

            )
            if self.early_stopping.stop:
                print(f"Training stopped early at epoch {epoch + 1}.")
                break

        best_model_path = f"{self.checkpoint_dir}/best.pth"
        logger.info(f"Loading best model for testing from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = self._run_epoch(
            mode="test",
            model=model,
            loader=test_loader,
            criterion=criterion,
            tb_writer=test_writer,
        )
        return metrics_summary, {"Test Loss": test_metrics[0], "Test Accuracy": test_metrics[1]}

    def _prepare_domain(
            self,
            idx: int
    ) -> Tuple:
        dataset = get_dataset(
            name=self.dataset,
            root_dir=self.args["dataset_dir"],
            test_domain=idx,
            augment=self.args["augment"],
        )
        train_loader, val_loader, test_loader = dataset.generate_loaders(
            batch_size=self.args["batch_size"]
        )
        model = get_model(self.model, dataset.num_classes).to(self.device)
        criterion = get_criterion(self.criterion)
        optimizer = get_optimizer(
            optimizer_name=self.optimizer,
            model_parameters=model.parameters(),
            **self.args,
        )
        return model, criterion, optimizer, train_loader, val_loader, test_loader

    def _setup_tb_writers(
            self,
            log_dir: str
    ) -> Tuple[Optional[SummaryWriter], Optional[SummaryWriter], Optional[SummaryWriter]]:
        if not self.args.get("tensorboard", False):
            return None, None, None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        train_writer = SummaryWriter(f"{log_dir}/train_{timestamp}")
        val_writer = SummaryWriter(f"{log_dir}/val_{timestamp}")
        test_writer = SummaryWriter(f"{log_dir}/test_{timestamp}")
        return train_writer, val_writer, test_writer

    def print_metrics(self):
        print('\nTRAINING METRICS')
        pprint.pprint(self.train_metrics)
        print('\nTEST METRICS')
        pprint.pprint(self.test_metrics)

    @classmethod
    def save_metrics(cls, metrics, base_log_dir):
        """
        Converts the metrics dictionary to separate DataFrames for each domain and run, then saves them to CSV files.
        :param metrics: The metrics dictionary containing the metrics for each run.
        :param base_log_dir: The base directory where the run folders should be created.
        """
        for run_idx, run_metrics in metrics.items():
            run_folder = os.path.join(base_log_dir, f'run_{run_idx}')
            os.makedirs(run_folder, exist_ok=True)

            for domain in run_metrics['train'].keys():
                train_df = pd.DataFrame(run_metrics['train'][domain])
                train_file = f'{domain}_train_metrics.csv'
                train_df.to_csv(os.path.join(run_folder, train_file), index=False)

                test_metrics = run_metrics['test'].get(domain, {})
                test_df = pd.DataFrame([test_metrics])
                test_file = f'{domain}_test_metrics.csv'
                test_df.to_csv(os.path.join(run_folder, test_file), index=False)

                print(f'Saved training metrics for {domain} in {run_folder}/{train_file}')
                print(f'Saved testing metrics for {domain} in {run_folder}/{test_file}')

    def save_config(self, filename):
        config = vars(self)

        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {filename}")