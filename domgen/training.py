import os
import csv
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict
from pprint import pprint

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights

from domgen.data import PACS
from domgen.eval import plot_training_curves, plot_accuracies

from tqdm import tqdm
import logging

# remember to set tensorboard logdir to experiments
logdir = 'experiments'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Experiment Logger')


def train_model(args, model, optimizer, criterion, train_loader, val_loader, test_loader, test_domain, i):
    training_metrics = _training(model=model,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 num_epochs=args.epochs,
                                 domain_name=test_domain,
                                 device=args.device,
                                 train_loader=train_loader,
                                 val_loader=val_loader,
                                 experiment_name=args.experiment,
                                 experiment_number=i)

    test_metrics = _testing(model=model,
                            criterion=criterion,
                            device=args.device,
                            dataloader=test_loader,
                            domain_name=test_domain,
                            experiment_name=args.experiment,
                            experiment_number=i)

    return training_metrics, test_metrics


def run_epoch(model: torch.nn.Module,
              tb_writer: SummaryWriter,
              dataloader: DataLoader,
              criterion,
              mode: str,
              optimizer: torch.optim.Optimizer = None,
              epoch: int = None,
              num_epochs: int = None,
              device: str = 'cpu') -> Tuple[float, float]:
    """Trains the model for one epoch and logs metrics to tensorboard.

    :param epoch: Index of the current epoch
    :param model: Model to train.
    :param tb_writer: Tensorboard summary writer.
    :param dataloader: Training dataloader.
    :param device: Device to train the model on. Can be either CPU, GPU, or MPS.
    :param optimizer: Torch optimizer to train the model.
    :param criterion: Loss function.
    :param num_epochs: Number of epochs.
    :return: Average loss and accuracy of the epoch.
    """

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with tqdm(dataloader, unit='batch') as batch:
        for i, (inputs, labels) in enumerate(batch):
            batch.set_description(f'Epoch {epoch + 1}/{num_epochs}' if epoch is not None else 'TESTING')
            inputs, labels = inputs.to(device), labels.to(device)

            if mode == 'train':
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if mode == 'train':
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            correct_predictions += correct
            total_predictions += labels.size(0)

            running_loss += loss.item()
            denominator = len(predictions) if predictions.dim() != 0 else 1
            accuracy = correct / denominator
            batch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

    avg_loss = running_loss / len(dataloader)
    avg_accuracy = 100. * correct_predictions / total_predictions

    # Log to TensorBoard
    if mode != 'test':
        tb_writer.add_scalar(f'Loss/{mode}', avg_loss, epoch + 1)
        tb_writer.add_scalar(f'Accuracy/{mode}', avg_accuracy, epoch + 1)
    else:
        tb_writer.add_scalar(f'Loss/{mode}', avg_loss, 1)
        tb_writer.add_scalar(f'Accuracy/{mode}', avg_accuracy, 1)

    return avg_loss, avg_accuracy


def _training(model: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              criterion,
              num_epochs: int,
              domain_name: str,
              device: str,
              train_loader,
              val_loader,
              experiment_name: str,
              experiment_number: int) -> List[dict[str, float | int]]:
    """Training loop. Trains the model and evaluates on the validation set.
    :param model: Model to train.
    :param optimizer: Torch optimizer to train.
    :param criterion: Loss function.
    :param num_epochs: Number of epochs.
    :param domain_name: Domain name.
    :param device: Device to train the model on.
    :param train_loader: Training dataloader.
    :param val_loader: Validation dataloader.
    :param experiment_name: Name of the experiment.
    :param experiment_number: Number of the experiment.
    :return: Average loss and accuracy of the epoch.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'{logdir}/{experiment_name}/run_{experiment_number}/{domain_name}/train_{timestamp}')
    best_vloss = float('inf')
    metrics_summary = []

    for epoch in range(num_epochs):
        logger.info(f'EPOCH {epoch + 1}:')

        model.train()
        avg_loss, avg_accuracy = run_epoch(epoch=epoch,
                                           tb_writer=writer,
                                           mode='train',
                                           model=model,
                                           criterion=criterion,
                                           optimizer=optimizer,
                                           dataloader=train_loader,
                                           num_epochs=num_epochs,
                                           device=device)

        model.eval()
        with torch.no_grad():
            avg_vloss, avg_vaccuracy = run_epoch(epoch=epoch,
                                                 tb_writer=writer,
                                                 mode='val',
                                                 model=model,
                                                 criterion=criterion,
                                                 optimizer=optimizer,
                                                 dataloader=val_loader,
                                                 num_epochs=num_epochs,
                                                 device=device)

        logger.info(f'LOSS: TRAIN: {avg_loss} VAL: {avg_vloss}')
        logger.info(f'ACCURACY: TRAIN: {avg_accuracy} VAL: {avg_vaccuracy}\n')

        # Logging to TensorBoard
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

        metrics = {
            'epoch': epoch,
            'avg_training_loss': avg_loss,
            'avg_validation_loss': avg_vloss,
            'avg_training_accuracy': avg_accuracy,
            'avg_validation_accuracy': avg_vaccuracy,
            'best_validation_loss': best_vloss
        }
        metrics_summary.append(metrics)

    return metrics_summary


def _testing(model: torch.nn.Module,
             criterion,
             device: str,
             dataloader: DataLoader,
             experiment_name: str,
             domain_name: str,
             experiment_number: int) -> Dict[str, float]:
    """Evaluates the model on the test set.
    :param model: Model to train.
    :param criterion: Loss function.
    :param device: Torch device to train the model.
    :param dataloader: Testing dataloader.
    :param experiment_name: Name of the experiment.
    :param domain_name: Domain name.
    :param experiment_number: Number of the experiment.
    :param device: Device to train the model on.
    :param experiment_name: Name of the experiment.
    :return: Average loss and accuracy on the test set.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'{logdir}/{experiment_name}/run_{experiment_number}/{domain_name}/test_{timestamp}')
    with torch.no_grad():
        model.eval()
        test_loss, test_accuracy = run_epoch(tb_writer=writer,
                                             mode='test',
                                             model=model,
                                             criterion=criterion,
                                             dataloader=dataloader,
                                             device=device)
    writer.add_scalars('Test Loss and Accuracy',
                       {'Loss': test_loss, 'Accuracy': test_accuracy},
                       1)
    writer.flush()
    return {'Test Loss': test_loss, 'Test Accuracy': test_accuracy}
