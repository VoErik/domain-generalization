from datetime import datetime
from typing import Tuple, List, Dict
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from ._utils import EarlyStopping

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


def train_model(
        args,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        tensorboard: bool = True,
        device: str = 'cpu',
) -> Tuple[list[dict[str, float | int]], dict[str, float]]:
    """
    Training function. Runs the training and validation functions and tests the model afterwards on the holdout set.
    :param args: Command line + configuration arguments.
    :param model: Model instance to train.
    :param optimizer: Optimizing algorithm.
    :param criterion: Loss function
    :param train_loader: Training data loader.
    :param val_loader: Validation data loader.
    :param test_loader: Test data loader.
    :param tensorboard: Whether to log the run with tensorboard. Default: True.
    :param device: Device to train on.
    :return: Training and testing metrics.
    """

    train_writer, test_writer = None, None
    if tensorboard:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        train_writer = SummaryWriter(
            f'{args.log_dir}/{args.experiment}/run_{args.experiment_number}/{args.domain_name}/train_{timestamp}'
        )
        test_writer = SummaryWriter(
            f'{args.log_dir}/{args.experiment}/run_{args.experiment_number}/{args.domain_name}/test_{timestamp}'
        )

    training_metrics = _training(args,
                                 model=model,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 num_epochs=args.epochs,
                                 device=device,
                                 train_loader=train_loader,
                                 val_loader=val_loader,
                                 tb_writer=train_writer, )

    test_metrics = test(args=args,
                        model=model,
                        criterion=criterion,
                        device=device,
                        test_loader=test_loader,
                        tb_writer=test_writer)

    return training_metrics, test_metrics


def train_epoch(
        args,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: str,
        tb_writer: SummaryWriter = None,
        epoch: int = None
) -> Tuple[float, float]:
    """
    Runs a training epoch.
    :param args: Namespace arguments.
    :param model: Instance of nn.Module
    :param optimizer: Optimizing algorithm.
    :param criterion: Loss function.
    :param train_loader: Training data loader.
    :param device: CPU, CUDA, MPS
    :param tb_writer: Tensorboard summary writer.
    :param epoch: Current epoch.
    :return: Tuple of loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    scheduler = None
    if args.use_scheduling:
        scheduler = ReduceLROnPlateau(optimizer, patience=args.patience)

    with tqdm(train_loader, unit='batch', disable=args.silent) as batch:
        for i, (inputs, labels) in enumerate(batch):
            if not args.silent:
                batch.set_description(f'Epoch {epoch}')
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step(loss)

            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            correct_predictions += correct
            total_predictions += labels.size(0)

            running_loss += loss.item()
            denominator = len(predictions) if predictions.dim() != 0 else 1
            accuracy = correct / denominator
            if not args.silent:
                batch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

    avg_loss = running_loss / len(train_loader)
    avg_accuracy = 100. * correct_predictions / total_predictions

    if tb_writer is not None:
        tb_writer.add_scalar('Loss/train', avg_loss, epoch)
        tb_writer.add_scalar('Accuracy/train', avg_accuracy, epoch)

    print(f"Epoch [{epoch}] - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")

    return avg_loss, avg_accuracy


def validate(
        args,
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: str,
        tb_writer: SummaryWriter = None,
) -> Tuple[float, float]:
    """
    Runs the model on the validation set and calculates loss and accuracy across all batches.
    :param model: Instance of nn.Module
    :param val_loader: Validation data loader.
    :param criterion: Loss function.
    :param device: CPU, CUDA, MPS
    :param tb_writer: Tensorboard summary writer.
    :return: Tuple of loss and accuracy.
    """
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with tqdm(val_loader, unit='batch', desc='Validation', disable=args.silent) as batch:
        for inputs, labels in batch:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            batch_accuracy = correct_predictions / total_predictions
            if not args.silent:
                batch.set_postfix(loss=loss.item(), accuracy=100. * batch_accuracy)

    avg_loss = val_loss / total_predictions
    avg_accuracy = 100. * correct_predictions / total_predictions

    if tb_writer is not None:
        tb_writer.add_scalar('Loss/val', avg_loss, 1)
        tb_writer.add_scalar('Accuracy/val', avg_accuracy, 1)

    print(f'Validation - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%')

    return avg_loss, avg_accuracy


def test(
        args,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: str,
        tb_writer: SummaryWriter = None,
) -> Dict[str, float]:
    """
    Runs the model on the test set and calculates the average test loss and accuracy for the test set.
    :param model: Model (instance of nn.Module)
    :param test_loader: Test data loader.
    :param criterion: Loss function.
    :param device: CPU, CUDA, MPS
    :param tb_writer: Summary Writer for Tensorboard.
    :return: Dict with Test Loss and Test Accuracy
    """
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with tqdm(test_loader, unit='batch', desc='Testing', disable=args.silent) as batch:
        for inputs, labels in batch:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            batch_accuracy = correct_predictions / total_predictions
            if not args.silent:
                batch.set_postfix(loss=loss.item(), accuracy=100. * batch_accuracy)

    avg_loss = test_loss / total_predictions
    avg_accuracy = 100. * correct_predictions / total_predictions

    if tb_writer is not None:
        tb_writer.add_scalars('Test Loss and Accuracy',
                              {'Loss': avg_loss, 'Accuracy': avg_accuracy},
                              1)
        tb_writer.flush()
    return {'Test Loss': avg_loss, 'Test Accuracy': avg_accuracy}


def _training(
        args,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        num_epochs: int,
        device: str,
        train_loader,
        val_loader,
        tb_writer=None,
) -> List[dict[str, float | int]]:
    """Training loop. Trains the model and evaluates on the validation set.
    :param model: Model to train.
    :param optimizer: Torch optimizer to train.
    :param criterion: Loss function.
    :param num_epochs: Number of epochs.
    :param device: Device to train the model on.
    :param train_loader: Training dataloader.
    :param val_loader: Validation dataloader.
    :return: Average loss and accuracy of the epoch.
    """

    best_vloss = float('inf')
    best_vacc = 0.0
    metrics_summary = []

    early_stopping = None
    if args.use_early_stopping:
        early_stopping= EarlyStopping(patience=args.patience)

    for epoch in range(num_epochs):
        logger.info(f'EPOCH {epoch + 1}:')

        avg_loss, avg_accuracy = train_epoch(
            args=args,
            epoch=epoch,
            tb_writer=tb_writer,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device
        )

        avg_vloss, avg_vaccuracy = validate(
            args=args,
            tb_writer=tb_writer,
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device
        )

        logger.info(f'LOSS: TRAIN: {avg_loss} VAL: {avg_vloss}')
        logger.info(f'ACCURACY: TRAIN: {avg_accuracy} VAL: {avg_vaccuracy}\n')

        if tb_writer is not None:
            tb_writer.add_scalars('Training vs. Validation Loss',
                                  {'Training': avg_loss, 'Validation': avg_vloss},
                                  epoch + 1)
            tb_writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
        if avg_vaccuracy > best_vacc:
            best_vacc = avg_vaccuracy

        metrics = {
            'epoch': epoch,
            'avg_training_loss': avg_loss,
            'avg_validation_loss': avg_vloss,
            'avg_training_accuracy': avg_accuracy,
            'avg_validation_accuracy': avg_vaccuracy,
            'best_validation_accuracy': best_vacc,
            'best_validation_loss': best_vloss
        }
        metrics_summary.append(metrics)

        if early_stopping and early_stopping.stop:
            break

    return metrics_summary
