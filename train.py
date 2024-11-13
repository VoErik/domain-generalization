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

from domgen.data import PACSData
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


def training(model: torch.nn.Module,
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

        # Save model with best validation loss
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'{EXPERIMENT_DIR}/model_{timestamp}_{epoch}.pth'
            torch.save(model.state_dict(), model_path)

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


def testing(model: torch.nn.Module,
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


# TODO factor out into main() function and add either config file or argparser
if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    DATASET = PACSData('datasets/PACS')  # TODO: implement get dataset function
    DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']  # TODO: implement get test domains function
    TRAIN_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 32
    NUM_EPOCHS = 3
    NUM_EXPERIMENTS = 3
    EXPERIMENT_NAME = 'First'

    # CRITERION
    # OPTIMIZER
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    # LR SCHEDULER # TODO: implement LR scheduling
    field_names = ['epoch',
                   'avg_training_loss',
                   'avg_validation_loss',
                   'avg_training_accuracy',
                   'avg_validation_accuracy',
                   'best_validation_loss']
    test_accuracies = {DOMAIN: [] for DOMAIN in DOMAINS}
    for i in range(NUM_EXPERIMENTS):
        logger.info(f'RUNNING EXPERIMENT {i + 1}/{NUM_EXPERIMENTS}')
        logger.info(f'TRAINING ON {DEVICE}.\n')
        logger.info('STARTING...')
        for DOMAIN in DOMAINS:
            EXPERIMENT_DIR = os.path.join(f'experiments/{EXPERIMENT_NAME}/run_{i}/' + DOMAIN)
            if not os.path.exists(EXPERIMENT_DIR):
                os.makedirs(EXPERIMENT_DIR)

            train, val, test = DATASET.create_dataloaders(test_domain=DOMAIN,
                                                          train_batch_size=TRAIN_BATCH_SIZE,
                                                          val_batch_size=VAL_BATCH_SIZE)

            # TODO: swap with own implementation?
            model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT, progress=True).to(DEVICE)
            # TODO: get criterion & optimizer functions?
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

            logger.info(f'TRAINING MODEL ON EVERYTHING BUT {DOMAIN}.')
            logger.info(f'TRAINING FOR {NUM_EPOCHS} EPOCHS.')
            training_metrics = training(model=model,
                                        optimizer=optimizer,
                                        criterion=criterion,
                                        num_epochs=NUM_EPOCHS,
                                        domain_name=DOMAIN,
                                        device=DEVICE,
                                        train_loader=train,
                                        val_loader=val,
                                        experiment_name=EXPERIMENT_NAME,
                                        experiment_number=i)

            logger.info(f'TESTING MODEL ON {DOMAIN}.')
            test_metrics = testing(model=model,
                                   criterion=criterion,
                                   device=DEVICE,
                                   dataloader=test,
                                   domain_name=DOMAIN,
                                   experiment_name=EXPERIMENT_NAME,
                                   experiment_number=i)
            logger.info(f'TEST LOSS: {test_metrics["Test Loss"]}')
            logger.info(f'TEST ACCURACY: {test_metrics["Test Accuracy"]}\n')

            with open(f'{EXPERIMENT_DIR}/training_metrics.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()
                writer.writerows(training_metrics)
            with open(f'{EXPERIMENT_DIR}/test_metrics.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['Test Loss', 'Test Accuracy'])
                writer.writeheader()
                writer.writerows([test_metrics])
            test_accuracies[DOMAIN].append(test_metrics['Test Accuracy'])
        logger.info('FINISHED...')

    domain_names = []
    average_accuracies = []
    worst_case_accuracies = []
    best_case_accuracies = []

    for domain, acc_list in test_accuracies.items():
        domain_names.append(domain)
        average_accuracies.append(sum(acc_list) / len(acc_list) if acc_list else 0)
        worst_case_accuracies.append(min(acc_list) if acc_list else None)
        best_case_accuracies.append(max(acc_list) if acc_list else None)

    df = pd.DataFrame({
        'Domain': domain_names,
        'Average Accuracy': average_accuracies,
        'Worst Case Accuracy': worst_case_accuracies,
        'Best Case Accuracy': best_case_accuracies
    })
    df.to_csv(f'experiments/{EXPERIMENT_NAME}/results.csv', index=False)
    general_average_accuracy = df['Average Accuracy'].mean()
    overall_worst_case_performance = df['Worst Case Accuracy'].min()

    logger.info('Metrics per Domain:')
    pprint(df)
    logger.info(f'General Average Accuracy: {general_average_accuracy}')
    logger.info(f'Overall Worst Case Performance: {overall_worst_case_performance}')
    logger.info(f'Saving plots to {EXPERIMENT_NAME}/plots/')
    plot_training_curves(f'experiments/{EXPERIMENT_NAME}/results.csv')
    plot_accuracies(f'experiments/{EXPERIMENT_NAME}/')
