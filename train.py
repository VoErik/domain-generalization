import os
import csv
import torch
import logging
import argparse
import pandas as pd
from ruamel.yaml import YAML

from domgen.data import DOMAIN_NAMES, get_dataset
from domgen.eval import plot_accuracies, plot_training_curves
from domgen.models import train_model, get_model, get_optimizer, get_criterion, get_device, delete_model_dirs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Experiment Logger')


def main(args):
    if args.deterministic:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # --- setup --- #
    if args.config:
        yaml = YAML(typ="safe")
        with open(args.config, "r") as f:
            conf = yaml.load(f)
        args.model = conf["model"]
        args.lr = conf['lr']
        args.batch_size = conf['batch_size']
        args.momentum = conf['momentum']
        args.weight_decay = conf['weight_decay']
        args.epochs = conf['epochs']
        args.eps = conf['eps']
        args.betas = conf['betas']
        args.criterion = conf['criterion']
        args.optimizer = conf['optimizer']
        args.nesterov = conf['nesterov']
        args.use_early_stopping = conf['use_early_stopping']
        args.use_scheduling = conf['use_scheduling']
        args.pretrained = conf['pretrained']
        args.experiment = conf['experiment']
        args.patience = conf['patience']
        args.num_runs = conf['num_runs']
        args.dataset = conf['dataset']

    if args.device:
        device = args.device
    else:
        device = get_device()

    domains = DOMAIN_NAMES[args.dataset]
    test_accuracies = {domain: [] for domain in domains}

    field_names = ['epoch',
                   'avg_training_loss',
                   'avg_validation_loss',
                   'avg_training_accuracy',
                   'avg_validation_accuracy',
                   'best_validation_loss',
                   'best_validation_accuracy']

    # --- loop over different seeds --- #
    for i in range(args.num_runs):
        args.experiment_number = i
        logger.info(f'RUNNING EXPERIMENT {i + 1}/{args.num_runs}')
        logger.info(f'TRAINING ON {device}.\n')
        logger.info('STARTING...')

        # --- start training over domain splits --- #
        for idx, domain in enumerate(domains):
            args.domain_name = domain
            logger.info(f'LEAVE OUT {domain}.')
            logger.info(f'TRAINING FOR {args.epochs} EPOCHS.')

            # --- setup --- #
            dataset = get_dataset(
                name=args.dataset,
                root_dir=args.dataset_dir,
                test_domain=idx
            )

            train, val, test = dataset.generate_loaders(
                batch_size=args.batch_size
            )

            model = get_model(
                model_name=args.model,
                num_classes=dataset.num_classes
            ).to(device)

            criterion = get_criterion(
                criterion_name=args.criterion
            )

            optimizer = get_optimizer(
                optimizer_name=args.optimizer,
                model_parameters=model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                betas=args.betas,
                weight_decay=args.weight_decay,
                nesterov=args.nesterov,
                eps=args.eps,
            )

            # --- actual training happens here --- #
            training_metrics, test_metrics = train_model(
                args=args,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train,
                val_loader=val,
                test_loader=test,
                device=device
            )

            logger.info(f'TEST LOSS: {test_metrics["Test Loss"]}')
            logger.info(f'TEST ACCURACY: {test_metrics["Test Accuracy"]}\n')

            # --- log metrics to files --- #
            experiment_dir = os.path.join(f'{args.log_dir}/{args.experiment}/run_{i}/' + domain)
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)
            with open(f'{experiment_dir}/training_metrics.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()
                writer.writerows(training_metrics)
            with open(f'{experiment_dir}/test_metrics.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['Test Loss', 'Test Accuracy'])
                writer.writeheader()
                writer.writerows([test_metrics])

            test_accuracies[domain].append(test_metrics['Test Accuracy'])

        logger.info('FINISHED...')

    # --- extract averages and log them to file --- #
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
        'avg_acc': average_accuracies,
        'worst_case_acc': worst_case_accuracies,
        'best_case_acc': best_case_accuracies
    })
    df.to_csv(f'{args.log_dir}/{args.experiment}/results.csv', index=False)
    general_average_accuracy = df['avg_acc'].mean()
    overall_worst_case_performance = df['worst_case_acc'].min()
    overall_best_case_performance = df['best_case_acc'].max()

    logger.info('Metrics per Domain:')
    logger.info(f'General Average Accuracy: {general_average_accuracy}')
    logger.info(f'Overall Worst Case Performance: {overall_worst_case_performance}')
    logger.info(f'Overall Best Case Performance: {overall_best_case_performance}')
    logger.info(f'Saving plots to {args.log_dir}/{args.experiment}/plots/')

    # --- create plots --- #
    plot_training_curves(f'{args.log_dir}/{args.experiment}/', show=False)
    plot_accuracies(f'{args.log_dir}/{args.experiment}/results.csv', show=False)

    # --- clean up --- #
    if args.delete_checkpoints:
        directory = f'{args.log_dir}/{args.experiment}'
        if os.path.isdir(directory):
            delete_model_dirs(directory)
            logger.info("Completed deleting all 'model' directories.")
        else:
            logger.info("The provided path is not a valid directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='datasets', help='path to datasets')
    parser.add_argument('--dataset', type=str, default='PACS', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='loss criterion')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer name')
    parser.add_argument('--patience', type=int, default=5, help='patience for lr scheduling and early stopping')
    parser.add_argument('--use_scheduling', action='store_true', default=True, help='use scheduling')
    parser.add_argument('--use_early_stopping', action='store_true', default=True, help='use early stopping')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs per split')
    parser.add_argument('--device', type=str, help='Device to train on')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--deterministic', action='store_true', default=False, help='use seed or not')
    parser.add_argument('--experiment', type=str, default='exp', help='dir of the experiment')
    parser.add_argument('--log_dir', type=str, default='experiments', help='log directory')
    parser.add_argument('--model', type=str, default='resnet18', help='base model')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pretrained model')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs per experiment')
    parser.add_argument('--silent', action='store_true', default=False, help='silent mode')
    parser.add_argument('--config', type=str, default=None, help='config file')
    parser.add_argument('--delete_checkpoints', action='store_true', default=True, help='delete checkpoints')
    args = parser.parse_args()

    main(args)
