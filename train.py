import argparse
import albumentations as A
from types import SimpleNamespace
from domgen.models import DomGenTrainer
from domgen.eval import plot_accuracies, plot_training_curves
from domgen.utils import config_to_namespace, merge_namespace

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
    parser.add_argument('--model', type=str, default='ResNet18', help='base model')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pretrained model')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs per experiment')
    parser.add_argument('--silent', action='store_true', default=False, help='silent mode')
    parser.add_argument('--config', type=str, default=None, help='config file')
    parser.add_argument('--delete_checkpoints', action='store_true', default=True, help='delete checkpoints')

    # --- get training arguments from either cmd line or config file (yaml / json) --- #
    cmd_args = parser.parse_args()
    config_namespace = SimpleNamespace()
    if cmd_args.config is not None:
        config_namespace = config_to_namespace(cmd_args.config)
    args = merge_namespace(config_namespace, cmd_args)  # cmd line args take precedence
    experiment_path = f'{args.log_dir}/{args.experiment}'

    # --- load augment based on configuration --> currently dummy --- #
    augment = {'flip': A.HorizontalFlip(p=1), 'noop': A.NoOp()}

    # --- train --- #
    trainer = DomGenTrainer(
        model=args.model,
        optimizer=args.optimizer,
        criterion=args.criterion,
        dataset=args.dataset,
        epochs_per_experiment=1,
        log_dir=args.log_dir,
        checkpoint_dir=f'{experiment_path}/checkpoints',
        lr=args.lr,
        pretrained=args.pretrained,
        momentum=args.momentum,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_experiments=1,
        augment=augment
    )
    trainer.fit()

    # --- save, plot, etc --- #
    trainer.save_metrics(trainer.metrics, experiment_path)
    plot_accuracies(root_path=experiment_path, save=True, show=False)
    plot_training_curves(base_dir=experiment_path, show=False)
    trainer.save_config(f'{experiment_path}/trainer_config.json')
