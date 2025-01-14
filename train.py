import argparse
import albumentations as A
from types import SimpleNamespace
from domgen.models import DomGenTrainer
from domgen.eval import plot_accuracies, plot_training_curves
from domgen.utils import config_to_namespace, merge_namespace

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='path to datasets')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--criterion', type=str, help='loss criterion')
    parser.add_argument('--optimizer', type=str, help='optimizer name')
    parser.add_argument('--patience', type=int, help='patience for lr scheduling and early stopping')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--momentum', type=float, help='momentum')
    parser.add_argument('--epochs', type=int, help='number of epochs per split')
    parser.add_argument('--device', type=str, help='Device to train on')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--deterministic', action='store_true', default=False, help='use seed or not')
    parser.add_argument('--experiment', type=str, help='dir of the experiment')
    parser.add_argument('--log_dir', type=str, help='log directory')
    parser.add_argument('--model', type=str, help='base model')
    parser.add_argument('--num_runs', type=int, help='Number of runs per experiment')
    parser.add_argument('--silent', action='store_true', help='silent mode')
    parser.add_argument('--config', type=str, default=None, help='config file')

    # --- get training arguments from either cmd line or config file (yaml / json) --- #
    cmd_args = parser.parse_args()
    config_namespace = SimpleNamespace()
    if cmd_args.config is not None:
        config_namespace = config_to_namespace(cmd_args.config)
    args = merge_namespace(config_namespace, cmd_args)  # cmd line args take precedence
    experiment_path = f'{args.log_dir}/{args.experiment}'

    # --- load augment based on configuration --> currently dummy --- #
    augment = {"hflip": A.HorizontalFlip(p=1), "noop": A.NoOp(p=1)}

    # --- train --- #
    trainer = DomGenTrainer(
        model=args.model,
        optimizer=args.optimizer,
        criterion=args.criterion,
        dataset=args.dataset,
        epochs_per_experiment=args.epochs,
        log_dir=args.log_dir,
        experiment=args.experiment,
        checkpoint_dir=f'{args.experiment}/checkpoints',
        lr=float(args.lr),
        pretrained=args.pretrained,
        momentum=args.momentum,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_experiments=args.num_runs,
        mixstyle_layers=args.mixstyle_layers,
        mixstyle_alpha=args.mixstyle_alpha,
        mixstyle_p=args.mixstyle_p,
        mix_type=args.mix_type,
        augment=augment,
        visualize_latent=args.visualize_latent,
        patience=args.patience,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
    )
    trainer.fit()

    # --- save, plot, etc --- #
    trainer.save_metrics(trainer.metrics, experiment_path)
    plot_accuracies(root_path=experiment_path, save=True, show=False)
    plot_training_curves(base_dir=experiment_path, show=False)
    trainer.save_config(f'{experiment_path}/trainer_config.json')
