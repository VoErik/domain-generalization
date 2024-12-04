import argparse
from datetime import datetime

from ray.air import RunConfig

from domgen.data import get_dataset
from domgen.models import train_epoch, validate, get_model, get_optimizer, get_criterion, get_device

from ray import tune
from ray import train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from functools import partial
import os
import tempfile
import torch
import yaml

def tune_augments(augment_config, base_config):
    pass


def tune_params(hp_config, base_config):
    device = get_device()
    model = get_model(model_name=base_config["model"], num_classes=base_config["num_classes"]).to(device)
    criterion = get_criterion(hp_config["criterion"])
    optimizer = get_optimizer(hp_config["optimizer"], model.parameters(),
                              lr=hp_config["lr"], momentum=hp_config["momentum"])
    dataset = get_dataset(
        root_dir=base_config["datadir"],
        name=base_config["dataset"],
        test_domain=base_config["test_domain"]
    )
    train_loader, val_loader, _ = dataset.generate_loaders(batch_size=hp_config["batch_size"])

    for i in range(base_config["num_epochs"]):
        train_epoch(model, optimizer, criterion, train_loader, device)
        val_loss, acc = validate(model, val_loader, criterion, device)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if (i + 1) % 5 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(temp_checkpoint_dir, "model.pth")
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            train.report({"mean_accuracy": acc, "val_loss": val_loss},  checkpoint=checkpoint)


def tune_hyperparameters(conf) -> None:
    with open(conf.hp_config, "r") as f:
        search_space = yaml.safe_load(f)

    with open(conf.data_config, "r") as f:
        base_config = yaml.safe_load(f)

    trial_dir = conf.trial_dir
    trial_name = f'{base_config["model"]}-{base_config["test_domain"]}-{datetime.now().strftime('%Y%m%d_%H%M%S')}'

    search_space = {
        "lr": tune.choice(search_space["lr"]),
        "momentum": tune.uniform(search_space["momentum"]["min"], search_space["momentum"]["max"]),
        "batch_size": tune.choice(search_space["batch_size"]),
        "weight_decay": tune.uniform(search_space["weight_decay"]["min"], search_space["weight_decay"]["max"]),
        "criterion": tune.choice(search_space["criterion"]),
        "optimizer": tune.choice(search_space["optimizer"]),
        "betas": tune.choice(search_space["betas"]),
        "eps": tune.choice(search_space["eps"]),
        "nesterov": tune.choice(search_space["nesterov"]),
    }

    base_config["datadir"] = os.path.abspath(base_config["datadir"])

    tune_params_with_config = partial(tune_params, base_config=base_config)
    trainable_with_gpu = tune.with_resources(tune_params_with_config, {"gpu": conf.num_gpu})
    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            num_samples=conf.num_samples,
            scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
            max_concurrent_trials=conf.max_concurrent
        ),
        run_config=RunConfig(
            storage_path=trial_dir,
            name=trial_name),
        param_space=search_space,
    )
    results = tuner.fit()

    df = results.get_dataframe()
    df.to_csv(f"{trial_dir}/{trial_name}/results.csv")
    best = results.get_best_result(metric="mean_accuracy", mode="max")
    print(best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['hp', 'augment'])
    parser.add_argument("--hp_config", default="./config/hp/hp_search_space.yaml")
    parser.add_argument("--data_config", default="./config/base/pacs-resnet18-pretrained-td=0.yaml")
    parser.add_argument("--augment_config", default="")
    parser.add_argument("--num_samples", default=20, type=int)
    parser.add_argument("--max_concurrent", default=5, type=int)
    parser.add_argument("--trial_dir", default="./experiments/trials", type=str)
    parser.add_argument("--num_gpu", default=0.5, type=float)
    args = parser.parse_args()

    if args.mode == 'hp':
        tune_hyperparameters(args)
    if args.mode == 'augment':
        tune_hyperparameters(args)

