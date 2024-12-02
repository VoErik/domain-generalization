from domgen.data import PACS, DOMAIN_NAMES, get_dataset
from domgen.models import train_epoch, validate, get_model, get_optimizer, get_criterion, get_device

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler

from functools import partial
import os
import tempfile
import torch


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

            train.report({"mean_accuracy": acc}, checkpoint=checkpoint)


def main():
    search_space = {
        "lr": tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        "momentum": tune.uniform(0.1, 0.95),
        "batch_size": tune.choice([8, 16, 32, 64, 128]),
        "weight_decay": tune.uniform(1e-6, 1e-2),
        "criterion": tune.choice(["cross_entropy", "nll"]),
        "optimizer": tune.choice(["sgd", "adam", "adamw"]),
        "betas": tune.choice([(0.9, 0.999), (0.8, 0.888), (0.85, 0.95)]),  # For Adam/AdamW
        "eps": tune.choice([1e-8, 1e-7, 1e-6]),  # For Adam/AdamW
        "nesterov": tune.choice([True, False]),  # For SGD
    }

    base_config = {
        "dataset": 'PACS',
        "datadir": os.path.abspath("./datasets"),
        "num_classes": 7,
        "test_domain": 0,
        "model": 'resnet18',
        "num_epochs": 5,
    }
    tune_params_with_config = partial(tune_params, base_config=base_config)
    tuner = tune.Tuner(
        partial(tune_params_with_config),
        tune_config=tune.TuneConfig(
            num_samples=10,
            scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    dfs = {result.path: result.metrics_dataframe for result in results}
    df = results.get_dataframe()
    df.to_csv("results.csv")


if __name__ == '__main__':
    main()
