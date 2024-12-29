import os
import tempfile
from argparse import Namespace
from datetime import datetime

import torch
from ray import train, tune
from ray.air import RunConfig, Result
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ruamel.yaml import YAML
from domgen.data import get_dataset
from domgen.models import get_model, get_criterion, get_optimizer, get_device, train_epoch, validate


class Tuner:
    """
    Base tuner class for both hyperparameter tuning and augmentation tuning.
    :param trial_dir: Directory to save the trials in. Default: trials.
    :param num_gpu: Number of GPUs to use. Can be fractional. Default: 1.
    :param num_trials: Number of trials to run. Default: 20.
    :param max_concurrent: Maximum number of concurrent trials to run. Default: 20.
    :param silent: Suppresses some logging activity. Default: True.
    :param mode: Evaluation mode. Default: max.
    """
    def __init__(
            self,
            trial_dir: str = './experiments/trials',
            num_gpu: int | float = 1,
            num_trials: int = 20,
            max_concurrent: int = 20,
            silent: bool = True,
            mode: str = 'max',
            **kwargs,
    ):
        self.trial_dir = trial_dir
        self.num_gpu = num_gpu
        self.num_trials = num_trials
        self.max_concurrent = max_concurrent
        self.silent = silent
        self.mode = mode
        self.additional_config = kwargs

    def get_trainable(self):
        raise NotImplementedError('You need to implement get_trainable()')

    def run(self):
        raise NotImplementedError('You need to implement tune()')


class ParamTuner(Tuner):
    def __init__(self, base_config, tune_config, **kwargs):
        super().__init__(**kwargs)
        yaml = YAML(typ="safe")
        with open(base_config, "r") as f:
            self.base_config = yaml.load(f)
        self.tune_config = tune_config


    def tune_params(self, model, criterion, optimizer, train_loader, val_loader, device):
        """
        Function to handle training and validation steps.
        """
        args = Namespace()
        args.use_scheduling = False
        args.silent = self.silent
        for i in range(self.base_config["num_epochs"]):
            train_epoch(
                model=model, optimizer=optimizer, criterion=criterion,
                train_loader=train_loader, device=device, args=args
            )
            val_loss, acc = validate(args, model, val_loader, criterion, device)

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                if (i + 1) % 5 == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(temp_checkpoint_dir, "model.pth")
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                train.report({"mean_accuracy": acc, "val_loss": val_loss}, checkpoint=checkpoint)

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
            criterion = get_criterion(conf["criterion"])
            optimizer = get_optimizer(
                conf["optimizer"], model.parameters(), lr=conf["lr"]
            )
            dataset = get_dataset(
                root_dir=self.base_config["datadir"], name=self.base_config["dataset"],
                test_domain=self.base_config["test_domain"]
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
        yaml = YAML(typ="safe")
        with open(self.tune_config, "r") as f:
            search_space = yaml.load(f)

        ray_search_space = {}
        for key, value in search_space.items():
            if isinstance(value, dict):
                # ranges or distributions
                if "min" in value and "max" in value:
                    if value.get("type") == "loguniform":
                        ray_search_space[key] = tune.loguniform(float(value["min"]), float(value["max"]))
                    elif value.get("type") == "uniform":
                        ray_search_space[key] = tune.uniform(float(value["min"]), float(value["max"]))
                elif "choices" in value:
                    ray_search_space[key] = tune.choice(value["choices"])
            elif isinstance(value, list):
                # choices
                ray_search_space[key] = tune.choice(value)
            else:
                raise ValueError(f"Unsupported format for search space entry: {key} -> {value}")

        return ray_search_space