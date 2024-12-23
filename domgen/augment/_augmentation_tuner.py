import os
import tempfile
from argparse import Namespace
from datetime import datetime

import torch
import albumentations as A
from ray import train, tune
from ray.air import RunConfig, Result
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ruamel.yaml import YAML

from domgen.data import get_dataset
from domgen.models import train_epoch, validate, get_device, get_model, get_criterion, get_optimizer
from domgen.models._tuning import Tuner


class AugmentationTuner(Tuner):
    """
    AugmentationTuner class for fine-tuning data augmentations.
    Integrates into the albumentations pipeline.
    """
    def __init__(self, base_config, tune_config, **kwargs):
        super().__init__(**kwargs)
        yaml = YAML(typ="safe")
        with open(base_config, "r") as f:
            self.base_config = yaml.load(f)
        self.tune_config = tune_config

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
            # Todo: Add **kwargs to get_criterion function
            criterion = get_criterion(conf["criterion"])
            # Todo: Add **kwargs to get_optimizer function
            optimizer = get_optimizer(
                conf["optimizer"], model.parameters(), lr=conf["lr"]
            )
            # Todo: Here, the logic for creating the augmentation that is passed to the get_dataset function is located
            # It should work for the following scenarios:
            # 1. Pass in an entire torchvision.Compose object.
            # 2. Pass in list of possible augmentation steps, then create a Compose object from them
            # 3. Pass in a Compose object but tune the parameters of the augmentation steps.
            augmentation = None


            dataset = get_dataset(
                root_dir=self.base_config["datadir"], name=self.base_config["dataset"],
                test_domain=self.base_config["test_domain"], augmentation=augmentation,
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
        # Todo: Create this function!
        # This function creates the search space dynamically based on the config. It should be able to handle the
        # three scenarios described in the get_trainable function.
        # It should return a suitable representation for the search space. Usually a dictionary-like object.
        pass

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

augmentations = {
    "blur": A.Blur((3,7), 1),
    "channel_dropout": A.ChannelDropout((1, 2), 128, p=1),
    "clahe": A.CLAHE((1,4), (8,8), False, 0.5),
    "color_jitter": A.ColorJitter(0.2, 0.2, 0.2, 0.1, 1),
    "defocus": A.Defocus((4, 8), (0.2, 0.4), True),
    "fancyPCA": A.FancyPCA(0.1, 1, True),
    "glass_blur": A.GlassBlur(0.7, 4, 3, "fast", False, 1),
    "gaussian_noise": A.GaussNoise(std_range=(0.1, 0.2), p=1.),
    "grid_distortion": A.GridDistortion(5, (-0.3, 0.3), p=0.5),
    "grid_dropout": A.GridDropout(ratio=0.3, unit_size_range=(10, 20), random_offset=True, p=1.0),
    "grid_elastic_deform": A.GridElasticDeform(num_grid_xy=(4, 4), magnitude=10, p=1),
    "horizontal_flip": A.HorizontalFlip(p=0.5),
    "hue_saturation_value": A.HueSaturationValue((-20,20), (-30,30), (-20,20), p=0.5),
    "iso_noise": A.ISONoise((0.01,0.05), (0.1,0.5),p=0.2),
    "median_blur": A.MedianBlur(7, 1),
    "pixel_dropout": A.PixelDropout(0.01, False, 0, None, 0.5),
    "random_brightness_contrast": A.RandomBrightnessContrast((-0.2, 0.2), (-0.2,0.2), True, False, p=0.5),
    "random_gamma": A.RandomGamma((80,120), p=0.5),
    "random_resized_crop": A.RandomResizedCrop((512,320), scale=(0.08,1), ratio=(0.75,1.3333333333333333), p=1),
    "random_tone_curve": A.RandomToneCurve(0.1, False, p=0.1),
    "rotate": A.Rotate((-90,90), interpolation=1, border_mode=4, rotate_method='largest_box', crop_border=False, mask_interpolation=0, fill=0, fill_mask=0, p=0.5),
    "sharpen": A.Sharpen((0.2,0.5), (0.5,1), 'kernel', 5, 1, 0.5),
    "shift_scale_rotate": A.ShiftScaleRotate((-0.0625,0.0625), (-0.1,0.1), (-90,90), 1, 4, shift_limit_x=None, shift_limit_y=None, mask_interpolation=0, fill=0, fill_mask=0, p=0.5),
    "solarize": A.Solarize(threshold_range=(0.5,0.5), p=0.5),
    "spatter": A.Spatter((0.65,0.65), (0.3,0.3), (2,2), (0.68,0.68), (0.6,0.6), 'rain', color=None, p=0.5),
    "transpose": A.Transpose(p=0.5),
    "xy_masking": A.XYMasking((1,3), (1,3), (10,100), (10,100), fill=0, fill_mask=0, p=0.5)
}