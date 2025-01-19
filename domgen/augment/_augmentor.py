from torchvision.transforms import InterpolationMode, v2
from domgen.augment import create_augmentation_pipeline
from domgen.augment._transforms import PACS_CUSTOM, TransformsWrapper

from medmnistc.augmentation import AugMedMNISTC
from medmnistc.corruptions.registry import CORRUPTIONS_DS
import torchvision.transforms as transforms


class Strategy:
    """Base class for augmentation strategies."""

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class MixUpStrategy(Strategy):
    def __init__(self, **kwargs):
        super().__init__()
        self.alpha = kwargs.get("alpha", 0.3)
        self.num_classes = kwargs.get("classes", 7)
        self.strat = v2.MixUp(alpha=self.alpha, num_classes=self.num_classes)

    def __call__(self, img, labels):
        return self.strat(img, labels)


class CutMixStrategy(Strategy):
    def __init__(self, **kwargs):
        super().__init__()
        self.alpha = kwargs.get("alpha", 1.0)
        self.num_classes = kwargs.get("classes", 7)
        self.strat = v2.CutMix(alpha=self.alpha, num_classes=self.num_classes)

    def __call__(self, img, labels):
        return self.strat(img, labels)


class AugMixStrategy(Strategy):
    def __init__(self, **kwargs):
        super().__init__()
        self.severity = kwargs.get("severity", 3)
        self.mixture_width = kwargs.get("mixture_width", 3)
        self.chain_depth = kwargs.get("chain_depth", -1)
        self.alpha = kwargs.get("alpha", 1.0)
        self.all_ops = kwargs.get("all_ops", False)
        self.interpolation = kwargs.get("interpolation", InterpolationMode.NEAREST)
        self.fill = kwargs.get("fill", 0)
        self.strat = v2.AugMix(
            alpha=self.alpha,
            severity=self.severity,
            mixture_width=self.mixture_width,
            chain_depth=self.chain_depth,
            interpolation=self.interpolation,
            fill=self.fill,
            all_ops=self.all_ops,
        )

    def __call__(self, img, labels):
        return self.strat(img), labels


class RandAugmentStrategy(Strategy):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_ops = kwargs.get("num_ops", 2)
        self.magnitude = kwargs.get("magnitude", 9)
        self.num_magnitude_bins = kwargs.get("num_magnitude_bins", 31)
        self.interpolation = kwargs.get("interpolation", InterpolationMode.NEAREST)
        self.fill = kwargs.get("fill", 0)
        self.strat = v2.RandAugment(
            num_ops=self.num_ops,
            magnitude=self.magnitude,
            num_magnitude_bins=self.num_magnitude_bins,
            interpolation=self.interpolation,
            fill=self.fill,
        )

    def __call__(self, img, labels):
        return self.strat(img), labels


class MixStyleStrategy(Strategy):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def __call__(self, img, labels):
        return img, labels

class NoAugment(Strategy):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def __call__(self, img, labels):
        return img, labels

class CustomAugment(Strategy):
    def __init__(self, **kwargs):
        super().__init__()
        self.aug_dict = kwargs.get("aug_dict", {})
        self.strat = create_augmentation_pipeline(self.aug_dict)

    def __call__(self, img, labels):
        return img, labels

class PACSCustom(Strategy):
    augments = PACS_CUSTOM
    def __init__(self, **kwargs):
        super().__init__()
        self.aug_dict = PACS_CUSTOM[kwargs.get("aug_dict", {})]
        self.strat = create_augmentation_pipeline(self.aug_dict)

    def __call__(self, img, labels):
        return img, labels

class MedMNISTC(Strategy):
    """
    Wrapper class for MedMNISTC.
    DiSalvo et al. 2024: https://github.com/francescodisalvo05/medmnistc-api/tree/main
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = kwargs.get("aug_dict", "pathmnist")
        self.augments = CORRUPTIONS_DS[self.dataset]
        self.strat = TransformsWrapper(torchvision_transform=transforms.Compose([
            AugMedMNISTC(self.augments),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]))

    def __call__(self, img, labels):
        return img, labels

AUG_STRATEGIES = {
    "no_augment": NoAugment,
    "mixup": MixUpStrategy,
    "cutmix": CutMixStrategy,
    "augmix": AugMixStrategy,
    "randaugment": RandAugmentStrategy,
    "mixstyle": MixStyleStrategy,
    "pacs_custom": PACSCustom,
    "medmnistc": MedMNISTC
}

class Augmentor:
    """Handler class for augmentation strategies."""
    strategy_name = None
    strategy = None

    def __init__(self, strategy_name, strategies=AUG_STRATEGIES, **kwargs):
        self.strategy_name = strategy_name.lower()
        self.strategies = strategies
        self.config = kwargs
        self.get_strategy()

    def get_strategy(self):
        if self.strategy_name is None or self.strategy_name not in self.strategies:
            raise AttributeError(f"Strategy name '{self.strategy_name}' is not defined."
                                 f" Supported strategies are: {self.strategies}.")
        self.strategy = self.strategies[self.strategy_name](**self.config)

    def apply_augment(self, img, labels):
        return self.strategy(img, labels)




