from domgen.augment._strategies import AUG_STRATEGIES

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
