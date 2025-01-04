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