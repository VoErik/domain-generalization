class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given number of epochs (patience).'
    """
    def __init__(self, patience=5, delta=0):
        """
        Initializes the EarlyStopping class.
        :param patience: Epochs to wait before early stopping.
        :param delta: Minimum improvement between two consecutive epochs to count as general improvement.
        """
        self.patience = patience
        self.min_delta = delta
        self.counter = 0
        self.best_loss = None
        self.stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        elif self.best_loss - loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'Early stopping after {self.counter} epochs.')
                self.stop = True
