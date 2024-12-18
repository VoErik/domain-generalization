import os
import shutil


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given number of epochs (patience).
    """
    def __init__(self, patience=5, delta=0.01):
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
            print('Loss did not improve.')
            self.counter += 1
            if self.counter >= self.patience:
                print(f'Early stopping after {self.counter} epochs.')
                self.stop = True


def delete_model_dirs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if dirname == "models":
                dir_to_delete = os.path.join(dirpath, dirname)
                shutil.rmtree(dir_to_delete)