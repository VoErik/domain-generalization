import os
import shutil
import torch



class EarlyStopping:
    """
    Early stops the training if the monitored metric doesn't improve after a given number of epochs.
    """

    def __init__(self, patience=5, delta=0.01, mode='min'):
        """
        :param patience: Epochs to wait before early stopping.
        :param delta: Minimum improvement to qualify as significant.
        :param mode: 'min' for minimizing (e.g., loss), 'max' for maximizing (e.g., accuracy).
        """
        self.patience = patience
        self.min_delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.is_improvement = (
            lambda new, best: new < best - self.min_delta if mode == 'min' else new > best + self.min_delta
        )

    def __call__(self, score, epoch=None, model=None, optimizer=None, checkpoint_path=None):
        if self.best_score is None or self.is_improvement(score, self.best_score):
            improvement = abs(self.best_score - score) if self.best_score is not None else 0
            print(f'Epoch {epoch}: Metric improved by {improvement:.4f}. New best: {score:.4f}')
            self.best_score = score
            self.counter = 0
            if model and checkpoint_path:
                print(f'Saving best model to {checkpoint_path}')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                    'best_score': self.best_score
                }, checkpoint_path)
        else:
            self.counter += 1
            print(f'Epoch {epoch}: No improvement. Patience counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                print(f'Early stopping at epoch {epoch}.')
                self.stop = True

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.stop = False


def delete_model_dirs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if dirname == "models":
                dir_to_delete = os.path.join(dirpath, dirname)
                shutil.rmtree(dir_to_delete)


