import torch
import torch.optim as optim
import torch.nn as nn
from domgen.models import resnet18, resnet34, resnet50, resnet101, resnet152, densenet121, densenet169, densenet201


def get_optimizer(
        optimizer_name: str,
        model_parameters,
        **kwargs
) -> optim.Optimizer:
    """
    Returns a PyTorch optimizer based on the given name.

    :param optimizer_name: Name of the optimizer (e.g., 'adam', 'sgd').
    :param model_parameters: Parameters of the model to optimize.

    :returns: Optimizer: PyTorch optimizer object.
    """
    optimizer_name = optimizer_name.lower()
    optimizer_args = {
        'sgd': ['lr', 'momentum', 'weight_decay', 'nesterov'],
        'adam': ['lr', 'betas', 'eps'],
        'adamw': ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad'],
    }

    if optimizer_name not in optimizer_args:
        raise ValueError(f"Optimizer '{optimizer_name}' not recognized. "
                         f"Available options: {list(optimizer_args.keys())}")

    valid_args = {key: kwargs[key] for key in optimizer_args[optimizer_name] if key in kwargs}

    if optimizer_name == 'sgd':
        return torch.optim.SGD(model_parameters, **valid_args)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(model_parameters, **valid_args)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(model_parameters, **valid_args)



def get_criterion(
        criterion_name: str,
        **kwargs
) -> nn.Module:
    """
    Returns a PyTorch criterion based on the given name.

    :param criterion_name: Name of the criterion (e.g., 'cross_entropy').
    :return: Criterion: PyTorch loss function.
    """
    criteria = {
        'cross_entropy': nn.CrossEntropyLoss,
        'nll': nn.NLLLoss,
        'bce_with_logits': nn.BCEWithLogitsLoss,  # For binary classification
    }

    criterion_name = criterion_name.lower()
    if criterion_name in criteria:
        return criteria[criterion_name](**kwargs)
    else:
        raise ValueError(f"Criterion '{criterion_name}' not recognized. "
                         f"Available options: {list(criteria.keys())}")


def get_model(
        model_name: str,
        num_classes: int,
        **kwargs
) -> nn.Module:
    """
    Returns a PyTorch model based on the given name.

    :param model_name: Name of the model (e.g., 'resnet18', 'densenet121').
    :param num_classes: Number of output classes for classification.
    :returns: model: PyTorch model object.
    """

    models_dict = {
        # ResNet Variants
        'ResNet18': resnet18,
        'ResNet34': resnet34,
        'ResNet50': resnet50,
        'ResNet101': resnet101,
        'ResNet152': resnet152,
        # DenseNet Variants
        'DenseNet121': densenet121,
        'DenseNet161': densenet121,
        'DenseNet169': densenet169,
        'DenseNet201': densenet201,
    }

    if model_name not in models_dict:
        raise ValueError(f"Model '{model_name}' not recognized. "
                         f"Available options: {list(models_dict.keys())}")

    print(kwargs)
    model = models_dict[model_name](
        num_classes=num_classes,
        **kwargs
    )
    return model


def get_device():
    device = 'cuda' \
        if torch.cuda.is_available() \
        else 'mps' \
        if torch.backends.mps.is_available() \
        else 'cpu'
    return device

