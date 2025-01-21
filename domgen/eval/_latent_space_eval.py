from platform import architecture

import numpy as np
from matplotlib import colors, pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


def normalize_features(features):
    """
    Normalizes features using z-score normalization (mean=0, std=1).

    :param features: Input feature matrix.
    :return: Normalized features.
    """
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)

def get_style_statistics_with_reduction(
        model,
        dataloader,
        device,
        reduction='avg_pool',  # or 'flatten'
        n_components=2,
        architecture: str = 'resnet'
):
    model.eval()
    features = {f'block_{i + 1}': [] for i in range(4)}
    clslabels = []
    domlabels = []

    # Define block layers (resnet or densenet)
    if architecture == 'resnet':
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    if architecture == 'densenet':
        layer_names = ['DenseBlock_1', 'DenseBlock_2', 'DenseBlock_3', 'DenseBlock_4']
    last_layers = [list(getattr(model, name).children())[-1] for name in layer_names]

    def get_hook(block_name):
        def hook(module, input, output):
            if reduction == 'avg_pool':
                pooled = F.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)  # Shape (batch_size, num_channels)
            elif reduction == 'flatten':
                pooled = output.view(output.size(0), -1)  # Flatten to (batch_size, num_flattened_features)
            else:
                raise ValueError("Invalid reduction method. Use 'avg_pool' or 'flatten'.")
            features[block_name].append(pooled.detach().cpu())

        return hook

    # Register hooks
    hooks = []
    for i, last_layer in enumerate(last_layers):
        block_name = f'block_{i + 1}'
        hooks.append(last_layer.register_forward_hook(get_hook(block_name)))

    # Process data
    with torch.no_grad():
        for inputs, clstargets, domtargets in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            clslabels.append(clstargets)
            domlabels.append(domtargets)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Concatenate and reduce dimensionality
    reduced_features = {}
    for block, feats in features.items():
        feats = torch.cat(feats, dim=0).numpy()  # Concatenate features

        # Normalize features
        feats = normalize_features(feats)

        if reduction == 'flatten' and n_components is not None:
            pca = PCA(n_components=50)
            tsne = TSNE(n_components=n_components)
            feats = pca.fit_transform(feats)
            feats = tsne.fit_transform(feats)
        reduced_features[block] = feats

    clslabels = torch.cat(clslabels, dim=0).numpy()
    domlabels = torch.cat(domlabels, dim=0).numpy()
    return reduced_features, clslabels, domlabels


def get_features_with_reduction(
        model,
        dataloader,
        device,
        reduction='avg_pool',
        n_components=2,
        architecture: str = 'resnet'
):
    model.eval()
    features = {f'block_{i+1}': [] for i in range(4)}
    clslabels = []
    domlabels = []

    if architecture == 'resnet':
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    if architecture == 'densenet':
        layer_names = ['DenseBlock_1', 'DenseBlock_2', 'DenseBlock_3', 'DenseBlock_4']
    last_layers = [list(getattr(model, name).children())[-1] for name in layer_names]

    def get_hook(block_name):
        def hook(module, input,output):
            if reduction == 'avg_pool':
                pooled = F.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)
            elif reduction == 'flatten':
                pooled = output.view(output.size(0), -1)
            else:
                raise ValueError("Invalid reduction method. Use 'avg_pool' or 'flatten'.")
            features[block_name].append(pooled.detach().cpu())
        return hook

    hooks = []
    for i, last_layer in enumerate(last_layers):
        block_name = f'block_{i+1}'
        hooks.append(last_layer.register_forward_hook(get_hook(block_name)))

    with torch.no_grad():
        for inputs, clstargets, domtargets in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            clslabels.append(clstargets)
            domlabels.append(domtargets)

    for hook in hooks:
        hook.remove()

    # Concatenate and reduce dimensionality
    reduced_features = {}
    for block, feats in features.items():
        feats = torch.cat(feats, dim=0).numpy()
        if reduction == 'flatten' and n_components is not None:
            reducer = PCA(n_components=n_components) if n_components > 2 else TSNE(n_components=n_components)
            feats = reducer.fit_transform(feats)
        reduced_features[block] = feats

    clslabels = torch.cat(clslabels, dim=0).numpy()
    domlabels = torch.cat(domlabels, dim=0).numpy()
    return reduced_features, clslabels, domlabels


def reduce_dimensionality(
        features,
        method='pca',
        n_components=2
):
    """
    Reduces the dimensionality of features using PCA or t-SNE.

    :param features: High-dimensional features.
    :param method: Dimensionality reduction method ('pca' or 'tsne').
    :param n_components: Number of dimensions to reduce to (2 or 3).
    :return: Reduced features.
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    reduced_features = reducer.fit_transform(features)
    return reduced_features


def reduce_features_by_block(
        features_dict,
        method='pca',
        n_components=2
):
    """
    Reduces dimensionality of features from each block.

    :param features_dict: Dictionary of features from each block.
    :param method: Dimensionality reduction method ('pca' or 'tsne').
    :param n_components: Number of dimensions to reduce to (2 or 3).
    :return: Dictionary of reduced features.
    """
    reduced_features = {}
    for block_name, features in features_dict.items():
        print(features.shape)
        reduced_features[block_name] = reduce_dimensionality(features, method, n_components)
    return reduced_features


def visualize_features_by_block(
        reduced_features,
        labels,
        savepath: str = '.',
        show: bool = False,
        labelmap: dict = None,
):
    """
    Visualizes reduced features from each block in a grid of subplots.

    :param reduced_features: Dictionary of reduced features from each block.
    :param labels: Corresponding labels for coloring.
    """
    num_blocks = len(reduced_features)
    fig, axes = plt.subplots(1, num_blocks, figsize=(5 * num_blocks, 5))

    unique_labels = np.unique(labels)
    num_colors = len(unique_labels)

    cmap = plt.get_cmap('tab10')  # 'tab10' supports up to 10 unique values
    norm = colors.BoundaryNorm(boundaries=np.arange(num_colors + 1) - 0.5, ncolors=num_colors)

    for i, (block_name, features) in enumerate(reduced_features.items()):
        ax = axes[i] if num_blocks > 1 else axes
        scatter = ax.scatter(features[:, 0], features[:, 1], c=labels, cmap=cmap, norm=norm, alpha=0.7)
        ax.set_title(f'{block_name} Features')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True)

    cbar = plt.colorbar(scatter, ax=axes, orientation='horizontal', pad=0.15)

    cbar.set_ticks(np.arange(num_colors))  # Set the number of ticks
    cbar.set_ticklabels([f'{labelmap[label.item()]}' for label in unique_labels])  # Set the tick labels
    cbar.set_label('Domains')  # Colorbar label

    plt.savefig(f'{savepath}/latentspace.png', dpi=300)
    if show:
        plt.show()

