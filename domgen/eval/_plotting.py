import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pprint import pprint
from typing import Dict, List


def plot_accuracies(path: str, save: bool = True, cmap: str = 'cividis') -> None:
    """Creates bar plots of accuracies across domains for one experiment.
    :param path: path to csv file.
    :param save: whether to save the figure. Default is True.
    :param cmap: matplotlib colormap. Careful when changing the colormap. Not all maps are suited for people with color blindness. Good choices are: 'cividis' and 'viridis'. Default is 'cividis'
    :return: None"""

    df = pd.read_csv(path)

    avg_accuracy = df["Average Accuracy"].mean()
    avg_worst_case = df["Worst Case Accuracy"].mean()
    avg_best_case = df["Best Case Accuracy"].mean()

    features = ["Average Accuracy", "Worst Case Accuracy", "Best Case Accuracy"]
    x = range(len(features))

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.2
    num_domains = len(df["Domain"])
    positions = [(i - (num_domains - 1) / 2) * bar_width for i in range(num_domains)]
    colormap = cm.get_cmap(cmap, num_domains)

    line_avg = ax.axhline(y=avg_accuracy, color="blue", linestyle="--", linewidth=2,
                          label=f"Average Accuracy ({avg_accuracy:.2f})")
    line_worst = ax.axhline(y=avg_worst_case, color="orange", linestyle="--", linewidth=2,
                            label=f"Worst Case Accuracy ({avg_worst_case:.2f})")
    line_best = ax.axhline(y=avg_best_case, color="green", linestyle="--", linewidth=2,
                           label=f"Best Case Accuracy ({avg_best_case:.2f})")

    bar_handles = []
    for i, domain in enumerate(df["Domain"]):
        accuracies = df.loc[i, features]
        color = colormap(i)
        bar = ax.bar(
            [pos + x_val for x_val, pos in zip(x, [positions[i]] * len(features))],
            accuracies,
            width=bar_width,
            label=domain,
            color=color
        )
        bar_handles.append(bar)

    domain_handles = [plt.Rectangle((0, 0), 1, 1, color=colormap(i), label=domain)
                      for i, domain in enumerate(df["Domain"])]
    separator = plt.Line2D([0], [0], color='none', label=" ")  # dummy line as a separator

    ax.legend(handles=domain_handles + [separator, line_avg, line_worst, line_best],
              title="Domains & Means", bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Across Domains")
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))

    if save:
        _save_plot(path, fig, name='accuracies')

    plt.show()


def _load_run_data(base_dir: str) -> Dict[str, List]:
    """Load training metrics from each run and domain folder and return as a nested dictionary.
    :param base_dir: path to experiment directory.
    :return: nested dictionary with run and domain data."""
    domain_data = {}
    for run_folder in sorted(os.listdir(base_dir)):
        run_path = os.path.join(base_dir, run_folder)
        if os.path.isdir(run_path):
            for domain_folder in sorted(os.listdir(run_path)):
                domain_path = os.path.join(run_path, domain_folder)
                csv_path = os.path.join(domain_path, 'training_metrics.csv')
                if os.path.isfile(csv_path):
                    if domain_folder not in domain_data:
                        domain_data[domain_folder] = []
                    data = pd.read_csv(csv_path)
                    domain_data[domain_folder].append(data)
    return domain_data


def _plot_metric_curves(run_data: Dict[str, List],
                        metric_name: str,
                        ylabel: str,
                        title: str,
                        save_path: str = None):
    """Plot the metric curves for each domain with an average line and margin.
    :param run_data: nested dictionary with run and domain data.
    :param metric_name: name of the metric to plot.
    :param ylabel: name of the y-axis label.
    :param title: title of the plot.
    :param save_path: path to save the figure. Default is None.
    :return: None"""
    for domain, runs in run_data.items():
        epochs = runs[0]['epoch']  # Assuming each run has the same number of epochs
        metrics = np.array([run[metric_name].values for run in runs])

        mean_curve = np.mean(metrics, axis=0)
        std_curve = np.std(metrics, axis=0)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(epochs, mean_curve, label='Average', color='black', linestyle='--', linewidth=1.5)
        plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, color='green', alpha=0.4,
                         label='±1 std dev')

        # Individual runs (for reference)
        for run_idx, run_metrics in enumerate(metrics):
            plt.plot(epochs,
                     run_metrics,
                     color='red',
                     alpha=0.7,
                     linewidth=0.8,
                     label=f'Runs' if run_idx == 0 else '')

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        if 'Accuracy' in title:
            plt.ylim(0, 100)
            plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))
            plt.legend(loc="lower right")
        elif 'Loss' in title:
            plt.ylim(bottom=0)
            plt.legend(loc="upper right")
        plt.title(f"{title} for {domain}")

        if save_path:
            save_name = f'{domain}_{metric_name}_plot'
            _save_plot(save_path, fig, save_name)
        plt.show()


# TODO: integrate into _plot_metric_curves function
def _plot_all_domain_averages(run_data: Dict[str, List],
                              metric_name: str,
                              ylabel: str,
                              title: str,
                              save_path=None):
    """Plot the average curves for each domain in a single plot, plus an overall average curve.
    :param run_data: nested dictionary with run and domain data.
    :param metric_name: name of the metric to plot.
    :param ylabel: name of the y-axis label.
    :param title: title of the plot.
    :param save_path: path to save the figure. Default is None.
    :return: None"""
    fig = plt.figure(figsize=(10, 6))

    all_domain_means = []
    epochs = run_data[list(run_data.keys())[0]][0]['epoch']  # gets epochs from the first domains first run

    for domain, runs in run_data.items():
        metrics = np.array([run[metric_name].values for run in runs])
        mean_curve = np.mean(metrics, axis=0)
        all_domain_means.append(mean_curve)
        plt.plot(epochs, mean_curve, label=f"{domain} Average", alpha=0.7)

    overall_mean_curve = np.mean(all_domain_means, axis=0)
    plt.plot(epochs, overall_mean_curve, label="Overall Average", color="black", linewidth=2, linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Average {title} Across Domains")

    if 'Accuracy' in title:
        plt.ylim(0, 100)
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))
        plt.legend(loc="lower right")
    elif 'Loss' in title:
        plt.ylim(bottom=0)
        plt.legend(loc="upper right")

    if save_path:
        save_name = f'global_{metric_name}_plot'
        _save_plot(save_path, fig, save_name)
    plt.show()


def plot_training_curves(base_dir: str):
    """Plot the training curves for each domain.
    :param base_dir: path to experiment directory.
    :return: None"""
    run_data = _load_run_data(base_dir)

    metrics = [
        ("avg_training_loss", "Training Loss", "Training Loss Across Runs"),
        ("avg_validation_loss", "Validation Loss", "Validation Loss Across Runs"),
        ("avg_training_accuracy", "Training Accuracy", "Training Accuracy Across Runs"),
        ("avg_validation_accuracy", "Validation Accuracy", "Validation Accuracy Across Runs")
    ]

    for metric_name, ylabel, title in metrics:
        _plot_metric_curves(run_data, metric_name, ylabel, title,
                            save_path=base_dir)
        _plot_all_domain_averages(run_data, metric_name, ylabel, title,
                                  save_path=base_dir)


def _save_plot(path: str, fig: plt.Figure, name: str) -> None:
    """Creates plot dir and saves the figure.
    :param path: path to save the figure.
    :param fig: matplotlib figure.
    :param name: name of the plot.
    :return: None
    """
    if not name:
        name = 'figure'
    plot_dir = os.path.join(os.path.dirname(path), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(plot_dir + f'/{name}.png', bbox_inches='tight', dpi=300)
