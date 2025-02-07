import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Dict, List, Tuple

from matplotlib.ticker import MaxNLocator, MultipleLocator
from parcoords import plot_parcoords
from scipy.signal import savgol_filter
from typing_extensions import deprecated

import random
from PIL import Image


def plot_accuracies(
        root_path: str,
        save: bool = True,
        cmap: str = 'cividis',
        show: bool = True
) -> None:
    """Plots worst-case, average, and best-case accuracies for each domain across multiple runs.
    :param root_path: Path to the root directory containing run folders.
    :param save: Whether to save the plot. Default is True.
    :param cmap: Colormap for the plot. Default is 'cividis'.
    :param show: Whether to display the plot. Default is True.
    :return: None"""

    test_files = []
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.endswith("_test_metrics.csv"):
                test_files.append(os.path.join(root, file))

    # aggregate data across all runs
    data = {}
    for file_path in test_files:
        domain = os.path.basename(file_path).split("_test_metrics.csv")[0]
        df = pd.read_csv(file_path)
        if domain not in data:
            data[domain] = []
        data[domain].extend(df["Test Accuracy"].tolist())

    stats = {
        domain: {
            "worst_case": min(accuracies),
            "average": sum(accuracies) / len(accuracies),
            "best_case": max(accuracies)
        }
        for domain, accuracies in data.items()
    }

    worst_cases = [stat["worst_case"] for stat in stats.values()]
    averages = [stat["average"] for stat in stats.values()]
    best_cases = [stat["best_case"] for stat in stats.values()]

    global_stats = {
        "worst_case": sum(worst_cases) / len(worst_cases),
        "average": sum(averages) / len(averages),
        "best_case": sum(best_cases) / len(best_cases)
    }

    domains = list(stats.keys())
    metrics = ["worst_case", "average", "best_case"]
    values = [[stats[domain][metric] for metric in metrics] for domain in domains]

    x = range(len(metrics))

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.2
    colormap = cm.get_cmap(cmap, len(domains))

    for i, (domain, domain_values) in enumerate(zip(domains, values)):
        ax.bar(
            [pos + i * bar_width for pos in x],
            domain_values,
            width=bar_width,
            label=domain,
            color=colormap(i)
        )

    # global average lines
    for metric, color in zip(metrics, ["orange", "blue", "green"]):
        ax.axhline(y=global_stats[metric], color=color, linestyle="--", linewidth=2,
                   label=f"Global {metric.replace('_', ' ').title()} ({global_stats[metric]:.2f})")

    ax.set_title("Accuracy Across Domains")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Cases")
    ax.set_xticks([pos + (len(domains) - 1) * bar_width / 2 for pos in x])
    ax.set_xticklabels(["Worst Case", "Average Case", "Best Case"])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))

    ax.legend(title="Domains & Means", bbox_to_anchor=(1.05, 1), loc='upper left')

    if save:
        os.makedirs(f'{root_path}/plots', exist_ok=True)
        plot_path = os.path.join(f'{root_path}/plots', "accuracy_across_domains.png")
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Plot saved at {plot_path}")

    if show:
        plt.show()
    plt.close()


def _load_run_data(base_dir: str) -> Dict[str, List[Dict[str, List]]]:
    """
    Load training metrics from each run folder and return as a nested dictionary.
    Each run folder contains training metrics files for multiple domains.

    :param base_dir: Path to experiment directory.
    :return: Dictionary where each key is a domain, and the value is a list of dictionaries
             containing the training data for all runs:
             {
                "domain1": [
                    {"epoch": [...], "accuracy": [...], ...},  # Run 1 data
                    {"epoch": [...], "accuracy": [...], ...},  # Run 2 data
                    ...
                ],
                "domain2": [
                    {"epoch": [...], "accuracy": [...], ...},
                    ...
                ]
             }
    """
    domain_data = {}
    for run_folder in sorted(os.listdir(base_dir)):
        run_path = os.path.join(base_dir, run_folder)
        if os.path.isdir(run_path):
            for file in sorted(os.listdir(run_path)):
                if file.endswith("_train_metrics.csv"):
                    domain_name = file.replace("_train_metrics.csv", "")
                    csv_path = os.path.join(run_path, file)
                    if os.path.isfile(csv_path):
                        if domain_name not in domain_data:
                            domain_data[domain_name] = []
                        df = pd.read_csv(csv_path)
                        run_dict = {col: df[col].tolist() for col in df.columns}
                        domain_data[domain_name].append(run_dict)
    return domain_data


def _plot_metric_curves(
        root_path: str,
        metric_name: str,
        ylabel: str,
        title: str,
        save_path: str = None,
        show: bool = True,
        margin_type: str = "std-dev",
        smooth: bool = True,
        smoothing_window: int = 3,
        polyorder: int = 2
) -> None:
    """
    Plot metric curves with margin (either standard deviation or min-max) as well as average
    across multiple runs for each domain.

    :param root_path: Path to the root directory containing run folders.
    :param metric_name: Name of the metric to plot (e.g., "Test Loss" or "Test Accuracy").
    :param ylabel: Label for the y-axis.
    :param title: Title for the plot.
    :param save_path: Path to save the figure.
    :param show: Whether to show the plot. Default is True.
    :param margin_type: Margin type. Options: [std-dev, min-max]. Default is std-dev.
    :param smooth: Whether to smooth the curve. Default is True.
    :param smoothing_window: Smoothing window parameter. Default is 5.
    :param polyorder: Polynomial order for smoothing curve. Default is 2.
    :return: None
    """
    test_files = []
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.endswith("_train_metrics.csv"):
                test_files.append(os.path.join(root, file))

    run_data = {}
    for file_path in test_files:
        domain = os.path.basename(file_path).split("_train_metrics.csv")[0]
        df = pd.read_csv(file_path)
        epochs = np.arange(len(df))
        metrics = df[metric_name].to_numpy()

        if domain not in run_data:
            run_data[domain] = []
        run_data[domain].append({"epoch": epochs, metric_name: metrics})

    for domain, runs in run_data.items():
        max_epochs = max(len(run["epoch"]) for run in runs)
        all_epochs = np.arange(0, max_epochs)

        extended_metrics = []

        for run in runs:
            run_epochs = np.array(run['epoch'])
            run_metrics = np.array(run[metric_name])

            if len(run_epochs) < max_epochs:
                missing_epochs = np.arange(len(run_epochs), max_epochs)
                run_epochs = np.concatenate([run_epochs, missing_epochs])
                last_metric = run_metrics[-1]
                run_metrics = np.concatenate([run_metrics, [last_metric] * len(missing_epochs)])

            extended_metrics.append(run_metrics)

        metrics = np.vstack(extended_metrics)

        mean_curve = np.mean(metrics, axis=0)

        if margin_type == "min-max":
            min_curve = np.min(metrics, axis=0)
            max_curve = np.max(metrics, axis=0)
            lower_bound, upper_bound = min_curve, max_curve
        elif margin_type == "std-dev":
            std_dev = np.std(metrics, axis=0)
            lower_bound, upper_bound = mean_curve - std_dev, mean_curve + std_dev
        else:
            raise ValueError("Invalid margin type. Use 'min-max' or 'std-dev'.")

        if smooth:
            mean_curve = savgol_filter(mean_curve, smoothing_window, polyorder)
            lower_bound = savgol_filter(lower_bound, smoothing_window, polyorder)
            upper_bound = savgol_filter(upper_bound, smoothing_window, polyorder)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(all_epochs, mean_curve, label='Average', color='black', linestyle='--', linewidth=1.5)
        ax.fill_between(all_epochs, lower_bound, upper_bound, color='blue', alpha=0.4,
                        label=f'{margin_type.title()} Range')

        colormap = cm.get_cmap('cividis', len(metrics))
        for run_idx, run_metrics in enumerate(metrics):
            if smooth:
                run_metrics = savgol_filter(run_metrics, smoothing_window, polyorder)
            ax.plot(all_epochs,
                    run_metrics,
                    color=colormap(run_idx),
                    alpha=0.7,
                    linewidth=0.8,
                    linestyle=':',
                    label=f'Runs' if run_idx == 0 else '')

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, max_epochs - 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=max_epochs - 1))

        if 'Accuracy' in title:
            ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5, alpha=0.8, color='gray')
            ax.grid(visible=True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5, color='lightgray')
            ax.minorticks_on()
            spacing = 0.5 if 'Validation' in title else 1
            ax.yaxis.set_major_locator(MultipleLocator(spacing))
            ax.legend(loc="lower right")
        elif 'Loss' in title:
            ax.set_ylim(bottom=0)
            ax.legend(loc="upper right")
        ax.set_title(f"{title} without {domain}")

        if save_path:
            save_name = f'{domain}_{metric_name}_plot.png'
            plot_path = os.path.join(save_path, 'plots', save_name)
            plt.savefig(plot_path, bbox_inches="tight")
            print(f"Plot saved at {plot_path}")
        if show:
            plt.show()
        plt.close()


def _plot_all_domain_averages(
        run_data: Dict[str, List],
        metric_name: str,
        ylabel: str,
        title: str,
        save_path: str = None,
        show: bool = True
):
    """
    Plot the average curves for each domain in a single plot, plus an overall average curve.
    Handles runs with varying epoch lengths by extending shorter runs.

    :param run_data: Nested dictionary with run and domain data.
    :param metric_name: Name of the metric to plot.
    :param ylabel: Label for the y-axis.
    :param title: Title of the plot.
    :param save_path: Path to save the figure. Default is None.
    :param show: Whether to display the plot. Default is True.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    max_epochs = max(
        max(len(run['epoch']) for run in runs)
        for runs in run_data.values()
    )
    all_epochs = np.arange(0, max_epochs)

    all_domain_means = []

    for domain, runs in run_data.items():
        extended_metrics = []

        for run in runs:
            run_epochs = np.array(run['epoch'])
            run_metrics = np.array(run[metric_name])

            if len(run_epochs) < max_epochs:
                missing_epochs = np.arange(len(run_epochs), max_epochs)
                run_epochs = np.concatenate([run_epochs, missing_epochs])
                last_metric = run_metrics[-1]
                run_metrics = np.concatenate([run_metrics, [last_metric] * len(missing_epochs)])

            extended_metrics.append(run_metrics)

        metrics = np.vstack(extended_metrics)

        mean_curve = np.mean(metrics, axis=0)
        all_domain_means.append(mean_curve)

        ax.plot(all_epochs, mean_curve, label=f"{domain} Average", alpha=0.7, linestyle="--")

    overall_mean_curve = np.mean(all_domain_means, axis=0)
    ax.plot(all_epochs, overall_mean_curve, label="Overall Average", color="black", linewidth=2, linestyle="-")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Average {title} Across Domains")
    ax.set_xlim(0, max_epochs - 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=max_epochs - 1))

    if 'Accuracy' in title:
        ax.grid(visible=True, which='major', linestyle='-', linewidth=0.3, alpha=0.8)
        ax.grid(visible=True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.minorticks_on()
        spacing = 0.5 if 'Validation' in title else 1
        ax.yaxis.set_major_locator(MultipleLocator(spacing))
    elif 'Loss' in title:
        ax.set_ylim(bottom=0)

    ax.legend(title="Domains & Overall Average", bbox_to_anchor=(1.05, 1), loc='upper left')

    if save_path:
        save_name = f'global_{metric_name}_plot.png'
        os.makedirs(f'{save_path}/plots', exist_ok=True)
        full_save_path = os.path.join(f'{save_path}/plots', save_name)
        plt.savefig(full_save_path, bbox_inches="tight")
        print(f"Plot saved at {full_save_path}")

    if show:
        plt.show()
    plt.close()


def plot_training_curves(
        base_dir: str,
        show: bool = True,
        save: bool = True
) -> None:
    """
    Plot the training curves for each domain.
    :param show: Whether to display the plot. Default is True.
    :param base_dir: Path to experiment directory.
    :param save_path: Path to save the figure. Default is None.
    :return: None
    """

    save_path = base_dir if save else None

    run_data = _load_run_data(base_dir)

    metrics = [
        ("train_loss", "Training Loss", "Training Loss Across Runs"),
        ("val_loss", "Validation Loss", "Validation Loss Across Runs"),
        ("train_accuracy", "Training Accuracy", "Training Accuracy Across Runs"),
        ("val_accuracy", "Validation Accuracy", "Validation Accuracy Across Runs")
    ]

    for metric_name, ylabel, title in metrics:
        _plot_metric_curves(
            base_dir, metric_name, ylabel, title, save_path=save_path, show=show
        )
        _plot_all_domain_averages(
            run_data, metric_name, ylabel, title, save_path=save_path, show=show
        )

def plot_domain_images(
        dataset_dir: str,
        domain: str,
        class_name: str,
        num_images: int = 5
) -> None:
    """
    Plots random images from the specified domain and class.

    :param dataset_dir: Path to the dataset directory.
    :param domain: Name of the domain (e.g., 'domain1').
    :param class_name: Name of the class (e.g., 'class1').
    :param num_images: Number of random images to plot. Default is 5.
    """
    class_dir = os.path.join(dataset_dir, domain, class_name)

    if not os.path.exists(class_dir):
        raise ValueError(f"The directory {class_dir} does not exist.")

    image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

    if not image_files:
        raise ValueError(f"No images found in the directory {class_dir}.")

    num_images = min(num_images, len(image_files))
    random_images = random.sample(image_files, num_images)

    plt.figure(figsize=(10, 5))
    for i, image_name in enumerate(random_images):
        image_path = os.path.join(class_dir, image_name)
        img = Image.open(image_path)

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{domain}_{class_name}")

    plt.tight_layout()
    plt.show()