import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Dict, List, Tuple
from parcoords import plot_parcoords


def plot_accuracies(
        path: str,
        save: bool = True,
        cmap: str = 'cividis',
        show: bool = True,
) -> None:
    """Creates bar plots of accuracies across domains for one experiment.
    :param path: path to csv file.
    :param save: whether to save the figure. Default is True.
    :param cmap: matplotlib colormap. Careful when changing the colormap. Not all maps are suited for people with color blindness. Good choices are: 'cividis' and 'viridis'. Default is 'cividis'
    :return: None"""

    df = pd.read_csv(path)

    avg_worst_case = df["worst_case_acc"].mean()
    avg_accuracy = df["avg_acc"].mean()
    avg_best_case = df["best_case_acc"].mean()

    features = ["worst_case_acc","avg_acc", "best_case_acc"]
    x = range(len(features))

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.2
    num_domains = len(df["Domain"])
    positions = [(i - (num_domains - 1) / 2) * bar_width for i in range(num_domains)]
    colormap = cm.get_cmap(cmap, num_domains)

    line_worst = ax.axhline(y=avg_worst_case, color="orange", linestyle="--", linewidth=2,
                            label=f"Worst Case Accuracy ({avg_worst_case:.2f})")
    line_avg = ax.axhline(y=avg_accuracy, color="blue", linestyle="--", linewidth=2,
                          label=f"Average Accuracy ({avg_accuracy:.2f})")
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
    if show:
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

def _plot_metric_curves(
        run_data: Dict[str, List],
        metric_name: str,
        ylabel: str,
        title: str,
        save_path: str = None,
        show: bool = True,
) -> None:
    """Plot the metric curves for each domain with padding for runs with different epochs."""
    for domain, runs in run_data.items():
        # Find the union of all epochs across runs
        all_epochs = sorted({epoch for run in runs for epoch in run['epoch']})

        # Align metrics by padding with NaN for missing epochs
        metrics = []
        for run in runs:
            run_epochs = run['epoch']
            run_metrics = run[metric_name].values

            # Create an array of NaNs for all epochs
            aligned_metrics = np.full(len(all_epochs), np.nan)
            # Find indices where run_epochs match all_epochs
            indices = [all_epochs.index(epoch) for epoch in run_epochs]
            aligned_metrics[indices] = run_metrics
            metrics.append(aligned_metrics)

        metrics = np.array(metrics)

        # Compute statistics while ignoring NaN
        mean_curve = np.nanmean(metrics, axis=0)
        min_curve = np.nanmin(metrics, axis=0)
        max_curve = np.nanmax(metrics, axis=0)

        # Plotting
        fig = plt.figure(figsize=(10, 6))
        plt.plot(all_epochs, mean_curve, label='Average', color='black', linestyle='--', linewidth=1.5)
        plt.fill_between(all_epochs, min_curve, max_curve, color='blue', alpha=0.4, label='Min-Max Range')

        colormap = cm.get_cmap('cividis', len(metrics))
        for run_idx, run_metrics in enumerate(metrics):
            plt.plot(all_epochs,
                     run_metrics,
                     color=colormap(run_idx),
                     alpha=0.7,
                     linewidth=0.8,
                     label=f'Run' if run_idx == 0 else '')

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
        if show:
            plt.show()


def _plot_metric_curves2(
        run_data: Dict[str, List],
        metric_name: str,
        ylabel: str,
        title: str,
        save_path: str = None,
        show: bool = True,
) -> None:
    """Plot the metric curves for each domain with an average line and min-max margin.
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
        min_curve = np.min(metrics, axis=0)
        max_curve = np.max(metrics, axis=0)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(epochs, mean_curve, label='Average', color='black', linestyle='--', linewidth=1.5)
        plt.fill_between(epochs, min_curve, max_curve, color='blue', alpha=0.4, label='Min-Max Range')

        colormap = cm.get_cmap('cividis', len(metrics))  # Use 'tab20' colormap for distinct colors
        for run_idx, run_metrics in enumerate(metrics):
            plt.plot(epochs,
                     run_metrics,
                     color=colormap(run_idx),
                     alpha=0.7,
                     linewidth=0.8,
                     label=f'Run' if run_idx == 0 else '')

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
        if show:
            plt.show()


# TODO: integrate into _plot_metric_curves function
def _plot_all_domain_averages(
        run_data: Dict[str, List],
        metric_name: str,
        ylabel: str,
        title: str,
        save_path=None,
        show: bool = True,
):
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
    if show:
        plt.show()


def plot_training_curves(
        base_dir: str,
        show: bool = True,
        save_path: str = None
) -> None:
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
        _plot_metric_curves(
            run_data, metric_name, ylabel, title, save_path=base_dir, show=show
        )
        _plot_all_domain_averages(
            run_data, metric_name, ylabel, title, save_path=base_dir, show=show
        )


def _save_plot(
        path: str,
        fig: plt.Figure,
        name: str
) -> None:
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


def plot_hyperparameters(
        data: str | pd.DataFrame,
        params: List[str] = None,
        metric_name: str = 'mean_accuracy',
        title: str = 'Hyperparameters',
        cmap: plt.cm = plt.cm.viridis,
        figsize: Tuple[int, int] = (15, 10),
        filter_optim: str = None
):
    if isinstance(data, pd.DataFrame):
        hp_df = data.copy()
    else:
        hp_df = pd.read_csv(data)

    rename_dict = {col: col.split('/')[-1] for col in hp_df.columns}
    hp_df = hp_df.rename(columns=rename_dict)

    if params:
        fields = params
    else:
        fields = ['lr', 'batch_size', 'momentum', 'weight_decay',
                  'optimizer', 'betas', 'eps', 'nesterov',
                  metric_name]

    if filter_optim:
        if filter_optim == 'adam':
            hp_df = hp_df[hp_df['optimizer'] == 'adam']
            fields = ['lr', 'batch_size','betas', 'eps', metric_name]
            scale = [("lr", "log"), ("eps", "log")]
        elif filter_optim == 'adamw':
            hp_df = hp_df[hp_df['optimizer'] == 'adamw']
            fields = ['lr', 'batch_size', 'weight_decay', 'betas', 'eps', metric_name]
            scale = [("lr", "log"), ("eps", "log"), ("weight_decay", "log")]
        elif filter_optim == 'sgd':
            hp_df = hp_df[hp_df['optimizer'] == 'sgd']
            fields = ['lr', 'batch_size', 'momentum', 'weight_decay', 'nesterov', metric_name]
            scale = [("lr", "log")]
        else:
            raise ValueError("Invalid filter value. Choose from 'adam', 'adamw', or 'sgd'.")
    else:
        scale = [("lr", "log"), ("eps", "log"), ("weight_decay", "log")]


    hp_df = hp_df[fields]
    plot_parcoords(
        hp_df,
        labels=fields,
        color_field=metric_name,
        scale=scale,
        title=title,
        cmap=cmap,
        figsize=figsize
    )

    plt.show()


def load_results(
        model_trials: List[str],
        trial_dir: str,
        fields: List[str],
        filename: str = 'results.csv'
) -> pd.DataFrame:
    """
    Loads results from hyperparameter tuning into a single dataframe.
    :param model_trials: List of directories containing the results files.
    :param trial_dir: Path to base trial directory.
    :param fields: List of dataframe column names.
    :param filename: Name of the results file. Default: results.csv
    :return: Concatenated pd.Dataframe with additional "trial" column.
    """
    results = []
    for trial in model_trials:
        file_path = os.path.join(trial_dir, trial, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            filtered_df = df[fields].copy()
            filtered_df["trial"] = trial
            results.append(filtered_df)
    return pd.concat(results, ignore_index=True) if results else None


def find_common_configs(
        domain: pd.DataFrame,
        tolerance: float = 0.01,
        fields: List[str] = None,
        top_n: int = 5,
        filter_optim: str = None,
) -> List[str]:
    """
    Searches for common hyperparameter configurations. Expects a pd.Dataframe containing the configurations over all domains of a model.

    :param domain: pd.Dataframe containing the configurations.
    :param tolerance: Tolerance for numerical values. Default 0.01.
    :param fields: Fields of the dataframe.
    :param top_n: How many trials to consider from each domain. Default 5.
    :return: List of common configurations, if any exist.
    """
    top_n_by_domain = {}

    if fields is None:
        fields = ['config/lr', 'config/batch_size', 'config/momentum', 'config/weight_decay',
                  'config/optimizer', 'config/betas', 'config/eps', 'config/nesterov']

    for dom in domain:
        domain_results = domain[domain['trial'] == dom]

        grouped = domain_results.groupby(
            fields)['mean_accuracy'].mean().reset_index()

        top_n_configurations = grouped.sort_values(by="mean_accuracy", ascending=False).head(top_n)

        top_n_by_domain[dom] = top_n_configurations

    common_configs = []

    def is_approx_equal(value1, value2, tolerance):
        return abs(value1 - value2) <= tolerance

    # ugly, ugly, ugly! infer columns and their types automatically.
    # but then we also need to pass which cols to ignore!
    if filter_optim == 'adam':
        categorical_columns = ['config/betas']
        numerical_columns = ['config/lr', 'config/batch_size', 'config/eps']
    elif filter_optim == 'adamw':
        categorical_columns = ['config/betas']
        numerical_columns = ['config/lr', 'config/batch_size', 'config/weight_decay', 'config/eps']

    elif filter_optim == 'sgd':
        categorical_columns = ['config/nesterov']
        numerical_columns = ['config/lr', 'config/momentum', 'config/weight_decay']
    else:
        categorical_columns = ['config/optimizer', 'config/betas', 'config/nesterov']
        numerical_columns = ['config/lr', 'config/momentum', 'config/weight_decay', 'config/eps']

    for domain_name, domain_data in top_n_by_domain.items():
        domain_top_configs = domain_data.head(5)

        for idx1, config1 in domain_top_configs.iterrows():
            is_common = True
            for other_domain, other_data in top_n_by_domain.items():
                if other_domain == domain_name:
                    continue

                other_top_configs = other_data.head(5)
                match_found = False
                for idx2, config2 in other_top_configs.iterrows():
                    categorical_match = all(config1[col] == config2[col] for col in categorical_columns)

                    numerical_match = all(
                        is_approx_equal(config1[col], config2[col], tolerance) for col in numerical_columns
                    )

                    if categorical_match and numerical_match:
                        match_found = True
                        break

                if not match_found:
                    is_common = False
                    break

            if is_common:
                common_configs.append(config1)

    return common_configs
