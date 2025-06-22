import glob
import re
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from math import sqrt
import os

# Information about where wandb files are stored
ENTITY = 'augustinasj'
PROJECT = 'slow_regression_final'
NAME="saved_picked.pdf"
METRIC = 'error'
STEP_METRIC = 'step'
MAX_STEP = 3000000
BIN_SIZE = 40000
NUM_RUNS = 10 # runs per group
GROUPED_GROUPS = [
    # Relus
    ["relu-backprop-.*", "relu-cbp-.*", "relu-l2-.*"],
    ["sigmoid-backprop-.*", "sigmoid-cbp-.*", "sigmoid-.*"],
    ["tanh-backprop-.*", "tanh-cbp-.*", "tanh-.*"],
]


ROWS = 1
COLS = 3
COL_NAMES = [r"a. ReLU", r"b. Sigmoid", r"c. Tanh"]
ROW_NAMES = []

def PROCESS_NAME(x):
    parts = x.split("-")
    processed_parts = [s.replace("=", " = ") for i, s in enumerate(parts) if i not in [0, 2]]
    if not processed_parts:
        return ""
    first_part = processed_parts[0]
    if first_part == "cbp":
        first_part = "CBP"
    else:
        first_part = first_part.capitalize()
    first_part = "$\\bf{" + first_part + "}$"
    rest_parts = [s.capitalize() for s in processed_parts[1:]]

    if rest_parts:
        return first_part + ". " + ", ".join(rest_parts)
    else:
        return first_part + "."

def get_unique_group_names(entity, project):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    group_names = {run.group for run in runs if run.group}
    return sorted(list(group_names))


def fetch_runs(entity, project, limit=None, group=None):
    """Fetch runs from a W&B project."""
    api = wandb.Api(timeout=60)
    if group is None:
        runs = api.runs(f"{entity}/{project}")
    else:
        runs = api.runs(f"{entity}/{project}", filters={"group": group})
    if limit:
        runs = list(runs)[:limit]
    return runs


def extract_binned_errors(run, metric, step_metric, max_step, bin_size):
    """Download history and bin error values up to max_step into bins of size bin_size."""
    hist = run.history(samples=10000)
    hist = hist[[step_metric, metric]]
    hist = hist[hist[step_metric] <= max_step]
    num_bins = int(max_step // bin_size)
    binned = []
    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        bin_vals = hist[(hist[step_metric] > start) & (hist[step_metric] <= end)][metric]
        if not bin_vals.empty:
            binned.append(bin_vals.mean())
        else:
            binned.append(np.nan)
    return np.array(binned)

def get_group_data(groups):

    group_data = {}
    for group in groups:
        print(f"Fetching runs for group: {group}")
        runs = fetch_runs(ENTITY, PROJECT, limit=NUM_RUNS, group=group)
        print(f"Found {len(runs)} runs for group '{group}'")
        all_binned = []
        for run in runs:
            print(f"  Processing run: {run} | util type=", run.config["util_type"])
            if run.config["util_type"] == "adaptable_contribution":
                print("    Skipping")
                continue
            errs = extract_binned_errors(run, METRIC, STEP_METRIC, MAX_STEP, BIN_SIZE)
            all_binned.append(errs)
        group_data[group] = np.vstack(all_binned)

if __name__ == '__main__':


    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

    # Top row: two subplots occupying (0, 0:2) and (0, 2:4)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])

    # Bottom row: one subplot centered in (1, 1:3)
    ax3 = fig.add_subplot(gs[1, 1:3])

    axs = [ax1, ax2, ax3]
    for i, ax in enumerate(axs):
        ax.set_xlabel('Sample Index', fontsize=14)
        if i == 0 or i == 2: ax.set_ylabel('Error', fontsize=14)
        ax.grid(True, axis='both', linestyle='--', linewidth=0.5)
    for i, row_name in enumerate(ROW_NAMES):
        axs[i * COLS].set_ylabel(row_name, fontsize=14)
    for i, col_name in enumerate(COL_NAMES):
        axs[i].set_title(col_name, fontsize=14, fontweight='bold')

    # Get the names of all groups!
    all_group_names = get_unique_group_names(ENTITY, PROJECT)
    saved_groups_to_plot = {}

    # Next we fetch all necessary data and find the optimal groups for every group set.
    # Since this is a long process, if it was already done and cached, just load the cached data.
    if os.path.exists("saved_groups_to_plot_temp.pkl"):
        print("Loading saved_groups_to_plot_temp.pkl")
        with open("saved_groups_to_plot_temp.pkl", 'rb') as f:
            saved_groups_to_plot = pickle.load(f)

    # If no cached data, we need to fetch and compute the data
    if saved_groups_to_plot == {}:
        for i, group_set in enumerate(GROUPED_GROUPS):
            groups_to_plot = {}
            # for every entry in group_set, fetch the groups that match the regex
            for group_pattern in group_set:
                matching_group_names = [name for name in all_group_names if re.match(group_pattern, name)]
                print("For", group_pattern, "found", len(matching_group_names), "matching groups:", matching_group_names)

                # now the goal is to find the best group from all of these matching groups.
                # the best one is defined as the one with the lowest MSE.
                group_data = get_group_data(matching_group_names)
                best_group = None
                best_mse = float('inf')

                for group_name in matching_group_names:
                    # Calculate the starting index for the last 15% of the data
                    num_bins = group_data[group_name].shape[1]
                    start_index = int(num_bins * 0.85)

                    # Slice the data to get the last 15%
                    last_15_percent_data = group_data[group_name][:, start_index:]

                    # Calculate MSE over the last 15%
                    mse = np.nanmean(last_15_percent_data)
                    if mse < best_mse:
                        best_mse = mse
                        best_group = group_name

                groups_to_plot[best_group] = group_data[best_group]

            saved_groups_to_plot[i] = groups_to_plot

    # Dump saved_groups_to_plot to a pickle file, for temporary caching storage. This
    # allows only to change the plotting code without re-fetching the data.
    with open("saved_groups_to_plot_temp.pkl", 'wb') as f:
        pickle.dump(saved_groups_to_plot, f)
        print(f"Data saved to 'saved_groups_to_plot_temp.pkl'")

    # Do all the plotting
    for i, group_set in enumerate(GROUPED_GROUPS):
        for group_name, group_data in saved_groups_to_plot[i].items():
            mean_err = np.nanmean(group_data, axis=0)
            std_err = np.nanstd(group_data, axis=0) / sqrt(group_data.shape[0])
            x = np.arange(int(MAX_STEP // BIN_SIZE)) * BIN_SIZE
            axs[i].plot(x, mean_err, label=f'{PROCESS_NAME(group_name)}', linewidth=1.9)
            axs[i].fill_between(x, mean_err - std_err, mean_err + std_err, alpha=0.2)

        # add legend
        axs[i].legend(fontsize=7.2)
        # remove x ticks and x numbers

        # set x ticks to be 0 and "1M"
        axs[i].set_xticks([0, 1_000_000, 2000000, 3000000])
        axs[i].set_xticklabels([0, "1M", "2M", "3M"], fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(left=0.076, bottom=0.086, right=0.99, top=0.952, wspace=0.343, hspace=0.3)
    plt.savefig(NAME, dpi=600)
    plt.show()
