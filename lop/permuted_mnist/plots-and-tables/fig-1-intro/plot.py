import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from math import sqrt
import os
# Configuration
ENTITY = 'augustinasj'
PROJECT = 'mnist-hyperparam'
METRIC = 'accuracy'
STEP_METRIC = 'task'
MAX_STEP = 1100
BIN_SIZE = 10
NUM_RUNS = 3  # Number of runs to aggregate per group
GROUPS = [
    # ("agent=bp-activation=relu-layers=5-lr=0.0005-repl=0-decay=0-width=100", "Learning rate = 0.0005"),
    ("agent=bp-activation=relu-layers=5-lr=0.001-repl=0-decay=0-width=100", "Learning rate = 0.001"),
    ("agent=bp-activation=relu-layers=5-lr=0.005-repl=0-decay=0-width=100", "Learning rate = 0.005"),
    ("agent=bp-activation=relu-layers=5-lr=0.01-repl=0-decay=0-width=100", "Learning rate = 0.01"),

]

# Data caching
LOAD_FROM_PKL = True  # Set True to load fetched data from pickle instead of refetching
PKL_FILE = 'fetched_data.pkl'


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
    ret = np.array(binned)
    # print("ret shape", ret.shape)
    return ret

if __name__ == '__main__':
    # Load or fetch data
    if LOAD_FROM_PKL and os.path.exists(PKL_FILE):
        with open(PKL_FILE, 'rb') as f:
            group_data = pickle.load(f)
        print(f"Loaded data for groups: {list(group_data.keys())} from {PKL_FILE}")
    else:
        group_data = {}
        for group, name in GROUPS:
            print(f"Fetching runs for group: {group}")
            runs = fetch_runs(ENTITY, PROJECT, limit=NUM_RUNS, group=group)
            print(f"Found {len(runs)} runs for group '{group}'")
            all_binned = []
            for run in runs:
                print(f"  Processing run: {run.name}")
                errs = extract_binned_errors(run, METRIC, STEP_METRIC, MAX_STEP, BIN_SIZE)
                all_binned.append(errs)
            group_data[name] = np.vstack(all_binned)
        # Save fetched data
        with open(PKL_FILE, 'wb') as f:
            pickle.dump(group_data, f)
        print(f"Saved fetched data to {PKL_FILE}")

    # Plotting
    plt.figure(figsize=(7, 4))
    x = np.arange(int(MAX_STEP // BIN_SIZE)) * BIN_SIZE
    for group, performances in group_data.items():
        mean_err = np.nanmean(performances, axis=0) * 100
        std_err = np.nanstd(performances, axis=0) * 100
        print("std_err = ", std_err)
        plt.plot(x, mean_err, label=f'{group}', linewidth=2.5)
        plt.fill_between(x, mean_err - std_err, mean_err + std_err, alpha=0.5)

    plt.xlim(0, MAX_STEP)
    plt.xlabel('Task Index', fontsize=16)
    plt.ylabel('Mean Task Accuracy (%)', fontsize=16)
    plt.title('Task Accuracy for Continual Permuted MNIST', fontsize=18)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, axis='both', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('intro.pdf', dpi=500)
    plt.show()
