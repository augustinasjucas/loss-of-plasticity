import glob
import re
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from math import sqrt
import os
# Configuration
ENTITY = 'augustinasj'
PROJECT = 'slow_regression-increasing-layers-large-inputs'
METRIC = 'error'
STEP_METRIC = 'step'
MAX_STEP = 3_000_000
BIN_SIZE = 40_000
MULTIPLY_VALUES = 1
NUM_RUNS = 5  # Number of runs to aggregate per group

MULT = 1
PROCESS_NAME = lambda x : x
XLABEL = "Iteration"
YLABEL = "Error"

GROUP_THE_GROUPS = True

XLABEL = "Task Index"
YLABEL = "Task Accuracy"
MAX_STEP = 1100
BIN_SIZE = 5
MULT = -1
NUM_RUNS = 10
NAME="sanity_check.pdf"

# FIG: sanity check
PROJECT = 'mnist-sanity-check'
ROW_NAMES = []
COL_NAMES = []
STEP_METRIC = 'task'
METRIC = 'accuracy'
GROUPS = [                          # These are the selected best hyperparameters
    "bp-layers=5-lr=0.003", 
    "cbp-layers=5-lr=0.003-repl=0.0001", 
    "l2-layers=5-lr=0.001-decay=0.001",
]

LABEL = {
    "bp-layers=5-lr=0.003": "Backprop", 
    "cbp-layers=5-lr=0.003-repl=0.0001": "Continual Backprop", 
    "l2-layers=5-lr=0.001-decay=0.001": "L2 regularization",
}


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
    return group_data
if __name__ == '__main__':

    group_data = get_group_data(GROUPS)

    # Plotting
    plt.figure(figsize=(6*1.5, 4))
    x = np.arange(int(MAX_STEP // BIN_SIZE)) * BIN_SIZE
    for group, performances in group_data.items():
        performances = performances * MULTIPLY_VALUES
        mean_err = np.nanmean(performances, axis=0)
        std_err = np.nanstd(performances, axis=0) / sqrt(performances.shape[0])
        plt.plot(x, mean_err, label=f'{group} Mean Error' if LABEL is None else LABEL[group], linewidth=3)
        plt.fill_between(x, mean_err - std_err, mean_err + std_err, alpha=0.4)

    plt.xlabel(XLABEL, fontsize=20)
    plt.ylabel(YLABEL, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.xlim(0, MAX_STEP)
    plt.legend(loc='lower left', fontsize=15)
    plt.grid(True, axis='both')
    plt.tight_layout()
    plt.savefig(NAME, dpi=600)
    plt.show()
