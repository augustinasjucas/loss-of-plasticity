from collections import defaultdict
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
METRIC = 'accuracy'
YLABEL = "Accuracy (%)"

STEP_METRIC = 'task'
MAX_STEP = 1100
BIN_SIZE = 10
MULTIPLY_VALUES = 100
NUM_RUNS = 100  # Number of runs to aggregate per group
LOAD_TEMP = False
MULT = -1
PROCESS_NAME = lambda x : x
XLABEL = "Task index"


PKL = None
import pickle
with open('SAVED_POST_HYPERPARAM.pkl', 'rb') as f:
    PKL = pickle.load(f)


### FIG 1
GROUPED_GROUPS = [

    ## Changing the depth
    [
        ["^agent=bp-.*-activation=relu-layers=5.*-width=100$", "^agent=cbp-.*-activation=relu-layers=5.*-width=100$", "^agent=l2-.*-activation=relu-layers=5.*-width=100$", "111111"],
        ["^agent=bp-.*-activation=selu-layers=5.*-width=100$", "^agent=cbp-.*-activation=selu-layers=5.*-width=100$", "^agent=l2-.*-activation=selu-layers=5.*-width=100$", "111111"],
        ["^agent=bp-.*-activation=tanh-layers=5.*-width=100$", "^agent=cbp-.*-activation=tanh-layers=5.*-width=100$", "^agent=l2-.*-activation=tanh-layers=5.*-width=100$", "111111"],

    ],
]
PLOT_NAMES = [
    "Repacing All Layers Separately",
]

WITHIN_PLOT_NAMES = [

    [
        r"Activation: $\bf{ReLU}$",
        r"Activation: $\bf{SELU}$",
        r"Activation: $\bf{Tanh}$",
    ],
]

def bitstring_to_word(bitstring):
    bitstring = bitstring.replace(" ", "")
    num_ones = bitstring.count('1')
    if num_ones == 0:
        return "No replacement"
    elif num_ones == 1:
        index = bitstring.index('1')
        return "Replacing layer " + str(index + 1)
    elif num_ones == len(bitstring):
        return "Replacing all layers"
    elif num_ones == len(bitstring) - 1:
        indices = [str(i + 1) for i, bit in enumerate(bitstring) if bit == '0']
        return "Replacing all layers except " + indices[0]
    else:
        indices = [str(i + 1) for i, bit in enumerate(bitstring) if bit == '1']
        return "Replacing layers " + ", ".join(indices[:-1]) + " and " + indices[-1]

CURRENT_NAME_INDEX = 0
NAMES_TO_INDEX = {}

def PROCESS_NAME(x):
    # split on “-” only when NOT followed by a digit (so “1e-3” stays together)
    parts = re.split(r"-(?!\d)", x)
    processed_parts = [
        s.replace("=", " = ")
        for s in parts
        if (("agent" in s or "rates" in s )
            and s not in ("repl=0", "decay=0"))
    ]
    if not processed_parts:
        return ""
    first_part = processed_parts[0]
    if first_part == "agent = cbp":
        first_part = "CBP"
    elif first_part.lower() == "agent = bp":
        first_part = "BP"
    else:
        first_part = first_part[len("Agent = "):].capitalize()

    if first_part == "CBP":
        first_part = "Continual Backprop"
    elif first_part == "BP":
        first_part = "Backprop"
    elif first_part == "L2":
        first_part = "L2 Regularization"

    second_part = processed_parts[1]
    if second_part.startswith("rates"):
        second_part_end = second_part.split("=")[-1]
        word = bitstring_to_word(second_part_end)
        second_part = word
            
    id = first_part + "_" + second_part
    if id in NAMES_TO_INDEX:
        index = NAMES_TO_INDEX[id]
    else:
        global CURRENT_NAME_INDEX
        index = CURRENT_NAME_INDEX
        NAMES_TO_INDEX[id] = index
        CURRENT_NAME_INDEX = (1 + CURRENT_NAME_INDEX) % 10

    return first_part, index

# just for pickling
def default_list():
    return list()

def default_dd_list():
    return defaultdict(default_list)

def default_dd_dd_list():
    return defaultdict(default_dd_list)


def fetch_runs(group=None):
    # fetch from PKL dict
    return list(PKL[group].values())

def extract_binned_errors(run, metric, step_metric, max_step, bin_size):
    if run['config'].get('util_type') != 'contribution':
        raise ValueError(f"Run {run.id} is not a contribution run. Skipping.")

    df = run['run_history']
    df = df[df[step_metric] <= max_step]
    n = int(max_step // bin_size)
    binned = []
    for i in range(n):
        lo, hi = i * bin_size, (i + 1) * bin_size
        chunk = df[(df[step_metric] > lo) & (df[step_metric] <= hi)][metric]
        binned.append(chunk.mean() if not chunk.empty else np.nan)
    return np.array(binned)


def get_group_data(groups):
   
    group_data = {}
    for group in groups:
        print(f"      Fetching runs for group: {group}")
        runs = fetch_runs(group=group)
        print(f"      Found {len(runs)} runs for group '{group}'")
        all_binned = []
        for run in runs:
            run_name = run['run_name']
            print(f"        Processing run: {run_name} | util type=", run['config']["util_type"])
            if run["config"]["util_type"] == "adaptable_contribution":
                print("    Skipping")
                continue
            try:
                errs = extract_binned_errors(run, METRIC, STEP_METRIC, MAX_STEP, BIN_SIZE)
                all_binned.append(errs)
            except Exception as e:
                print(f"        [WARN] Error processing run {run_name}: {e}")
                continue
        group_data[group] = np.vstack(all_binned)
    return group_data


def fff(x):
    x, _ = PROCESS_NAME(x)
    # All layers -> 1
    # l2 -> 2
    # layer x -> 3
    # no replacement -> 4
    if "all" in x.lower():
        return 1
    elif "l2" in x.lower():
        return 2
    # ends with number
    elif re.search(r'\d$', x):
        return int(x[-1]) + 2  # layer 1 -> 3, layer 2 -> 4, etc.
    elif "no" in x.lower() or "none" in x.lower():
        return 20
    else:
        return 21

def order(group_list):
    return sorted(group_list, key=lambda x: fff(x[0]))

if __name__ == '__main__':
    figs = []
    axss = []
    for i, group_set in enumerate(GROUPED_GROUPS):
        fig, axs = plt.subplots(1, len(GROUPED_GROUPS[i]), figsize=(9, 4), sharey=True, constrained_layout=True)
        if len(GROUPED_GROUPS[i]) == 1:
            axs = [axs]
        figs.append(fig)
        axss.append(axs)

    wandb_data = defaultdict(default_dd_dd_list)    # wandb_data[figure_index][subplot_index][group_pattern_index] = {"group1": binned_metrics, "group2": binned_metrics, ...}
    
    # as the very first step, fetch all data from all groups
    for i, figure_list in enumerate(figs):
        print(f"Processing figure {i+1}/{len(figs)}: {PLOT_NAMES[i]}")
        for j, group_set in enumerate(GROUPED_GROUPS[i]):
            print(f"  Processing subplot {j+1}/{len(GROUPED_GROUPS[i])} in figure {i+1}: {WITHIN_PLOT_NAMES[i][j]}")
            for k, group_pattern in enumerate(group_set[:3]):
                print(f"    Processing group pattern {k+1}/{len(group_set)}: {group_pattern}")
                matching_groups = [name for name in PKL.keys() if re.match(group_pattern, name)]
                print("     Got the matching groups:", len(matching_groups))
                group_data = get_group_data(matching_groups)
                wandb_data[i][j][k] = group_data

    for i, fig in enumerate(figs):
        axs = axss[i]
        for j, group_set in enumerate(GROUPED_GROUPS[i]):
            current_ax = axs[j]
            groups_to_plot_on_this_ax = {}
            best_group_for_pattern = {}

            # group_set is GROUPED_GROUPS[i][j], a list of patterns for the current subplot
            for k, pattern_regex in enumerate(group_set[:3]):
                data_for_this_pattern_regex = wandb_data[i][j][k]

                if k == 2 or k == 0: # bp and l2
                    group_name, group_array = list(data_for_this_pattern_regex.items())[0]
                    groups_to_plot_on_this_ax[group_name] = group_array
                elif k == 1: # cbp
                    layer_configs_we_want = GROUPED_GROUPS[i][j][3:]

                    for actual_group_name, group_data_array in data_for_this_pattern_regex.items():
                        is_good = False
                        for layer_config in layer_configs_we_want:
                            if layer_config in actual_group_name:
                                is_good = True
                                break
                        if is_good:
                            groups_to_plot_on_this_ax[actual_group_name] = group_data_array
                else:
                    continue


            if not groups_to_plot_on_this_ax:
                current_ax.set_title(WITHIN_PLOT_NAMES[i][j], fontsize=19)
                
                current_ax.set_xlabel(XLABEL, fontsize=20)
                if j == 0: current_ax.set_ylabel(YLABEL, fontsize=20)
                current_ax.grid(True, linestyle='--', alpha=0.6)

            pallete = plt.get_cmap('tab10')

            for group_name_to_plot, (data_array_to_plot) in order(groups_to_plot_on_this_ax.items()):
                if data_array_to_plot.ndim == 1:
                    data_array_to_plot = data_array_to_plot.reshape(1, -1)

                if data_array_to_plot.shape[0] == 0:
                    continue
                mean_values = np.nanmean(data_array_to_plot, axis=0)
                if np.all(np.isnan(mean_values)):
                    continue

                std_dev = np.nanstd(data_array_to_plot, axis=0)
                std_err = std_dev / sqrt(data_array_to_plot.shape[0]) if data_array_to_plot.shape[0] > 0 else np.zeros_like(mean_values)
                
                x_coords = np.arange(len(mean_values)) * BIN_SIZE

                color_ind = PROCESS_NAME(group_name_to_plot)[1]
                current_ax.plot(x_coords, mean_values * MULTIPLY_VALUES, label=f'{PROCESS_NAME(group_name_to_plot)[0]}', color=pallete(color_ind), linewidth=3.5)
                current_ax.fill_between(x_coords, 
                                        (mean_values - std_err) * MULTIPLY_VALUES, 
                                        (mean_values + std_err) * MULTIPLY_VALUES, 
                                        alpha=0.2,
                                        color=pallete(color_ind)
                                        )

            if j == 0:
                lines_labels = [ax.get_legend_handles_labels() for ax in axs]
                lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]  # flatten
                fig.legend(lines, labels, loc='lower left', bbox_to_anchor=(0.1, 0.1975), fontsize=15, borderaxespad=0., ncol=1)

            current_ax.set_xlabel(XLABEL, fontsize=20)
            if j == 0: current_ax.set_ylabel(YLABEL, fontsize=20)
            current_ax.set_title(WITHIN_PLOT_NAMES[i][j], fontsize=20)
            current_ax.set_xlim(0, MAX_STEP)
            num_x_ticks = 3
            max_x_val_on_plot = 1100
            x_tick_values = np.linspace(0, max_x_val_on_plot, num_x_ticks, endpoint=True, dtype=int)
            x_tick_values = np.unique(np.clip(x_tick_values, 0, max_x_val_on_plot))

            x_tick_labels = []
            for val_tick in x_tick_values:
                if val_tick >= 1_000_000 and val_tick % 1_000_000 == 0:
                    x_tick_labels.append(f"{val_tick//1_000_000}M")
                elif val_tick >= 1_000 and val_tick % 1_000 == 0:
                        x_tick_labels.append(f"{val_tick//1_000}k")
                elif val_tick >= 1_000:
                    x_tick_labels.append(f"{val_tick/1000:.1f}k".replace(".0k","k"))
                else:
                    x_tick_labels.append(str(val_tick))
            
            current_ax.set_xticks(x_tick_values)
            current_ax.set_xticklabels(x_tick_labels, fontsize=20)
            current_ax.tick_params(axis='y', labelsize=20)
            current_ax.grid(True, linestyle='--', alpha=0.6)
        
        nm = PLOT_NAMES[i] + " " + YLABEL
        fig_name = f"activations.pdf"
        fig.savefig(fig_name, dpi=600)
        print(f"Saved figure {i+1}/{len(figs)} as '{fig_name}'")
