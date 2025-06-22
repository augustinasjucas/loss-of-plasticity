from collections import defaultdict
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle

################
# For figure six use "100", for seven - 784
WIDTH = 100
################


ENTITY = 'augustinasj'
METRIC = "weight_magnitude/layer_"
YLABEL = "Weight magnitude"
MULTS = None
STEP_METRIC = 'task'
MAX_STEP = 400
BIN_SIZE = 10
MULTIPLY_VALUES = 1
NUM_RUNS = 100  # Number of runs to aggregate per group
LOAD_TEMP = False
MULT = -1
PROCESS_NAME = lambda x : x
XLABEL = "Layer index"
GROUP_THE_GROUPS = True
PRINT_AT_THE_END = []

GROUPED_GROUPS = [
    [
        [f"^agent=bp-.*-activation=relu-layers=5.*-width={WIDTH}$", f"^agent=cbp-.*-activation=relu-layers=5.*-width={WIDTH}$", f"^agent=l2-.*-activation=relu-layers=5.*-width={WIDTH}$", "111111", "100000", "011111", "000000"], # 5
    ],
]
PLOT_NAMES = [
    "Simple case",
]

WITHIN_PLOT_NAMES = [
    [
        r"$\bf{c.\ Layerwise\ Weight\ Magnitude\ Distribution}$",
    ],
]

PKL = None
with open('SAVED_EXPLANATION.pkl', 'rb') as f:
    PKL = pickle.load(f)

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
        indices = [i + 1 for i, bit in enumerate(bitstring) if bit == '0']
        zero_index = indices[0]
        return "Replacing layers " + str(zero_index + 1) + "-" + str(len(bitstring))
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

    # first_part = "$\\bf{" + first_part + "}$"

    second_part = processed_parts[1]
    if second_part.startswith("rates"):
        second_part_end = second_part.split("=")[-1]
        # second_part_end is a 0101010 string, convert it to a word
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
    return second_part, index

def fff(x):
        # All layers -> 1
    # l2 -> 2
    # layer x -> 3
    # no replacement -> 4
    print ("Processing:", x)
    if "111111" in x.lower():
        return 1
    elif "100000" in x.lower():
        return 2
    elif "000000" in x.lower() or "none" in x.lower():
        return 20
    else:
        return 3

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
        
        total_layers = runs[0]['config']['num_hidden_layers']
        layers_dict = {}
        for layer in range(total_layers):
            all_binned = []

            for run in runs:
                run_name = run['run_name']
                print(f"        Processing run: {run_name} | util type=", run['config']["util_type"])
                if run["config"]["util_type"] == "adaptable_contribution":
                    print("    Skipping")
                    continue
                try:
                    metric = METRIC
                    metric += str(layer)

                    errs = extract_binned_errors(run, metric, STEP_METRIC, MAX_STEP, BIN_SIZE)
                    all_binned.append(errs)
                except Exception as e:
                    print(f"        [WARN] Error processing run {run_name}: {e}")
                    continue
            try:
                layers_dict[layer] = np.vstack(all_binned)
            except ValueError as e:
                print(f"        [ERROR] Could not stack data for group '{group}': {e}")
                layers_dict[layer] = np.array([])
        group_data[group] = layers_dict

    return group_data

if __name__ == '__main__':
    figs = []
    axss = []
    for i, group_set in enumerate(GROUPED_GROUPS):
        fig, axs = plt.subplots(1, len(GROUPED_GROUPS[i]), figsize=(9 * 1.05, 3.7 * 1.05))
        if len(GROUPED_GROUPS[i]) == 1:
            axs = [axs]
        figs.append(fig)
        axss.append(axs)

    wandb_data = defaultdict(default_dd_dd_list)
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
            groups_to_plot_on_this_ax = defaultdict(lambda : defaultdict(default_list))  # groups_to_plot_on_this_ax[layer][group_name] = (data_array_to_plot, color_index)
            best_group_for_pattern = {}
            for k, pattern_regex in enumerate(group_set[:3]):
                for actual_group_name, data_for_this_pattern_regex in wandb_data[i][j][k].items():
                    for layer, group_data_array in data_for_this_pattern_regex.items():
                        if  k == 2:
                            group_name, group_array = list(data_for_this_pattern_regex.items())[0]
                            groups_to_plot_on_this_ax[layer][group_name] = group_array
                        elif k == 1: # cbp
                            layer_configs_we_want = GROUPED_GROUPS[i][j][3:]

                            is_good = False
                            for layer_config in layer_configs_we_want:
                                if "=" + layer_config + "-" in actual_group_name:
                                    is_good = True
                                    break
                            if is_good:
                                groups_to_plot_on_this_ax[layer][actual_group_name] = group_data_array
                        else:
                            continue

            if not groups_to_plot_on_this_ax:
                current_ax.set_title(WITHIN_PLOT_NAMES[i][j], fontsize=14)
                current_ax.set_xlabel(XLABEL, fontsize=12)
                current_ax.set_ylabel(YLABEL, fontsize=12)

            pallete = plt.get_cmap('tab10')
            total_layers = len(groups_to_plot_on_this_ax)
            group_names = list(groups_to_plot_on_this_ax[0].keys())
            group_names = list(sorted(group_names, key=lambda x: fff(x)))

            for layer, layers_map in groups_to_plot_on_this_ax.items():
                for group_ind, (group_name_to_plot) in enumerate(group_names):
                    data_array_to_plot = layers_map[group_name_to_plot]

                    if data_array_to_plot.ndim == 1:
                        data_array_to_plot = data_array_to_plot.reshape(1, -1)

                    if data_array_to_plot.shape[0] == 0:
                        continue
                    
                    mean_of_last_15_percent_per_run = np.nanmean(data_array_to_plot[:, -int(data_array_to_plot.shape[1] * 0.15):], axis=1)
                    mean_over_runs = np.nanmean(mean_of_last_15_percent_per_run)
                    std_over_runs = np.nanstd(mean_of_last_15_percent_per_run, axis=0) 
                    
                    x_pos = (total_layers + 2) * group_ind + layer
                    processed_name, color_idx = PROCESS_NAME(group_name_to_plot)

                    label_to_use = None
                    bar_color = pallete(color_idx)
                    if layer == 0: label_to_use = processed_name
                    else: bar_color = pallete(color_idx)

                    if MULTS: 
                        mean_over_runs *= MULTS[layer]
                        std_over_runs *= MULTS[layer]

                    current_ax.bar(x_pos, mean_over_runs, yerr=std_over_runs, capsize=5,
                                    color=bar_color, label=label_to_use, ecolor='black', width=0.82, 
                                    edgecolor='black', linewidth=2)
            try:
                handles, labels = current_ax.get_legend_handles_labels()
                current_ax.legend(handles, labels, loc='upper left', fontsize=15)
            except Exception as e:
                print(f"  [WARN] Could not create legend for subplot '{WITHIN_PLOT_NAMES[i][j]}': {e}")
            current_ax.set_xlabel(XLABEL, fontsize=20)
            current_ax.set_ylabel(YLABEL, fontsize=20)
            current_ax.set_title(WITHIN_PLOT_NAMES[i][j], fontsize=19)
            num_x_ticks = 5 
            x_ticks = np.arange(0, (total_layers + 2) * len(layers_map.items()), 1)
            x_tick_labels = []
            for layer in range(total_layers):
                x_tick_labels.append(f"L{layer + 1}")
            x_tick_labels += [""] * 2
            x_tick_labels = x_tick_labels * len(layers_map.items())
            current_ax.set_xticks(x_ticks)
            current_ax.set_xticklabels(x_tick_labels, fontsize=16)
            current_ax.tick_params(axis='y', labelsize=20)
            current_ax.ticklabel_format(style='scientific', axis='y', scilimits=(4, 4)) 
            current_ax.yaxis.get_offset_text().set_fontsize(18)

            current_ax.grid(True, linestyle='--', alpha=0.6, axis='y')  # Add grid for better readability
        
        fig.tight_layout()
        fig.subplots_adjust(left=0.1, bottom=0.179, right=0.984, top=0.895)  # Adjust margins for the entire figure
        nm = YLABEL + "--" + PLOT_NAMES[i]
        fig_name = f"{WIDTH}_layerwise_weights.pdf"
        fig.savefig(fig_name, dpi=600)

        print(f"Saved figure {i+1}/{len(figs)} as '{fig_name}'")
