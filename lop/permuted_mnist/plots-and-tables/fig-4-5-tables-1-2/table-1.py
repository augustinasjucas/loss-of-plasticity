import numpy as np
import pandas as pd
import pickle

with open('SAVED_POST_HYPERPARAM.pkl', 'rb') as f:
    PKL = pickle.load(f)

METRIC = 'accuracy'
STEP_METRIC = 'task'
MAX_STEP = 1100
BIN_SIZE = 1

EXPERIMENTS = [
    (
        "Varying Widths",
        [
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.001-repl=0.001-decay=0-width=20', "Width 20", ["111111", "100000", "010000", "001000", "000010"]),
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=50', "Width 50", ["111111", "100000", "010000", "001000", "000010"]),
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=100', "Width 100", ["111111", "100000", "010000", "001000", "000010"]),
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=150', "Width 150", ["111111", "100000", "010000", "001000", "000010"]),
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=200', "Width 200", ["111111", "100000", "010000", "001000", "000010"])
        ]
    ),
    (
        "Varying Activation Functions",
        [
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=100', "ReLU", ["111111", "100000", "010000", "001000", "000010"]),
            ('agent=cbp-rates={}-activation=selu-layers=5-lr=0.0005-repl=0.0001-decay=0-width=100', "SELU", ["111111", "100000", "010000", "001000", "000010"]),
            ('agent=cbp-rates={}-activation=tanh-layers=5-lr=0.0005-repl=1e-06-decay=0-width=100', "Tanh", ["111111", "100000", "010000", "001000", "000010"]),
        ]
    ),
    (
        "Varying Depths",
        [
            ('agent=cbp-rates={}-activation=relu-layers=2-lr=0.005-repl=0.0001-decay=0-width=100', "Depth 2", ["111", "100", "010", "-", "010"]),
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=100', "Depth 5", ["111111", "100000", "010000", "001000", "000010"]),
            ('agent=cbp-rates={}-activation=relu-layers=8-lr=0.001-repl=0.0001-decay=0-width=100', "Depth 8", ["111111111", "100000000", "010000000", "001000000", "000000010"]),
        ]
    ),

    (
        r"Varying Depths, Same Number of Parameters",
        [
            ('agent=cbp-rates={}-activation=relu-layers=2-lr=0.005-repl=1e-05-decay=0-width=129', "Depth 2", ["111", "100", "010", "-", "010"]),
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=100', "Depth 5", ["111111", "100000", "010000", "001000", "000010"]),
            ('agent=cbp-rates={}-activation=relu-layers=8-lr=0.001-repl=0.0001-decay=0-width=86', "Depth 8", ["111111111", "100000000", "010000000", "001000000", "000000010"]),
        ]
    ),

]
#  agent=cbp-rates=100000000-activation=relu-layers=8-lr=0.001-repl=0.0001-decay=0-width=86


def fetch_runs(group=None):
    # fetch from PKL dict
    return list(PKL[group].values())

def extract_binned_errors(run, metric, step_metric, max_step, bin_size):
    if run['config'].get('util_type') != 'contribution':
        raise ValueError(f"Run {run.get('id', 'N/A')} is not a contribution run. Skipping.")

    df = run['run_history']
    df = df[df[step_metric] <= max_step].copy() # Use .copy() to avoid SettingWithCopyWarning

    if df.empty:
        return np.array([np.nan] * int(max_step // bin_size))
    num_bins = int(max_step // bin_size)
    bins = np.arange(0, max_step + bin_size, bin_size)
    labels = np.arange(num_bins) # Use simple integer labels for bins
    df['bin'] = pd.cut(df[step_metric], bins=bins, labels=labels, right=True, include_lowest=False)
    binned_means = df.dropna(subset=['bin']).groupby('bin', observed=True)[metric].mean()
    result = binned_means.reindex(labels, fill_value=np.nan)
    
    return result.to_numpy()


def get_group_data(group):
   
    runs = fetch_runs(group=group)
    all_binned = []
    for run in runs:
        run_name = run['run_name']
        if run["config"]["util_type"] == "adaptable_contribution":
            continue
        try:
            max_step = MAX_STEP
            errs = extract_binned_errors(run, METRIC, STEP_METRIC, max_step, BIN_SIZE)
            all_binned.append(errs)
        except Exception as e:
            continue
    return (np.vstack(all_binned), run_name)

def bf(text):
    """
    Format text for bold in LaTeX.
    """
    return r"\textbf{" + str(text) + "}"

def f (binary_string):
    # extract the position of the "1" string in the binary string
    # if is all ones, return "All Layers"
    if binary_string == "1" * len(binary_string):
        return r"\textit{Repl. All}"
    position = binary_string.find("1")
    return f"Layer {position + 1}"


if __name__ == "__main__":

    print(r"\toprule")
    print(r"& Baseline & \multicolumn{4}{c}{Replacing Individual Layers} \\")
    print(r"\cmidrule(lr){2-2} \cmidrule(l){3-6}")
    cols = ["All Layers", "Layer 1", "Layer 2", "Layer 3", "Last layer"]
    cols = [bf(col) for col in cols]
    print(f"{'':>10}", end='   ')
    print( "&" + " & ".join(f"{bf(col):>25}" for col in cols) + " \\\\")
    print(r"\midrule")

    for e, (experiment_name, rows) in enumerate(EXPERIMENTS):
        print(r"\multicolumn{4}{l}{\textbf{" + f"{experiment_name} " + r"}} \\ ")

        for i, (pattern, row_name, cols) in enumerate(rows):
            print("", end=' ')
            print(f"{row_name:>10}", end=' & ')
            
            row_means = []
            row_data_strings = []

            for col in cols:
                if col == "-":
                    row_means.append(np.nan)
                    row_data_strings.append("-")
                    continue
                group_name = pattern.format(col)
                data = get_group_data(group_name) 
                means = np.nanmean(data[0][:, -int(data[0].shape[1] * 0.15):], axis=1)
                avg = np.nanmean(means) * 100
                std = np.nanstd(means) * 100
                
                row_means.append(avg)
                row_data_strings.append(f"{avg:.1f} ± {std:.1f}")

            max_mean_in_row = np.nanmax(row_means[1:])

            for ii, stri in enumerate(row_data_strings):
                current_mean = row_means[ii]
                if not np.isnan(current_mean) and current_mean == max_mean_in_row:
                    stri = r"\underline{" + stri + "}"
                if ii == 0:
                    stri = r"\textit{" + stri + "}"
                end = ' & ' if ii < len(cols) - 1 else ' '
                print(f"{stri:>25}", end=end)
            print("\\\\")

            # add spacing between rows
            print(r"\addlinespace[0.2em]")

        print(r"\midrule")
        if e < len(EXPERIMENTS) - 1:
            print()
            print()
    print(r"\bottomrule")