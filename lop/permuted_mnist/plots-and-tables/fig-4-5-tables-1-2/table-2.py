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
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.001-repl=0.001-decay=0-width=20', "Width 20", ["111111", "100000", "011111", "000000"]),
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=50', "Width 50", ["111111", "100000", "011111", "000000"]),
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=100', "Width 100", ["111111", "100000", "011111", "000000"]),
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=150', "Width 150", ["111111", "100000", "011111", "000000"]),
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=200', "Width 200", ["111111", "100000", "011111", "000000"])
        ]
    ),
    # (
    #     "Varying Activation Functions",
    #     [
    #         ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=100', "ReLU", ["111111", "100000", "011111", "000000"]),
    #         ('agent=cbp-rates={}-activation=selu-layers=5-lr=0.0005-repl=0.0001-decay=0-width=100', "SELU", ["111111", "100000", "011111", "000000"]),
    #         ('agent=cbp-rates={}-activation=tanh-layers=5-lr=0.0005-repl=1e-06-decay=0-width=100', "Tanh", ["111111", "100000", "011111", "000000"]),
    #     ]
    # ),
    (
        "Varying Depths",
        [
            ('agent=cbp-rates={}-activation=relu-layers=2-lr=0.005-repl=0.0001-decay=0-width=100', "Depth 2", ["111", "100", "011", "000"]),
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=100', "Depth 5", ["111111", "100000", "011111", "000000"]),
            ('agent=cbp-rates={}-activation=relu-layers=8-lr=0.001-repl=0.0001-decay=0-width=100', "Depth 8", ["111111111", "100000000", "011111111", "000000000"]),
        ]
    ),

    (
        r"Varying Depths, Same Number of Parameters",
        [
            ('agent=cbp-rates={}-activation=relu-layers=2-lr=0.005-repl=1e-05-decay=0-width=129', "Depth 2", ["111", "100", "011", "000"]),
            ('agent=cbp-rates={}-activation=relu-layers=5-lr=0.005-repl=0.0001-decay=0-width=100', "Depth 5", ["111111", "100000", "011111", "000000"]),
            ('agent=cbp-rates={}-activation=relu-layers=8-lr=0.001-repl=0.0001-decay=0-width=86', "Depth 8", ["111111111", "100000000", "011111111", "000000000"]),
        ]
    ),
]

def fetch_runs(group=None):
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
            if "tanh" in group:
                max_step = 610
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
    ones = binary_string.count("1")
    if ones == 0:
        return "No layers"
    elif ones == 1:
        return "Layer 1"
    elif ones == len(binary_string):
        return "All Layers"
    else:
        return f"Layers 2-L"

if __name__ == "__main__":

    print(r"\toprule")

    print(f"{'':>10}", end='   ')
    print( "&" + " & ".join(f"{bf(f(col)):>25}" for col in EXPERIMENTS[0][1][0][2]) + " \\\\")
    print(r"\midrule")
    print()

    for e, (experiment_name, rows) in enumerate(EXPERIMENTS):
        print(r"\multicolumn{4}{l}{\textbf{" + f"{experiment_name} " + r"}} \\ ")

        for i, (pattern, row_name, cols) in enumerate(rows):
            print("", end=' ')
            print(f"{row_name:>10}", end=' & ')
            
            row_means = []
            row_data_strings = []

            for col in cols:
                group_name = pattern.format(col)
                data = get_group_data(group_name) 
                means = np.nanmean(data[0][:, -int(data[0].shape[1] * 0.15):], axis=1)
                avg = np.nanmean(means) * 100
                std = np.nanstd(means) * 100

                row_means.append(avg)
                row_data_strings.append(f"{avg:.1f} ± {std:.1f}")


            for ii, stri in enumerate(row_data_strings):
                current_mean = row_means[ii]
                end = ' & ' if ii < len(cols) - 1 else ' '
                print(f"{stri:>25}", end=end)
            print("\\\\")

            print(r"\addlinespace[0.2em]")

        print(r"\midrule")
        if e < len(EXPERIMENTS) - 1:
            print()
            print()
    print(r"\bottomrule")