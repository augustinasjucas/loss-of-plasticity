import numpy as np
import matplotlib.pyplot as plt
import pickle

# Configuration
# ENTITY = 'augustinasj'
# PROJECT = 'mnist-hyperparam'
METRIC = 'error'
STEP_METRIC = 'step'
MAX_STEP = 3000000
BIN_SIZE = 40000


GROUPS_ONE = [
    ("flip_one-lr=0.01", "Learning rate = 0.01"),
    ("flip_one-lr=0.003", "Learning rate = 0.003"),
    ("flip_one-lr=0.001", "Learning rate = 0.001"),
]
TITLE_ONE = r"a. Flip-One Dataset"
PNG_NAME_ONE = "groups_one.pdf"

GROUPS_ALL = [
    ("flip_all-lr=0.01", "Learning rate = 0.01"),
    ("flip_all-lr=0.003", "Learning rate = 0.003"),
    ("flip_all-lr=0.001", "Learning rate = 0.001"),

]
TITLE_ALL = r"b. Flip-All Dataset"
PNG_NAME_ALL = "groups_all.pdf"

# Load the data globally
PKL = None
import pickle
with open('SAVED_SCR.pkl', 'rb') as f:
    PKL = pickle.load(f)

for (GROUPS, TITLE, PNG_NAME) in [(GROUPS_ONE, TITLE_ONE, PNG_NAME_ONE), (GROUPS_ALL, TITLE_ALL, PNG_NAME_ALL)]:

    def fetch_runs(group):
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
            group_data[group] = (np.vstack(all_binned), run_name)
        return group_data

    if __name__ == '__main__':
        group_data = get_group_data([g[0] for g in GROUPS])

        plt.figure(figsize=(5, 5))
        x = np.arange(int(MAX_STEP // BIN_SIZE)) * BIN_SIZE
        for group, (performances, _) in group_data.items():
            mean_err = np.nanmean(performances, axis=0)
            std_err = np.nanstd(performances, axis=0)
            label = next((name for key, name in GROUPS if key == group), group)
            plt.plot(x, mean_err, label=label, linewidth=2.7)
            plt.fill_between(x, mean_err - std_err, mean_err + std_err, alpha=0.2)

        plt.xlim(0, MAX_STEP)
        plt.xlabel('Sample Index', fontsize=20)
        plt.ylabel('Error', fontsize=20)
        plt.ylim(0.4, 2.1)
        plt.title(TITLE, fontsize=22, fontweight='bold')
        plt.legend(fontsize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xticks(np.arange(0, MAX_STEP + 1, 1000000), [f"{i//1000000}M" for i in range(0, MAX_STEP + 1, 1000000)], fontsize=16)
        plt.grid(True, axis='both', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        plt.subplots_adjust(top=0.932, bottom=0.126, left=0.16, right=0.961)
        plt.savefig(PNG_NAME, dpi=400)
