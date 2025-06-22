# The Bit-Flipping Problem
This folder contains the implementation of the Continual Permuted MNIST experiment, also the dataset and data analysis scripts.


## Generating Data
The data needs to be generated just once at the beggining as follows. First, run:
```
python3 multi_param_expr.py -c cfg/data.json
```
This will generate 15  `.json` files at the folder `env_temp_cfg/`. Then, for each `.json` file f in that folder, run:
```
python3 generate.py -c <path_to_f>
```

This will generate the necessary data in `data` folder.

However, when running `one-vs-all` experiments (described later), different data needs to be generated. In particular, instead of using `cfg/data.json`, this whole process needs to be repeated for `1-one-vs-all/data-generation/flip_all.json` and then for `lop/bit_flipping/cfg/1-one-vs-all/data-generation/flip_one.json`.

## Experimental Setups
The `cfg` directory contains 3 folders, each representing an experimental setup:
- [cfg/1-one-vs-all/](cfg/1-one-vs-all) contains the experiment specifications from subsection *Flipping One Bit vs. Flipping All Bits*.
- [cfg/2-hyperparameters/](cfg/2-hyperparameters) contains the experiment specifications for the main experiment described in *Problems With Scaling Up the Model* subsection.
- [cfg/3-contribution-and-different-replacement-rates/](cfg/3-contribution-and-different-replacement-rates/) contains experiments relating to *Differing Utility Score Definitions* subsection as well as some undescribed experiments relating to layerwise replacement rates.

Note that logging is done using Weights and Biases. And the wandb entity, into which logs are pushed is a parameter of an experiment. Therefore, you will probably want to change the   `wandb_entity` field of every `.json` file within the [`cfg/`](cfg) folder to match your own.

## Running an Experiment

Every `.json` file in [cfg](cfg) folder correponds to some set of experimental settings.

After selecting which of these 4 experiments you want to run, the following algorithm needs to be performed:

For every `.json` file F (corresponding to a set of experimental settings) in the experiment's folder:
  1. Generate run configurations from file F, by running `python3 multi_param_expr.py -c <F>`. This will generate a number of json files in `temp_cfg/` folder, each file corresponding to a single run instance of the experiment.
  2. For each file R in `temp_cfg/` folder, run `python3 online_expr.py -c <R> --index <i>`. Note that the index `i` here is just for semantic purposes to earier distinguish between runs in Weights and Biases.

After performing these steps, the experimental results will all be residing in Weights and Biases, in the project defined in the experiment files.

For inspecting the results in Weights and Biases manually (and not using our analysis scripts), we suggest using the wandb panel to group all results by the *Group* field, since we give all runs which correspond to the same exact set of hyperparameters, the same group name, and the individual runs simply correspond to different random seeds.

## Analysing the Experimental Results

After experiments have been run, all necessary data for analysis will be residing in Weights and Biases. These results can then be analyzed by the scripts in the [plots](plots) folder, which contains all scripts we used for generating figures the paper.

To generate the figures we used, follow these steps:
1. Enter the wanted directory. For instance, `cd plots/fig-8-flip-one-vs-all`. This folder contains scripts for generating Figure 8 from the paper.
2. If the folder contains `fetch.py`, first this script must be run - it will download all data from the needed Weights and Biases project and store it in a local pickle file, so that every rerun of the plotting script does not have to perform expensive fetching from the internet. Note that the `fetch.py` file contains a variable `entity`, which corresponds to the wandb entity the data should be downloaded from - this likely needs to be changed to match your own.
3. Once `fetch.py` was (potentially) run, you can simply call `plot.py` which will generate a either `pdf` file. Note that the numbers in the folder names are named correspond to the Figures in the paper. For instance, `fig-8-flip-one-vs-all/plot.py` will generate the equivalent of Figure 8 from the paper.