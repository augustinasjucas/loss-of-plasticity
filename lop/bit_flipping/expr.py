import sys
import json
import pickle
import argparse
from lop.nets.ffnn import FFNN
from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from lop.utils.miscellaneous import *
import wandb
import os
import time

WANDB_LOGS_DIR = '/tmp/ajucas/wandb-mnist-continued/'

def mean_ignore_infs(tensor: torch.Tensor) -> torch.Tensor:
    finite_vals = tensor[torch.isfinite(tensor)]
    if finite_vals.numel() == 0:
        return torch.tensor(float('nan'), dtype=tensor.dtype, device=tensor.device)
    return finite_vals.mean()

def expr(params: {}, index):
    agent_type = params['agent']
    env_file = params['env_file']
    num_data_points = int(params['num_data_points'])
    to_log = False
    to_log_grad = False
    to_log_activation = False
    beta_1 = 0.9
    beta_2 = 0.999
    weight_decay = 0.0
    accumulate = False
    perturb_scale = 0
    if 'to_log' in params.keys():
        to_log = params['to_log']
    if 'to_log_grad' in params.keys():
        to_log_grad = params['to_log_grad']
    if 'to_log_activation' in params.keys():
        to_log_activation = params['to_log_activation']
    if 'beta_1' in params.keys():
        beta_1 = params['beta_1']
    if 'beta_2' in params.keys():
        beta_2 = params['beta_2']
    if 'weight_decay' in params.keys():
        weight_decay = params['weight_decay']
    if 'accumulate' in params.keys():
        accumulate = params['accumulate']
    if 'perturb_scale' in params.keys():
        perturb_scale = params['perturb_scale']

    num_inputs = params['num_inputs']
    num_features = params['num_features']
    hidden_activation = params['hidden_activation']
    step_size = params['step_size']
    opt = params['opt']
    replacement_rate = params["replacement_rate"]
    decay_rate = params["decay_rate"]
    additional_layers = params['additional_layers']
    replacement_rates = params['replacement_rates']
    compute_scores = True if 'compute_scores' not in params else params['compute_scores']
    mt = 10
    util_type='adaptable_contribution'
    init = 'kaiming'
    if "mt" in params.keys():
        mt = params["mt"]
    if "util_type" in params.keys():
        util_type = params["util_type"]
    if "init" in params.keys():
        init = params["init"]

    net = FFNN(
        input_size=num_inputs,
        num_features=num_features,
        hidden_activation=hidden_activation,
        additional_layers=additional_layers,
    )

    if agent_type == 'bp' or agent_type == 'linear' or agent_type == 'l2':
        
        if compute_scores:
            utility_params = {
                'replacement_rate': 0,
                'maturity_threshold': -1,
                'decay_rate': decay_rate,
                'util_type': util_type,
                'init': init,
                'accumulate': accumulate,
                'replacement_rates': []
            }
        else: utility_params = {}


        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            beta_1=beta_1,
            beta_2=beta_2,
            weight_decay=weight_decay,
            to_perturb=(perturb_scale > 0),
            perturb_scale=perturb_scale,
            utility_params=utility_params,
            compute_utility=compute_scores,
        )
    elif agent_type == 'cbp':
        learner = ContinualBackprop(
            net=net,
            step_size=step_size,
            opt=opt,
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            device='cpu',
            maturity_threshold=mt,
            util_type=util_type,
            init=init,
            accumulate=accumulate,
            replacement_rates=replacement_rates
        )
    os.makedirs(WANDB_LOGS_DIR + index, exist_ok=True)

    params['index'] = index

    # Initialize Weights & Biases
    wandb.init(
        project=params['wandb_project'],
        config=params,
        name=str(index) + " | " + params['wandb_run_name'],
        group=params['wandb_group'],
        dir=WANDB_LOGS_DIR + index
    )

    # Define wandb metrics
    wandb.define_metric("error", summary="mean", step_metric="step")
    if compute_scores:
        for i in range(0, len(learner.net.layers), 2):
            wandb.define_metric("avg_scores/layer_{}".format(i), summary="mean", step_metric="step")
            wandb.define_metric("corrected_avg_scores/layer_{}".format(i), summary="mean", step_metric="step")
            wandb.define_metric("median_scores/layer_{}".format(i), summary="mean", step_metric="step")
            wandb.define_metric("corrected_median_scores/layer_{}".format(i), summary="mean", step_metric="step")
            wandb.define_metric("min_scores/layer_{}".format(i), summary="mean", step_metric="step")
            wandb.define_metric("max_scores/layer_{}".format(i), summary="mean", step_metric="step")
            wandb.define_metric("weight_magnitude/layer_{}".format(i), summary="mean", step_metric="step")
            wandb.define_metric("last_features_act/layer_{}".format(i), summary="mean", step_metric="step")
            wandb.define_metric("mean_feature_act/layer_{}".format(i), summary="mean", step_metric="step")
        parts = 10
        for i in range(parts+1):
            x = 100 // parts * i
            wandb.define_metric(f"score-evolution/{x}%", summary="mean", step_metric="layer_index")
            wandb.define_metric(f"corrected-score-evolution/{x}%", summary="mean", step_metric="layer_index")
            wandb.define_metric(f"score-median-evolution/{x}%", summary="mean", step_metric="layer_index")
            wandb.define_metric(f"corrected-score-median-evolution/{x}%", summary="mean", step_metric="layer_index")
            wandb.define_metric(f"score-min-evolution/{x}%", summary="mean", step_metric="layer_index")
            wandb.define_metric(f"score-max-evolution/{x}%", summary="mean", step_metric="layer_index")
            wandb.define_metric(f"weight-magnitude-evolution/{x}%", summary="mean", step_metric="layer_index")
            wandb.define_metric(f"last-features-act-evolution/{x}%", summary="mean", step_metric="layer_index")
            wandb.define_metric(f"mean-feature-act-evolution/{x}%", summary="mean", step_metric="layer_index")

    with open(env_file, 'rb+') as f:
        inputs, outputs, _ = pickle.load(f)

    errs = torch.zeros((num_data_points), dtype=torch.float)
    if to_log: weight_mag = torch.zeros((num_data_points, 2), dtype=torch.float)
    if to_log_grad: grad_mag = torch.zeros((num_data_points, 2), dtype=torch.float)
    if to_log_activation: activation = torch.zeros((num_data_points, ), dtype=torch.float)

    start = time.time()

    for i in range(num_data_points):
        if i % 1000 == 0:
            now = time.time()
            elapsed = now - start
            remaining = (num_data_points - i) * (elapsed / (i + 1))
            print("i =", i, "Remaining:", remaining, "in minutes:", remaining/60, flush=True)

        x, y = inputs[i: i+1], outputs[i: i+1]
        err = learner.learn(x=x, target=y)
        if to_log:
            weight_mag[i][0] = learner.net.layers[0].weight.data.abs().mean()
            weight_mag[i][1] = learner.net.layers[-1].weight.data.abs().mean()
        if to_log_grad:
            grad_mag[i][0] = learner.net.layers[0].weight.grad.data.abs().mean()
            grad_mag[i][1] = learner.net.layers[-1].weight.grad.data.abs().mean()
        if to_log_activation:
            if hidden_activation == 'relu':
                activation[i] = (learner.previous_features[0] == 0).float().mean()
            if hidden_activation == 'tanh':
                activation[i] = (learner.previous_features[0].abs() > 0.9).float().mean()
        errs[i] = err

        # Log to wandb
        temp_log = {}
        if compute_scores:
            for j in range(0, len(learner.net.layers) // 2):
                temp_log["avg_scores/layer_{}".format(j)] = mean_ignore_infs(learner.gnt.util[j])
                temp_log["corrected_avg_scores/layer_{}".format(j)] = mean_ignore_infs(learner.gnt.bias_corrected_util[j])
                temp_log["median_scores/layer_{}".format(j)] = learner.gnt.util[j].median()
                temp_log["corrected_median_scores/layer_{}".format(j)] = learner.gnt.bias_corrected_util[j].median()
                temp_log["min_scores/layer_{}".format(j)] = learner.gnt.util[j].min()
                temp_log["max_scores/layer_{}".format(j)] = learner.gnt.util[j].max()
                temp_log["weight_magnitude/layer_{}".format(j)] = learner.net.layers[j*2].weight.data.abs().mean()
                temp_log["last_features_act/layer_{}".format(j)] = learner.gnt.last_features_act[j].abs().mean()
                temp_log["mean_feature_act/layer_{}".format(j)] = learner.gnt.mean_feature_act[j].abs().mean()

            if i == 0 or (i+1) % (num_data_points // parts) == 0:
                progress = (i+1) // (num_data_points // parts) 
                percentage = 100 // parts * progress
                # for every layer, log the score distribution
                for j in range(0, len(learner.net.layers) // 2):
                    wandb.log({
                        f"score-evolution/{percentage}%": mean_ignore_infs(learner.gnt.util[j]),
                        f"corrected-score-evolution/{percentage}%": mean_ignore_infs(learner.gnt.bias_corrected_util[j]),
                        f"score-median-evolution/{percentage}%": learner.gnt.util[j].median(),
                        f"corrected-score-median-evolution/{percentage}%": learner.gnt.bias_corrected_util[j].median(),
                        f"score-min-evolution/{percentage}%": learner.gnt.util[j].min(),
                        f"score-max-evolution/{percentage}%": learner.gnt.util[j].max(),
                        f"weight-magnitude-evolution/{percentage}%": learner.net.layers[j*2].weight.data.abs().mean(),
                        f"last-features-act-evolution/{percentage}%": learner.gnt.last_features_act[j].abs().mean(),
                        f"mean-feature-act-evolution/{percentage}%": learner.gnt.mean_feature_act[j].abs().mean(),
                        "layer_index": j,
                    })
        temp_log["error"] = err
        temp_log["step"] = i
        if i % 200 == 0:
            wandb.log(temp_log)

    data_to_save = {
        'errs': errs.numpy()
    }
    if to_log:
        data_to_save['weight_mag'] = weight_mag.numpy()
    if to_log_grad:
        data_to_save['grad_mag'] = grad_mag.numpy()
    if to_log_activation:
        data_to_save['activation'] = activation.numpy()
    return data_to_save


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path to the file containing the parameters for the experiment",
                        type=str, default='temp_cfg/0.json')
    parser.add_argument('--index', type=str)

    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    data = expr(params, index=args.index)

    os.makedirs(os.path.dirname(params['data_file']), exist_ok=True)

    with open(params['data_file'], 'wb+') as f:
        pickle.dump(data, f)

    wandb.finish()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
