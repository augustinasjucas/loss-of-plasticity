import sys
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from torch.nn.functional import softmax
from lop.nets.deep_ffnn import DeepFFNN
from lop.utils.miscellaneous import nll_accuracy, compute_matrix_rank_summaries
import wandb
import os
import time
import torch.nn as nn
from lop.algos.hessian import compute_effective_hessian_ranks
from lop.bit_flipping.expr import mean_ignore_infs

# Can be changed if needed
WANDB_LOGS_DIR = '/tmp/ajucas/wandb-mnist-continued/'

def compute_accuracy(X, y, net):
    with torch.no_grad():
        output = net.predict(X)[0]
        predictions = softmax(output, dim=1)
        accuracy = nll_accuracy(predictions, y).item()
    return accuracy

def get_probing_accuracies(net, x, y):
    """
    Computes probing accuracies for each layer of the network.
    Procedure defined in the paper.
    """
    original_mode_is_train = net.training
    net.eval()
    device = x.device
    num_classes = 10
    accuracies = []

    # Iterate through each layer whose output can be probed
    for i in range(1, len(net.layers), 2):
        # Create a sub-network that outputs the activations of the current layer
        # net.layers[0]...net.layers[i]
        current_probing_modules = net.layers[:i+1]
        probed_sub_network = nn.Sequential(*current_probing_modules).to(device)

        # Get activations of the whole dataset - this is the probe input
        with torch.no_grad():
            h = probed_sub_network(x)
        
        h = h.detach()

        probe_input_dim = h.shape[1]
        probe_classifier = nn.Linear(probe_input_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(probe_classifier.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        max_probe_epochs = 50  # Maximum number of epochs to train the probe
        epochs_without_improvement = 0
        best_accuracy = -1.0
        current_epoch_accuracy = torch.tensor(float('nan'), device='cpu')

        # Train the probe classifier
        for epoch in range(max_probe_epochs):
            optimizer.zero_grad()
            predictions = probe_classifier(h)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                current_epoch_accuracy = nll_accuracy(softmax(predictions, dim=1), y).cpu()

            if current_epoch_accuracy.item() <= best_accuracy:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0
                best_accuracy = current_epoch_accuracy.item()

            if epochs_without_improvement >= 15:
                break 
        
        # Append the best accuracy for this layer
        accuracies.append(best_accuracy)

    if original_mode_is_train:
        net.train()

    return accuracies


def online_expr(params: {}, index):
    torch.set_num_threads(1)

    agent_type = params['agent']
    num_tasks = 200
    if 'num_tasks' in params.keys():
        num_tasks = params['num_tasks']
    if 'num_examples' in params.keys() and "change_after" in params.keys():
        num_tasks = int(params["num_examples"]/params["change_after"])

    step_size = params['step_size']
    opt = params['opt']
    weight_decay = 0
    use_gpu = 0
    dev = 'cpu'
    to_log = False
    num_features = 2000
    change_after = 10 * 6000
    to_perturb = False
    perturb_scale = 0.1
    num_hidden_layers = 1
    mini_batch_size = 1
    replacement_rate = 0.0001
    decay_rate = 0.99
    maturity_threshold = 100
    util_type = 'adaptable_contribution'
    total_to_take = 60000
    compute_scores = False
    replacement_rates = []
    act_type = 'relu'

    if 'to_log' in params.keys():
        to_log = params['to_log']
    if 'weight_decay' in params.keys():
        weight_decay = params['weight_decay']
    if 'num_features' in params.keys():
        num_features = params['num_features']
    if 'change_after' in params.keys():
        change_after = params['change_after']
    if 'use_gpu' in params.keys():
        if params['use_gpu'] == 1:
            use_gpu = 1
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if dev == torch.device("cuda"):    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'to_perturb' in params.keys():
        to_perturb = params['to_perturb']
    if 'perturb_scale' in params.keys():
        perturb_scale = params['perturb_scale']
    if 'num_hidden_layers' in params.keys():
        num_hidden_layers = params['num_hidden_layers']
    if 'mini_batch_size' in params.keys():
        mini_batch_size = params['mini_batch_size']
    if 'replacement_rate' in params.keys():
        replacement_rate = params['replacement_rate']
    if 'decay_rate' in params.keys():
        decay_rate = params['decay_rate']
    if 'maturity_threshold' in params.keys():
        maturity_threshold = params['mt']
    if 'util_type' in params.keys():
        util_type = params['util_type']
    if 'total_to_take' in params.keys():
        total_to_take = params['total_to_take']
    if 'compute_scores' in params.keys():
        compute_scores = params['compute_scores']
    if 'replacement_rates' in params.keys():
        replacement_rates = params['replacement_rates']
    if 'activation' in params.keys():
        act_type = params['activation']

    classes_per_task = 10
    images_per_class = 6000
    input_size = 784
    num_hidden_layers = num_hidden_layers
    net = DeepFFNN(input_size=input_size, num_features=num_features, num_outputs=classes_per_task,
                   num_hidden_layers=num_hidden_layers, act_type=act_type)

    param_cnt = 0
    for layer in net.layers:
        if hasattr(layer, 'weight'):
            param_cnt += layer.weight.numel()
        if hasattr(layer, 'bias'):
            param_cnt += layer.bias.numel()

    print("Total number of parameters: ", param_cnt)

    if agent_type in ['bp', 'linear', "l2"]:
        if compute_scores:
            utility_params = {
                'replacement_rate': 0,
                'maturity_threshold': -1,
                'decay_rate': decay_rate,
                'util_type': util_type,
                'accumulate': True
            }
        else: utility_params = {}

        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            weight_decay=weight_decay,
            device=dev,
            to_perturb=to_perturb,
            perturb_scale=perturb_scale,
            utility_params=utility_params,
            compute_utility=compute_scores,
        )
    elif agent_type in ['cbp']:
        learner = ContinualBackprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            decay_rate=decay_rate,
            util_type=util_type,
            accumulate=True,
            device=dev,
            replacement_rates=replacement_rates,
        )

    accuracy = nll_accuracy
    examples_per_task = images_per_class * classes_per_task
    total_examples = int(num_tasks * change_after)
    total_iters = int(total_examples/mini_batch_size)
    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks/10)

    accuracies = torch.zeros(total_iters, dtype=torch.float)
    weight_mag_sum = torch.zeros((total_iters, num_hidden_layers+1), dtype=torch.float)

    rank_measure_period = total_to_take
    effective_ranks = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    approximate_ranks = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    approximate_ranks_abs = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    ranks = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    dead_neurons = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)

    # Make a directory for storing wandb logs for this run
    os.makedirs(WANDB_LOGS_DIR + index, exist_ok=True)

    params['index'] = index
    params['param_cnt'] = param_cnt

    # Initialize Weights & Biases
    wandb.init(
        project=params['wandb_project'],
        config=params,
        name=str(index) + " | " + params['wandb_run_name'],
        group=params['wandb_group'],
        dir=WANDB_LOGS_DIR + index
    )

    # Define the wandb metrics
    for end in ["", "_last_30"]:
        wandb.define_metric("accuracy" + end, summary="mean", step_metric="task")
        wandb.define_metric("weight_mag_sum" + end, summary="mean", step_metric="task")
        wandb.define_metric("dead_neurons_sum" + end, summary="mean", step_metric="task")
        wandb.define_metric("total_hessian" + end, summary="mean", step_metric="task")
        wandb.define_metric("post_train_accuracy" + end, summary="mean", step_metric="task")

        if compute_scores:
            for i in range(0, len(learner.net.layers) // 2):
                wandb.define_metric("avg_scores/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("corrected_avg_scores/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("median_scores/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("corrected_median_scores/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("min_scores/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("max_scores/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("weight_magnitude/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("last_features_act/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("mean_feature_act/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("dead_neurons/layer_{}".format(i) + end, summary="mean", step_metric="task")

                # rank, effective rank, approximate rank, and absolute approximate rank
                wandb.define_metric("rank/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("effective_rank/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("approximate_rank/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("abs_approximate_rank/layer_{}".format(i) + end, summary="mean", step_metric="task")

                # For probing accuracies
                wandb.define_metric("probing_accuracy_pre/layer_{}".format(i) + end, summary="mean", step_metric="task")
                wandb.define_metric("probing_accuracy_post/layer_{}".format(i) + end, summary="mean", step_metric="task")

                # Hessians
                wandb.define_metric("hessian/layer_{}".format(i) + end, summary="mean", step_metric="task")

    # These metrics can be set to False if not needed, as they slow down the training quite a lot
    do_hessian = True
    do_probing = True

    iter = 0
    with open('data/mnist_', 'rb+') as f:
        X, Y, _, _ = pickle.load(f)
        if use_gpu == 1:
            X = X.to(dev)
            Y = Y.to(dev)

    change_after = min(change_after, total_to_take)
    probing_accuracies_pre = []
    probing_accuracies_post = []
    
    # For time tracking
    START = time.time()
    
    for task_idx in (range(num_tasks)):
        start_time = time.time()

        # Initialize accumulators for data-dependent metrics
        if compute_scores:
            num_layers = len(learner.net.layers) // 2
            sum_avg_scores = torch.zeros(num_layers)
            sum_corrected_avg_scores = torch.zeros(num_layers)
            sum_medians = torch.zeros(num_layers)
            sum_corrected_medians = torch.zeros(num_layers)
            sum_mins = torch.zeros(num_layers)
            sum_maxs = torch.zeros(num_layers)
            sum_last_features_act = torch.zeros(num_layers)
            sum_mean_feature_act = torch.zeros(num_layers)
            sum_weight_magnitude = torch.zeros(num_layers)

            # Same thing just for the last 30% of the data
            sum_avg_scores_last_30 = torch.zeros(num_layers)
            sum_corrected_avg_scores_last_30 = torch.zeros(num_layers)
            sum_medians_last_30 = torch.zeros(num_layers)
            sum_corrected_medians_last_30 = torch.zeros(num_layers)
            sum_mins_last_30 = torch.zeros(num_layers)
            sum_maxs_last_30 = torch.zeros(num_layers)
            sum_last_features_act_last_30 = torch.zeros(num_layers)
            sum_mean_feature_act_last_30 = torch.zeros(num_layers)
            sum_weight_magnitude_last_30 = torch.zeros(num_layers)

            batch_count = 0
            batch_count_last_30 = 0

        new_iter_start = iter
        pixel_permutation = np.random.permutation(input_size)
        X = X[:, pixel_permutation]
        data_permutation = np.random.permutation(examples_per_task)
        x, y = X[data_permutation], Y[data_permutation]
        x = x[:total_to_take]
        y = y[:total_to_take]

        if agent_type != 'linear':
            with torch.no_grad():
                new_idx = int(iter / rank_measure_period)
                m = net.predict(x[:1000])[1]
                for rep_layer_idx in range(num_hidden_layers):
                    ranks[new_idx][rep_layer_idx], effective_ranks[new_idx][rep_layer_idx], \
                    approximate_ranks[new_idx][rep_layer_idx], approximate_ranks_abs[new_idx][rep_layer_idx] = \
                        compute_matrix_rank_summaries(m=m[rep_layer_idx], use_scipy=True)
                    dead_neurons[new_idx][rep_layer_idx] = (m[rep_layer_idx].abs().sum(dim=0) == 0).sum()
                
            # Hessian
            if do_hessian and task_idx % 5 == 0:
                hessian_ranks, hessian_rank_full = compute_effective_hessian_ranks(net, x[:2000], y[:2000])
        
        if do_probing and task_idx % 5 == 0:
            # Compute the probing accuracies for the current task
            start_time = time.time()
            probing_accuracies_pre = get_probing_accuracies(net, x, y)
            end_time = time.time()

        for start_idx in range(0, change_after, mini_batch_size):
            start_idx = start_idx % examples_per_task
            batch_x = x[start_idx: start_idx+mini_batch_size]
            batch_y = y[start_idx: start_idx+mini_batch_size]

            # train the network
            loss, network_output = learner.learn(x=batch_x, target=batch_y)

            if to_log and agent_type != 'linear':
                for idx, layer_idx in enumerate(learner.net.layers_to_log):
                    weight_mag_sum[iter][idx] = learner.net.layers[layer_idx].weight.data.abs().sum()
            # log accuracy
            with torch.no_grad():
                accuracies[iter] = accuracy(softmax(network_output, dim=1), batch_y).cpu()
            iter += 1

            # Compute the layerwise metrics
            if compute_scores and agent_type != 'linear':
                for j in range(0, len(learner.net.layers) // 2):
                    util_j = learner.gnt.util[j]
                    corrected_util_j = learner.gnt.bias_corrected_util[j]

                    # Compute metrics for this batch and accumulate
                    avg_score = mean_ignore_infs(util_j)
                    sum_avg_scores[j] += avg_score.item() if torch.is_tensor(avg_score) else avg_score

                    corrected_avg = mean_ignore_infs(corrected_util_j)
                    sum_corrected_avg_scores[j] += corrected_avg.item() if torch.is_tensor(corrected_avg) else corrected_avg

                    sum_medians[j] += util_j.median().item()
                    sum_corrected_medians[j] += corrected_util_j.median().item()

                    sum_mins[j] += util_j.min().item()
                    sum_maxs[j] += util_j.max().item()

                    sum_last_features_act[j] += learner.gnt.last_features_act[j].abs().mean().item()
                    sum_mean_feature_act[j] += learner.gnt.mean_feature_act[j].abs().mean().item()

                    sum_weight_magnitude[j] += learner.net.layers[j * 2].weight.data.abs().sum().item()

                    if iter > int(0.7 * change_after):
                        sum_avg_scores_last_30[j] += avg_score.item() if torch.is_tensor(avg_score) else avg_score
                        sum_corrected_avg_scores_last_30[j] += corrected_avg.item() if torch.is_tensor(corrected_avg) else corrected_avg
                        sum_medians_last_30[j] += util_j.median().item()
                        sum_corrected_medians_last_30[j] += corrected_util_j.median().item()
                        sum_mins_last_30[j] += util_j.min().item()
                        sum_maxs_last_30[j] += util_j.max().item()
                        sum_last_features_act_last_30[j] += learner.gnt.last_features_act[j].abs().mean().item()
                        sum_mean_feature_act_last_30[j] += learner.gnt.mean_feature_act[j].abs().mean().item()
                        sum_weight_magnitude_last_30[j] += learner.net.layers[j * 2].weight.data.abs().sum().item()
                        batch_count_last_30 += 1

                batch_count += 1

        if task_idx % 5 == 0:
            # Compute the probing accuracies for the current task
            if do_probing:
                start_time = time.time()
                probing_accuracies_post = get_probing_accuracies(net, x, y)
                end_time = time.time()

            # Compute post-training accuracies for the full model
            post_train_accuracy = compute_accuracy(x, y, net)
            end_time = time.time()

        # Log everything to wandb
        temp_log = {
            "accuracy": accuracies[new_iter_start:iter - 1].mean(),
            "weight_mag_sum": weight_mag_sum[new_iter_start:iter - 1].sum(dim=1).mean(),
            "dead_neurons_sum": dead_neurons[new_idx].sum(),
            "task": task_idx,
            "weight_mag_sum_last_30": weight_mag_sum[new_iter_start + int(0.7 * change_after):iter - 1].sum(dim=1).mean(),
            "accuracy_last_30": accuracies[new_iter_start + int(0.7 * change_after):iter - 1].mean(),
            "total_hessian":  hessian_rank_full if do_hessian else float('nan'),
            "post_train_accuracy": post_train_accuracy,
        }

        if compute_scores:
            for j in range(0, len(learner.net.layers) // 2):
                temp_log[f"avg_scores/layer_{j}"] = sum_avg_scores[j] / batch_count
                temp_log[f"corrected_avg_scores/layer_{j}"] = sum_corrected_avg_scores[j] / batch_count
                temp_log[f"median_scores/layer_{j}"] = sum_medians[j] / batch_count
                temp_log[f"corrected_median_scores/layer_{j}"] = sum_corrected_medians[j] / batch_count
                temp_log[f"min_scores/layer_{j}"] = sum_mins[j] / batch_count
                temp_log[f"max_scores/layer_{j}"] = sum_maxs[j] / batch_count
                temp_log[f"last_features_act/layer_{j}"] = sum_last_features_act[j] / batch_count
                temp_log[f"mean_feature_act/layer_{j}"] = sum_mean_feature_act[j] / batch_count
                temp_log[f"weight_magnitude/layer_{j}"] = sum_weight_magnitude[j] / batch_count
                temp_log[f"dead_neurons/layer_{j}"] = dead_neurons[new_idx][j].sum()
                temp_log[f"rank/layer_{j}"] = ranks[new_idx][j]
                temp_log[f"effective_rank/layer_{j}"] = effective_ranks[new_idx][j]
                temp_log[f"approximate_rank/layer_{j}"] = approximate_ranks[new_idx][j]
                temp_log[f"abs_approximate_rank/layer_{j}"] = approximate_ranks_abs[new_idx][j]

                # same thing for the last 30% of the data
                temp_log[f"avg_scores_last_30/layer_{j}"] = sum_avg_scores_last_30[j] / batch_count_last_30
                temp_log[f"corrected_avg_scores_last_30/layer_{j}"] = sum_corrected_avg_scores_last_30[j] / batch_count_last_30
                temp_log[f"median_scores_last_30/layer_{j}"] = sum_medians_last_30[j] / batch_count_last_30
                temp_log[f"corrected_median_scores_last_30/layer_{j}"] = sum_corrected_medians_last_30[j] / batch_count_last_30
                temp_log[f"min_scores_last_30/layer_{j}"] = sum_mins_last_30[j] / batch_count_last_30
                temp_log[f"max_scores_last_30/layer_{j}"] = sum_maxs_last_30[j] / batch_count_last_30
                temp_log[f"last_features_act_last_30/layer_{j}"] = sum_last_features_act_last_30[j] / batch_count_last_30
                temp_log[f"mean_feature_act_last_30/layer_{j}"] = sum_mean_feature_act_last_30[j] / batch_count_last_30
                temp_log[f"weight_magnitude_last_30/layer_{j}"] = sum_weight_magnitude_last_30[j] / batch_count_last_30

                temp_log[f"probing_accuracy_pre/layer_{j}"] = probing_accuracies_pre[j] if do_probing else float('nan')
                temp_log[f"probing_accuracy_post/layer_{j}"] = probing_accuracies_post[j] if do_probing else float('nan')

                temp_log[f"hessian/layer_{j}"] = hessian_ranks[j] if do_hessian else float('nan')
        
        wandb.log(temp_log)
        
        end_time = time.time()
        
        # Print some results to stdout
        print("Task ", task_idx, " took ", end_time - start_time, " seconds", flush=True)
        print("On average, a single task took ", (end_time - START) / (task_idx + 1), " seconds", flush=True)
        s = (end_time - START) / (task_idx + 1) * (num_tasks - task_idx - 1)
        print("Estimating time left: ", s , " seconds =", s / 60, "minutes =", s / 60/60, "hours", flush=True)
        print('Last accuracy', accuracies[new_iter_start:iter - 1].mean().item(), flush=True)
        print('Last accuracy last 30%', accuracies[new_iter_start + int(0.7 * change_after):iter - 1].mean().item(), flush=True)
        print("Post training accuracy: ", post_train_accuracy, flush=True)
        print()

    wandb.finish()

def save_data(file, data):
    with open(file, 'wb+') as f:
        pickle.dump(data, f)


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

    online_expr(params, index=args.index)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
