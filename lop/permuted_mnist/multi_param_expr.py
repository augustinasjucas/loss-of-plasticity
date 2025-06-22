import sys
import json
import copy
import argparse
import subprocess
from tqdm import tqdm
from lop.utils.miscellaneous import get_configurations

def f_rates(rates_list):
    if len(rates_list) == 0:
        return str(rates_list)
    else:
        # turn into a bistring: every element > 0 gets a "1", every element == 0 gets a "0"
        return ''.join(['1' if rate > 0 else '0' for rate in rates_list])



def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path of the file containing the parameters of the experiment",
                        type=str, default='cfg/a.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    list_params, hyper_param_settings_temp = get_configurations(params=params)
    # go over all settings, and remove those:
    #  - replacement_rate > 0.0 and agent != cbp
    #  - weight_decay > 0.0 and agent != l2
    hyper_param_settings = []
    for param_setting in hyper_param_settings_temp:
        new_params = copy.deepcopy(params)
        for idx, param in enumerate(list_params):
            new_params[param] = param_setting[idx]

        if float(new_params['replacement_rate']) > 0.0 and new_params['agent'] != 'cbp':
            continue
        if float(new_params['weight_decay']) > 0.0 and new_params['agent'] != 'l2':
            continue
        if float(new_params['agent'] == 'cbp') and new_params['replacement_rate'] == 0.0:
            continue
        if float(new_params['agent'] == 'l2') and new_params['weight_decay'] == 0.0:
            continue
        hyper_param_settings.append(param_setting)
    

    # make a directory for temp cfg files
    bash_command = "mkdir -p temp_cfg/"
    subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

    bash_command = "rm -r --force " + params['data_dir']
    subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    bash_command = "mkdir " + params['data_dir']
    subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

    """
        Set and write all the parameters for the individual config files
    """
    for setting_index, param_setting in enumerate(hyper_param_settings):
        new_params = copy.deepcopy(params)
        for idx, param in enumerate(list_params):
            new_params[param] = param_setting[idx]
            if param == 'depth_width':
                new_params['num_hidden_layers'] = param_setting[idx][0]
                new_params['num_features'] = param_setting[idx][1]
                del new_params['depth_width']
        new_params['index'] = setting_index
        new_params['data_dir'] = params['data_dir'] + str(setting_index) + '/'

        """
            Make the data directory
        """
        bash_command = "mkdir -p " + new_params['data_dir']
        subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

        for idx in tqdm(range(params['num_runs'])):
            new_params['data_file'] = new_params['data_dir'] + str(idx)
            new_params['run_index'] = idx

            # compute wandb group name: replace "{lr}" with params['step_size'] and "{decay}" with params['weight_decay']
            new_params['wandb_group'] = new_params['wandb_group'].format(
                lr=new_params['step_size'],
                decay=new_params['weight_decay'],
                repl=new_params['replacement_rate'],
                layers=new_params['num_hidden_layers'],
                rates=f_rates(new_params['replacement_rates']),
                activation=new_params['activation'],
                agent=new_params['agent'],
                run_index=idx,
                width=new_params['num_features'],
            )

            # compute the wandb run name: just the run index
            new_params['wandb_run_name'] = "run #" + str(idx)


            """
                write data in config files
            """
            new_cfg_file = 'temp_cfg/'+str(setting_index*params['num_runs']+idx)+'.json'
            try:    f = open(new_cfg_file, 'w+')
            except: f = open(new_cfg_file, 'w+')
            with open(new_cfg_file, 'w+') as f:
                json.dump(new_params, f, sort_keys=False, indent=4)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
