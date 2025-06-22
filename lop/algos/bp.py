from lop.algos.gnt import GnT
import torch
import torch.nn.functional as F
from torch import optim


class Backprop(object):
    def __init__(self, net, step_size=0.001, loss='mse', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
                 to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0,
                 compute_utility=False, utility_params=None):
        self.net = net
        self.to_perturb = to_perturb
        self.perturb_scale = perturb_scale
        self.device = device

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                  weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                   weight_decay=weight_decay)

        # define the loss function
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]

        # Placeholder
        self.previous_features = None

        self.compute_utility = compute_utility

        if compute_utility:
            # assert that utility_params contains decay_rate, util_type and init
            assert utility_params is not None, "utility_params must be provided when compute_utility is True"
            assert 'decay_rate' in utility_params, "decay_rate must be provided in utility_params"
            assert 'maturity_threshold' in utility_params, "maturity_threshold must be provided in utility_params"
            assert 'util_type' in utility_params, "util_type must be provided in utility_params"

            self.gnt = GnT(
                net=self.net.layers,
                hidden_activation=self.net.act_type,
                opt=self.opt,
                loss_func=self.loss_func,
                device=self.device,
                **utility_params
            )


    def learn(self, x, target):
        """
        Learn using one step of gradient-descent
        :param x: input
        :param target: desired output
        :return: loss
        """
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        loss.backward()
        self.opt.step()
        if self.to_perturb:
            self.perturb()

        if self.compute_utility:
            self.gnt.test_features(features=self.previous_features)

        if self.loss == 'nll':
            return loss.detach(), output.detach()
        return loss.detach()

    def perturb(self):
        with torch.no_grad():
            for i in range(int(len(self.net.layers)/2)+1):
                self.net.layers[i * 2].bias +=\
                    torch.empty(self.net.layers[i * 2].bias.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
                self.net.layers[i * 2].weight +=\
                    torch.empty(self.net.layers[i * 2].weight.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
