import torch.nn as nn


class FFNN(nn.Module):
    """
    A feed forward neural network with just one hidden layer (or more if specified).
    This network is used as the learning network in the Slowly Changing Regression problem
    """
    def __init__(self, input_size, num_features=5, num_outputs=1, hidden_activation='relu', additional_layers=0):
        super(FFNN, self).__init__()
        self.num_inputs = input_size
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.act_type = hidden_activation

        # define the hidden activation
        self.hidden_activation = {'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'selu': nn.SELU,
                                  'swish': nn.SiLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}[self.act_type]

        # define the architecture
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, num_features))
        self.layers.append(self.hidden_activation())
        # support for additional hidden layers
        for _ in range(additional_layers):
            self.layers.append(nn.Linear(num_features, num_features))
            self.layers.append(self.hidden_activation())
        self.layers.append(nn.Linear(num_features, num_outputs))

        # initialize all weights
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if i == 0:  # first layer
                    if hidden_activation in ['sigmoid', 'relu', 'tanh', 'leaky_relu']:
                        nn.init.kaiming_uniform_(layer.weight, nonlinearity=hidden_activation)
                    elif hidden_activation in ['swish', 'elu']:
                        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                elif i == len(self.layers) - 1:  # output layer
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='linear')
                else:  # hidden layers
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity=hidden_activation)
                layer.bias.data.fill_(0.0)

    def predict(self, x):
        """
        Forward pass
        :param x: input
        :return: estimated output
        """
        features = x
        activations = []
        for layer in self.layers[:-1]:
            features = layer(features)
            activations.append(features)
        out = self.layers[-1](features)
        return out, activations
