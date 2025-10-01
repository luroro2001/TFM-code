import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as pl
from encoding import GaussianEncoding, PositionalEncoding

def init_kaiming(m):
    if type(m) == nn.Linear:
        init.kaiming_uniform_(m.weight, nonlinearity='relu')

class MLP(nn.Module):
    def __init__(self, 
                 n_input, 
                 n_output, 
                 dim_hidden=1, 
                 n_hidden=1, 
                 activation=nn.ReLU(), 
                 bias=True, 
                 final_activation=nn.Identity(), 
                 last_bias=False, 
                 bn=False):
        """Simple fully connected network, potentially including FiLM conditioning

        Parameters
        ----------
        n_input : int
            Number of input neurons
        n_output : int
            Number of output neurons
        n_hidden : int, optional
            number of neurons per hidden layers, by default 1
        n_hidden_layers : int, optional
            Number of hidden layers, by default 1        
        activation : _type_, optional
            Activation function to be used at each layer, by default nn.Tanh()
        bias : bool, optional
            Include bias or not, by default True
        final_activation : _type_, optional
            Final activation function at the last layer, by default nn.Identity()
        """
        super().__init__()


        self.activation = activation
        self.final_activation = final_activation
        self.bn_active = bn

        self.initial_layer = nn.Linear(n_input, dim_hidden, bias=bias)

        self.hidden_layers = nn.ModuleList([])        

        if (self.bn_active):
            self.bn = nn.ModuleList([])
        
        for i in range(n_hidden):
            self.hidden_layers.append(nn.Linear(dim_hidden, dim_hidden, bias=bias))
            if (self.bn_active):
                self.bn.append(nn.BatchNorm1d(dim_hidden))
        
        self.last_layer = nn.Linear(dim_hidden, n_output, bias=last_bias)

        # self.initial_layer.apply(init_kaiming)
        # self.hidden_layers.apply(init_kaiming)
        # self.last_layer.apply(init_kaiming)
        
    def forward(self, x, gamma=None, beta=None):

        x = self.initial_layer(x)

        # Apply all layers
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)

            if (self.bn_active):
                x = self.bn[i](x)

            x = self.activation(x)
                    
        x = self.last_layer(x)
        x = self.final_activation(x)
        
        return x

    def weights_init(self, type='xavier', nonlinearity='relu'):
        for module in self.modules():
            if (type == 'xavier'):
                xavier_init(module)
            if (type == 'kaiming'):
                kaiming_init(module, nonlinearity=nonlinearity)


if (__name__ == '__main__'):

    mlp = MLP(n_input=2, n_output=1, dim_hidden=64, n_hidden=3, activation=nn.GELU())
        
    v = np.linspace(-1, 1, 1000)
    v = torch.tensor(v[:, None].astype('float32'))
        
        
    out = mlp(out_enc).detach().numpy()