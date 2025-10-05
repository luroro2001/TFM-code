import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

# L: I assume ResidualNet can model spectral profiles (1D) and ConvResidualNet can 
# model Stokes parameters' maps or images from telescopes 

class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features) for _ in range(2)]
            )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)

        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )

        self.dropout = nn.Dropout(p=dropout_probability)

        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs

        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)

        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)

        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)

        temps = self.activation(temps)

        temps = self.dropout(temps)

        temps = self.linear_layers[1](temps)

        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)

        return inputs + temps


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs


class ConvResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        context_channels=None,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
        reduce=True
    ):
        super().__init__()
        self.activation = activation

        if context_channels is not None:
            self.context_batch = nn.Conv2d(
                in_channels=context_channels,
                out_channels=channels,
                kernel_size=1,
                padding=0,                
            )
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_batchs = nn.ModuleList(
                [nn.BatchNorm2d(channels) for _ in range(2)]
            )
        self.conv_batchs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, padding=1) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.conv_batchs[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.conv_batchs[-1].bias, -1e-3, 1e-3)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.reduce = reduce

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_batchs[0](temps)
        temps = self.activation(temps)
        temps = self.conv_batchs[0](temps)        
        if self.use_batch_norm:
            temps = self.batch_norm_batchs[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.conv_batchs[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_batch(context)), dim=1), dim=1)
        if self.reduce:
            out = self.maxpool(inputs + temps)
        else:
            out = inputs + temps
            
        return out


class ConvResidualNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        context_channels=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        reduce=False
    ):
        super().__init__()
        self.context_channels = context_channels
        self.hidden_channels = hidden_channels
        if context_channels is not None:
            self.initial_batch = nn.Conv2d(
                in_channels=in_channels + context_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0,
            )
        else:
            self.initial_batch = nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0,
            )
        self.blocks = nn.ModuleList(
            [
                ConvResidualBlock(
                    channels=hidden_channels,
                    context_channels=context_channels,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    reduce=reduce
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_batch = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=1, padding=0
        )

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_batch(inputs)
        else:
            temps = self.initial_batch(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_batch(temps)
        return outputs




if __name__ == "__main__":
    
    tmp = ResidualNet(in_features=2, 
                      out_features=1,
                      hidden_features=64,
                      num_blocks=2,
                      activation=F.relu,                      
                      dropout_probability=0.0,
                      use_batch_norm=True)
    

    tmp = ConvResidualNet(in_channels=2, 
                      out_channels=1,
                      hidden_channels=64,
                      num_blocks=3,
                      reduce=True,
                      activation=F.relu,                      
                      dropout_probability=0.0,
                      use_batch_norm=True)