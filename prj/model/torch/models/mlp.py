import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, input_features, hidden_dims=[], dropout_rate=0.1, output_dim=1, 
                 use_tanh=False, final_mult=1.0, bn_momentum=0.1, 
                 initial_bn=True, use_dropout=True, use_norm=True):
        super(Mlp, self).__init__()
        self.final_mult = final_mult
        self.use_tanh = use_tanh
        self.bn_momentum = bn_momentum
        
        layers = []
        if initial_bn:
            layers.append(nn.BatchNorm1d(input_features, momentum=bn_momentum))
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
            
        in_features = input_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            if use_norm:
                layers.append(nn.BatchNorm1d(hidden_dim, momentum=bn_momentum))
            layers.append(nn.SiLU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
            
        layers.append(nn.Linear(in_features, output_dim))
        if self.use_tanh:
            layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.final_mult * self.model(x)