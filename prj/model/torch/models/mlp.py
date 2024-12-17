import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dims=[], 
                 use_dropout=True, 
                 dropout_rate=0.1, 
                 use_bn=True, 
                 output_dim=1
        ):
        super(Mlp, self).__init__()
        
        layers = []
        in_features = input_dim[0]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
            
        layers.append(nn.Linear(in_features, output_dim))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)