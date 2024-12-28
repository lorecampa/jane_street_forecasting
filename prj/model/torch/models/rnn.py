import torch
from torch import nn
import torch.nn.functional as F
from .layers import CausalPadding, TransposeLayer


class RecurrentModel(nn.Module):
    def __init__(
        self, 
        input_features, 
        output_dim=1, 
        hidden_dim=64, 
        num_layers=3, 
        kernel_size=3, 
        dilations=[1, 2, 4, 8, 16], 
        use_attention=False,
        attention_heads=4,
        dropout_rate=0.3,
    ):
        super(RecurrentModel, self).__init__()
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        self.feature_projector = nn.Sequential(
            TransposeLayer(),
            nn.BatchNorm1d(input_features), 
            TransposeLayer(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_features, hidden_dim),
            TransposeLayer(),
            nn.BatchNorm1d(hidden_dim), 
            TransposeLayer(),
            nn.Dropout(dropout_rate),
            nn.SiLU()
        ) 

        self.conv_modules = []
        for _ in range(num_layers):
            module = []
            module.append(TransposeLayer())
            for d in dilations:
                module += [
                    CausalPadding((kernel_size - 1) * d),
                    nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              kernel_size=kernel_size,
                              dilation=d,
                              padding=0),
                    nn.BatchNorm1d(hidden_dim), 
                    nn.Dropout(dropout_rate),
                    nn.SiLU()
                ]
            module.append(TransposeLayer())
            self.conv_modules.append(nn.Sequential(*module))
        self.conv_modules = nn.ModuleList(self.conv_modules)

        self.gru_layers = nn.ModuleList([
            nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
            for _ in range(num_layers)
        ])
        if use_attention:
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attention_heads, batch_first=True)
                for _ in range(num_layers)
            ])

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden = self.feature_projector(x)

        if self.use_attention:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        for layer in range(self.num_layers):
            hidden = self.conv_modules[layer](hidden)
            hidden, _ = self.gru_layers[layer](hidden)

            if self.use_attention:
                hidden, _ = self.attention_layers[layer](
                    hidden, hidden, hidden, attn_mask=causal_mask
                )

        last_hidden = hidden[:, -1, :]
        return 5 * self.predictor(last_hidden)