import torch
from torch import nn
from typing import Tuple, Union
from torch.nn import functional as F
import math


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super().__init__()
        self._input_size = input_size
        print(input_size)
        print(hidden_size)
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._activation = activation
        print(self._input_size)
        self._layers = nn.ModuleList(
            [nn.Linear(self._input_size, self._hidden_size),
             self._activation(),
             nn.Linear(self._hidden_size, self._output_size),
             self._activation()]
        )
        self.apply(self._init_weights)

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight.data)
            if m.bias.data is not None:
                nn.init.constant_(m.bias.data, 0)
                
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len):
        self.d_model, self.max_len = d_model, max_len
        super().__init__()
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 1)
                             * (-math.log(10000.0) / self.d_model))
        pe = torch.sin(position * div_term).permute(1, 0).unsqueeze(0)
        self._pe = nn.Parameter(pe,
                                requires_grad=False)

    def forward(self, x):
        x += self._pe.repeat(x.size(0), 1, 1)
        return x
    
class PositionalEncoderNoahExample(nn.Module):
    def __init__(self, d_model, max_len):
        self.d_model, self.max_len = d_model, max_len
        super().__init__()
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 1)
                             * (-math.log(10000.0) / self.d_model))
        pe = torch.sin(position * div_term).permute(1, 0).unsqueeze(0)
        self._pe = nn.Parameter(pe,
                                requires_grad=False)

    def forward(self, x):
        x += self._pe.repeat(x.size(0), 1, 1)
        return x
