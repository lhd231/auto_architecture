import torch
from torch import nn
from typing import Tuple, Union, Dict
from torch.nn import functional as F

from utility_funcs import model_factory, calculate_num_patches, init_weights
from utility_models import MLP, PositionalEncoder

class NeuroNet(nn.Module):
    def __init__(
        self,
        arch_dict,
        hp_dict,
        num_heads: int = 1
    ):
        super().__init__()
        self._input_shape = hp_dict["input_shape"]
        self._patch_sizes = hp_dict["patch_sizes"]
        self._conv_type = hp_dict["conv_type"]
        self._in_channels = hp_dict["in_channels"]
        self._out_channels = hp_dict["out_channels"]
        self._num_classes = hp_dict["num_classes"]
        self._first_stride = hp_dict["first_stride"]
        self._act = hp_dict["act"]
        assert (self._input_shape.count(self._input_shape[0])
                == len(self._input_shape)), f'Input shape sizes' \
                    f' should be the same for each dimension'
        assert (len(self._out_channels)
                == len(self._patch_sizes)), f'Please specify' \
            f' a number of output channels for each patch'


        if self._conv_type == nn.Conv2d:
            self._batch_norm = nn.LayerNorm
        else:
            self._batch_norm = nn.LayerNorm
        model_dict = model_factory(arch_dict, hp_dict)
        self._patch_layers = model_dict["patches"]
        self._positional_encoder = model_dict["pos_encoder"]
        self._num_patches, self._size_triu, size_patch \
            = calculate_num_patches(self._first_stride, self._input_shape, self._patch_sizes)
        
        #self._ln1 = nn.LayerNorm(out_channel)
        #self._mix_mlp = MLP(out_channel, 2 * out_channel, out_channel,
        #                    activation=self._act)
        #self._ln2 = nn.LayerNorm(out_channel)
        self._embedding_layers = model_dict["embedding"]
        self._double_mix_mlp = MLP(self._out_channels[-1] * 2,
                                   self._out_channels[-1] * 2,
                                   self._out_channels[-1] * 2,
                                   self._act)
        self._ln3 = nn.LayerNorm(self._out_channels[-1] * 2)
        self._num_heads = num_heads
        print(self._out_channels)
        #self._attention_heads = nn.ModuleList(
        #    [nn.Linear(out_channel * 2, 1, bias=False)
        #     for _ in range(self._num_heads)]
        #)
        self._interaction_mlp = nn.Linear(
            self._out_channels[-1] * 2, 1, bias=False,
        )
        self._prediction_mlp = nn.Linear(
            self._out_channels[-1] * 2,
            self._num_classes, bias=True)
        self._softmax = nn.Softmax(dim=1)
        self.apply(init_weights)


    @property
    def ix_to_grid(self):
        return self._ix

    def forward(self, x):
        for layer in self._patch_layers:
            x = layer(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self._positional_encoder(x)
        x = x.permute(0, 2, 1)
        batch_size, num_patches, num_features = x.size()
        x = torch.reshape(
            x,
            (batch_size * num_patches, num_features)
        )
        #emb = self._ln1(x)
        #emb = self._mix_mlp(emb)
        #x = self._ln2(emb) + x
        for layer in self._embedding_layers:
            x = layer(x)
        x += x
        x = x.view(batch_size, num_patches, num_features)
        tril_indices = torch.tril_indices(
            num_patches, num_patches, device=x.device,
            offset=-1)
        row_ix, col_ix = tril_indices
        patch_concats = torch.cat(
            (x[:, row_ix], x[:, col_ix]),
            dim=-1)
        patch_grid = patch_concats.permute(0, 2, 1)
        patch_concats = torch.reshape(
            patch_concats,
            (batch_size * self._size_triu, num_features * 2)
        )
        concat_emb = self._double_mix_mlp(patch_concats)
        patch_concats = self._ln3(concat_emb) + patch_concats
        interactions = self._interaction_mlp(patch_concats)
        interactions = interactions.view(
            batch_size, self._size_triu, 1)
        attention = self._softmax(interactions)
        patch_grid = torch.bmm(
            patch_grid, attention).squeeze()
        y_hat = self._prediction_mlp(patch_grid)
        return y_hat, (None, attention.squeeze())

