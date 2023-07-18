import torch
from torch import nn
from typing import Tuple, Union
from torch.nn import functional as F
import math
from utility_models import MLP, PositionalEncoder

"""
There are 3 examples of maker functions here:
"""


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)


"""
Shows how to add layers to a modulelist one at a time. If you wanted to change the embedding layer, you could switch out the value
in the layer_1 key. However, if you want to do more than one layer, you need to change this function.
"""
def embedding_maker(arch_params, hp_params):
    hidden_layers = 2*hp_params["out_channels"][-1] if "hidden_size_1" not in arch_params.keys() else arch_params["hidden_size_1"]
    embedding_layers = nn.ModuleList()
    print(arch_params["layer_1"])
    embedding_layers.append(arch_params["norm_1"](hp_params["out_channels"][-1]))
    te = arch_params["layer_1"](hp_params["out_channels"][-1],hidden_layers,hp_params["out_channels"][-1],hp_params["act"])
    print(type(te))
    embedding_layers.append(te)
    embedding_layers.append(arch_params["norm_final"](hp_params["out_channels"][-1]))
    return embedding_layers


"""
Takes in a list of params. This gives you the opportunity to create a series of layers. You can add specific
keys to each per-layer dict (element in arch_params) to define the hidden, input, output, etc values yourself
Here, I wrote it as close to what you had done as I found possible.
"""
def patch_maker(arch_params, hp_params):
    patch_layers = nn.ModuleList()
    in_channel = hp_params["in_channels"]
    conv_type = hp_params["conv_type"]
    for i,p in enumerate(arch_params):
        out_channels = hp_params["out_channels"] if "out_channels" not in p.keys() else p["out_channels"]
        kernel_sizes = hp_params["patch_sizes"] if "patch_sizes" not in p.keys() else p["patch_sizes"]
        act = hp_params["act"] if "act" not in p.keys() else p["act"]
        strides = hp_params["patch_sizes"] if "first_stride" not in p.keys() else p["first_stride"]
        p_layer = conv_type(
                        in_channels=in_channel,
                        out_channels=out_channels[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=0,
                        dilation=1,
                        bias=False)
        patch_layers.append(p_layer)
        patch_layers.append(act())
        in_channel = out_channels[i]
    return patch_layers

"""
The arch_params dict is just an nn.Module class (your PositionalEncoder class)
"""
def positional_encoding_maker(arch_params,hp_params):
    num_patches, size_triu, size_patch \
            = calculate_num_patches(None,hp_params["input_shape"],hp_params["patch_sizes"])
    pos_layer = arch_params['func'](hp_params["out_channels"][-1],num_patches)
    return pos_layer

def model_factory(arch_params, hp_params):

    model_dict = {}
    for k,(v0,v1) in arch_params.items():
        pl = v0(v1, hp_params)
        model_dict[k] = pl
    return model_dict
    
    
def calculate_num_patches(first_stride, input_shape, patch_sizes):
    if first_stride is not None:
        size_patch = ((input_shape[0] - patch_sizes[0])
                        / first_stride) + 1
    else:
        size_patch = input_shape[0] // patch_sizes[0]
    if len(patch_sizes) > 1:
        for patch_size in patch_sizes[1:]:
            size_patch = size_patch // patch_size
    num_patches = size_patch ** len(input_shape)
    triu_size = num_patches * (num_patches - 1) // 2
    return num_patches, triu_size, size_patch


