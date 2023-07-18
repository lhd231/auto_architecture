import torch
import pickle 
from utility_funcs import patch_maker, calculate_num_patches, embedding_maker
from utility_models import PositionalEncoder, MLP
from torch import nn



patches = (5, 2)
input_sizes = (60, 60)
in_channels = 3
out_channels = (16, 32)
first_stride = None
num_patches, _size_triu, size_patch \
            = calculate_num_patches(first_stride, input_sizes, patches)
patches = (5, 2)
input_sizes = (60, 60)
in_channels = 3
out_channels = (16, 32)
print(num_patches)
first_stride = None
print([out_channels[1],2*out_channels[1],out_channels[1],nn.ReLU])
test_dict = {"patch_sizes":(5,2),
             "input_shape":(60,60),
             "in_channels":in_channels,
             "out_channels":out_channels,
             "conv_type":nn.Conv2d,
             "first_stride":first_stride,
             "num_classes":1,
             "act":nn.ReLU,
    "patches":(patch_maker,[[in_channels,out_channels[0],patches[0],patches[0],nn.ReLU],[out_channels[0],out_channels[1],patches[1],patches[1],nn.ReLU]]),
    "pos_encoder":(PositionalEncoder,[out_channels[1],num_patches]),
    "embedding":(embedding_maker,([nn.LayerNorm,MLP,nn.LayerNorm],[out_channels[1],[out_channels[1],2*out_channels[1],out_channels[1],nn.ReLU],out_channels[1]]))}

pickle.dump(test_dict,open("test.p",'wb'))

'''
model = NeuroNet((60, 60), 
                 (5, 2), 
                 nn.Conv2d, 
                 3, 
                 (16, 32), 
                 nn.ReLU, 
                 1,
                 hp_dict).to(device)


                        in_channels=in_channels,
                        out_channels=out_channel,
                        kernel_size=patch_size,
                        stride=self._first_stride,
                        
                        
    def __init__(
        self,
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        patch_sizes: Tuple[int],
        conv_type: Union[nn.Conv2d, nn.Conv3d],
        in_channels: int,
        out_channels: Tuple[int],
        activation_function: nn.Module,
        num_classes: int,
        hp_dict: dict,
        num_heads: int = 1,
        first_stride: Union[None, int] = None
    ):
    
'''