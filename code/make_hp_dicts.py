import torch
import pickle 
from utility_funcs import patch_maker, calculate_num_patches, embedding_maker, positional_encoding_maker
from utility_models import PositionalEncoder, MLP, PositionalEncoderNoahExample
from torch import nn
from typing import Tuple, Union, Dict, List
from types import FunctionType


'''
Dear Eloy,
I think this works. Basically, in the hp_dict, you add a new possible parameter set. 
So, let's say you want another embedding architecture, it would look like: 
    "embedding":[[nn.LayerNorm,MLP,nn.LayerNorm], [your, new, list, of, things]]
The for loop at the end adds the unique function. This is kind of where it goes off the rails.
If it's too much of a shit show, I'll put refactor at the top of my list.
Anyway, let's say you want to add a new piece of architecture. 
You would make a new function in utility_funcs, called "CNN_1x1_maker". 
You would add that and the possible hps here. The factory function in utility_funcs
will unravel that for you. and give you your new architecture as a key:model/model list
in a dict. You can then throw this thing wherever you want in the forward function.
I've done the patches, pos_encoder, and embeddings already, just to get a baseline.
Oh, another example, let's say you want to add a new position encoder. Make a PositionalEncoderNew
class in utility_models and add it to this pos_encoder list. You'll have to contend with
the last for loop in this file, so enjoy that. 

Sincerely,
lhd231
'''

class hpDictMaker:
    def __init__(self, hp_d):
        self.hp_dict = hp_d
        self.hp_combos = 1
    def add_hp_set(self, hp_label : str,
                   hp_set : list[int, float, str, None]):
        if hp_label not in self.hp_dict.keys():
            self.hp_dict[hp_label] = [hp_set]
        else:
            self.hp_dict[hp_label].append(hp_set)
            self.hp_combos*=2
    
    def add_architecture(self, hp_label: str, 
                         func: FunctionType, 
                         hp_set: list[int, float, str, None]):
        if hp_label not in self.hp_dict.keys():
            self.hp_dict[hp_label] = [{func:[hp_set]}]
        else:
            self.hp_dict[hp_label].append({func:[hp_set]})
            self.hp_combos*=2
        
        
    def add_arch_hp_set(self, hp_label: str, func: FunctionType, hp_set):
        self.hp_dict[hp_label].append({func:[hp_set]})
        self.hp_combos*=2
        
    def make_all_dicts(self):
        all_dicts = []
        for i in range(self.hp_combos):
            all_dicts.append([dict(),dict()])
        
        for i, (k,v) in enumerate(self.hp_dict.items()):
            for j, ele in enumerate(v):
                print("break")
                if isinstance(ele, dict):
                    for k2,v2 in ele.items():
                        for l, it in enumerate(v2):
                            for l2 in range(int(self.hp_combos/(len(v2)*len(v)))):
                                all_dicts[j*len(v)+l2][1][k] = (k2,it)
                        print((k2,v2))
                else:
                    if k == "out_channels":
                        print("OUT CHANS "+str(ele))
                        print(j*len(v)+j)
                    for l2 in range(int(self.hp_combos/len(v))):
                        all_dicts[j*len(v)+l2][0][k] = ele
        return all_dicts 
    



"""
Eloy:
These variables are any hp your heart could desire. Throw them into a dict like this (they have to go in as a list because it's not done).
add_hp_set() adds a new hp set, in this example, I add a new out_channels hp set. 
add_architecture() adds an entire architecture (function defined by you elsewhere). Also takes in a dict of architecture specific hps
add_arch_hp_set() adds a new hp set to an existing architecture.
The shitty parts: 
-It currently only takes in one hp set per hp label in the init. It should take more
-Same is true for add_architecture function.
-It puts the hps and architectures in two separate dicts in one list. This should change.
"""
patch_size = (5,2)
input_shape = (60,60)
in_channels = 3
out_channels = (16,32)
num_classes = 1
act = nn.ReLU
conv_type = nn.Conv2d
num_patches, _size_triu, size_patch \
        = calculate_num_patches(None, input_shape, patch_size)

hp_dict = {"patch_sizes":[patch_size],
        "input_shape":[input_shape],
        "in_channels":[in_channels],
        "out_channels":[out_channels],
            "num_classes":[num_classes],
            "act":[act],
            "conv_type":[nn.Conv2d],
            "first_stride":[None]
            }
DM = hpDictMaker(hp_dict)
DM.add_hp_set("out_channels", (8,16))
DM.add_architecture("patches", patch_maker,[{"act":nn.ReLU}, {"act":nn.ReLU} ])
DM.add_architecture("embedding", embedding_maker,{"norm_1":nn.LayerNorm, "layer_1":MLP, "norm_final":nn.LayerNorm, "hidden_size_1":64})

DM.add_architecture("pos_encoder", positional_encoding_maker, {"func":PositionalEncoder})
DM.add_arch_hp_set("pos_encoder", positional_encoding_maker, {"func":PositionalEncoderNoahExample})
all_dicts = DM.make_all_dicts()
for i,D in enumerate(all_dicts):
    print(D[0]["out_channels"])
    pickle.dump(D,open("test"+str(i)+".p",'wb'))
    
'''
total_perms = 1
for k,v in hp_dict.items(): total_perms*=len(v)
all_dicts = [dict()]*total_perms

for i, (k,v) in enumerate(hp_dict.items()):
    for j,ele in enumerate(v):
        all_dicts[int(j*len(v)+j)][k] = ele 
        
for D in all_dicts:
    num_patches, _size_triu, size_patch \
            = calculate_num_patches(D["first_stride"], D["input_shape"], D["patch_sizes"])
    D["patches"] = (patch_maker,[[D["in_channels"],D["out_channels"][0],D["patch_sizes"][0],D["patch_sizes"][0],nn.ReLU],[D["out_channels"][0],D["out_channels"][1],D["patch_sizes"][1],D["patch_sizes"][1],nn.ReLU]])
    D["pos_encoder"] = (D["pos_encoder"],[D["out_channels"][-1],num_patches])
    D["embedding"] = (embedding_maker,(D['embedding'],[D["out_channels"][-1],[D["out_channels"][-1],2*D["out_channels"][-1],D["out_channels"][-1],nn.ReLU],D["out_channels"][-1]]))


for i,D in enumerate(all_dicts):
    pickle.dump(D,open("test"+str(i)+".p",'wb'))
'''
