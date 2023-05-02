import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...python import cache
from ..init import xavier_uniform


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch : int, patch_size : int, base_dim=32, max_downs=5):
        super().__init__()

        layers = find_archi(patch_size, max_downs=max_downs)

        layers_count = len(layers)

        dim_mults=[ min(2**i, 8) for i in range(layers_count+1) ]
        dims = [ base_dim * mult for mult in dim_mults ]

        self._in_beta = nn.parameter.Parameter( torch.zeros(in_ch,), requires_grad=True)
        self._in_gamma = nn.parameter.Parameter( torch.ones(in_ch,), requires_grad=True)
        self._in = nn.Conv2d(in_ch, base_dim, 1, 1, 0, bias=False)

        down_c_list = self._down_c_list = nn.ModuleList()
        up_c_list = self._up_c_list = nn.ModuleList()
        up_s_list = self._up_s_list = nn.ModuleList()
        logits_list = self._logits_list = nn.ModuleList()

        for level, ((ks, stride), up_ch, down_ch) in enumerate(list(zip(layers, dims[:-1], dims[1:]))):
            down_c_list.append( nn.Conv2d(up_ch, down_ch, ks, stride, ks//2) )
            up_c_list.insert(0, nn.ConvTranspose2d(down_ch, up_ch, ks, stride, padding=ks//2, output_padding=stride-1) )
            up_s_list.insert(0, nn.ConvTranspose2d(down_ch, up_ch, ks, stride, padding=ks//2, output_padding=stride-1) )

            logits_list.insert(0, nn.Conv2d(up_ch, 1, 1, 1) )

            if level == layers_count-1:
                self._mid_logit = nn.Conv2d(down_ch, 1, 1, 1)

        xavier_uniform(self)
        
        self._grad_params = [x for x in self.parameters() if x.requires_grad]

    def set_requires_grad(self, r : bool):
        for param in self._grad_params:
            param.requires_grad_(r)

    def forward(self, inp : torch.Tensor):
        x = inp

        x = x + self._in_beta[None,:,None,None]
        x = x * self._in_gamma[None,:,None,None]
        x = self._in(x)

        shortcuts = []
        for down_c in self._down_c_list:
            x = F.leaky_relu(down_c(x), 0.2)
            shortcuts.insert(0, x)

        logits = [ self._mid_logit(x) ]

        for shortcut_x, up_c, up_s, logit in zip(shortcuts, self._up_c_list, self._up_s_list, self._logits_list):
            x = F.leaky_relu(up_c(x) + up_s(shortcut_x), 0.2 )
            logits.append( logit(x) )

        return logits
    
    @staticmethod
    def get_max_patch_size(max_downs=5): return max(list(get_table(max_downs).keys()))


@cache
def get_table(max_downs):
    def calc_receptive_field_size(layers):
        rf = 0
        ts = 1
        for i, (k, s) in enumerate(layers):
            if i == 0:
                rf = k
            else:
                rf += (k-1)*ts
            ts *= s
        return rf
    
    
    def gen(layer_count : int, layer_id : int = 0):
        if layer_id == layer_count:
            yield ()
        else:
            k_gen = [3,5,7]
            s_gen = [1,2]
            for ks in k_gen:
                for s in s_gen:
                    for sub in gen(layer_count, layer_id+1):
                        yield ((ks, s),) + sub
                        
    d = {}
    for layers_count in range(1,max(1,max_downs)+1):
        for layers in gen(layers_count):

            ks_sum = 0
            s_sum = 0
            for ks, s in layers:
                ks_sum += ks
                s_sum += s

            rf = calc_receptive_field_size(layers)

            s_rf = d.get(rf, None)
            if s_rf is None:
                d[rf] = (layers, ks_sum, s_sum)
            else:
                if layers_count > len(s_rf[0]) or ks_sum <= s_rf[1] and s_sum >= s_rf[2]:
                    d[rf] = (layers, ks_sum, s_sum)
                    
        
    return { rf : info[0] for rf, info in d.items() }
                
    
def find_archi(patch_size, max_downs=5):
    """"""
    table = get_table(max_downs)
        
    x = sorted(list(table.keys()))
    q=x[np.abs(np.array(x)-patch_size).argmin()]
    return table[q]


# max_p = PatchDiscriminator.get_max_patch_size()
# print('max ',  )

# for i in range(1, max_p+1):
#    print(f'{i} {_find_archi(i)}')

# import code
# code.interact(local=dict(globals(), **locals()))
