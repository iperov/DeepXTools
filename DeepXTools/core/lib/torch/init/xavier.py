import torch
import torch.nn as nn

def init_with_xavier_uniform(module : nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear) ):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)