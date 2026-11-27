"""Frozen-condition utilities and generator-neutral moment pretraining."""
import torch
from torch import nn
FREEZE=('source_to_hidden','drug_emb','dose_mlp','line_emb','core.DisentanglementEncoder')

def condition_modules(m):
    return [m.get_submodule(n) for n in FREEZE]

def is_condition(n):
    return n in ('dag','dose_values') or any(n.startswith(x+'.') for x in FREEZE)

def freeze_condition(m):
    for module in condition_modules(m):module.requires_grad_(False);module.eval()

def freeze_generator(m):
    m.requires_grad_(False)
    for module in condition_modules(m):module.requires_grad_(True)

class MomentHead(nn.Module):
    """Disposable conditional first/second raw-moment regression; no diffusion/CFM target."""
    def __init__(self,genes):
        super().__init__();self.net=nn.Sequential(nn.Linear(160,128),nn.SiLU(),nn.Linear(128,genes*2))
    def forward(self,f,target):
        a,b=self.net(f.flatten(1)).chunk(2,dim=1)
        return (a-target).square().mean()+(b-target.square()).square().mean()
