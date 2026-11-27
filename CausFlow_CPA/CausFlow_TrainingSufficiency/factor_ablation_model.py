"""Two-by-two intervention on fixed five-factor FactorSeparated PILOT.

The base factor encoder, Denoise_net, training loss and gene gate remain unchanged;
only DAG propagation and explicit direct condition-to-gene velocity are toggled.
This is a model engineering ablation, NOT proof of causal identification.
"""
import torch
from factor_base_pilot import FactorSeparatedCausFlow


class FactorAblationModel(FactorSeparatedCausFlow):
    def __init__(self, *args, use_direct=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_direct = bool(use_direct)
        if not self.use_direct:
            self.direct_to_gene.requires_grad_(False)

    def velocity(self, x_t, t, factors):
        hidden = self.core(self.path_to_hidden(x_t), t, labels=None, concept_embs=factors)
        v = self.to_gene(hidden)
        if self.use_direct:
            drug, dose, line = (factors[:,i,:] for i in range(3))
            direct = torch.cat((drug, dose, line, drug * dose), dim=-1)
            v = v + self.direct_to_gene(direct)
        if self.fullrank_gate:
            v = v + x_t * torch.tanh(self.gate(hidden))
        return v
