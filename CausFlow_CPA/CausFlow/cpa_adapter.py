"""Source-anchored, full-rank-gated CausFlow adaptation for CPA-like prediction.

This is an explicitly adapted architecture, NOT a claim that the unmodified
CausFlow encoder can counterfactually extrapolate an unseen dose. Flow starts
from matched TRAIN controls, and a gene-wise gate lets all genes participate
in the velocity update instead of restricting changes to a low-rank decoder.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


def check_dag(a: torch.Tensor) -> None:
    if a.shape != (4, 4):
        raise ValueError('SCM adjacency must be 4x4: nodes [drug,dose,cell_line,response].')
    if not torch.isfinite(a).all() or a.diag().abs().max() > 0:
        raise ValueError('DAG must be finite with zero diagonal')
    # A[i,j] = parent j -> child i. Reject cyclic graphs.
    if torch.linalg.matrix_power(a.abs(), 4).diag().max() > 1e-7:
        raise ValueError('DAG has a directed cycle (4-step check)')
    m = a.abs().clone()
    for _ in range(4):
        if m.diag().max() > 1e-7:
            raise ValueError('DAG has a directed cycle')
        m = m @ a.abs()


def default_dag():
    # Drug, dose, and line jointly feed the downstream response factor.
    # This is an architectural assumption, not an inferred biological DAG.
    return [[0.,0.,0.,0.], [0.,0.,0.,0.], [0.,0.,0.,0.], [1.,1.,1.,0.]]


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, t):
        half = self.width // 2
        frequencies = torch.exp(-math.log(10000.) * torch.arange(half,device=t.device) / max(half-1,1))
        e = t[:,None] * frequencies[None,:] * 1000.
        z = torch.cat((e.sin(), e.cos()),dim=-1)
        if z.size(1) < self.width:
            z = F.pad(z, (0,self.width-z.size(1)))
        return self.net(z)


class CausFlowCPADose(nn.Module):
    """Compact CausFlow SCM-conditioned velocity model.

    SCM nodes are [drug, dose, cell_line, response/base].
    Exogenous source state comes only from an independently sampled
    vehicle/control cell of the same cell line, never from target cells.
    """
    def __init__(self, genes, num_drugs, num_doses, num_lines,
                 width=192, factor_dim=48, dag=None):
        super().__init__()
        self.genes = int(genes)
        self.width = int(width)
        self.factor_dim = int(factor_dim)
        dag_tensor = torch.as_tensor(default_dag() if dag is None else dag,dtype=torch.float32)
        check_dag(dag_tensor)
        self.register_buffer('dag',dag_tensor)
        self.register_buffer('graph_transform',torch.linalg.inv(torch.eye(4)-dag_tensor))
        self.drug_emb = nn.Embedding(num_drugs, factor_dim)
        self.dose_emb = nn.Embedding(num_doses, factor_dim)
        self.line_emb = nn.Embedding(num_lines, factor_dim)
        self.source_encoder = nn.Sequential(nn.Linear(genes,width),nn.LayerNorm(width),nn.SiLU(),
                                            nn.Linear(width,factor_dim))
        self.time_encoder = TimeEmbedding(width)
        self.input_proj = nn.Linear(genes,width)
        self.context_proj = nn.Linear(factor_dim,width)
        self.attention = nn.MultiheadAttention(width,num_heads=4,batch_first=True)
        self.net = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.SiLU(),
                                 nn.Linear(width*2,width),nn.SiLU())
        self.output_proj = nn.Linear(width,genes)
        self.gene_gate = nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,genes))
        # Start with a zero vector field. The gene-wise multiplicative path
        # supplies a full-rank diagonal update; output_proj supplies a learned
        # low-rank cross-gene correction.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.gene_gate[-1].weight)
        nn.init.zeros_(self.gene_gate[-1].bias)

    def condition(self, source_control_x, drug_idx, dose_idx, line_idx):
        # Positions: known drug, dose, cell line, and source-cell state.
        exog = torch.stack((self.drug_emb(drug_idx), self.dose_emb(dose_idx),
                            self.line_emb(line_idx),self.source_encoder(source_control_x)),dim=1)
        # Structural propagation using fixed DAG; enables explicit DAG ablation.
        return torch.einsum('ij,bjd->bid',self.graph_transform,exog)

    def velocity(self, x_t, t, cond):
        h = self.input_proj(x_t) + self.time_encoder(t)
        context = self.context_proj(cond)
        # One query token attends to all four causal factors, no O(G^2) layers.
        attn,_ = self.attention(h[:,None,:],context,context,need_weights=False)
        h = h + attn[:,0,:]
        h = self.net(h) + h
        low_rank = self.output_proj(h)
        gene_gate = torch.tanh(self.gene_gate(h))
        return low_rank + x_t * gene_gate

    def cfm_loss(self, x_target, x_source_control, drug, dose, line):
        # Empirical base distribution: independent matched-line train controls.
        # x_target is the endpoint only, never a condition/context input.
        z = self.condition(x_source_control,drug,dose,line)
        t = torch.rand(x_target.size(0),device=x_target.device)
        target_velocity = x_target-x_source_control
        x_t = x_source_control + t[:,None]*target_velocity
        pred_velocity = self.velocity(x_t,t,z)
        return F.mse_loss(pred_velocity,target_velocity)

    @torch.no_grad()
    def generate(self, x_source_control, drug, dose, line, steps=40, seed=None):
        if steps < 1:
            raise ValueError('steps must be >= 1')
        self.eval()
        device=x_source_control.device
        z=self.condition(x_source_control,drug,dose,line)
        # The empirical source distribution is the matched training control
        # population, exactly as in the conditional flow-matching path.
        x=x_source_control.clone()
        dt=1.0/steps
        for i in range(steps):
            t=torch.full((x.size(0),),i*dt,device=device)
            x=x+dt*self.velocity(x,t,z)
        # No rank thresholding or averaging: preserve independent generated cells
        # for true variance / distribution calculations. Negative predictions
        # are retained for diagnostic metrics; clipping requires explicit study.
        return x
