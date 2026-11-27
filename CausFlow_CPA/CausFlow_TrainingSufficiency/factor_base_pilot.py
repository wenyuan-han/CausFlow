"""FACTOR-SEPARATED PILOT v1. NOT original CausFlow, not an established improvement.

Designs *train-only source* conditioning with explicit drug/dose/line/source/response
nodes, 5-node fixed DAG, original Denoise_net, and a direct drug-dose velocity
route. No label classifier on labels injected as embeddings, no tautological
SCM-loss and no contradictory adversarial label-removal on the response node.

This pilot changes multiple factors simultaneously and is for engineering
feasibility. Controlled ablations are required before any causal claim.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from CausFlow.Modules import Denoise_net


def default_dag5():
    a=torch.zeros(5,5)
    a[4,0:4]=1.  # drug,dose,line,source -> response
    return a


class FactorSeparatedCausFlow(nn.Module):
    def __init__(self, genes, n_drugs, n_doses, n_lines, width=192, dag=None,
                 auxiliary=1.0, fullrank_gate=True, dose_values=None,
                 cfm_weight=1.0):
        super().__init__()
        if width < 8 or width%4:
            raise ValueError('width must be divisible by 4 and at least 8')
        dag=torch.as_tensor(default_dag5() if dag is None else dag,dtype=torch.float32)
        if dag.shape!=(5,5) or torch.any(dag.diag()!=0):raise ValueError('Require 5-node DAG')
        if (torch.linalg.matrix_power(dag.abs(),5).abs()>1e-6).any():raise ValueError('DAG cyclic')
        self.register_buffer('dag',dag)
        if dose_values is None:dose_values=list(range(n_doses))
        if len(dose_values)!=n_doses:raise ValueError('dose values count mismatch')
        dose_values=torch.tensor([float(x) for x in dose_values],dtype=torch.float32)
        if not torch.isfinite(dose_values).all() or (dose_values<0).any():
            raise ValueError('Non-finite or negative doses')
        self.register_buffer('dose_values',dose_values)
        self.genes=int(genes);self.width=int(width)
        self.auxiliary=float(auxiliary);self.fullrank_gate=bool(fullrank_gate)
        self.cfm_weight=float(cfm_weight)
        if self.cfm_weight<1:raise ValueError('cfm_weight should be >=1')
        latent=32
        self.source_to_hidden=nn.Sequential(nn.Linear(genes,width),nn.LayerNorm(width),nn.SiLU())
        self.drug_emb=nn.Embedding(n_drugs,latent)
        self.dose_mlp=nn.Sequential(nn.Linear(1,latent),nn.SiLU(),nn.Linear(latent,latent))
        self.line_emb=nn.Embedding(n_lines,latent)
        self.path_to_hidden=nn.Linear(genes,width)
        # Original CausFlow Denoise_net and its source-exogenous encoder.
        self.core=Denoise_net(dim=width,out_dim=width,num_factor=5,
                              causal_dag=dag,label_categories=[n_drugs,n_doses,n_lines],
                              depth=2,num_heads=4,dim_head=32)
        self.to_gene=nn.Linear(width,genes)
        nn.init.zeros_(self.to_gene.weight);nn.init.zeros_(self.to_gene.bias)
        self.direct_to_gene=nn.Linear(latent*4,genes)
        nn.init.zeros_(self.direct_to_gene.weight);nn.init.zeros_(self.direct_to_gene.bias)
        if self.fullrank_gate:
            self.gate=nn.Linear(width,genes)
            nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)

    def components(self, source, drug, dose, line):
        # No target expression anywhere on the condition path.
        h=self.source_to_hidden(source)
        enc=self.core.DisentanglementEncoder
        mlv=enc.exogenous_encoder_m_v(h)
        d=enc.out_dim;f=5
        means,logvars=mlv[:,:f*d],mlv[:,f*d:]
        mu=means.reshape(source.shape[0],f,d)
        lv=logvars.reshape(source.shape[0],f,d)
        drug_e=self.drug_emb(drug)
        dose_e=self.dose_mlp(torch.log1p(self.dose_values[dose]).unsqueeze(-1))
        line_e=self.line_emb(line)
        source_e=mu[:,3,:]
        response_e=torch.zeros_like(source_e)
        E=torch.stack((drug_e,dose_e,line_e,source_e,response_e),dim=1)
        factors=torch.einsum('ij,bjd->bid',torch.linalg.inv(enc.I-enc.causal_dag),E)
        # Nontrivial source prior KL; no latent ``Graph Loss`` identity.
        src_lv=lv[:,3,:].clamp(min=-10,max=10)
        src_kl=0.5*(source_e.square()+src_lv.exp()-1-src_lv).mean()
        return factors,src_kl,E

    def velocity(self,x_t,t,factors):
        v_hidden=self.core(self.path_to_hidden(x_t),t,labels=None,concept_embs=factors)
        drug,dose,line=factors[:,0,:],factors[:,1,:],factors[:,2,:]
        direct=torch.cat((drug,dose,line,drug*dose),dim=-1)
        v=self.to_gene(v_hidden)+self.direct_to_gene(direct)
        if self.fullrank_gate:v=v+x_t*torch.tanh(self.gate(v_hidden))
        return v

    def loss_parts(self,target,source,drug,dose,line):
        factors,kl,_=self.components(source,drug,dose,line)
        t=torch.rand(target.shape[0],device=target.device)
        target_velocity=target-source
        x_t=source+t[:,None]*target_velocity
        v=self.velocity(x_t,t,factors)
        if self.cfm_weight==1:
            cfmloss=F.mse_loss(v,target_velocity)
        else:
            w=torch.where(target>0,self.cfm_weight,1.)
            cfmloss=((v-target_velocity).square()*w).mean()
        l1=v.abs().sum(dim=1).mean()
        aux_total=0.01*kl+1e-4*l1
        total=cfmloss+self.auxiliary*aux_total
        zero=cfmloss.new_zeros(())
        return total,dict(cfm=cfmloss,classification=zero,adversary=zero,
                          graph=zero,kl=kl,l1=l1,aux_total=aux_total)

    def cfm_loss(self,target,source,drug,dose,line):
        return self.loss_parts(target,source,drug,dose,line)[0]

    @torch.no_grad()
    def generate(self,source,drug,dose,line,steps=40,seed=None):
        if steps<1:raise ValueError('steps should be >=1')
        factors,_,_=self.components(source,drug,dose,line)
        x=source.clone();dt=1/steps
        for i in range(steps):
            t=torch.full((len(x),),i*dt,device=x.device,dtype=x.dtype)
            x=x+dt*self.velocity(x,t,factors)
        return x
