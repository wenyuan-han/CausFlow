"""Response-aware engineering adaptation of FactorSeparated B; fixed DAG, not causal discovery."""
import torch
from torch.nn import functional as F
from factor_ablation_model import FactorAblationModel

class ResponseModel(FactorAblationModel):
    def __init__(self,*args,l1_weight=1e-4,**kwargs):
        super().__init__(*args,**kwargs)
        self.l1_weight=float(l1_weight)

    def loss_parts(self,target,source,drug,dose,line):
        _,p=super().loss_parts(target,source,drug,dose,line)
        p['aux_total']=.01*p['kl']+self.l1_weight*p['l1']
        return p['cfm']+self.auxiliary*p['aux_total'],p

    def response_loss(self,source,labels,true_delta,steps=4,floor=1e-4):
        # Shared condition across the entire response batch. Only source and labels
        # enter the flow. The fit-only target statistic is used solely in the loss.
        if not bool((labels==labels[0]).all()):raise ValueError('Mixed conditions in response batch')
        factors,_,_=self.components(source,*labels.unbind(1))
        pred=source
        for i in range(steps):
            t=source.new_full((len(source),),i/steps)
            pred=pred+self.velocity(pred,t,factors)/steps
        delta=pred.mean(0)-source.mean(0)
        mse=F.mse_loss(delta,true_delta)
        relative=mse/true_delta.square().mean().clamp_min(floor)
        return relative,mse,delta.norm()/(true_delta.norm()+1e-12)
