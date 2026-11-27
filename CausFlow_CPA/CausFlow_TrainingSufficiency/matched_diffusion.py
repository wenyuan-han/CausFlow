"""Matched epsilon-prediction diffusion generator using the actual ResponseModel backbone.
This is a controlled DDPM/DDIM baseline, NOT the CausCell implementation.
"""
from types import MethodType
import math
import torch

T=1000

def schedule(device,dtype):
    grid=torch.arange(T+1,device=device,dtype=torch.float64)/T
    a=torch.cos((grid+.008)/1.008*math.pi/2).square();a=a/a[0]
    beta=(1-a[1:]/a[:-1]).clamp(1e-8,.999)
    return (1-beta).cumprod(0).to(dtype=dtype)

def loss_parts(self,target,source,drug,dose,line):
    f,kl,_=self.components(source,drug,dose,line)
    a=schedule(target.device,target.dtype);k=torch.randint(T,(len(target),),device=target.device);ab=a[k,None];eps=torch.randn_like(target)
    xt=ab.sqrt()*target+(1-ab).sqrt()*eps
    pred=self.velocity(xt,(k.to(target.dtype)+1)/T,f)
    loss=(pred-eps).square().mean()
    return loss+.01*kl,{'cfm':loss,'kl':kl}

@torch.no_grad()
def generate(self,source,drug,dose,line,steps=40,seed=None):
    if not 2<=steps<=T:raise ValueError('DDIM steps must be within 2..1000')
    f,_,_=self.components(source,drug,dose,line);a=schedule(source.device,source.dtype)
    if seed is None:x=torch.randn_like(source)
    else:
        g=torch.Generator(device=source.device);g.manual_seed(seed);x=torch.randn(source.shape,device=source.device,dtype=source.dtype,generator=g)
    times=torch.linspace(T-1,0,steps).round().long().tolist()
    for i,k in enumerate(times):
        ab=a[k];t=source.new_full((len(source),),(k+1)/T);eps=self.velocity(x,t,f)
        x0=((x-(1-ab).sqrt()*eps)/ab.sqrt()).clamp(0.,8.)
        # Standard clipped-denoised DDIM update, fixed nonnegative log-expression support.
        eps=(x-ab.sqrt()*x0)/(1-ab).sqrt();prev=a[times[i+1]] if i+1<len(times) else x.new_tensor(1.)
        x=prev.sqrt()*x0+(1-prev).sqrt()*eps
    return x

def adapt(model):
    model.loss_parts=MethodType(loss_parts,model);model.generate=MethodType(generate,model);return model
