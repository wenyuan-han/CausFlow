"""Official scheduler oracle; no surrogate sampler. Executed by full test."""
import torch
import diffusers
from diffusers import DDIMScheduler
assert diffusers.__version__=='0.30.3'
torch.manual_seed(7)
s=DDIMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2',prediction_type='v_prediction',clip_sample=False,thresholding=False,timestep_spacing='trailing',set_alpha_to_one=True)
x=torch.randn(4,8);eps=torch.randn_like(x)
for n in (2,5,10,20,40,50,100):
    s.set_timesteps(n);assert len(s.timesteps)==n and int(s.timesteps[0])==999
    for k in (int(s.timesteps[0]),int(s.timesteps[-1])):
        ts=torch.full((len(x),),k,dtype=torch.long);xt=s.add_noise(x,eps,ts);v=s.get_velocity(x,eps,ts);r=s.step(v,k,xt,eta=0.);assert torch.allclose(r.pred_original_sample,x,atol=2e-5,rtol=2e-5)
print('PASS official Diffusers DDIM v-prediction oracle and NFE schedules')
