"""Isolated PyTorch process; no scoring or target-file access during generation."""
import sys,json,hashlib,os,copy,time,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def require(c,msg):
    if not c:raise ValueError(msg)
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8')
def statehash(m,cond):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):
        if pm.is_condition(k)==cond:
            z=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(z.dtype).encode());h.update(str(tuple(z.shape)).encode());h.update(z.numpy().tobytes())
    return h.hexdigest()
def reset(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def save(p,obj):
    tmp=p.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,p)
def main(pp,device):
    pp=Path(pp);out=pp.parent;p=json.loads(pp.read_text(encoding='utf8'));ph=sha(pp)
    require(p['schema']=='br65_outer_v1','Wrong schema')
    origin=Path(p['origin_dir']);original_hash=p['origin_plan_sha256']
    require(sha(origin/'plan.json')==original_hash,'Original plan changed')
    require(sha(origin/'train_status.json')==p['origin_train_status_sha256'],'Original training record changed')
    sys.path.insert(0,str(Path(p['root'])/'CausFlow_BR65'));import pilot_model as pm
    require(Path(pm.__file__).resolve()==Path(p['root'])/'CausFlow_BR65'/'pilot_model.py','Wrong pilot model module')
    globals()['pm']=pm
    require(sha(pp)==json.loads((out/'preflight_status.json').read_text(encoding='utf8'))['plan_sha256'],'Plan changed')
    data_dir=Path(p['lock_dir']);require(sha(data_dir/'lock.json')==p['lock_sha256'],'V2 lock changed')
    require(device!='cuda' or torch.cuda.is_available(),'CUDA not available')
    for f,h in p['code'].items():require(sha(f)==h,'Changed source '+f)
    sys.path.insert(0,p['root']);sys.path.insert(0,p['source'])
    import run_locked_benchmark as b
    require(Path(b.__file__).resolve()==Path(p['source'])/'run_locked_benchmark.py','Wrong source')
    torch.set_num_threads(1);t,dev,make=b.torch_context(SimpleNamespace(repo=Path(p['root']),device=device))
    has_ddim='DDIM' in p['models'];require(has_ddim or p['fixture'],'No diffusion fallback allowed')
    if has_ddim:
        import diffusers
        from diffusers import DDIMScheduler
        require(diffusers.__version__=='0.30.3','Install diffusers==0.30.3')
        def scheduler():return DDIMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2',prediction_type='v_prediction',clip_sample=False,thresholding=False,timestep_spacing='trailing',set_alpha_to_one=True)
        schedule=scheduler()
    meta=dict(device=str(dev),torch_version=str(t.__version__),diffusers_version=diffusers.__version__ if has_ddim else None,gpu=t.cuda.get_device_name(dev) if dev.type=='cuda' else None,cuda_version=t.version.cuda,threads=t.get_num_threads())
    if dev.type=='cuda':t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
    def standard(x):return (x-p['mean'])/p['sd']
    def inverse(x):return (x*p['sd']+p['mean']).clamp_min(0)
    seed=p['seed']
    print('BR65 DEVICE',meta,'PHASE','outer_predict',flush=True)
    def new():return make('E1',p['vocab'],p['n_genes'],p['width'])
    def load(cp,h):
        require(sha(cp)==h,'Checkpoint changed');ck=torch.load(cp,map_location='cpu',weights_only=True);require(ck['plan_sha256']==original_hash,'Wrong checkpoint plan');return ck
    tr=json.loads((origin/'train_status.json').read_text(encoding='utf8'));require(tr['status']=='COMPLETE' and tr['plan_sha256']==original_hash and tr['meta']==meta,'Training/inference environment mismatch');dest=out/'predictions';dest.mkdir(exist_ok=False);files={};times=[]
    def sync():
        if dev.type=='cuda':t.cuda.synchronize(dev)
    for kind in p['models']:
        for step in p['stages']:
            fn=origin/'models'/f'{kind}_{step}.pt';ck=load(fn,tr['checkpoint_hashes'][fn.name]);require(ck['kind']==kind and ck['step']==step,'Wrong checkpoint');m=new();m.load_state_dict(ck['model'],strict=True);m.eval();require(statehash(m,True)==tr['shared_hash'],'Condition mismatch');del ck
            for budget in ([1] if kind=='NonFlow' else p['budgets'] if step==p['selected_steps'][kind] else [p['selection_nfe']]):
                sched=scheduler() if kind=='DDIM' else None
                if sched is not None:
                    sched.set_timesteps(budget,device='cpu');ks=[int(k) for k in sched.timesteps];require(len(ks)==budget,'Unexpected DDIM NFE')
                count=[0];h=m.path_to_hidden.register_forward_hook(lambda *args:count.__setitem__(0,count[0]+1))
                @t.no_grad()
                def generate(src,lab,noise):
                    f,_,_=m.components(src,*lab.unbind(1));x=standard(src)
                    if kind=='NonFlow':x=x+m.velocity(x,x.new_zeros(len(x)),f)
                    elif kind=='CFM':
                        for i in range(budget):x=x+m.velocity(x,x.new_full((len(x),),i/budget),f)/budget
                    else:
                        x=noise.clone()
                        for k in ks:
                            pred=m.velocity(sched.scale_model_input(x,k),x.new_full((len(x),),(k+1)/1000),f);x=sched.step(pred,k,x,eta=0.,return_dict=True).prev_sample
                    return inverse(x)
                for j,q in enumerate(p['queries']):
                    key=q['query_id'];fn=data_dir/'outer/sources'/(key+'.npy');require(sha(fn)==p['arrays']['sources'][key],'Source changed');src=np.load(fn,allow_pickle=False);lab=[list(map(str,p['vocab'][c])).index(str(q[c])) for c in b.KEYS];noisegen=t.Generator(device=dev).manual_seed(b.seedof('br5_noise',seed,key));batches=[]
                    for i in range(0,len(src),64):
                        ss=t.tensor(src[i:i+64],device=dev);ll=t.tensor(lab,dtype=t.long,device=dev).expand(len(ss),-1);noise=t.randn(ss.shape,device=dev,generator=noisegen) if kind=='DDIM' else None;batches.append((ss,ll,noise))
                    if j==0:generate(*batches[0]);sync()
                    count[0]=0;sync();base=t.cuda.memory_allocated(dev) if dev.type=='cuda' else None
                    if dev.type=='cuda':t.cuda.reset_peak_memory_stats(dev)
                    start=time.perf_counter();ys=[generate(*batch) for batch in batches];sync();elapsed=time.perf_counter()-start;peak=t.cuda.max_memory_allocated(dev) if dev.type=='cuda' else None
                    require(count[0]==len(batches)*budget,'Measured NFE mismatch');y=t.cat(ys).cpu().numpy();require(y.shape==src.shape and np.isfinite(y).all(),'Nonfinite or wrong prediction');name=f'{kind}_{step}_{budget}_{key}.npy';np.save(dest/name,y,allow_pickle=False);files[name]=sha(dest/name);times.append(dict(model=kind,step=step,nfe=budget,query_id=key,split=q['split'],seconds=elapsed,cells_per_second=len(src)/max(elapsed,1e-12),n_batches=len(batches),measured_nfe_total=count[0],measured_nfe_per_batch=count[0]//len(batches),peak_allocated_bytes=peak,incremental_peak_bytes=peak-base if peak is not None else None));del ys,batches,ss,ll,noise;print('PREDICTED',kind,step,budget,key,flush=True)
                h.remove()
            del m
    dump(out/'timing.json',times);dump(out/'predict_status.json',dict(status='COMPLETE',plan_sha256=ph,train_status_sha256=sha(origin/'train_status.json'),meta=meta,files=files,timing_sha256=sha(out/'timing.json'),target_expression_as_inference_input=False));print('BR65 OUTER PREDICT COMPLETE',flush=True)
if __name__=='__main__':main(*sys.argv[1:])
