"""Isolated PyTorch process; no scoring or target-file access during generation."""
import sys,json,hashlib,os,copy,time,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import pilot_model as pm

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
def main(pp,phase,device):
    pp=Path(pp);out=pp.parent;p=json.loads(pp.read_text(encoding='utf8'));ph=sha(pp)
    require(p['schema']=='br65_v2','Wrong schema')
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
    print('BR65 DEVICE',meta,'PHASE',phase,flush=True)
    def new():return make('E1',p['vocab'],p['n_genes'],p['width'])
    def load(cp,h):
        require(sha(cp)==h,'Checkpoint changed');ck=torch.load(cp,map_location='cpu',weights_only=True);require(ck['plan_sha256']==ph,'Wrong checkpoint plan');return ck
    if phase=='train':
        require(sha(data_dir/'fit_x.npy')==p['train_sha256'] and sha(data_dir/'fit_scale.json')==p['scale_sha256'],'Training data/scale changed');require(sha(data_dir/'fit_index.npz')==p['fit_sha256'],'Fit index changed');data=np.load(data_dir/'fit_x.npy',mmap_mode='r',allow_pickle=False)
        with np.load(data_dir/'fit_index.npz',allow_pickle=False) as f:fit,labels,control=f['fit'],f['labels'],f['control']
        require(len(fit)==p['n_fit'] and len(np.unique(fit))==len(fit) and data.shape==(len(fit),p['n_genes']),'Fit invalid');require(labels.shape==(len(fit),3) and control.shape==(len(fit),),'Fit metadata invalid');pools={int(l):np.flatnonzero((labels[:,2]==l)&control) for l in np.unique(labels[:,2])};require(all(len(v)>=2 for v in pools.values()),'Insufficient controls')
        def batch(r,h):
            ix=r.choice(len(fit),p['batch'],replace=len(fit)<p['batch']);si=np.array([r.choice(pools[int(labels[i,2])]) for i in ix]);
            for j in np.flatnonzero(ix==si):si[j]=r.choice(pools[int(labels[ix[j],2])][pools[int(labels[ix[j],2])]!=ix[j]])
            h.update(fit[ix].tobytes()+fit[si].tobytes());y=np.asarray(data[ix],np.float32);s=np.asarray(data[si],np.float32);require(np.isfinite(y).all() and np.isfinite(s).all() and (y>=0).all() and (s>=0).all(),'Fit expression invalid');return t.tensor(s,device=dev),t.tensor(y,device=dev),t.tensor(labels[ix],device=dev)
        dest=out/'models';dest.mkdir(exist_ok=False);reset(3000+seed);m=new();gen_before=statehash(m,False);pm.freeze_generator(m);head=pm.MomentHead(p['n_genes']).to(dev);m.eval();head.train();opt=t.optim.AdamW([z for z in m.parameters() if z.requires_grad]+list(head.parameters()),lr=p['lr'],weight_decay=1e-4);r=np.random.default_rng(8300+seed);draws=hashlib.sha256();history=[]
        def optimize(loss,params):
            require(bool(t.isfinite(loss)),'Nonfinite loss');opt.zero_grad(set_to_none=True);loss.backward();gn=t.nn.utils.clip_grad_norm_(params,1.);require(bool(t.isfinite(gn)),'Nonfinite gradient');opt.step();return float(gn)
        for step in range(1,p['shared_steps']+1):
            src,y,lab=batch(r,draws);f,kl,_=m.components(src,*lab.unbind(1));loss=head(f,y)+.01*kl;gn=optimize(loss,[z for z in m.parameters() if z.requires_grad]+list(head.parameters()))
            if step==1 or step%100==0 or step==p['shared_steps']:history.append(dict(model='shared',step=step,loss=float(loss.detach()),grad_norm=gn));print('SHARED',step,float(loss.detach()),flush=True)
        require(statehash(m,False)==gen_before,'Generator changed in representation pretraining');shared_hash=statehash(m,True);initial=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()});save(dest/'shared.pt',dict(model=initial,plan_sha256=ph,condition_hash=shared_hash));del m,head,opt
        report=dict(status='COMPLETE',plan_sha256=ph,meta=meta,shared_hash=shared_hash,condition_hashes={},initial_hashes={},draw_hashes={},parameter_counts={},shared_checkpoint_sha256=sha(dest/'shared.pt'),shared_draw_hash=draws.hexdigest(),checkpoint_hashes={})
        for kind in p['models']:
            reset(4000+seed);m=new();m.load_state_dict(initial,strict=True);m.train();pm.freeze_condition(m);# Original frozen condition deterministic even with generator train mode.
            require(statehash(m,True)==shared_hash,'Initial condition mismatch');report['initial_hashes'][kind]=statehash(m,False);params=[z for z in m.parameters() if z.requires_grad];report['parameter_counts'][kind]=sum(z.numel() for z in params);opt=t.optim.AdamW(params,lr=p['lr'],weight_decay=1e-4);r=np.random.default_rng(8400+seed);draws=hashlib.sha256();noisegen=t.Generator(device=dev);noisegen.manual_seed(9400+seed)
            for step in range(1,max(p['stages'])+1):
                src,y,lab=batch(r,draws);u,v=standard(src),standard(y)
                with t.no_grad():f,_,_=m.components(src,*lab.unbind(1))
                if kind=='CFM':
                    ti=t.rand(len(y),device=dev,generator=noisegen);target=v-u;loss=(m.velocity(u+ti[:,None]*target,ti,f)-target).square().mean()
                elif kind=='NonFlow':loss=(u+m.velocity(u,u.new_zeros(len(u)),f)-v).square().mean()
                else:
                    k=t.randint(1000,(len(y),),device=dev,generator=noisegen);eps=t.randn(y.shape,device=dev,generator=noisegen);x=schedule.add_noise(v,eps,k);target=schedule.get_velocity(v,eps,k);loss=(m.velocity(x,(k.float()+1)/1000,f)-target).square().mean()
                gn=optimize(loss,params)
                if step==1 or step%100==0 or step in p['stages']:history.append(dict(model=kind,step=step,loss=float(loss.detach()),grad_norm=gn));print(kind,step,float(loss.detach()),flush=True)
                if step in p['stages']:
                    require(statehash(m,True)==shared_hash,'Frozen representation changed');f=dest/f'{kind}_{step}.pt';save(f,dict(model=m.state_dict(),kind=kind,step=step,plan_sha256=ph,condition_hash=shared_hash));report['checkpoint_hashes'][f.name]=sha(f)
            report['condition_hashes'][kind]=statehash(m,True);report['draw_hashes'][kind]=draws.hexdigest();del m,opt
        require(all(len(set(report[k].values()))==1 for k in ('draw_hashes','initial_hashes','parameter_counts','condition_hashes')),'Arms not matched');dump(out/'training_history.json',history);dump(out/'train_status.json',report);print('BR65 TRAIN COMPLETE',flush=True)
    elif phase=='predict':
        tr=json.loads((out/'train_status.json').read_text(encoding='utf8'));require(tr['status']=='COMPLETE' and tr['plan_sha256']==ph and tr['meta']==meta,'Training/inference environment mismatch');dest=out/'predictions';dest.mkdir(exist_ok=False);files={};times=[]
        def sync():
            if dev.type=='cuda':t.cuda.synchronize(dev)
        for kind in p['models']:
            for step in p['stages']:
                fn=out/'models'/f'{kind}_{step}.pt';ck=load(fn,tr['checkpoint_hashes'][fn.name]);require(ck['kind']==kind and ck['step']==step,'Wrong checkpoint');m=new();m.load_state_dict(ck['model'],strict=True);m.eval();require(statehash(m,True)==tr['shared_hash'],'Condition mismatch');del ck
                for budget in ([1] if kind=='NonFlow' else p['budgets']):
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
                        key=q['query_id'];fn=data_dir/'inner/sources'/(key+'.npy');require(sha(fn)==p['arrays']['sources'][key],'Source changed');src=np.load(fn,allow_pickle=False);lab=[list(map(str,p['vocab'][c])).index(str(q[c])) for c in b.KEYS];noisegen=t.Generator(device=dev).manual_seed(b.seedof('br5_noise',seed,key));batches=[]
                        for i in range(0,len(src),64):
                            ss=t.tensor(src[i:i+64],device=dev);ll=t.tensor(lab,dtype=t.long,device=dev).expand(len(ss),-1);noise=t.randn(ss.shape,device=dev,generator=noisegen) if kind=='DDIM' else None;batches.append((ss,ll,noise))
                        if j==0:generate(*batches[0]);sync()
                        count[0]=0;sync();base=t.cuda.memory_allocated(dev) if dev.type=='cuda' else None
                        if dev.type=='cuda':t.cuda.reset_peak_memory_stats(dev)
                        start=time.perf_counter();ys=[generate(*batch) for batch in batches];sync();elapsed=time.perf_counter()-start;peak=t.cuda.max_memory_allocated(dev) if dev.type=='cuda' else None
                        require(count[0]==len(batches)*budget,'Measured NFE mismatch');y=t.cat(ys).cpu().numpy();require(y.shape==src.shape and np.isfinite(y).all(),'Nonfinite or wrong prediction');name=f'{kind}_{step}_{budget}_{key}.npy';np.save(dest/name,y,allow_pickle=False);files[name]=sha(dest/name);times.append(dict(model=kind,step=step,nfe=budget,query_id=key,split=q['split'],seconds=elapsed,cells_per_second=len(src)/max(elapsed,1e-12),n_batches=len(batches),measured_nfe_total=count[0],measured_nfe_per_batch=count[0]//len(batches),peak_allocated_bytes=peak,incremental_peak_bytes=peak-base if peak is not None else None));del ys,batches,ss,ll,noise;print('PREDICTED',kind,step,budget,key,flush=True)
                    h.remove()
                del m
        dump(out/'timing.json',times);dump(out/'predict_status.json',dict(status='COMPLETE',plan_sha256=ph,train_status_sha256=sha(out/'train_status.json'),meta=meta,files=files,timing_sha256=sha(out/'timing.json'),target_expression_as_inference_input=False));print('BR65 PREDICT COMPLETE',flush=True)
    else:raise ValueError(phase)
if __name__=='__main__':main(*sys.argv[1:])
