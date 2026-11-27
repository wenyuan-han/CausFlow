"""Training/prediction: no raw H5AD or target expression read."""
import os,sys,hashlib,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch as t
from run_br64 import verify_plan,require,sha,check,read,dump

def main(out,phase,device):
    out=Path(out).resolve();p,lock=verify_plan(out);ph=sha(out/'plan.json');l=read(lock/'lock.json');root=Path(p['root']);src=root/'CausFlow_TrainingSufficiency';sys.path.insert(0,str(root));sys.path.insert(0,str(src));import run_locked_benchmark as b
    require(Path(b.__file__).resolve()==src/'run_locked_benchmark.py','Wrong model source');t.set_num_threads(1);_,dev,make=b.torch_context(SimpleNamespace(repo=root,device=device));require(dev.type==device,'Wrong device');meta=dict(torch=str(t.__version__),device=str(dev),gpu=t.cuda.get_device_name(dev) if dev.type=='cuda' else None,cuda=t.version.cuda)
    def model():return make('E1',p['vocab'],p['n_genes'],p['width'])
    if phase=='train':
        dest=out/'models';require(not dest.exists(),'Model directory exists; preserve');check(lock/'fit_x.npy',l['files']['fit_x.npy']);check(lock/'fit_index.npz',l['files']['fit_index.npz']);check(lock/'vocab.json',l['files']['vocab.json']);x=np.load(lock/'fit_x.npy',mmap_mode='r',allow_pickle=False)
        with np.load(lock/'fit_index.npz',allow_pickle=False) as f:fit,labels,control=f['fit'],f['labels'],f['control']
        require(x.shape==(p['n_fit'],p['n_genes']) and len(np.unique(fit))==p['n_fit'] and labels.shape==(p['n_fit'],3) and control.shape==(p['n_fit'],),'Fit shape mismatch');pools={int(i):np.flatnonzero((labels[:,2]==i)&control) for i in np.unique(labels[:,2])};require(all(len(v)>=2 for v in pools.values()),'Insufficient line controls')
        b.reset(t,p['seed']);rng=np.random.default_rng(b.seedof('train-v2',p['seed']));m=model();m.train();params=[z for z in m.parameters() if z.requires_grad];opt=t.optim.AdamW(params,lr=p['lr'],weight_decay=1e-4);dest.mkdir();history=[];buf=[];hashes={};draws=hashlib.sha256()
        for step in range(1,max(p['stages'])+1):
            ix=rng.choice(len(fit),p['batch'],replace=len(fit)<p['batch']);si=np.array([rng.choice(pools[int(labels[i,2])]) for i in ix]);
            for j in np.flatnonzero(ix==si):si[j]=rng.choice(pools[int(labels[ix[j],2])][pools[int(labels[ix[j],2])]!=ix[j]])
            draws.update(fit[ix].tobytes()+fit[si].tobytes());target=t.tensor(np.asarray(x[ix],np.float32),device=dev);source=t.tensor(np.asarray(x[si],np.float32),device=dev);lab=t.tensor(labels[ix],device=dev);opt.zero_grad(set_to_none=True);loss,parts=m.loss_parts(target,source,*lab.unbind(1));require(bool(t.isfinite(loss)),'Nonfinite loss');loss.backward();gn=t.nn.utils.clip_grad_norm_(params,1.);require(bool(t.isfinite(gn)),'Nonfinite grad');opt.step();buf.append([float(loss.detach()),float(parts['cfm'].detach()),float(parts['kl'].detach()),float(gn)])
            if step==1 or step%200==0 or step in p['stages']:
                v=np.mean(buf,axis=0);history.append(dict(step=step,loss=float(v[0]),cfm=float(v[1]),kl=float(v[2]),grad_norm=float(v[3]),window=len(buf)));buf=[];print('E1',step,float(v[0]),flush=True)
            if step in p['stages']:
                fn=dest/f'E1_{step}.pt';tmp=fn.with_suffix('.tmp');t.save(dict(model=m.state_dict(),step=step,plan_sha256=ph,lock_sha256=p['lock_sha256']),tmp);os.replace(tmp,fn);hashes[str(step)]=sha(fn)
        dump(out/'training_history.json',history);dump(out/'train_status.json',dict(status='COMPLETE',plan_sha256=ph,meta=meta,checkpoints=hashes,draw_sha256=draws.hexdigest(),parameter_count=sum(v.numel() for v in params),objective='original E1 adapter source-anchored CFM + source KL',target_expression_scope='core fit only'));print('BR64 TRAIN COMPLETE',flush=True)
    elif phase=='predict':
        tr=read(out/'train_status.json');require(tr['status']=='COMPLETE' and tr['plan_sha256']==ph,'Wrong training record');dest=out/'predictions';require(not dest.exists(),'Predictions exist; preserve');dest.mkdir();files={}
        for step in p['stages']:
            fn=out/'models'/f'E1_{step}.pt';check(fn,tr['checkpoints'][str(step)]);ck=t.load(fn,map_location='cpu',weights_only=True);require(ck['step']==step and ck['plan_sha256']==ph and ck['lock_sha256']==p['lock_sha256'],'Checkpoint mismatch');m=model();m.load_state_dict(ck['model'],strict=True);m.eval();del ck
            for q in p['queries']:
                k=q['query_id'];fn=lock/'inner/sources'/(k+'.npy');check(fn,l['files'][f'inner/sources/{k}.npy']);x=np.load(fn,allow_pickle=False);require(x.shape==(256,p['n_genes']),'Source shape mismatch');labs=[list(map(str,p['vocab'][c])).index(str(q[c])) for c in ['drug','dose','cell_line']];ys=[]
                with t.no_grad():
                    for i in range(0,len(x),64):
                        s=t.tensor(x[i:i+64],device=dev);lab=[t.full((len(s),),v,device=dev,dtype=t.long) for v in labs];ys.append(m.generate(s,*lab,steps=p['ode_steps']).cpu().numpy())
                y=np.concatenate(ys);require(y.shape==x.shape and np.isfinite(y).all(),'Bad prediction');name=f'E1_{step}_{k}.npy';np.save(dest/name,y,allow_pickle=False);files[name]=sha(dest/name);print('PREDICTED',step,k,flush=True)
            del m
        dump(out/'predict_status.json',dict(status='COMPLETE',plan_sha256=ph,train_status_sha256=sha(out/'train_status.json'),meta=meta,files=files,target_expression_as_inference_input=False));print('BR64 PREDICT COMPLETE',flush=True)
    else:raise ValueError(phase)
if __name__=='__main__':main(*sys.argv[1:])
