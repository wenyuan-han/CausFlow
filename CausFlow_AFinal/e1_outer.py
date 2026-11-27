"""Re-use BR64 seed 0/1/2 weights; lock common inner-based selection before outer inference."""
import argparse,sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
from run_afinal import HERE,BR6,req,sha,check,read,dump,verify,csv

SEED0_PLAN='3e09763ba13fdd27a2dc21d5b4b9f2924e9e948e0c76638d1955bc3f72cfe3ad'
SEED0_TRAIN='809f1e3be50731e889ff7d7d7aab4c49d5b9b5986360df37fddbde821b04bb97'

def select_step(d):
    req(not d.duplicated(['seed','query_id','step']).any(),'Duplicate selection rows');req(set(d.seed)=={0,1,2} and set(d.step)=={2000,10000} and len(d)==72 and (d.groupby(['seed','step']).size()==12).all(),'Incomplete inner selection');req(np.isfinite(d.W1_DEG50_exact).all(),'Nonfinite inner W1');v=d.groupby('step').W1_DEG50_exact.mean();return int(min(v[v<=v.min()*1.01].index)),{str(k):float(x) for k,x in v.items()}

def selection(out):
    l,_,lock,bl=verify(out);root=Path(l['root']);inner=csv(lock/'inner_queries.csv');check(lock/'inner_queries.csv',bl['files']['inner_queries.csv']);ss=read(lock/'inner_baseline_score/score_status.json');req(ss['lock_sha256']==BR6 and ss['status']=='COMPLETE','Bad inner target status');sys.path.insert(0,str(root/'CausFlow_BR6_Lock'));import locked_metrics as m;check(Path(m.__file__),bl['code']['locked_metrics.py']);rows=[];runs={}
    for seed in (0,1,2):
        d=root/f'Output/BR64_E1_seed{seed}_v1';p=read(d/'plan.json');ph=sha(d/'plan.json');check(d/'plan.json',read(d/'plan_status.json')['sha256']);tr=read(d/'train_status.json');pr=read(d/'predict_status.json');req(p['schema']=='br64_e1_v1' and p['seed']==seed and p['smoke'] is False and p['lock_sha256']==BR6 and p['stages']==[2000,10000] and p['ode_steps']==40 and (p['width'],p['batch'],p['lr'])==(192,32,.0001) and p['queries']==inner.to_dict('records') and p['n_fit']==220998 and p['n_genes']==5000 and Path(p['root']).resolve()==root and Path(p['lock_dir']).resolve()==lock,'Invalid E1 plan');req(tr['status']==pr['status']=='COMPLETE' and tr['plan_sha256']==pr['plan_sha256']==ph and pr['target_expression_as_inference_input'] is False,'Bad E1 records');check(d/'train_status.json',pr['train_status_sha256']);req(set(pr['files'])=={f'E1_{s}_{k}.npy' for s in (2000,10000) for k in inner.query_id},'Missing E1 inner predictions')
        for path,h in p['code'].items():check(path,h)
        if seed==0:check(d/'plan.json',SEED0_PLAN);check(d/'train_status.json',SEED0_TRAIN)
        for step in (2000,10000):
            check(d/'models'/f'E1_{step}.pt',tr['checkpoints'][str(step)])
            for q in inner[inner.split=='val_OOD'].to_dict('records'):
                k=q['query_id'];tp=lock/'inner_baseline_score/targets'/f'{k}.npy';cp=lock/'inner/controls'/f'{k}.npy';check(tp,ss['target_hashes'][tp.name]);check(cp,bl['files'][f'inner/controls/{k}.npy']);real=np.load(tp,allow_pickle=False).astype(float);ctl=np.load(cp,allow_pickle=False).astype(float);pname=f'E1_{step}_{k}.npy';check(d/'predictions'/pname,pr['files'][pname]);y=np.load(d/'predictions'/pname,allow_pickle=False).astype(float);req(y.shape==(256,5000) and np.isfinite(y).all(),'Bad inner generation');rows.append(dict(seed=seed,step=step,query_id=k,W1_DEG50_exact=m.metrics(real,ctl,np.maximum(y,0),m.rank_deg(real,ctl))['W1_DEG50_exact']))
        runs[str(seed)]=dict(path=str(d),plan_sha256=ph,train_status_sha256=sha(d/'train_status.json'),predict_status_sha256=sha(d/'predict_status.json'),checkpoints=tr['checkpoints'])
    df=pd.DataFrame(rows);step,curve=select_step(df);dest=out/'e1_selection.json';req(not dest.exists(),'Selection exists, preserve');df.to_csv(out/'e1_inner_selection_metrics.csv',index=False);dump(dest,dict(br6_lock_sha256=BR6,derived_lock_sha256=sha(out/'lock.json'),selected_step=step,curve=curve,rule='earliest stage within 1% of min mean inner val_OOD raw W1 over 3 seeds and 12 conditions',nfe=40,scope='retrospective outer, no outer targets used for selection',inner_metrics_sha256=sha(out/'e1_inner_selection_metrics.csv'),runs=runs));dump(out/'e1_selection_status.json',dict(sha256=sha(dest)));print('E1 LOCKED',step,curve,flush=True)

def getsel(out):
    p=out/'e1_selection.json';check(p,read(out/'e1_selection_status.json')['sha256']);s=read(p);req(s['br6_lock_sha256']==BR6 and s['derived_lock_sha256']==sha(out/'lock.json') and s['selected_step'] in (2000,10000) and s['nfe']==40,'Bad E1 selection');check(out/'e1_inner_selection_metrics.csv',s['inner_metrics_sha256']);step,curve=select_step(pd.read_csv(out/'e1_inner_selection_metrics.csv'));req(step==s['selected_step'] and all(abs(curve[k]-s['curve'][k])<1e-12 for k in curve),'Selection mismatch');return s

def predict(out):
    import torch as t
    l,_,lock,bl=verify(out);s=getsel(out);root=Path(l['root']);src=root/'CausFlow_TrainingSufficiency';sys.path.insert(0,str(root));sys.path.insert(0,str(src));import run_locked_benchmark as b;req(Path(b.__file__).resolve()==src/'run_locked_benchmark.py','Wrong model source');req(t.cuda.is_available(),'CUDA unavailable');t.set_num_threads(1);_,dev,make=b.torch_context(SimpleNamespace(repo=root,device='cuda'));step=s['selected_step'];check(lock/'outer_queries.csv',bl['files']['outer_queries.csv']);qs=csv(lock/'outer_queries.csv').to_dict('records');req(len(qs)==48,'Wrong outer size')
    for seed in (0,1,2):
        rec=s['runs'][str(seed)];d=Path(rec['path']);check(d/'plan.json',rec['plan_sha256']);check(d/'train_status.json',rec['train_status_sha256']);p=read(d/'plan.json');dest=out/'predictions/E1'/f'seed_{seed}';req(not dest.exists(),'E1 predictions exist, preserve');dest.mkdir(parents=True);fn=d/'models'/f'E1_{step}.pt';check(fn,rec['checkpoints'][str(step)]);
        for path,h in p['code'].items():check(path,h)
        ck=t.load(fn,map_location='cpu',weights_only=True);req(ck['step']==step and ck['plan_sha256']==rec['plan_sha256'] and ck['lock_sha256']==BR6,'Bad checkpoint');b.reset(t,seed);model=make('E1',p['vocab'],p['n_genes'],p['width']);model.load_state_dict(ck['model'],strict=True);model.eval();del ck;hashes={}
        for q in qs:
            k=q['query_id'];rel=f'outer/sources/{k}.npy';check(lock/rel,bl['files'][rel]);x=np.load(lock/rel,allow_pickle=False);req(x.shape==(256,5000),'Bad source');labs=[list(map(str,p['vocab'][c])).index(str(q[c])) for c in ['drug','dose','cell_line']];ys=[]
            with t.no_grad():
                for i in range(0,len(x),64):
                    a=t.tensor(x[i:i+64],device=dev);lab=[t.full((len(a),),v,device=dev,dtype=t.long) for v in labs];ys.append(model.generate(a,*lab,steps=40).cpu().numpy())
            y=np.concatenate(ys);req(y.shape==x.shape and np.isfinite(y).all(),'Bad E1 predictions');np.save(dest/f'{k}.npy',y,allow_pickle=False);hashes[k]=sha(dest/f'{k}.npy');print('E1 OUTER',seed,k,flush=True)
        dump(dest/'manifest.json',dict(model='E1',seed=seed,br6_lock_sha256=BR6,selection_sha256=sha(out/'e1_selection.json'),checkpoint_sha256=sha(fn),files=hashes,target_expression_as_inference_input=False));del model
    print('E1 OUTER COMPLETE',flush=True)

def verify_e1(out,seed,qs):
    s=getsel(out);p=out/'predictions/E1'/f'seed_{seed}';m=read(p/'manifest.json');r=s['runs'][str(seed)];req(m['model']=='E1' and m['seed']==seed and m['br6_lock_sha256']==BR6 and m['target_expression_as_inference_input'] is False and m['checkpoint_sha256']==r['checkpoints'][str(s['selected_step'])] and set(m['files'])==set(qs),'E1 prediction provenance');check(out/'e1_selection.json',m['selection_sha256']);check(Path(r['path'])/'models'/f"E1_{s['selected_step']}.pt",m['checkpoint_sha256']);
    for k,h in m['files'].items():check(p/f'{k}.npy',h)
    return p

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['select','predict']);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve();selection(out) if a.phase=='select' else predict(out)
if __name__=='__main__':main()
