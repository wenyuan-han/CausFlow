"""BR6.4 E1 joint training on reviewed BR6 V2 lock; inner validation only."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
LOCK='f27281346dc457f602acab6b78a5caaf61b60e038f46b229827b56d0d9d8ca4b'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8')
def check(p,h):require(sha(p)==h,'Hash mismatch: '+str(p))
def verify_plan(out):
    p=read(out/'plan.json');require(p['schema']=='br64_e1_v1','Wrong plan schema');check(out/'plan.json',read(out/'plan_status.json')['sha256'])
    for path,h in p['code'].items():check(path,h)
    lock=Path(p['lock_dir']);check(lock/'lock.json',p['lock_sha256']);return p,lock

def prepare(root,lock,out,smoke):
    require(not out.exists() and out.parent==root/'Output','Fresh direct child of Output required')
    check(lock/'lock.json',LOCK);l=read(lock/'lock.json');require(l['schema']=='BR6_strict_panel_v2' and l['n_fit']==220998 and l['n_genes']==5000,'Wrong V2 data')
    sys.path.insert(0,str(root/'CausFlow_BR6_Lock'));import build_br6 as b
    b.verify(lock);ss=read(lock/'inner_baseline_score/score_status.json');require(ss['lock_sha256']==LOCK and ss['status']=='COMPLETE' and ss['n_queries']==24,'Wrong scoring target lock')
    code={str(root/k):v for k,v in read(HERE/'source_lock.json').items()};code.update({str(HERE/n):sha(HERE/n) for n in ['run_br64.py','worker.py','source_lock.json']});code[str(root/'CausFlow_BR6_Lock/locked_metrics.py')]=l['code']['locked_metrics.py']
    for path,h in code.items():check(path,h)
    import csv
    with (lock/'inner_queries.csv').open(encoding='utf8',newline='') as f:qs=list(csv.DictReader(f))
    require(len(qs)==24 and len({q['query_id'] for q in qs})==24,'Inner queries invalid')
    if smoke:qs=[next(q for q in qs if q['split']==s) for s in ['val_ID','val_OOD']]
    out.mkdir();p=dict(schema='br64_e1_v1',root=str(root),lock_dir=str(lock),lock_sha256=LOCK,code=code,seed=0,width=192,batch=32,lr=.0001,stages=[2] if smoke else [2000,10000],ode_steps=40,smoke=smoke,queries=qs,n_genes=l['n_genes'],n_fit=l['n_fit'],vocab=read(lock/'vocab.json'),scope='E1 joint-trained historical source-anchored adapter; V2 inner only; not native reconstruction; no outer evaluation')
    dump(out/'plan.json',p);dump(out/'plan_status.json',{'sha256':sha(out/'plan.json')});print('BR64 PREFLIGHT COMPLETE',flush=True)

def score(out):
    p,lock=verify_plan(out);sys.path.insert(0,str(Path(p['root'])/'CausFlow_BR6_Lock'));import locked_metrics as m
    import numpy as np,pandas as pd
    l=read(lock/'lock.json');tr=read(out/'train_status.json');pr=read(out/'predict_status.json');ph=sha(out/'plan.json');require(tr['plan_sha256']==pr['plan_sha256']==ph and tr['status']==pr['status']=='COMPLETE','Wrong run records');check(out/'train_status.json',pr['train_status_sha256'])
    dest=out/'score';require(not dest.exists(),'Score exists; preserve')
    base=lock/'inner_baseline_score';ss=read(base/'score_status.json');require(ss['lock_sha256']==p['lock_sha256'] and ss['status']=='COMPLETE','Wrong target lock');rows=[];qids={q['query_id'] for q in p['queries']};expected={f'E1_{step}_{k}.npy' for step in p['stages'] for k in qids};require(set(pr['files'])==expected,'Prediction completeness mismatch')
    for q in p['queries']:
        k=q['query_id'];tp=base/'targets'/(k+'.npy');check(tp,ss['target_hashes'][tp.name]);cp=lock/'inner/controls'/(k+'.npy');check(cp,l['files'][f'inner/controls/{k}.npy']);real=np.load(tp,allow_pickle=False).astype(float);ctl=np.load(cp,allow_pickle=False).astype(float);deg=m.rank_deg(real,ctl);require(real.shape==(int(q['n_real']),p['n_genes']),'Target shape mismatch')
        for model,kind in [('No-change','sources'),('Same/nearest-dose','nearest')]:
            fn=lock/'inner'/kind/(k+'.npy');check(fn,l['files'][f'inner/{kind}/{k}.npy']);y=np.load(fn,allow_pickle=False).astype(float);rows.append({**q,'model':model,'step':0,'output':'raw',**m.metrics(real,ctl,y,deg)})
        for step in p['stages']:
            fn=out/'predictions'/f'E1_{step}_{k}.npy';check(fn,pr['files'][fn.name]);y=np.load(fn,allow_pickle=False).astype(float);require(y.shape==(256,p['n_genes']) and np.isfinite(y).all(),'Bad prediction');raw=np.maximum(y,0);s4=np.where(raw<.01,0,raw)
            for tag,v in [('raw',raw),('S4',s4)]:rows.append({**q,'model':'E1','step':step,'output':tag,**m.metrics(real,ctl,v,deg)})
    df=pd.DataFrame(rows);require(len(df)==len(qids)*(2+2*len(p['stages'])),'Incomplete scores');dest.mkdir();df.to_csv(dest/'metrics_by_condition.csv',index=False);sm=df.groupby(['split','model','step','output'])[list(m.METRICS)].mean().reset_index();sm.to_csv(dest/'summary.csv',index=False);df.groupby(['split','model','step','output'])[list(m.METRICS)].count().to_csv(dest/'finite_counts.csv');v=sm[(sm.split=='val_OOD')&(sm.model=='E1')&(sm.output=='raw')].set_index('step').W1_DEG50_exact;require(np.isfinite(v).all(),'Nonfinite W1');chosen=int(min(v[v<=v.min()*1.01].index));dump(dest/'decision.json',dict(verdict='SMOKE_ONLY' if p['smoke'] else 'REVIEW_PILOT',selection='earliest step within 1% of min inner OOD raw DEG50 W1',chosen_step=chosen,checkpoint_sha256=tr['checkpoints'][str(chosen)],boundary_optimum=int(v.idxmin())==max(p['stages']),lock_sha256=p['lock_sha256'],plan_sha256=ph,score_count=len(df),outer_evaluated=False));print(sm.to_string(index=False));print('BR64 SCORE COMPLETE',flush=True)

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--phase',required=True,choices=['preflight','train','predict','score']);a.add_argument('--root',default='.');a.add_argument('--lock',default='Output/BR6_StrictLocked_v2');a.add_argument('--out',required=True);a.add_argument('--device',default='cuda',choices=['cpu','cuda']);a.add_argument('--smoke',action='store_true');x=a.parse_args();root=Path(x.root).resolve();out=Path(x.out).resolve();lock=(root/x.lock).resolve()
    if x.phase=='preflight':prepare(root,lock,out,x.smoke)
    elif x.phase=='score':score(out)
    else:subprocess.run([sys.executable,str(HERE/'worker.py'),str(out),x.phase,x.device],check=True)
if __name__=='__main__':main()
