"""Reuse reviewed BR65 checkpoints for retrospective V2 outer evaluation; never train."""
import argparse, hashlib, itertools, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
SELECTION_SHA='06490b28b1ae3e2b8116563a41a2ac061da5d2cda90a3119a9b5fb51afd1d104'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def req(ok,msg):
    if not ok:raise ValueError(msg)
def check(p,h):req(sha(p)==h,'Hash mismatch: '+str(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8')
def qframe(p):return pd.read_csv(p,dtype={'dose':str,'drug':str,'cell_line':str},keep_default_na=False)
def expected(p):
    return {(a,s,n,q['query_id']) for a in p['models'] for s in p['stages'] for n in ([1] if a=='NonFlow' else p['budgets'] if s==p['selected_steps'][a] else [p['selection_nfe']]) for q in p['queries']}
def selection():
    check(HERE/'selection_lock.json',SELECTION_SHA);return read(HERE/'selection_lock.json')
def preflight(root,lock,out):
    req(not out.exists() and out.parent==root/'Output','Use a fresh direct child of Output')
    s=selection();check(lock/'lock.json',s['lock_sha256']);l=read(lock/'lock.json')
    req(l['schema']=='BR6_strict_panel_v2' and (l['n_inner'],l['n_outer'],l['n_genes'],l['n_fit'])==(24,48,5000,220998),'Wrong V2 lock')
    check(lock/'outer_queries.csv',l['files']['outer_queries.csv']);qs=qframe(lock/'outer_queries.csv');req(len(qs)==48 and qs.query_id.is_unique and qs.groupby('split').size().to_dict()=={'ID':24,'dose_OOD':24},'Wrong outer query set')
    req(qs.query_id.str.fullmatch(r'q\d+').all(),'Unexpected outer query ID')
    arrays={k:{} for k in ('sources','controls','nearest')}
    for k in arrays:
        for q in qs.to_dict('records'):
            key=q['query_id'];rel=f'outer/{k}/{key}.npy';check(lock/rel,l['files'][rel]);arrays[k][key]=l['files'][rel]
            x=np.load(lock/rel,mmap_mode='r',allow_pickle=False);req(x.ndim==2 and x.shape[1]==5000 and len(x)>0 and (k=='controls' or len(x)==256) and np.isfinite(x).all() and (x>=0).all(),'Invalid reference');del x
    allplans={}
    for seed in (0,1,2):
        ref=s['seeds'][str(seed)];orig=root/f'Output/BR65_Pilot_seed{seed}_v1';check(orig/'plan.json',ref['plan_sha256']);check(orig/'train_status.json',ref['train_status_sha256']);check(orig/'predict_status.json',ref['predict_status_sha256']);check(orig/'score/metrics_by_condition.csv',ref['metrics_sha256']);p=read(orig/'plan.json');t=read(orig/'train_status.json')
        req(p['seed']==seed and not p['fixture'] and not p['smoke'] and p['lock_sha256']==s['lock_sha256'] and Path(p['root']).resolve()==root and Path(p['lock_dir']).resolve()==lock,'Origin mismatch')
        req(t['status']=='COMPLETE' and t['checkpoint_hashes']==ref['checkpoint_hashes'] and t['shared_hash']==ref['shared_hash'],'Origin weights record mismatch')
        for f,h in ref['checkpoint_hashes'].items():check(orig/'models'/f,h)
        for f,h in p['code'].items():check(f,h)
        for f,h in l['code'].items():check(root/'CausFlow_BR6_Lock'/f,h)
        p.update(schema='br65_outer_v1',origin_dir=str(orig),origin_plan_sha256=ref['plan_sha256'],origin_train_status_sha256=ref['train_status_sha256'],selected_steps=s['selected_steps'],queries=qs.to_dict('records'),arrays=arrays,scope=s['scope'])
        p['code'].update({str(HERE/n):sha(HERE/n) for n in ('run_final.py','outer_worker.py','selection_lock.json')});p['code'].update({str(root/'CausFlow_BR6_Lock'/f):h for f,h in l['code'].items()});req(len(expected(p))==768,'Unexpected prediction budget');allplans[seed]=p
    out.mkdir();plans={}
    for seed,p in allplans.items():
        d=out/f'seed_{seed}';d.mkdir();dump(d/'plan.json',p);ph=sha(d/'plan.json');dump(d/'preflight_status.json',dict(status='COMPLETE',plan_sha256=ph,target_values_read=False));plans[str(seed)]=ph
    dump(out/'preflight_status.json',dict(status='COMPLETE',selection_sha256=SELECTION_SHA,lock_sha256=s['lock_sha256'],plans=plans,target_values_read=False));print('BR65 FINAL PREFLIGHT COMPLETE',flush=True)

def getplan(out,seed):
    d=out/f'seed_{seed}';p=read(d/'plan.json');st=read(out/'preflight_status.json');check(d/'plan.json',st['plans'][str(seed)]);check(d/'plan.json',read(d/'preflight_status.json')['plan_sha256']);req(p['seed']==seed and p['schema']=='br65_outer_v1','Wrong outer plan')
    for f,h in p['code'].items():check(f,h)
    check(Path(p['lock_dir'])/'lock.json',p['lock_sha256']);return d,p

def verify_predictions(d,p):
    g=read(d/'predict_status.json');req(g['status']=='COMPLETE' and g['plan_sha256']==sha(d/'plan.json') and g['train_status_sha256']==p['origin_train_status_sha256'] and g['target_expression_as_inference_input'] is False,'Bad prediction provenance')
    check(d/'timing.json',g['timing_sha256']);times=pd.DataFrame(read(d/'timing.json'));ex=expected(p);req(len(times)==len(ex) and set(times[['model','step','nfe','query_id']].itertuples(index=False,name=None))==ex and (times.nfe==times.measured_nfe_per_batch).all() and (times.measured_nfe_total==times.nfe*times.n_batches).all() and (times.seconds>0).all(),'NFE or timing mismatch')
    req(set(g['files'])=={f'{a}_{s}_{n}_{k}.npy' for a,s,n,k in ex},'Incomplete predictions')
    for n,h in g['files'].items():check(d/'predictions'/n,h)
    times['seed']=p['seed'];return times

def predict(out,device):
    selection();env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[k]='1'
    for seed in (0,1,2):
        d,p=getplan(out,seed)
        if not (d/'predict_status.json').exists():
            req(not (d/'predictions').exists(),'Incomplete predictions; preserve failed run and use a fresh output directory')
            subprocess.run([sys.executable,str(HERE/'outer_worker.py'),str(d/'plan.json'),device],env=env,check=True)
        verify_predictions(d,p);print('OUTER SEED COMPLETE',seed,flush=True)
    print('BR65 FINAL PREDICT COMPLETE',flush=True)

def stats(d,chosen,primary,metric_signs):
    # Average seeds and cell lines within drug before inference. No treating repeated seeds as independent drugs.
    sub=d[((d.step==d.model.map(chosen))&(d.nfe==d.model.map(primary)))|d.model.isin(['No-change','Same/nearest-dose'])]
    rows=[];drugrows=[];rng=np.random.default_rng(6502)
    for split in sorted(sub.split.unique()):
        z=sub[(sub.split==split)&(sub.output=='raw')]
        for other in ('NonFlow','DDIM','No-change','Same/nearest-dose'):
            j=z[z.model=='CFM'].merge(z[z.model==other],on=['query_id','seed','drug','cell_line'],suffixes=('_cfm','_other'),validate='one_to_one');req(len(j)==len(z[z.model=='CFM']),'Missing pairing')
            for m in ('W1_DEG50_exact','R2_var_all','response_pcc_deg50'):
                benefit=(j[m+'_cfm']-j[m+'_other'])*metric_signs[m];h=j.assign(benefit=benefit).groupby('drug').benefit.mean();x=h.to_numpy();n=len(x);req(np.isfinite(x).all() and n>=2,'Invalid drug effects');boot=x[rng.integers(0,n,size=(10000,n))].mean(1);sign=np.asarray(list(itertools.product((-1,1),repeat=n)));pv=float(np.mean(np.abs(sign@x/n)>=abs(x.mean())-1e-12));rows.append(dict(split=split,comparison='CFM-'+other,metric=m,n_drugs=n,benefit=float(x.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),p_signflip=pv,n_positive=int((x>0).sum())))
                for drug,val in h.items():drugrows.append(dict(split=split,comparison='CFM-'+other,metric=m,drug=drug,benefit=val))
    t=pd.DataFrame(rows);t['p_BH']=np.nan
    for _,idx in t.groupby('split').groups.items():
        ix=np.asarray(list(idx));pv=t.loc[ix,'p_signflip'].to_numpy();order=np.argsort(pv);adj=np.minimum.accumulate((pv[order]*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1];t.loc[ix[order],'p_BH']=np.minimum(adj,1.)
    return t,pd.DataFrame(drugrows)

def score(out):
    s=selection();plans=[getplan(out,i) for i in range(3)];times=pd.concat([verify_predictions(d,p) for d,p in plans],ignore_index=True);d0,p0=plans[0];lock=Path(p0['lock_dir']);l=read(lock/'lock.json');dest=out/'score';req(not dest.exists(),'Score exists; preserve')
    # Only here may target data be opened, after all predictions and their hashes are verified.
    import h5py
    sys.path.insert(0,str(Path(p0['root'])/'CausFlow_BR6_Lock'));import build_br6 as b;import locked_metrics as m
    for mod in (b,m):req(Path(mod.__file__).resolve().parent==Path(p0['root'])/'CausFlow_BR6_Lock','Wrong metric/reader source')
    for n in ('panel.csv','score_indices.npz','outer_queries.csv'):check(lock/n,l['files'][n])
    raw=Path(l['raw_h5ad']);check(raw,l['raw_sha256']);panel=pd.read_csv(lock/'panel.csv');qs=p0['queries'];rows=[]
    with h5py.File(raw,'r') as f,np.load(lock/'score_indices.npz',allow_pickle=False) as ix:
        reader=b.Raw(f,panel.raw_col_0based.to_numpy(),l['normalization_target'])
        for q in qs:
            key=q['query_id'];real=np.empty((int(q['n_real']),p0['n_genes']),np.float32)
            req(len(ix[key])==len(real),'Target row count mismatch')
            for inds,v,_ in reader.batches(ix[key]):real[inds]=np.log1p(v).astype(np.float32)
            real=real.astype(float);refs={}
            for k in ('sources','controls','nearest'):
                path=lock/'outer'/k/(key+'.npy');check(path,p0['arrays'][k][key]);refs[k]=np.load(path,allow_pickle=False).astype(float)
            ctl=refs['controls'];deg=m.rank_deg(real,ctl)
            for d,p in plans:
                def record(y,model,step,nfe,tag):
                    y=np.asarray(y,dtype=float);req(y.shape==refs['sources'].shape and np.isfinite(y).all() and (y>=0).all(),'Invalid generation');return {**q,'seed':p['seed'],'model':model,'step':step,'nfe':nfe,'output':tag,**m.metrics(real,ctl,y,deg),'mean_expression':float(y.mean()),'true_mean_expression':float(real.mean()),'zero_fraction':float((y==0).mean()),'true_zero_fraction':float((real==0).mean())}
                for name,k in [('No-change','sources'),('Same/nearest-dose','nearest')]:rows.append(record(refs[k],name,0,0,'raw'))
                for a,step,n,k in sorted(expected(p)):
                    if k!=key:continue
                    y=np.load(d/'predictions'/f'{a}_{step}_{n}_{key}.npy',allow_pickle=False);rows.append(record(y,a,step,n,'raw'));rows.append(record(np.where(y>=.01,y,0),a,step,n,'S4'))
            print('SCORED',key,flush=True)
    d=pd.DataFrame(rows);req(len(d)==4896 and not d.duplicated(['seed','query_id','model','step','nfe','output']).any(),'Incomplete scores')
    ms=list(m.METRICS)+['mean_expression','true_mean_expression','zero_fraction','true_zero_fraction'];req(np.isfinite(d[ms]).all().all(),'Nonfinite metrics; inspect before reporting');keys=['split','model','step','nfe','output'];summary=d.groupby(keys)[ms].mean().reset_index();chosen=s['selected_steps'];primary=s['primary_nfe'];selected=d[((d.step==d.model.map(chosen))&(d.nfe==d.model.map(primary)))|d.model.isin(['No-change','Same/nearest-dose'])];seedmeans=selected.groupby(['split','model','output','seed'])[ms].mean().reset_index();tests,bydrug=stats(d,chosen,primary,m.METRICS)
    dest.mkdir();d.to_csv(dest/'metrics_by_condition.csv',index=False);summary.to_csv(dest/'summary_all_budgets.csv',index=False);seedmeans.to_csv(dest/'selected_by_seed.csv',index=False);seedmeans.groupby(['split','model','output'])[ms].agg(['mean','std']).to_csv(dest/'selected_summary.csv');tests.to_csv(dest/'paired_stats.csv',index=False);bydrug.to_csv(dest/'paired_by_drug.csv',index=False);times.to_csv(dest/'timing_by_condition.csv',index=False);times.groupby(['split','seed','model','step','nfe'])[['seconds','cells_per_second','peak_allocated_bytes','incremental_peak_bytes']].mean().to_csv(dest/'timing_summary.csv');dump(dest/'score_status.json',dict(status='COMPLETE',selection_sha256=SELECTION_SHA,lock_sha256=s['lock_sha256'],score_count=len(d),n_predictions=len(times),scope=s['scope'],selected_steps=chosen,primary_nfe=primary,timing='provisional; independent repeat timing required',stats='drug cluster paired sign flip, 10000 cluster bootstrap, BH over 12 tests within each split; retrospective descriptive inference'));print('BR65 FINAL SCORE COMPLETE',flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--phase',required=True,choices=['preflight','predict','score']);ap.add_argument('--root',default='.');ap.add_argument('--lock',default='Output/BR6_StrictLocked_v2');ap.add_argument('--out',required=True);ap.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=ap.parse_args();root=Path(a.root).resolve();out=Path(a.out).resolve();lock=(root/a.lock).resolve()
    if a.phase=='preflight':preflight(root,lock,out)
    elif a.phase=='predict':predict(out,a.device)
    else:score(out)
if __name__=='__main__':main()
