"""BR6.5: matched frozen generators on reviewed V2, inner only."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
MODELS=['CFM','NonFlow','DDIM']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def req(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8')
def check(p,h):req(sha(p)==h,'Hash mismatch: '+str(p))
def query_frame(p):return pd.read_csv(p,dtype={'dose':str,'drug':str,'cell_line':str},keep_default_na=False)
def preflight(root,lock,out,smoke,seed,*,fixture=False,fixture_ddim=False):
    req(not out.exists() and out.parent==root/'Output' and out!=lock,'Fresh direct child of Output required')
    reviewed=read(HERE/'reviewed_lock.json') if not fixture else {'lock_sha256':sha(lock/'lock.json'),'score_status_sha256':None}
    check(lock/'lock.json',reviewed['lock_sha256']);l=read(lock/'lock.json')
    req(l['schema']=='BR6_strict_panel_v2' and l['target_expression_exported'] is False,'Wrong data schema')
    if not fixture:
        req((l['n_fit'],l['n_genes'],l['n_inner'],l['n_outer'])==(220998,5000,24,48),'Wrong V2 data dimensions')
        check(lock/'lock.json',read(lock/'build_status.json')['lock_sha256'])
    # Only fit and reference files; no read of target expression, score indices or outer files.
    refs=['fit_x.npy','fit_index.npz','fit_scale.json','vocab.json','inner_queries.csv','genes.csv']
    for n in refs:check(lock/n,l['files'][n])
    qs=query_frame(lock/'inner_queries.csv');req(len(qs)==l['n_inner'] and qs.query_id.is_unique and set(qs.split)=={'val_ID','val_OOD'},'Inner queries invalid')
    req(qs.query_id.str.fullmatch(r'v\d+').all(),'Unexpected query id')
    if not fixture:req(qs.groupby('split').size().eq(12).all(),'Inner group sizes differ')
    if smoke:qs=qs.groupby('split',sort=False).head(1)
    code={str(root/k):v for k,v in read(HERE/'source_lock.json').items()} if not fixture else {str(f):sha(f) for folder in ['CausFlow','CausFlow_TrainingSufficiency'] for f in (root/folder).glob('*.py')}
    code.update({str(HERE/n):sha(HERE/n) for n in ['run_br65.py','worker.py','pilot_model.py','source_lock.json','reviewed_lock.json','check_diffusers.py']});code[str(root/'CausFlow_BR6_Lock/locked_metrics.py')]=l['code']['locked_metrics.py']
    for f,h in code.items():check(f,h)
    scale=read(lock/'fit_scale.json');req(scale['n_cells']==l['n_fit'] and scale['n_genes']==l['n_genes'] and scale['sd']>1e-6 and scale['never_detected']==0,'Wrong fit scale')
    if not fixture:req(scale['min_detected']>=100 and scale['median_library']==1555,'Wrong V2 feature support')
    arrays={k:{} for k in ('sources','controls','nearest')}
    for k in arrays:
        for q in qs.to_dict('records'):
            key=q['query_id'];rel=f'inner/{k}/{key}.npy';h=l['files'][rel];check(lock/rel,h);arrays[k][key]=h
    n_pred=3 if fixture else 256
    for q in qs.to_dict('records'):
        k=q['query_id']
        for kind in arrays:
            x=np.load(lock/'inner'/kind/(k+'.npy'),mmap_mode='r',allow_pickle=False);req(x.ndim==2 and x.shape[1]==l['n_genes'] and (kind=='controls' or len(x)==n_pred),'Reference shape mismatch');req(np.isfinite(x).all() and (x>=0).all(),'Invalid reference');del x
    out.mkdir();p=dict(schema='br65_v2',root=str(root),source=str(root/'CausFlow_TrainingSufficiency'),lock_dir=str(lock),lock_sha256=reviewed['lock_sha256'],expected_score_status_sha256=reviewed['score_status_sha256'],code=code,train_sha256=l['files']['fit_x.npy'],fit_sha256=l['files']['fit_index.npz'],scale_sha256=l['files']['fit_scale.json'],n_fit=l['n_fit'],n_genes=l['n_genes'],n_pred=n_pred,vocab=read(lock/'vocab.json'),width=16 if fixture else 192,batch=8 if fixture else 32,lr=.0001,seed=seed,shared_steps=2 if smoke else 2000,stages=[2,4] if smoke else [2000,10000],budgets=[2,5] if smoke else [5,10,20,40,50,100],selection_nfe=2 if smoke else 40,queries=qs.to_dict('records'),arrays=arrays,smoke=smoke,fixture=fixture,mean=scale['mean'],sd=scale['sd'],models=MODELS if not fixture or fixture_ddim else MODELS[:2],scope='V2 inner only, retrospective; standardized source-anchored CFM versus Gaussian DDIM versus one-shot residual; not pure objective attribution; Raw primary, S4 sensitivity')
    dump(out/'plan.json',p);dump(out/'preflight_status.json',dict(status='COMPLETE',plan_sha256=sha(out/'plan.json'),lock_sha256=p['lock_sha256'],target_values_read=False));print('BR65 PREFLIGHT COMPLETE',len(qs),'inner queries',flush=True)
def plan(out):
    check(out/'plan.json',read(out/'preflight_status.json')['plan_sha256']);p=read(out/'plan.json');req(p['schema']=='br65_v2','Wrong schema')
    for f,h in p['code'].items():check(f,h)
    check(Path(p['lock_dir'])/'lock.json',p['lock_sha256']);return p

def worker(out,phase,device):
    plan(out);env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[k]='1'
    subprocess.run([sys.executable,str(HERE/'worker.py'),str(out/'plan.json'),phase,device],env=env,check=True)

def score(out):
    p=plan(out);lock=Path(p['lock_dir']);ph=sha(out/'plan.json');t=read(out/'train_status.json');g=read(out/'predict_status.json');req(all(s['status']=='COMPLETE' and s['plan_sha256']==ph for s in (t,g)),'Incomplete phases');req(g['train_status_sha256']==sha(out/'train_status.json') and g['target_expression_as_inference_input'] is False,'Invalid provenance')
    for key in ('condition_hashes','draw_hashes','initial_hashes','parameter_counts'):
        req(set(t[key])==set(p['models']) and len(set(t[key].values()))==1,'Unmatched arms: '+key)
    req(all(h==t['shared_hash'] for h in t['condition_hashes'].values()),'Frozen weights changed')
    sys.path.insert(0,str(Path(p['root'])/'CausFlow_BR6_Lock'));import locked_metrics as m
    req(Path(m.__file__).resolve()==Path(p['root'])/'CausFlow_BR6_Lock/locked_metrics.py','Wrong metric module')
    base=lock/'inner_baseline_score';sp=base/'score_status.json'
    if not p['fixture']:check(sp,p['expected_score_status_sha256'])
    ss=read(sp);req(ss['lock_sha256']==p['lock_sha256'] and ss['status']=='COMPLETE' and ss['n_queries']==(2 if p['fixture'] else 24),'Wrong target provenance')
    check(out/'timing.json',g['timing_sha256']);times=pd.DataFrame(read(out/'timing.json'));times[['peak_allocated_bytes','incremental_peak_bytes']]=times[['peak_allocated_bytes','incremental_peak_bytes']].apply(pd.to_numeric)
    expected={(a,s,n,q['query_id']) for a in p['models'] for s in p['stages'] for n in ([1] if a=='NonFlow' else p['budgets']) for q in p['queries']}
    req(set(g['files'])=={f'{a}_{s}_{n}_{k}.npy' for a,s,n,k in expected},'Prediction set mismatch')
    req(len(times)==len(expected) and set(times[['model','step','nfe','query_id']].itertuples(index=False,name=None))==expected and (times.nfe==times.measured_nfe_per_batch).all() and (times.measured_nfe_total==times.nfe*times.n_batches).all() and (times.seconds>0).all(),'Timing or NFE invalid')
    dest=out/'score';req(not dest.exists(),'Score exists; preserve');rows=[]
    def record(q,y,real,ctl,deg,model,step,budget,output):
        return {**q,'seed':p['seed'],'model':model,'step':step,'nfe':budget,'output':output,**m.metrics(real,ctl,y,deg),'mean_expression':float(y.mean()),'true_mean_expression':float(real.mean()),'zero_fraction':float((y==0).mean()),'true_zero_fraction':float((real==0).mean())}
    for q in p['queries']:
        k=q['query_id'];z={}
        for kind,hs in p['arrays'].items():
            f=lock/'inner'/kind/(k+'.npy');check(f,hs[k]);z[kind]=np.load(f,allow_pickle=False).astype(float)
        f=base/'targets'/(k+'.npy');check(f,ss['target_hashes'][f.name]);real=np.load(f,allow_pickle=False).astype(float);ctl=z['controls'];src=z['sources'];req(real.shape==(int(q['n_real']),p['n_genes']) and np.isfinite(real).all() and (real>=0).all(),'Invalid target');deg=m.rank_deg(real,ctl)
        for name,kind in [('No-change','sources'),('Same/nearest-dose','nearest')]:rows.append(record(q,z[kind],real,ctl,deg,name,0,0,'raw'))
        for a,s,n,key in sorted(expected):
            if key!=k:continue
            name=f'{a}_{s}_{n}_{k}.npy';f=out/'predictions'/name;check(f,g['files'][name]);y=np.load(f,allow_pickle=False).astype(float);req(y.shape==src.shape and np.isfinite(y).all() and (y>=0).all(),'Invalid generation');rows.extend([record(q,y,real,ctl,deg,a,s,n,'raw'),record(q,np.where(y>=.01,y,0),real,ctl,deg,a,s,n,'S4')])
    d=pd.DataFrame(rows);req(len(d)==2*len(expected)+2*len(p['queries']) and not d.duplicated(['query_id','model','step','nfe','output']).any(),'Score count mismatch');req(np.isfinite(d.W1_DEG50_exact).all(),'Nonfinite W1');ms=list(m.METRICS)+['mean_expression','true_mean_expression','zero_fraction','true_zero_fraction'];keys=['split','model','step','nfe','output'];summary=d.groupby(keys)[ms].mean().reset_index();own={};gates={}
    for model in p['models']:
        z=summary[(summary.model==model)&(summary.split=='val_OOD')&(summary.output=='raw')&(summary.nfe==(1 if model=='NonFlow' else p['selection_nfe']))];req(len(z)==len(p['stages']) and np.isfinite(z.W1_DEG50_exact).all(),'Invalid validation curve');s=int(z.loc[z.W1_DEG50_exact<=1.01*z.W1_DEG50_exact.min(),'step'].min());own[model]=s;gates[model]={'boundary_minimum':bool(int(z.loc[z.W1_DEG50_exact.idxmin(),'step'])==max(p['stages']))}
        for tag in ('raw','S4'):
            r=summary[(summary.model==model)&(summary.split=='val_OOD')&(summary.output==tag)&(summary.step==s)&(summary.nfe==(1 if model=='NonFlow' else p['selection_nfe']))].iloc[0];ratio=float(r.mean_expression/max(r.true_mean_expression,1e-12));gap=float(abs(r.zero_fraction-r.true_zero_fraction));gates[model][tag]={'mean_ratio':ratio,'zero_gap':gap,'distribution_warning':bool(not .5<=ratio<=2 or gap>.10 or r.R2_mean_all<0 or r.R2_var_all<0)}
    dest.mkdir();d.to_csv(dest/'metrics_by_condition.csv',index=False);summary.to_csv(dest/'summary.csv',index=False);d.groupby(keys)[ms].count().to_csv(dest/'finite_counts.csv');times.to_csv(dest/'timing_by_condition.csv',index=False);tim=times.groupby(['split','model','step','nfe'])[['seconds','cells_per_second','peak_allocated_bytes','incremental_peak_bytes']].mean().reset_index();tim.to_csv(dest/'timing_summary.csv',index=False)
    curves=pd.concat([summary[(summary.model==a)&(summary.step==s)] for a,s in own.items()]);curves.merge(tim,on=['split','model','step','nfe'],validate='many_to_one').to_csv(dest/'quality_efficiency_selected.csv',index=False)
    pairs=[]
    for policy in ['own']+[str(s) for s in p['stages']]:
        for tag in ('raw','S4'):
            sel=pd.concat([d[(d.model==a)&(d.step==(s if policy=='own' else int(policy)))&(d.output==tag)&(d.nfe==(1 if a=='NonFlow' else p['selection_nfe']))] for a,s in own.items()]);sel=pd.concat([sel,d[d.model.isin(['No-change','Same/nearest-dose'])]])
            for other in [a for a in p['models'] if a!='CFM']+['No-change','Same/nearest-dose']:
                j=sel[sel.model=='CFM'].merge(sel[sel.model==other],on=['query_id','drug','split','cell_line'],suffixes=('_cfm','_other'),validate='one_to_one');req(len(j)==len(p['queries']),'Missing pair')
                for metric,sign in m.METRICS.items():
                    h=j.assign(benefit=(j[metric+'_cfm']-j[metric+'_other'])*sign).groupby(['split','drug']).benefit.agg(['mean','count']).reset_index().rename(columns={'mean':'benefit','count':'n_finite'});h['metric']=metric;h['comparison']='CFM-'+other;h['policy']=policy;h['output']=tag;pairs.append(h)
    pd.concat(pairs).to_csv(dest/'paired_by_drug.csv',index=False)
    dump(dest/'decision.json',dict(status='COMPLETE',verdict='SMOKE_ONLY' if p['smoke'] else 'REVIEW_PILOT_BEFORE_EXPANSION',own_steps=own,gates=gates,selection='earliest checkpoint within 1% of min inner Dose-OOD Raw DEG50 W1 at 40 NFE (NonFlow 1; smoke 2)',scope=p['scope'],lock_sha256=p['lock_sha256'],plan_sha256=ph,score_count=len(d),prediction_count=len(expected),outer_evaluated=False,timing_status='provisional; require unloaded GPU repeated timing for claims',notes=['No significance claim from single seed. Compare raw as primary; S4 uses same raw-selected checkpoint.','NonFlow variance collapse is not a reason to exclude the arm. DDIM failure limits claims.','No target expression used for training or generation according to isolated worker.']));print('BR65 SCORE COMPLETE',flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--phase',required=True,choices=['preflight','train','predict','score']);ap.add_argument('--root',default='.');ap.add_argument('--lock',default='Output/BR6_StrictLocked_v2');ap.add_argument('--out',required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--smoke',action='store_true');ap.add_argument('--seed',type=int,choices=[0,1,2],default=0);a=ap.parse_args();root=Path(a.root).resolve();out=Path(a.out).resolve();lock=(root/a.lock).resolve()
    if a.phase=='preflight':preflight(root,lock,out,a.smoke,a.seed)
    elif a.phase=='score':score(out)
    else:worker(out,a.phase,a.device)
    dump(out/'run_status.json',dict(status='COMPLETE',phase=a.phase))
if __name__=='__main__':main()
