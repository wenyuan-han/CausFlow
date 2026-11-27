"""BR6 professional input export and audited scoring. No target reads in export/train."""
import argparse, hashlib, json, shutil, sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
HERE=Path(__file__).resolve().parent
BR6='f27281346dc457f602acab6b78a5caaf61b60e038f46b229827b56d0d9d8ca4b'
ORIGINAL='d2dedfb1c5a7b3670bbb52ed3ea66bbda7ff3de838945f604a53a10a82c3189a'
REPAIR='c798533c2e9d2c94a5cd915676f98909044f96ba5c71991ef0993b34292fcabf'
MODELS=('CPA','CPA_mean','scGen_LowerDose')
POLICY={'CPA':'cpa-tools 0.8.8 Gaussian model; posterior predictive draws; decoded mean diagnostic','scGen_LowerDose':'scGen 2.1.1 official VAE+latent arithmetic; same-line same-dose ID or nearest lower-dose OOD, not native dose extrapolation','CPA_max_epochs':50,'scGen_max_epochs':100,'batch_size':128,'native_validation':'random validation sampled only within common permitted core pool; actual gradient-fit subset can differ from CausFlow','primary':'raw nonnegative clipping, no threshold; S4 sensitivity'}
def req(x,msg):
    if not x:raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def check(p,h):req(sha(p)==h,'Hash mismatch: '+str(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8')
def csv(p):return pd.read_csv(p,dtype=str,keep_default_na=False)
def files(d):return {p.relative_to(d).as_posix():sha(p) for p in sorted(d.rglob('*')) if p.is_file()}
def safe(d,k):
    p=(d/k).resolve();req(p.is_relative_to(d.resolve()),'Unsafe path');return p

def original_code():
    check(HERE/'reference/original_locked_source/run_professional.py',ORIGINAL);check(HERE/'reference/cpa_save_repair.py',REPAIR)
def lock_read(lock):
    check(lock/'lock.json',BR6);l=read(lock/'lock.json');req(l['schema']=='BR6_strict_panel_v2' and l['target_expression_exported'] is False and (l['n_fit'],l['n_genes'],l['n_inner'],l['n_outer'])==(220998,5000,24,48),'Wrong BR6 schema');return l

def make_obs(obs,genes,x,fit):
    req(len(obs)==len(fit)==len(x) and x.shape[1]==len(genes) and len(set(fit))==len(fit),'Fit shape/index mismatch')
    req(obs.source_cell_id.is_unique and obs.source_cell_id.ne('').all() and genes.gene.is_unique and genes.gene.str.startswith('ENSG').all(),'Invalid cell or gene IDs')
    o=obs[['source_cell_id','drug','cell_line']].copy();o['dose_numeric']=obs.dose.astype(float);req(np.isfinite(o.dose_numeric).all(),'Invalid dose');o.index=pd.Index(o.source_cell_id.to_numpy(),name=None);return o

def mat_csr(x):
    chunks=[]
    for i in range(0,len(x),512):
        v=np.asarray(x[i:i+512],np.float32);req(np.isfinite(v).all() and (v>=0).all(),'Bad fit values');chunks.append(sparse.csr_matrix(v))
    return sparse.vstack(chunks,format='csr')

def reference_rows(fit,refs,query,obs,x,source,ctrl):
    lookup=pd.Index(fit);ii=lookup.get_indexer(refs);req(len(ii)==256 and (ii>=0).all(),'Source not in fit')
    req((obs.iloc[ii].drug.str.casefold()==ctrl.casefold()).all() and (obs.iloc[ii].cell_line==query['cell_line']).all(),'Not matched controls')
    req(source.shape==(256,x.shape[1]) and np.array_equal(x[ii],source),'Source expression/ordering mismatch')
    subset=obs[(obs.drug==query['drug'])&(obs.cell_line==query['cell_line'])].dose.astype(float)
    dose=float(query['dose']);req(len(subset)>0,'Drug/line absent')
    if 'OOD' in query['split']:req((subset<dose).any() and not (subset>=dose).any(),'Dose OOD invalid')
    else:req((subset==dose).any(),'ID missing')
    return obs.iloc[ii].source_cell_id.tolist()

def export(root,out):
    original_code();lock=root/'Output/BR6_StrictLocked_v2';l=lock_read(lock);req(not out.exists() and out.parent==root/'Output','Fresh direct child of Output required')
    import anndata as ad
    for k in ['fit_x.npy','fit_obs.csv','fit_index.npz','genes.csv','reference_indices.npz','inner_queries.csv','outer_queries.csv']:
        check(lock/k,l['files'][k])
    x=np.load(lock/'fit_x.npy',mmap_mode='r',allow_pickle=False);obs=csv(lock/'fit_obs.csv');genes=csv(lock/'genes.csv');req(x.shape==(220998,5000),'Wrong fit dimensions')
    with np.load(lock/'fit_index.npz',allow_pickle=False) as z:fit=z['fit'];req(np.array_equal(obs[['drug_idx','dose_idx','cell_line_idx']].astype(int).to_numpy(),z['labels']),'Label mismatch')
    o=make_obs(obs,genes,x,fit);out.mkdir();d=out/'external_input';d.mkdir();qs=[];qorig=[]
    with np.load(lock/'reference_indices.npz',allow_pickle=False) as rr:
        for scope in ('outer','inner'):
            frame=csv(lock/f'{scope}_queries.csv');req(len(frame)==(48 if scope=='outer' else 24),'Bad queries')
            for q in frame.to_dict('records'):
                k=q['query_id'];req(k[0] in 'qv' and k[1:].isdigit(),'Invalid query ID');rel=f'{scope}/sources/{k}.npy';check(lock/rel,l['files'][rel]);source=np.load(lock/rel,allow_pickle=False);ids=reference_rows(fit,rr[k+'_source'],q,obs,x,source,l['control_drug']);shutil.copyfile(lock/rel,d/f'{k}_source.npy');pd.DataFrame({'source_cell_id':ids}).to_csv(d/f'{k}_source_ids.csv',index=False);qorig.append(q);qs.append({**q,'split': 'dose_OOD' if 'OOD' in q['split'] else 'ID'})
    req(len({q['query_id'] for q in qs})==72,'Duplicate queries');pd.DataFrame(qs).to_csv(d/'queries.csv',index=False);pd.DataFrame(qorig).to_csv(out/'original_queries.csv',index=False);shutil.copyfile(lock/'genes.csv',d/'genes.csv');print('Writing core-fit sparse H5AD',flush=True)
    a=ad.AnnData(mat_csr(x),obs=o,var=pd.DataFrame(index=pd.Index(genes.gene.tolist())));a.write_h5ad(d/'fit.h5ad');del a
    code={n:sha(HERE/n) for n in ('run_afinal.py','professional_worker.py','e1_outer.py','collect_audit.py')};new=dict(schema='afinal_br6_professional_v1',br6_lock_sha256=BR6,br6_dir=str(lock),root=str(root),source_hashes={'run_professional.py':ORIGINAL,'cpa_save_repair.py':REPAIR},code=code,professional_policy=POLICY,original_queries_sha256=sha(out/'original_queries.csv'),scope='retrospective, fit and control-source only');dump(out/'lock.json',new)
    dump(d/'manifest.json',dict(lock_sha256=sha(out/'lock.json'),br6_lock_sha256=BR6,files=files(d),n_fit=220998,n_pred=256,control_drug=l['control_drug'],seeds=[0,1,2],target_expression_included=False,professional_policy=POLICY))
    dump(out/'export_status.json',dict(status='COMPLETE',derived_lock_sha256=sha(out/'lock.json'),export_manifest_sha256=sha(d/'manifest.json'),br6_lock_sha256=BR6));print('A-FINAL EXPORT COMPLETE',flush=True)

def verify(out,full=False):
    original_code();l=read(out/'lock.json');s=read(out/'export_status.json');d=out/'external_input';m=read(d/'manifest.json');req(l['schema']=='afinal_br6_professional_v1' and l['br6_lock_sha256']==m['br6_lock_sha256']==BR6 and l['professional_policy']==m['professional_policy']==POLICY and m['target_expression_included'] is False and s['status']=='COMPLETE','Export identity mismatch');check(out/'lock.json',s['derived_lock_sha256']);req(m['lock_sha256']==s['derived_lock_sha256'],'Derived lock mismatch');check(d/'manifest.json',s['export_manifest_sha256']);check(out/'original_queries.csv',l['original_queries_sha256']);lock=Path(l['br6_dir']);bl=lock_read(lock)
    for k,h in l['code'].items():check(HERE/k,h)
    for k,h in m['files'].items():
        if full or k!='fit.h5ad':check(safe(d,k),h)
    return l,m,lock,bl

def pred_receipt(out,model,seed,queries):
    l,m,_,_=verify(out);dest=out/'professional_models'/('CPA' if model=='CPA_mean' else model)/f'seed_{seed}';p=out/'predictions'/model/f'seed_{seed}';r=read(p/'manifest.json');cfg=read(dest/'train_config.json');req(r['model']==model and r['seed']==seed and r['lock_sha256']==m['lock_sha256'] and r['target_expression_as_inference_input'] is False and cfg['smoke'] is False and cfg['lock_sha256']==m['lock_sha256'],'Model provenance mismatch');check(dest/'train_config.json',r['train_config_sha256']);check(out/'external_input/manifest.json',r['export_manifest_sha256']);req(set(r['files'])==set(queries),'Incomplete predictions');req(bool(r['model_artifacts']),'Missing model artifacts')
    for k,h in r['model_artifacts'].items():check(safe(dest,k),h)
    for k,h in r['files'].items():check(p/f'{k}.npy',h)
    return p

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=['export','verify','score']);p.add_argument('--root',default='.');p.add_argument('--out',default='Output/BR6_AFinal_v1');a=p.parse_args();root=Path(a.root).resolve();out=(root/a.out).resolve()
    if a.phase=='export':export(root,out)
    elif a.phase=='verify':verify(out,True);print('EXPORT VERIFIED')
    else:score(out)

def paired(df,metrics):
    results=[];drugs=[];rng=np.random.default_rng(20261124);primary=df[df.output=='raw'];direction={k:(-1 if 'W1' in k or 'rmse' in k.lower() else 1) for k in metrics}
    for split,z in primary.groupby('split'):
        a=z[z.model=='E1'].set_index(['seed','query_id']);
        for model in ['CPA','scGen_LowerDose','No-change','Same/nearest-dose']:
            b=z[z.model==model].set_index(['seed','query_id']);req(a.index.is_unique and b.index.is_unique and set(a.index)==set(b.index),'Unpaired primary rows');b=b.reindex(a.index)
            for k in ('W1_DEG50_exact','response_pcc_deg50','R2_var_all'):
                v=(a[k]-b[k])*direction[k];h=pd.DataFrame({'drug':a.drug,'benefit':v}).groupby('drug').benefit.mean();x=h.to_numpy();req(len(x)==8 and np.isfinite(x).all(),'Invalid eight-drug pairing');s=np.asarray(list(itertools.product([-1,1],repeat=8)));p=float(np.mean(np.abs(s@x)>=abs(x.sum())-1e-12));boot=x[rng.integers(0,8,(10000,8))].mean(1);results.append(dict(split=split,comparison='E1-'+model,metric=k,n_drugs=8,benefit=float(x.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),p_signflip=p,n_positive=int((x>0).sum())));drugs.extend(dict(split=split,comparison='E1-'+model,metric=k,drug=d,benefit=float(v)) for d,v in h.items())
    t=pd.DataFrame(results);t['p_BH']=np.nan
    for _,idx in t.groupby('split').groups.items():
        v=t.loc[idx,'p_signflip'].to_numpy();o=np.argsort(v);adj=np.minimum.accumulate((v[o]*len(v)/np.arange(1,len(v)+1))[::-1])[::-1];t.loc[np.asarray(idx)[o],'p_BH']=np.minimum(1,adj)
    return t,pd.DataFrame(drugs)

def score(out):
    l,m,lock,bl=verify(out,True);root=Path(l['root']);from e1_outer import verify_e1,getsel
    s=getsel(out);qs=csv(out/'original_queries.csv');outer=qs[qs.query_id.str.startswith('q')];req(len(outer)==48 and len(qs)==72,'Query count mismatch');dest=out/'score';req(not dest.exists(),'Score exists; preserve')
    pred={};ids=qs.query_id.tolist();outerids=outer.query_id.tolist()
    # All generation manifests and prediction files checked before opening raw target data.
    for seed in (0,1,2):
        for name in MODELS:pred[name,seed]=pred_receipt(out,name,seed,ids)
        pred['E1',seed]=verify_e1(out,seed,outerids)
    sys.path.insert(0,str(root/'CausFlow_BR6_Lock'));import build_br6 as b;import locked_metrics as met;import panel_utils as u;import h5py
    for mod in (b,met,u):check(Path(mod.__file__),bl['code'][Path(mod.__file__).name])
    for k in ('panel.csv','score_indices.npz','outer_queries.csv'):check(lock/k,bl['files'][k])
    raw=Path(bl['raw_h5ad']);check(raw,bl['raw_sha256']);panel=pd.read_csv(lock/'panel.csv');rows=[]
    with h5py.File(raw,'r') as f,np.load(lock/'score_indices.npz',allow_pickle=False) as ix:
        reader=b.Raw(f,panel.raw_col_0based.to_numpy(),bl['normalization_target'])
        for q in outer.to_dict('records'):
            k=q['query_id'];real=np.empty((int(q['n_real']),5000),np.float32);req(len(ix[k])==len(real),'Target count mismatch')
            for inds,v,_ in reader.batches(ix[k]):real[inds]=np.log1p(v).astype(np.float32)
            real=real.astype(float);refs={}
            for kind in ('sources','controls','nearest'):
                rel=f'outer/{kind}/{k}.npy';check(lock/rel,bl['files'][rel]);refs[kind]=np.load(lock/rel,allow_pickle=False).astype(float)
            ctl=refs['controls'];deg=met.rank_deg(real,ctl)
            def record(a,name,seed,tag,neg):
                req(a.shape==(256,5000) and np.isfinite(a).all() and (a>=0).all(),'Bad scoring array');rows.append({**q,'model':name,'seed':seed,'output':tag,**met.metrics(real,ctl,a,deg),'native_negative_fraction':neg,'zero_fraction':float((a==0).mean()),'true_zero_fraction':float((real==0).mean()),'mean_expression':float(a.mean()),'true_mean_expression':float(real.mean())})
            for seed in (0,1,2):
                for name,kind in [('No-change','sources'),('Same/nearest-dose','nearest')]:record(refs[kind],name,seed,'raw',0.)
                for name in ('E1',*MODELS):
                    y=np.load(pred[name,seed]/f'{k}.npy',allow_pickle=False).astype(float);req(np.isfinite(y).all(),'Bad native outputs');neg=float((y<0).mean());y=np.maximum(y,0);record(y,name,seed,'raw',neg);record(np.where(y>=.01,y,0),name,seed,'S4',neg)
            print('A-FINAL SCORED',k,flush=True)
    df=pd.DataFrame(rows);metrics=list(met.METRICS);req(len(df)==1440 and not df.duplicated(['model','seed','query_id','output']).any() and np.isfinite(df[metrics]).all().all(),'Incomplete score');t,g=paired(df,metrics);dest.mkdir();df.to_csv(dest/'metrics_by_condition.csv',index=False);seedmeans=df.groupby(['split','model','output','seed'])[metrics].mean().reset_index();seedmeans.to_csv(dest/'summary_by_seed.csv',index=False);seedmeans.groupby(['split','model','output'])[metrics].agg(['mean','std']).to_csv(dest/'summary.csv');t.to_csv(dest/'paired_stats.csv',index=False);g.to_csv(dest/'paired_by_drug.csv',index=False);dump(dest/'score_status.json',dict(status='COMPLETE',br6_lock_sha256=BR6,derived_lock_sha256=sha(out/'lock.json'),e1_selection_sha256=sha(out/'e1_selection.json'),e1_step=s['selected_step'],rows=len(df),scope='retrospective outer only',primary='raw nonnegative, no threshold',stats='8 drug-level paired sign flips and 10000 bootstrap, BH 12 per split, CPA_mean diagnostic',files=files(dest)));print('A-FINAL SCORE COMPLETE',flush=True)

if __name__=='__main__':main()
