#!/usr/bin/env python3
"""Locked, retrospective re-split of existing CPA-like train cache. No official CPA claim."""
import argparse, hashlib, json, sys, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from scipy import sparse

HERE=Path(__file__).resolve().parent
MODELS=('E1','E4','NonFlow')
KEYS=['drug','dose','cell_line']
COLS=['source_cell_id',*KEYS,'drug_idx','dose_idx','cell_line_idx']

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def seedof(*s):return int(hashlib.sha256('|'.join(map(str,s)).encode()).hexdigest()[:8],16)
def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['prepare','audit','smoke','train','predict','export','score','summarize'])
    p.add_argument('--repo',default='.')
    p.add_argument('--cache-dir',required=True);p.add_argument('--out',required=True)
    p.add_argument('--h5ad',default=None,help='Optional processed H5AD; audit reads gene metadata and provenance only')
    p.add_argument('--prior-pseudo',default='Output/ResponseAware_2000/pseudo_manifest.json')
    p.add_argument('--n-drugs',type=int,default=8);p.add_argument('--split-seed',type=int,default=60429)
    p.add_argument('--min-ood',type=int,default=25);p.add_argument('--min-id',type=int,default=20)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--width',type=int,default=192);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--n-pred',type=int,default=256)
    p.add_argument('--ode-steps',type=int,default=40);p.add_argument('--device',default='auto')
    p.add_argument('--eval-tag',default='internal',help='Separate result directory; do not choose model after scoring')
    p.add_argument('--models',nargs='+',default=list(MODELS),help='Train/predict internal models; score registered external names too')
    a=p.parse_args();a.cache_dir=Path(a.cache_dir).resolve();a.out=Path(a.out).resolve();a.repo=Path(a.repo).resolve()
    if min(a.n_drugs,a.steps,a.batch_size,a.n_pred,a.ode_steps,a.min_ood,a.min_id)<2 or a.width<8 or a.width%4: p.error('Invalid numeric settings')
    if len(set(a.seeds))!=len(a.seeds) or len(set(a.models))!=len(a.models):p.error('Duplicate seed/model')
    if any(not s.replace('_','').replace('-','').isalnum() for s in [*a.models,a.eval_tag]):p.error('Invalid model name')
    return a

def readmeta(a):
    c=a.cache_dir; report=read(c.parent/'audit.json'); vocab=read(c/'vocab.json')
    frames={s:pd.read_csv(c/f'{s}_obs.csv',dtype={'dose':str},keep_default_na=False) for s in ('train','val','test','ood')}
    for s,d in frames.items():
        if not set(COLS)<=set(d):raise ValueError(f'{s}: missing columns')
        if not np.isfinite(d.dose.astype(float)).all():raise ValueError('Invalid dose')
        for k in KEYS:
            m={str(v):i for i,v in enumerate(vocab[k])}
            if not np.array_equal(d[k].astype(str).map(m).to_numpy(),d[k+'_idx'].to_numpy()):raise ValueError(f'{s} {k} vocab mismatch')
    allobs=pd.concat(frames.values(),ignore_index=True)
    if allobs.source_cell_id.duplicated().any():raise ValueError('Duplicate cell IDs across partitions')
    genes=pd.read_csv(c/'genes.csv',dtype=str,keep_default_na=False).gene.tolist()
    if len(set(genes))!=len(genes):raise ValueError('Duplicate genes')
    if hashlib.sha256('\n'.join(genes).encode()).hexdigest()!=report['gene_sha256']:raise ValueError('Gene fingerprint mismatch')
    x=np.load(c/'train_x.npy',mmap_mode='r')
    if x.shape!=(len(frames['train']),len(genes)):raise ValueError('Shape mismatch')
    hashes={f:digest(c/f) for f in ['genes.csv','vocab.json',*[f'{s}_obs.csv' for s in frames]]}
    hashes['audit.json']=digest(c.parent/'audit.json')
    hashes['train_x.npy_size']= (c/'train_x.npy').stat().st_size
    return report,vocab,frames,allobs,genes,x,hashes

def code_hashes(a):
    fs=[HERE/'run_locked_benchmark.py',HERE/'response_model.py',HERE/'factor_ablation_model.py',HERE/'factor_base_pilot.py']
    fs += [a.repo/'CausFlow'/n for n in ('Modules.py','utils.py','Dataset.py','cpa_pipeline.py','cpa_adapter.py','__init__.py')]
    return {str(p.name):digest(p) for p in fs}

def prepare(a,data):
    report,vocab,frames,allobs,genes,x,hashes=data;obs=frames['train']
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'lock.json').exists():raise FileExistsError('Already locked; do not redraw a split after inspection')
    old=read(a.prior_pseudo);excluded=set(frames['ood'].drug)|{c['drug'] for c in old['pseudo_conditions']}
    control=str(report['control_drug']);lines=sorted(obs.cell_line.unique());rng=np.random.default_rng(a.split_seed)
    g={tuple(k):np.asarray(v,dtype=np.int64) for k,v in obs.groupby(KEYS,sort=True).indices.items()}
    # Whole drug-dose across all lines. Maximum dose checked against ALL original metadata.
    candidates=[]
    for drug,dd in allobs.groupby('drug',sort=True):
        if drug in excluded or str(drug).casefold()==control.casefold():continue
        top=float(dd.dose.astype(float).max())
        if top<=0:continue
        entries=[]
        for line in lines:
            hi=[(k,v) for k,v in g.items() if k[0]==drug and k[2]==line and float(k[1])==top]
            lo=[(k,v) for k,v in g.items() if k[0]==drug and k[2]==line and float(k[1])<top and len(v)>=2*a.min_id]
            if len(hi)!=1 or len(hi[0][1])<a.min_ood or not lo:break
            lkey,lidx=max(lo,key=lambda kv:float(kv[0][1]));entries.append((hi[0][0],hi[0][1],lkey,lidx))
        if len(entries)==len(lines):candidates.append((str(drug),entries))
    if len(candidates)<a.n_drugs:raise ValueError(f'Only {len(candidates)} eligible drugs; need {a.n_drugs}. Report failure before making a new lock.')
    chosen=[candidates[i] for i in sorted(rng.choice(len(candidates),a.n_drugs,replace=False))]
    queries=[];arrays={};held=[]
    for drug,entries in chosen:
        for hk,hidx,lk,lidx in entries:
            # ID lower condition: fixed held-out cell sample, at least min-id fit retained.
            n=min(max(a.min_id,int(.2*len(lidx))),len(lidx)-a.min_id)
            itest=np.sort(rng.choice(lidx,n,replace=False))
            for split,key,idx in [('dose_OOD',hk,np.sort(hidx)),('ID',lk,itest)]:
                qid=f'q{len(queries):03d}';arrays[qid]=idx;held.append(idx)
                queries.append(dict(query_id=qid,split=split,drug=key[0],dose=key[1],cell_line=key[2],n_real=len(idx)))
    hold=np.concatenate(held);fit=np.setdiff1d(np.arange(len(obs)),hold)
    if len(np.unique(hold))!=len(hold):raise ValueError('Held-out overlap')
    for q in queries:
        mask=(obs.drug==q['drug'])&(obs.cell_line==q['cell_line'])&(obs.dose.astype(float)==float(q['dose']))
        n=int(mask.iloc[fit].sum())
        if q['split']=='dose_OOD' and n:raise ValueError('Dose condition leakage')
        if q['split']=='ID' and n<a.min_id:raise ValueError('ID train condition insufficient')
    arrays['fit']=fit
    # Fixed draws, independent of training seed. Same query/source/control for all models.
    for q in queries:
        qid=q['query_id'];pool=fit[(obs.iloc[fit].drug.str.casefold()==control.casefold()).to_numpy()&(obs.iloc[fit].cell_line==q['cell_line']).to_numpy()]
        if len(pool)<2:raise ValueError('Missing fit controls')
        r=np.random.default_rng(seedof('draw',a.split_seed,qid))
        arrays[qid+'_source']=r.choice(pool,a.n_pred,replace=True)
        arrays[qid+'_control']=r.choice(pool,min(512,len(pool)),replace=False)
        lower=fit[(obs.iloc[fit].drug==q['drug']).to_numpy()&(obs.iloc[fit].cell_line==q['cell_line']).to_numpy()&(obs.iloc[fit].dose.astype(float)<float(q['dose'])).to_numpy()]
        if len(lower):
            ld=obs.iloc[lower].dose.astype(float).to_numpy();lower=lower[ld==ld.max()];arrays[qid+'_lower']=r.choice(lower,a.n_pred,replace=True)
        same=fit[(obs.iloc[fit].drug==q['drug']).to_numpy()&(obs.iloc[fit].cell_line==q['cell_line']).to_numpy()&(obs.iloc[fit].dose.astype(float)==float(q['dose'])).to_numpy()]
        if len(same):arrays[qid+'_same']=r.choice(same,a.n_pred,replace=True)
    np.savez_compressed(a.out/'indices.npz',**arrays)
    pd.DataFrame(queries).to_csv(a.out/'queries.csv',index=False)
    lock=dict(schema='locked_retrospective_v1',scope='new retrospective split of previously used original train; NOT independent external test; NOT certified CPA official split/HVG',source='original train cache only; other split metadata only',cache_hashes=hashes,code_hashes=code_hashes(a),indices_sha256=digest(a.out/'indices.npz'),queries_sha256=digest(a.out/'queries.csv'),prior_pseudo_sha256=digest(a.prior_pseudo),excluded_drugs=sorted(excluded),eligible_drugs=len(candidates),selected_drugs=[d for d,_ in chosen],n_fit=len(fit),n_id=sum(q['n_real'] for q in queries if q['split']=='ID'),n_ood=sum(q['n_real'] for q in queries if q['split']=='dose_OOD'),split_seed=a.split_seed,steps=a.steps,batch_size=a.batch_size,width=a.width,lr=a.lr,seeds=a.seeds,n_pred=a.n_pred,ode_steps=a.ode_steps,primary_output='S4: max(0,x), then values <0.01 set to zero; raw secondary; same transform for learning models',metrics='legacy Welch DEG50 R2/W1; response PCC; absolute-change top50; diagnostic MWU BH q<=0.05 and |delta|>=0.01; fixed final checkpoint',feature_policy='fixed inherited 5000 (or actual cache count), no new HVG selection; upstream selection not certified train-only',models=list(MODELS))
    dump(a.out/'lock.json',lock);print('LOCKED',len(fit),'fit rows;',len(queries),'queries;',lock['selected_drugs'],flush=True)

def load(a,data):
    lock=read(a.out/'lock.json');assert lock['schema']=='locked_retrospective_v1'
    if data[-1]!=lock['cache_hashes'] or code_hashes(a)!=lock['code_hashes']:raise ValueError('Cache metadata/code changed after lock')
    if digest(a.out/'indices.npz')!=lock['indices_sha256'] or digest(a.out/'queries.csv')!=lock['queries_sha256']:raise ValueError('Split altered')
    with np.load(a.out/'indices.npz') as z:idx={k:z[k] for k in z.files}
    qs=pd.read_csv(a.out/'queries.csv',dtype={'dose':str},keep_default_na=False).to_dict('records')
    for k in ('steps','width','batch_size','lr','n_pred','ode_steps','seeds'):setattr(a,k,lock[k])
    return lock,idx,qs

def audit(a,data,lock,idx,qs):
    report,vocab,frames,allobs,genes,x,_=data;fit=idx['fit'];n=np.zeros(len(genes),np.int64);mi=float('inf');ma=-float('inf')
    for i in range(0,len(fit),512):
        b=np.asarray(x[fit[i:i+512]],dtype=np.float32)
        if not np.isfinite(b).all() or np.any(b<0):raise ValueError('Nonfinite or negative fit expression')
        n+=(b!=0).sum(0);mi=min(mi,float(b.min()));ma=max(ma,float(b.max()))
    pd.DataFrame({'gene':genes,'fit_detected_cells':n}).to_csv(a.out/'fit_gene_support.csv',index=False)
    dump(a.out/'feature_audit.json',dict(n_fit=len(fit),genes=len(genes),never_detected=int((n==0).sum()),detected_le5=int((n<=5).sum()),detected_le20=int((n<=20).sum()),mean_nnz=float(n.sum()/len(fit)),fit_min=mi,fit_max=ma,feature_policy=lock['feature_policy'],limitation='Cannot verify HVG identity or upstream leakage without full-gene source and selection provenance. No automatic dropping or reordering.'))
    if a.h5ad:
        import h5py
        with h5py.File(a.h5ad,'r') as h:
            def plain(node,depth=0):
                if depth>5:return '<nested>'
                if isinstance(node,h5py.Group):return {k:plain(node[k],depth+1) for k in list(node.keys())[:100]}
                if node.size>200:return {'shape':list(node.shape),'note':'array omitted'}
                raw=node[()]
                def conv(z):
                    if isinstance(z,bytes):return z.decode('utf8',errors='replace')
                    if isinstance(z,np.ndarray):return [conv(v) for v in z.tolist()] if z.ndim else conv(z.item())
                    if isinstance(z,np.generic):return conv(z.item())
                    if isinstance(z,list):return [conv(v) for v in z]
                    if isinstance(z,float) and not np.isfinite(z):return str(z)
                    return z
                return conv(raw)
            vg=h['var'];key=vg.attrs.get('_index','_index');key=key.decode() if isinstance(key,bytes) else key
            names=vg[key].asstr()[:].tolist()
            if names!=genes:raise ValueError('H5AD gene order differs from cache')
            uns=h.get('uns');prov={k:plain(uns[k]) for k in ('dataset_provenance','provenance','log1p') if uns is not None and k in uns}
            dump(a.out/'h5ad_metadata_audit.json',dict(gene_order_matches=True,variable_fields=list(vg.keys()),provenance=prov,limitation='Processed metadata only; no certification of upstream full-gene HVG selection.'))
    print('AUDIT SAVED',flush=True)

def torch_context(a):
    sys.path.insert(0,str(a.repo));sys.path.insert(0,str(HERE))
    import torch
    from response_model import ResponseModel
    dev=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    class BenchmarkModel(ResponseModel):
        # Same exact parameterized representation/backbone as E1; non-flow one-shot residual.
        def __init__(self,*args,nonflow=False,**kw):super().__init__(*args,**kw);self.nonflow=nonflow
        def loss_parts(self,target,source,drug,dose,line):
            if not self.nonflow:return super().loss_parts(target,source,drug,dose,line)
            f,kl,_=self.components(source,drug,dose,line)
            pred=source+self.velocity(source,source.new_zeros(len(source)),f)
            mse=(pred-target).square().mean();return mse+.01*kl,{'cfm':mse,'kl':kl}
        @torch.no_grad()
        def generate(self,source,drug,dose,line,steps=40,seed=None):
            if not self.nonflow:return super().generate(source,drug,dose,line,steps,seed)
            f,_,_=self.components(source,drug,dose,line);return source+self.velocity(source,source.new_zeros(len(source)),f)
    def make(model,vocab,G,width):
        dag=torch.zeros(5,5)
        if model!='E4':dag[4,:4]=1
        return BenchmarkModel(G,len(vocab['drug']),len(vocab['dose']),len(vocab['cell_line']),width=width,dag=dag,auxiliary=1.,cfm_weight=1.,dose_values=vocab['dose'],fullrank_gate=True,use_direct=False,l1_weight=0.,nonflow=model=='NonFlow').to(dev)
    return torch,dev,make

def reset(torch,s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None

def train(a,data,lock,idx,qs,smoke=False):
    torch,dev,make=torch_context(a);report,vocab,frames,_,genes,x,_=data;obs=frames['train'];fit=idx['fit'];labels=obs[['drug_idx','dose_idx','cell_line_idx']].to_numpy(np.int64);lines=obs.cell_line.to_numpy();control=report['control_drug']
    pools={l:fit[(lines[fit]==l)&(obs.iloc[fit].drug.str.casefold()==control.casefold()).to_numpy()] for l in set(lines)}
    if any(len(p)<2 for p in pools.values()):raise ValueError('Insufficient fit controls')
    for model in (MODELS if smoke else a.models):
        if model not in MODELS:raise ValueError('External model training not implemented here')
        for s in ([0] if smoke else a.seeds):
            reset(torch,s);r=np.random.default_rng(seedof('train',s));m=make(model,vocab,len(genes),a.width);m.train();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4)
            dest=a.out/'models'/model/f'seed_{s}';history=[];nsteps=2 if smoke else a.steps
            if not smoke:
                if (dest/'final.pt').exists():raise FileExistsError(dest/'final.pt')
                dest.mkdir(parents=True,exist_ok=True)
            for step in range(1,nsteps+1):
                tix=r.choice(fit,a.batch_size,replace=len(fit)<a.batch_size);srcix=np.array([r.choice(pools[lines[i]]) for i in tix]);same=tix==srcix
                for j in np.flatnonzero(same):srcix[j]=r.choice(pools[lines[tix[j]]][pools[lines[tix[j]]]!=tix[j]])
                target=torch.tensor(np.asarray(x[tix],np.float32),device=dev);source=torch.tensor(np.asarray(x[srcix],np.float32),device=dev);lab=torch.tensor(labels[tix],device=dev)
                opt.zero_grad(set_to_none=True);loss,parts=m.loss_parts(target,source,*lab.unbind(1));loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                if not torch.isfinite(loss) or not torch.isfinite(gn):raise ValueError('Nonfinite loss/gradient')
                opt.step()
                if step==1 or step%200==0 or step==nsteps:
                    history.append(dict(step=step,loss=float(loss.detach()),main_loss=float(parts['cfm'].detach()),kl=float(parts['kl'].detach()),grad_norm=float(gn)))
                    print(model,s,step,'loss',history[-1]['loss'],flush=True)
            if smoke:
                m.eval();pred=m.generate(source,*lab.unbind(1),steps=2);assert pred.shape==source.shape and torch.isfinite(pred).all();continue
            cfg=dict(model=model,seed=s,step=a.steps,lock_sha256=digest(a.out/'lock.json'),n_params=sum(p.numel() for p in m.parameters() if p.requires_grad),train_samples=len(fit),train_draws=a.steps*a.batch_size,source='fit only',objective=('one-shot conditional residual MSE + source KL' if model=='NonFlow' else 'source-anchored CFM + source KL'))
            pd.DataFrame(history).to_csv(dest/'training_history.csv',index=False);dump(dest/'train_config.json',cfg);torch.save({'model':m.state_dict(),'config':cfg},dest/'final.pt')
    if smoke:print('SMOKE PASS (three actual models, gradient and generation)',flush=True)

def pred_dir(a,m,s):return a.out/'predictions'/m/f'seed_{s}'
def predict(a,data,lock,idx,qs):
    torch,dev,make=torch_context(a);_,vocab,frames,_,genes,x,_=data
    for model in a.models:
        if model not in MODELS:raise ValueError('Use external prediction contract; not an internal model')
        for s in a.seeds:
            dest=pred_dir(a,model,s);cp=a.out/'models'/model/f'seed_{s}'/'final.pt';ck=torch.load(cp,map_location='cpu',weights_only=True)
            if ck['config']['lock_sha256']!=digest(a.out/'lock.json') or ck['config']['model']!=model or ck['config']['seed']!=s or ck['config']['step']!=a.steps:raise ValueError('Checkpoint lock mismatch')
            if (dest/'manifest.json').exists():raise FileExistsError(dest/'manifest.json')
            dest.mkdir(parents=True,exist_ok=True);m=make(model,vocab,len(genes),a.width);m.load_state_dict(ck['model'],strict=True);m.eval();files={}
            for q in qs:
                k=q['query_id'];src=idx[k+'_source'];outs=[];labs=[list(map(str,vocab[c])).index(str(q[c])) for c in KEYS]
                for j in range(0,len(src),64):
                    u=torch.tensor(np.asarray(x[src[j:j+64]],np.float32),device=dev);lab=[torch.full((len(u),),v,dtype=torch.long,device=dev) for v in labs];y=m.generate(u,*lab,steps=a.ode_steps);outs.append(y.cpu().numpy())
                y=np.concatenate(outs);assert y.shape==(a.n_pred,len(genes)) and np.isfinite(y).all();np.save(dest/f'{k}.npy',y);files[k]=digest(dest/f'{k}.npy')
            dump(dest/'manifest.json',dict(model=model,seed=s,lock_sha256=digest(a.out/'lock.json'),checkpoint_sha256=digest(cp),target_expression_as_inference_input=False,files=files));print('PREDICTED',model,s,flush=True)

def export(a,data,lock,idx,qs):
    import anndata as ad
    _,_,frames,_,genes,x,_=data;obs=frames['train'];d=a.out/'external_input';d.mkdir(parents=True,exist_ok=True)
    if (d/'fit.h5ad').exists():raise FileExistsError(d/'fit.h5ad')
    fit=idx['fit'];mat=sparse.vstack([sparse.csr_matrix(np.asarray(x[fit[i:i+512]],dtype=np.float32)) for i in range(0,len(fit),512)],format='csr');o=obs.iloc[fit].copy();o.index=o.source_cell_id.astype(str);o['dose_numeric']=o.dose.astype(float)
    ad.AnnData(mat,obs=o,var=pd.DataFrame(index=genes)).write_h5ad(d/'fit.h5ad',compression='gzip');del mat
    for q in qs:np.save(d/(q['query_id']+'_source.npy'),np.asarray(x[idx[q['query_id']+'_source']],dtype=np.float32))
    pd.DataFrame(qs).to_csv(d/'queries.csv',index=False);pd.DataFrame({'gene':genes}).to_csv(d/'genes.csv',index=False)
    dump(d/'manifest.json',dict(lock_sha256=digest(a.out/'lock.json'),fit_h5ad_sha256=digest(d/'fit.h5ad'),target_expression_included=False,n_fit=len(fit),n_pred=a.n_pred));print('EXPORTED fit only + queries; not a trained external model')

def bh(p):
    p=np.where(np.isfinite(p),p,1.);o=np.argsort(p,kind='stable');q=np.empty_like(p);q[o]=np.minimum(1.,np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return q

def ranking(arr,ctl):
    delta=arr.mean(0)-ctl.mean(0);ab=np.abs(delta);g=np.arange(len(delta))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);p=ttest_ind(arr,ctl,axis=0,equal_var=False).pvalue
    p=np.where(np.isfinite(p),p,1.);u=[]
    for j in range(0,len(g),256):u.extend(mannwhitneyu(arr[:,j:j+256],ctl[:,j:j+256],axis=0,alternative='two-sided',method='asymptotic').pvalue)
    return delta,np.lexsort((-ab,p))[:50],np.lexsort((g,-ab))[:50],(bh(np.array(u))<=.05)&(ab>=.01)

def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else np.nan

def score(a,data,lock,idx,qs):
    sys.path.insert(0,str(a.repo));from CausFlow.cpa_pipeline import condition_metrics
    _,_,_,_,genes,x,_=data;out=a.out/('evaluation_'+a.eval_tag);out.mkdir(exist_ok=True);files={}
    # Verify every submitted prediction and hash before reading any target row.
    for m in a.models:
        for s in a.seeds:
            d=pred_dir(a,m,s);manifest=read(d/'manifest.json')
            if manifest['model']!=m or manifest['seed']!=s or manifest['lock_sha256']!=digest(a.out/'lock.json') or manifest['target_expression_as_inference_input'] is not False:raise ValueError('Prediction provenance mismatch')
            if set(manifest['files'])!={q['query_id'] for q in qs}:raise ValueError('Incomplete prediction set')
            for q in qs:
                k=q['query_id'];p=d/f'{k}.npy'
                if digest(p)!=manifest['files'][k]:raise ValueError('Prediction hash mismatch')
                y=np.load(p,mmap_mode='r');assert y.shape==(a.n_pred,len(genes)) and np.isfinite(y).all();files[(m,s,k)]=p
    path=out/'metrics_by_condition.csv'
    if path.exists():raise FileExistsError('Evaluation exists; preserve it. No silent overwrites.')
    rows=[]
    for q in qs:
        k=q['query_id'];real=np.asarray(x[idx[k]],np.float32);ctl=np.asarray(x[idx[k+'_control']],np.float32);tr,deg,ab,td=ranking(real,ctl)
        baselines=[('Vehicle',-1,'native',np.asarray(x[idx[k+'_source']],np.float32))]
        for suffix,name in [('_lower','NearestLower'),('_same','ObservedCondition')]:
            if k+suffix in idx:baselines.append((name,-1,'native',np.asarray(x[idx[k+suffix]],np.float32)))
        def add(m,s,post,y):
            pdlt,pdeg,pab,dp=ranking(y,ctl);r=condition_metrics(real,y,ctl,degs=deg);ratio=float(np.linalg.norm(pdlt)/(np.linalg.norm(tr)+1e-12));shared=td&dp
            r.update(response_pcc_all=corr(tr,pdlt),response_pcc_deg50=corr(tr[deg],pdlt[deg]),response_norm_ratio=ratio,response_amplitude_error=abs(np.log(max(ratio,1e-12))),top50_welch_overlap=len(np.intersect1d(deg,pdeg))/len(deg),top50_abs_overlap=len(np.intersect1d(ab,pab))/len(ab),n_deg_true=int(td.sum()),n_deg_pred=int(dp.sum()),deg_recall=float(shared.sum()/td.sum()) if td.any() else np.nan,deg_precision=float(shared.sum()/dp.sum()) if dp.any() else np.nan,deg_signed_recall=float((shared&(np.sign(tr)==np.sign(pdlt))).sum()/td.sum()) if td.any() else np.nan,negative_fraction=float((y<0).mean()),zero_fraction=float((y==0).mean()),real_zero_fraction=float((real==0).mean()))
            rows.append({**q,'model':m,'seed':s,'output':post,**r})
        for m,s,p,y in baselines:add(m,s,p,y)
        for m in a.models:
            for s in a.seeds:
                y=np.load(files[(m,s,k)]);add(m,s,'raw',y);z=np.maximum(y,0);z[z<.01]=0;add(m,s,'S4',z)
        print('SCORED',k,flush=True)
    pd.DataFrame(rows).to_csv(path,index=False);dump(out/'provenance.json',dict(lock_sha256=digest(a.out/'lock.json'),models=a.models,seeds=a.seeds,primary='S4',external_provenance='Self-declared external methods require independent code/config audit; hashes prove file integrity, not no leakage'))
    summarize(a)

def summarize(a):
    out=a.out/('evaluation_'+a.eval_tag);d=pd.read_csv(out/'metrics_by_condition.csv');metrics=['R2_mean_all','R2_var_all','R2_mean_DEG50','R2_var_DEG50','W1_quantile_all_approx','W1_DEG50_exact','response_pcc_all','response_pcc_deg50','response_amplitude_error','top50_welch_overlap','top50_abs_overlap','deg_recall','deg_precision','deg_signed_recall','negative_fraction','zero_fraction']
    g=['split','model','output','seed'];s=d.groupby(g)[metrics].mean().reset_index();s.to_csv(out/'metrics_by_seed.csv',index=False);s.groupby(g[:-1])[metrics].agg(['mean','std','count']).to_csv(out/'metrics_summary.csv')
    # Aggregate seeds within drug first; drug-cluster bootstrap ignores seed uncertainty, report seed dispersion separately.
    paired=[];main=d[(d.model=='E1')&(d.output=='S4')]
    for other in sorted(set(d.model)-{'E1'}):
        rhs=d[(d.model==other)&(d.output==('native' if other in ('Vehicle','NearestLower','ObservedCondition') else 'S4'))]
        keys=['split','query_id','drug'];l=main.groupby(keys)[metrics].mean();r=rhs.groupby(keys)[metrics].mean();u=l.join(r,lsuffix='_a',rsuffix='_b',how='inner').reset_index()
        for split,z in u.groupby('split'):
            for m in metrics:
                if m=='zero_fraction':continue
                delta=(z[m+'_a']-z[m+'_b'])*(-1 if m.startswith('W1') or m in ('response_amplitude_error','negative_fraction') else 1)
                drug=pd.DataFrame({'drug':z.drug,'delta':delta}).groupby('drug').delta.mean().dropna().to_numpy()
                if not len(drug):continue
                b=np.random.default_rng(20261121).choice(drug,size=(10000,len(drug)),replace=True).mean(1);paired.append(dict(split=split,comparison='E1-'+other,metric=m,n_drugs=len(drug),n_conditions=int(delta.notna().sum()),mean_drug_improvement=float(drug.mean()),ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975))))
    pd.DataFrame(paired).to_csv(out/'paired_drug_bootstrap.csv',index=False);print('SUMMARY SAVED',flush=True)

def main():
    a=cli();data=readmeta(a)
    if a.mode=='prepare':prepare(a,data);return
    lock,idx,qs=load(a,data)
    if a.mode=='audit':audit(a,data,lock,idx,qs)
    elif a.mode in ('train','smoke'):train(a,data,lock,idx,qs,a.mode=='smoke')
    elif a.mode=='predict':predict(a,data,lock,idx,qs)
    elif a.mode=='export':export(a,data,lock,idx,qs)
    elif a.mode=='score':score(a,data,lock,idx,qs)
    else:summarize(a)
if __name__=='__main__':main()
