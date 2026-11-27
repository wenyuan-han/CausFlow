#!/usr/bin/env python
"""Real upstream CPA/scGen adapters. Run in separate pinned environments.
Reads exported core-fit H5AD and query control cells ONLY; no held-out expression file.
CPA Gaussian predictive draws and decoder mean; scGen nearest observed dose latent shift.
"""
import argparse, hashlib, importlib.metadata as md, json, os, random, sys, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import torch
import scvi
from scipy import sparse

HERE=Path(__file__).resolve().parent

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for v in iter(lambda:f.read(1<<20),b''):h.update(v)
    return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def reset(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);scvi.settings.seed=seed
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def versions():
    result={'python':sys.version,'torch':str(torch.__version__),'cuda_available':bool(torch.cuda.is_available())}
    for p in ['cpa-tools','scgen','scvi-tools','anndata','scanpy','numpy','pytorch-lightning','lightning','torchmetrics']:
        try:result[p]=md.version(p)
        except md.PackageNotFoundError:pass
    return result

def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['preflight','smoke','run'],required=True);p.add_argument('--out',required=True);p.add_argument('--model',choices=['CPA','scGen_LowerDose'],required=True);p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--device',default='auto',choices=['auto','cpu','cuda']);a=p.parse_args();a.out=Path(a.out).resolve();return a

def load(a):
    d=a.out/'external_input';m=read(d/'manifest.json');l=read(a.out/'lock.json')
    if m['lock_sha256']!=digest(a.out/'lock.json') or m['target_expression_included'] is not False or l['professional_policy']!=m['professional_policy']:raise ValueError('Protocol/export mismatch')
    if l['source_hashes']['run_professional.py']!=digest(HERE/'run_professional.py'):raise ValueError('Professional script changed after freeze')
    for f,h in m['files'].items():
        if Path(f).name!=f or digest(d/f)!=h:raise ValueError('Export file mismatch: '+f)
    fit=ad.read_h5ad(d/'fit.h5ad');genes=pd.read_csv(d/'genes.csv',dtype=str,keep_default_na=False).gene.tolist();q=pd.read_csv(d/'queries.csv',dtype={'dose':str},keep_default_na=False)
    assert fit.n_obs==m['n_fit'] and list(fit.var_names)==genes and fit.obs_names.is_unique
    assert set(['drug','dose_numeric','cell_line','source_cell_id'])<=set(fit.obs.columns)
    values=fit.X.data if sparse.issparse(fit.X) else fit.X
    assert np.isfinite(values).all() and np.min(values)>=0
    fit.obs['drug']=fit.obs.drug.astype(str);fit.obs.loc[fit.obs.drug.str.casefold()==m['control_drug'].casefold(),'drug']=m['control_drug'];fit.obs['cell_line']=fit.obs.cell_line.astype(str);fit.obs['dose_numeric']=fit.obs.dose_numeric.astype(float);fit.obs['dose_str']=fit.obs.dose_numeric.map(lambda x:format(float(x),'.12g'))
    fit.obs['condition_key']=np.where(fit.obs.drug==m['control_drug'],'CONTROL',fit.obs.drug+'|'+fit.obs.dose_str)
    for _,r in q.iterrows():
        same=(fit.obs.drug==r.drug)&(fit.obs.cell_line==r.cell_line)&(fit.obs.dose_numeric==float(r.dose))
        if r.split=='dose_OOD' and same.any():raise ValueError('OOD rows leaked into permitted fit')
        if r.split=='ID' and not same.any():raise ValueError('Missing ID train condition')
    return d,m,l,fit,q,genes

def source(d,m,fit,q):
    k=q.query_id;ids=pd.read_csv(d/f'{k}_source_ids.csv',dtype=str,keep_default_na=False).source_cell_id.tolist();ix=fit.obs_names.get_indexer(ids);assert (ix>=0).all() and len(ix)==m['n_pred'];o=fit.obs.iloc[ix];assert (o.drug==m['control_drug']).all() and (o.cell_line==q.cell_line).all();x=np.load(d/f'{k}_source.npy',allow_pickle=False);z=fit.X[ix];z=z.toarray() if sparse.issparse(z) else np.asarray(z);assert np.array_equal(x,z);return np.asarray(x,np.float32),ix

def record_environment(dest):
    dump(dest/'environment.json',versions())
    p=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=False);(dest/'pip_freeze.txt').write_text(p.stdout,encoding='utf8')

def check_upstream(model):
    if model=='CPA':
        import cpa
        if md.version('cpa-tools')!='0.8.8':raise RuntimeError('Expected cpa-tools==0.8.8')
        import inspect
        if 'perturbation_key' not in inspect.signature(cpa.CPA.setup_anndata).parameters:raise RuntimeError('Unexpected CPA API')
        return cpa.CPA
    import scgen
    if md.version('scgen')!='2.1.1':raise RuntimeError('Expected scgen==2.1.1')
    return scgen.SCGEN

def cpa_predict(model,x,drug,dose,line,seed):
    """Official CPA module: encode source X, add target perturbation+covariate, decode Gaussian.
    Use actual decoder distribution, not an approximation or hand-implemented CPA.
    """
    module=model.module;module.eval();device=next(module.parameters()).device;means=[];samples=[];vars=[]
    if drug not in model.pert_encoder or line not in model.covars_encoder['cell_line']:raise ValueError('Unknown condition for CPA')
    g=torch.Generator(device=device);g.manual_seed(seed)
    with torch.no_grad():
        for i in range(0,len(x),128):
            xx=torch.tensor(x[i:i+128],device=device);n=len(xx);dr=torch.full((n,1),model.pert_encoder[drug],dtype=torch.long,device=device);ds=torch.full((n,1),float(dose),dtype=torch.float32,device=device);li=torch.full((n,),model.covars_encoder['cell_line'][line],dtype=torch.long,device=device)
            inf=module.inference(x=xx,perts={'true':dr,'mixup':dr},perts_doses={'true':ds,'mixup':ds},covars_dict={'cell_line':li,'cell_line_mixup':li},mixup_lambda=1.,n_samples=1)
            px=module.generative(z=inf['z'],library=inf['library'])['px'];mu=px.loc;sd=px.scale
            means.append(mu.detach().cpu().numpy());vars.append(sd.square().detach().cpu().numpy());samples.append((mu+sd*torch.randn(mu.shape,device=device,dtype=mu.dtype,generator=g)).cpu().numpy())
    return np.concatenate(samples),np.concatenate(means),np.concatenate(vars)

def fit_cpa(CPA,fit,m,seed,gpu,epochs,dest,smoke):
    # Reset global categorical encoders for an independent model per seed.
    CPA.pert_encoder=None;CPA.covars_encoder=None;CPA.pert_smiles_map=None
    fit.obs['drug_cpa']=fit.obs.drug.map(lambda x:str(x).replace('+','__PLUS__'))
    # Avoid CPA's '+' syntax for drug combinations. Names mapped deterministically both train and query.
    fit.obs['dose_cpa']=fit.obs.dose_numeric.map(lambda x:format(float(x),'.12g'))
    control=m['control_drug'].replace('+','__PLUS__');
    if fit.obs.drug_cpa.nunique()!=fit.obs.drug.nunique():raise ValueError('CPA drug name collision')
    CPA.setup_anndata(fit,perturbation_key='drug_cpa',control_group=control,dosage_key='dose_cpa',categorical_covariate_keys=['cell_line'],is_count_data=False,max_comb_len=1)
    model=CPA(fit,n_latent=256,recon_loss='gauss',doser_type='logsigm',variational=False,seed=seed)
    # Use CPA's own trainer and only permitted core rows. Validation sampled from this pool by library.
    model.train(max_epochs=epochs,use_gpu=gpu,train_size=.9,validation_size=.1,batch_size=128,check_val_every_n_epoch=1,early_stopping_patience=10,save_path=False,enable_progress_bar=True,**({'limit_train_batches':3,'limit_val_batches':2} if smoke else {}))
    if not smoke:
        model.save(str(dest/'model'),overwrite=False,save_anndata=False)
        if model.epoch_history is not None:model.epoch_history.to_csv(dest/'epoch_history.csv',index=False)
        split=getattr(model,'train_indices',None);valid=getattr(model,'validation_indices',None)
        if split is not None:np.save(dest/'native_train_indices.npy',np.asarray(split))
        if valid is not None:np.save(dest/'native_validation_indices.npy',np.asarray(valid))
    return model

def fit_scgen(SCGEN,fit,seed,gpu,epochs,dest,smoke):
    # Shared official VAE trained once per seed on all permitted drug/dose states.
    SCGEN.setup_anndata(fit,batch_key='condition_key',labels_key='cell_line')
    model=SCGEN(fit)
    model.train(max_epochs=epochs,use_gpu=gpu,train_size=.9,validation_size=.1,batch_size=128,early_stopping=True,early_stopping_patience=10,check_val_every_n_epoch=1,**({'limit_train_batches':3,'limit_val_batches':2} if smoke else {}))
    if not smoke:
        model.save(str(dest/'model'),overwrite=False,save_anndata=False)
        for k,v in model.history.items():
            if hasattr(v,'to_csv'):v.to_csv(dest/f'history_{k}.csv')
        for k in ['train_indices','validation_indices']:
            v=getattr(model,k,None)
            if v is not None:np.save(dest/f'native_{k}.npy',np.asarray(v))
    # The upstream predict method constructs CPU tensors in some releases. Explicit CPU inference.
    model.to_device('cpu');model.module.eval();return model

def scgen_predict(model,fit,x,ix,q):
    mask=(fit.obs.drug==q.drug)&(fit.obs.cell_line==q.cell_line)
    doses=fit.obs.loc[mask,'dose_numeric'].to_numpy();target=float(q.dose)
    if q.split=='dose_OOD':
        choices=doses[doses<target]
    else:choices=doses[doses==target]
    if not len(choices):raise ValueError('No observed dose for scGen transfer')
    dose=float(np.max(choices));stim=str(q.drug)+'|'+format(dose,'.12g')
    # q source drawn from permitted fit; only source controls are passed to official predict.
    adata=fit[ix].copy();adata.obs_names_make_unique();assert np.array_equal(adata.X.toarray() if sparse.issparse(adata.X) else adata.X,x)
    pred,delta=model.predict(ctrl_key='CONTROL',stim_key=stim,adata_to_predict=adata,restrict_arithmetic_to={'cell_line':[str(q.cell_line)]})
    return np.asarray(pred.X,np.float32),dose

def main():
    a=parse();cls=check_upstream(a.model);print('VERSIONS',versions(),flush=True)
    if a.mode=='preflight':
        print('IMPORT/API PASS; NOT a training smoke test',flush=True);return
    d,m,l,fit,q,genes=load(a);seeds=m['seeds'] if a.seeds is None else a.seeds
    if not set(seeds)<=set(m['seeds']) or len(set(seeds))!=len(seeds):raise ValueError('Invalid seed list')
    gpu=torch.cuda.is_available() if a.device=='auto' else a.device=='cuda'
    if gpu and not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    smoke=a.mode=='smoke';start=time.time()
    for seed in ([seeds[0]] if smoke else seeds):
        reset(seed);dest=a.out/('professional_smoke' if smoke else 'professional_models')/a.model/f'seed_{seed}'
        if dest.exists():raise FileExistsError(f'{dest}; move incomplete attempt aside before rerun')
        dest.mkdir(parents=True);record_environment(dest);work=fit.copy()
        # Smoke uses an explicit small training subset while retaining all fixed sources and first query condition.
        # It must not produce formal prediction manifests or train on external targets.
        qs=q.iloc[:1] if smoke else q
        if smoke:
            row=qs.iloc[0];sx,ix=source(d,m,fit,row);r=np.random.default_rng(seed)
            mask=(fit.obs.drug==row.drug)&(fit.obs.cell_line==row.cell_line)
            chosen=np.unique(np.concatenate([ix,np.flatnonzero(mask),r.choice(fit.n_obs,min(1000,fit.n_obs),replace=False)]));work=fit[chosen].copy()
        if a.model=='CPA':model=fit_cpa(cls,work,m,seed,gpu,1 if smoke else l['professional_policy']['CPA_max_epochs'],dest,smoke)
        else:model=fit_scgen(cls,work,seed,gpu,1 if smoke else l['professional_policy']['scGen_max_epochs'],dest,smoke)
        names=['CPA','CPA_mean'] if a.model=='CPA' else [a.model];files={n:{} for n in names};pred_dirs={n:a.out/'predictions'/n/f'seed_{seed}' for n in names};dose_log=[]
        if not smoke:
            for p in pred_dirs.values():
                if (p/'manifest.json').exists():raise FileExistsError(p/'manifest.json')
                p.mkdir(parents=True,exist_ok=True)
        for _,row in qs.iterrows():
            x,ix=source(d,m,fit,row);k=row.query_id
            if a.model=='CPA':
                y,mean,var=cpa_predict(model,x,row.drug.replace('+','__PLUS__'),float(row.dose),row.cell_line,seed*10000+int(k[1:]));result={'CPA':y,'CPA_mean':mean}
                if not smoke:np.save(dest/f'{k}_conditional_variance.npy',var.astype(np.float32))
            else:
                ix_work=work.obs_names.get_indexer(fit.obs_names[ix]);assert (ix_work>=0).all();y,dose=scgen_predict(model,work,x,ix_work,row);result={a.model:y};dose_log.append({'query_id':k,'query_dose':float(row.dose),'observed_dose_used':dose,'mode':'same dose ID' if row.split=='ID' else 'nearest observed lower-dose proxy'})
            for n,arr in result.items():
                arr=np.asarray(arr,np.float32)
                if arr.shape!=(m['n_pred'],len(genes)) or not np.isfinite(arr).all():raise ValueError('Wrong shape/nonfinite predictions')
                if not smoke:
                    p=pred_dirs[n]/f'{k}.npy';np.save(p,arr);files[n][k]=digest(p)
            print('PREDICTED' if not smoke else 'SMOKE PREDICTED',a.model,seed,k,flush=True)
        if dose_log:pd.DataFrame(dose_log).to_csv(dest/'dose_proxy.csv',index=False)
        cfg={'model':a.model,'seed':seed,'lock_sha256':m['lock_sha256'],'export_manifest_sha256':digest(d/'manifest.json'),'versions':versions(),'n_allowed_fit':fit.n_obs,'native_train_fraction':.9,'max_epochs':1 if smoke else l['professional_policy']['CPA_max_epochs' if a.model=='CPA' else 'scGen_max_epochs'],'n_params':sum(p.numel() for p in model.module.parameters() if p.requires_grad),'elapsed_sec':time.time()-start,'device_training':'cuda' if gpu else 'cpu','smoke':smoke,'policy':l['professional_policy'][a.model],'target_expression_as_inference_input':False,'note':'Native library validation on permitted core rows only; check learning curves and stopping before interpreting performance.'}
        dump(dest/'train_config.json',cfg)
        if not smoke:
            artifacts={str(p.relative_to(dest)):digest(p) for p in (dest/'model').rglob('*') if p.is_file()};cfg_hash=digest(dest/'train_config.json')
            for n in names:dump(pred_dirs[n]/'manifest.json',dict(model=n,seed=seed,lock_sha256=m['lock_sha256'],export_manifest_sha256=cfg['export_manifest_sha256'],train_config_sha256=cfg_hash,model_artifacts=artifacts,target_expression_as_inference_input=False,files=files[n],output_definition='Gaussian predictive draw' if n=='CPA' else 'Gaussian decoder mean' if n=='CPA_mean' else 'official scGen latent delta, nearest dose proxy for OOD'))
        else:print('PROFESSIONAL SMOKE PASS',a.model,flush=True)
if __name__=='__main__':main()
