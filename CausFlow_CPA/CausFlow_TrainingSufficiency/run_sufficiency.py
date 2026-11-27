#!/usr/bin/env python
"""Child of LockedBenchmark v1. Select common internal-model budget on new inner validation only.
Original 48 queries have already been scored in v1: retrospective, not a fresh final test.
"""
import argparse, json, sys, warnings, os, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ttest_ind, wasserstein_distance
import run_locked_benchmark as b
HERE=Path(__file__).resolve().parent
MODELS=(*b.MODELS,'MatchedDiffusion')
PRO=('CPA','scGen_LowerDose')

def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['prepare','audit','smoke','train','inspect','select','predict','export','score','summarize'])
    p.add_argument('--repo',default='.');p.add_argument('--cache-dir',required=True);p.add_argument('--parent',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='auto');p.add_argument('--models',nargs='+',default=list(MODELS));p.add_argument('--seeds',nargs='+',type=int,default=None)
    p.add_argument('--stages',nargs='+',type=int,default=[2000,10000,30000]);p.add_argument('--until',type=int,default=None,help='Stop at a predeclared stage, resume later');p.add_argument('--val-drugs',type=int,default=4);p.add_argument('--val-seed',type=int,default=81273)
    p.add_argument('--min-ood',type=int,default=25);p.add_argument('--min-id',type=int,default=20);p.add_argument('--val-pred',type=int,default=256)
    p.add_argument('--eval-tag',default='sufficiency');p.add_argument('--hash-array',action='store_true',help='Full array SHA256 at prepare and subsequent invocations (optional expensive integrity check)')
    a=p.parse_args()
    for k in ['repo','cache_dir','parent','out']:setattr(a,k,Path(getattr(a,k)).resolve())
    if a.out==a.parent:raise ValueError('Child output must be separate')
    if not a.stages or sorted(set(a.stages))!=a.stages or min(a.stages)<1:raise ValueError('Stages must increase')
    if len(set(a.models))!=len(a.models) or not set(a.models)<=set(MODELS+PRO+('CPA_mean',)):raise ValueError('Invalid models')
    if not a.eval_tag.replace('_','').replace('-','').isalnum():raise ValueError('Invalid eval tag')
    return a

def parent(a,data):
    d=a.out;a.out=a.parent;lock,idx,qs=b.load(a,data);a.out=d;return lock,idx,qs

def locksources():return {p.name:b.digest(p) for p in [HERE/'run_sufficiency.py',HERE/'run_professional.py',HERE/'run_locked_benchmark.py',HERE/'response_model.py',HERE/'factor_ablation_model.py',HERE/'factor_base_pilot.py',HERE/'matched_diffusion.py']}

def prepare(a,data,pl,pi,pq):
    if (a.out/'lock.json').exists():raise FileExistsError('Already prepared; never redraw after viewing')
    report,vocab,frames,allobs,genes,x,hashes=data;obs=frames['train'];fit=np.asarray(pi['fit']);fobs=obs.iloc[fit]
    excluded=set(pl['selected_drugs'])|set(pl['excluded_drugs']);control=str(report['control_drug']);lines=sorted(fobs.cell_line.unique())
    groups={tuple(k):fit[np.asarray(v)] for k,v in fobs.groupby(b.KEYS,sort=True).indices.items()}
    rng=np.random.default_rng(a.val_seed);eligible=[]
    for drug,dd in allobs.groupby('drug',sort=True):
        if drug in excluded or drug.casefold()==control.casefold():continue
        top=float(dd.dose.astype(float).max());entries=[]
        if top<=0:continue
        for line in lines:
            hi=[(k,v) for k,v in groups.items() if k[0]==drug and k[2]==line and float(k[1])==top and len(v)>=a.min_ood]
            lo=[(k,v) for k,v in groups.items() if k[0]==drug and k[2]==line and 0<float(k[1])<top and len(v)>=2*a.min_id]
            if len(hi)!=1 or not lo:break
            entries.append((hi[0],max(lo,key=lambda kv:float(kv[0][1]))))
        if len(entries)==len(lines):eligible.append((drug,entries))
    if len(eligible)<a.val_drugs:raise ValueError(f'Need {a.val_drugs} inner drugs; got {len(eligible)}. Do not redraw casually.')
    sel=[eligible[i] for i in sorted(rng.choice(len(eligible),a.val_drugs,replace=False))];arr={k:v.copy() for k,v in pi.items()};vq=[];held=[]
    for drug,entries in sel:
        for (hk,hidx),(lk,lidx) in entries:
            ni=min(max(a.min_id,int(.2*len(lidx))),len(lidx)-a.min_id)
            for split,k,ix in [('val_OOD',hk,np.sort(hidx)),('val_ID',lk,np.sort(rng.choice(lidx,ni,replace=False)))]:
                q=f'v{len(vq):03d}';arr[q]=ix;held.append(ix);vq.append(dict(query_id=q,split=split,drug=k[0],dose=k[1],cell_line=k[2],n_real=len(ix)))
    val=np.concatenate(held);core=np.setdiff1d(fit,val);arr['fit']=core
    assert len(np.unique(val))==len(val) and len(core)+len(val)==len(fit)
    for q in pq:
        k=q['query_id']
        assert not np.intersect1d(arr[k],core).size
        for suffix in ['_source','_control','_lower','_same']:
            if k+suffix in arr and not np.isin(arr[k+suffix],core).all():raise ValueError('Inner validation overlaps fixed parent reference draw')
    for q in vq:
        k=q['query_id'];line=q['cell_line'];mask=(obs.drug==q['drug'])&(obs.cell_line==line)&(obs.dose.astype(float)==float(q['dose']))
        n=int(mask.iloc[core].sum());assert n==0 if q['split']=='val_OOD' else n>=a.min_id
        pool=core[(obs.iloc[core].drug.str.casefold()==control.casefold()).to_numpy()&(obs.iloc[core].cell_line==line).to_numpy()]
        if len(pool)<2:raise ValueError('No source controls')
        r=np.random.default_rng(b.seedof('inner',a.val_seed,k));arr[k+'_source']=r.choice(pool,a.val_pred,replace=True);arr[k+'_control']=r.choice(pool,min(512,len(pool)),replace=False)
    a.out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.parent/'queries.csv',a.out/'queries.csv');pd.DataFrame(vq).to_csv(a.out/'validation_queries.csv',index=False);np.savez_compressed(a.out/'indices.npz',**arr)
    lock={**pl,'schema':'training_sufficiency_v2','scope':'retrospective extension; parent 48 queries already evaluated before this phase, NOT independent test','parent_lock_sha256':b.digest(a.parent/'lock.json'),'parent_indices_sha256':pl['indices_sha256'],'source_hashes':locksources(),'indices_sha256':b.digest(a.out/'indices.npz'),'queries_sha256':b.digest(a.out/'queries.csv'),'validation_queries_sha256':b.digest(a.out/'validation_queries.csv'),'n_fit':len(core),'n_inner_val':len(val),'inner_drugs':[d for d,_ in sel],'inner_seed':a.val_seed,'val_pred':a.val_pred,'stages':a.stages,'steps':max(a.stages),'matched_diffusion':'DDPM noise prediction, cosine T=1000, DDIM with clipped x0 in [0,8] (fixed log-space bound), no target input; same representation architecture, independent parameters; not original CausCell','internal_models':a.models if set(a.models)<=set(MODELS) and set(b.MODELS)<=set(a.models) else list(MODELS),'selection_rule':'Each model: mean S4 inner val_OOD exact DEG50 W1 across all seeds and queries; earliest stage within 1% of its minimum. Common budget is max of these stages to avoid cutting off slower models; report own-best and common-budget deviations. report all validation curves; optional predeclared last stage 50000 is required if any internal model improves >1% at 30000 compared with its best earlier stage. Boundary optimum is not convergence proof.','array_sha256':b.digest(a.cache_dir/'train_x.npy') if a.hash_array else None,'professional_policy':{'CPA':'cpa-tools 0.8.8 Gaussian model; posterior predictive draws; decoded mean diagnostic','scGen_LowerDose':'scGen 2.1.1 official VAE+latent arithmetic; same-line same-dose ID or nearest lower-dose OOD, not native dose extrapolation','CPA_max_epochs':50,'scGen_max_epochs':100,'batch_size':128,'native_validation':'random validation sampled only within common permitted core pool; actual gradient-fit subset can differ from CausFlow','primary':'S4 for all learned models; native/raw secondary'}}
    b.dump(a.out/'lock.json',lock);print('PREPARED',len(core),'core fit;',len(vq),'inner queries;',len(pq),'parent queries;',lock['inner_drugs'],flush=True)

def load(a,data,pl,pi,pq):
    l=b.read(a.out/'lock.json')
    for k,f in [('indices_sha256','indices.npz'),('queries_sha256','queries.csv'),('validation_queries_sha256','validation_queries.csv')]:
        if b.digest(a.out/f)!=l[k]:raise ValueError(f'{f} fingerprint mismatch')
    if l['parent_lock_sha256']!=b.digest(a.parent/'lock.json') or l['source_hashes']!=locksources():raise ValueError('Changed parent or code')
    if l['array_sha256'] is not None and b.digest(a.cache_dir/'train_x.npy')!=l['array_sha256']:raise ValueError('Array changed')
    with np.load(a.out/'indices.npz',allow_pickle=False) as z:idx={k:z[k] for k in z.files}
    vq=pd.read_csv(a.out/'validation_queries.csv',dtype={'dose':str},keep_default_na=False).to_dict('records')
    a.stages=l['stages'];a.steps=max(a.stages);a.val_pred=l['val_pred'];a.internal_models=l['internal_models'];return l,idx,vq

def audit(a,data,l,idx,vq,pq):
    report,vocab,frames,_,genes,x,_=data;obs=frames['train'];core=idx['fit'];inner=np.concatenate([idx[q['query_id']] for q in vq]);outer=np.concatenate([idx[q['query_id']] for q in pq])
    assert len(np.unique(np.concatenate([core,inner,outer])))==len(obs)==len(core)+len(inner)+len(outer)
    rows=[];lo=float('inf');hi=-float('inf')
    for i in range(0,len(core),512):
        z=np.asarray(x[core[i:i+512]],np.float32)
        if not np.isfinite(z).all() or (z<0).any():raise ValueError('Core data must be finite and nonnegative')
        lo=min(lo,float(z.min()));hi=max(hi,float(z.max()))
    if 'MatchedDiffusion' in a.internal_models and hi>8.:raise ValueError('Core exceeds predeclared diffusion x0 bound: audit required')
    for q in vq+pq:
        k=q['query_id'];z=obs.iloc[idx[k]];assert len(z)==q['n_real'] and (z.drug==q['drug']).all() and (z.cell_line==q['cell_line']).all() and (z.dose.astype(float)==float(q['dose'])).all()
        pool=obs.iloc[core];m=(pool.drug==q['drug'])&(pool.cell_line==q['cell_line']);same=int((m&(pool.dose.astype(float)==float(q['dose']))).sum());higher=int((m&(pool.dose.astype(float)>float(q['dose']))).sum())
        if 'OOD' in q['split']:assert same==0 and higher==0
        else:assert same>0
        for suf in ('_source','_control'):
            assert np.isin(idx[k+suf],core).all();s=obs.iloc[idx[k+suf]];assert (s.cell_line==q['cell_line']).all() and (s.drug.str.casefold()==str(report['control_drug']).casefold()).all()
        rows.append({**q,'core_same':same,'core_higher':higher})
    pd.DataFrame(rows).to_csv(a.out/'split_audit.csv',index=False);b.dump(a.out/'audit.json',dict(core=len(core),inner=len(inner),outer=len(outer),pairwise_disjoint=True,outer_previously_evaluated=True,fit_min=lo,fit_max=hi,feature_policy=l['feature_policy']));print('AUDIT PASS',flush=True)

def model_context(a):
    torch,dev,base_make=b.torch_context(a)
    from matched_diffusion import adapt
    def make(name,vocab,G,width):
        return adapt(base_make('E1',vocab,G,width)) if name=='MatchedDiffusion' else base_make(name,vocab,G,width)
    return torch,dev,make

def gen(torch,dev,m,x,ix,q,vocab,steps):
    labels=[list(map(str,vocab[c])).index(str(q[c])) for c in b.KEYS];out=[]
    for j in range(0,len(ix),64):
        src=torch.tensor(np.asarray(x[ix[j:j+64]],np.float32),device=dev);labs=[torch.full((len(src),),v,dtype=torch.long,device=dev) for v in labels];out.append(m.generate(src,*labs,steps=steps).cpu().numpy())
    y=np.concatenate(out);assert np.isfinite(y).all();return y

def validate(a,torch,dev,m,data,idx,vq,model,s,step):
    _,vocab,_,_,_,x,_=data;rows=[];m.eval()
    with torch.no_grad():
        for q in vq:
            k=q['query_id'];real=np.asarray(x[idx[k]],np.float32);ctl=np.asarray(x[idx[k+'_control']],np.float32);y=gen(torch,dev,m,x,idx[k+'_source'],q,vocab,a.ode_steps);y=np.maximum(y,0);y[y<.01]=0
            r=real.mean(0,dtype=np.float64);p=y.mean(0,dtype=np.float64);ct=ctl.mean(0,dtype=np.float64);delta=r-ct;pred=p-ct
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');pv=ttest_ind(real,ctl,axis=0,equal_var=False).pvalue
            pv=np.where(np.isfinite(pv),pv,1.);deg=np.lexsort((-np.abs(delta),pv))[:50]
            labs=[list(map(str,vocab[c])).index(str(q[c])) for c in b.KEYS];srcidx=np.resize(idx[k+'_source'],len(real));obj=[];weights=[]
            for j in range(0,len(real),64):
                target=torch.tensor(real[j:j+64],device=dev);source=torch.tensor(np.asarray(x[srcidx[j:j+64]],np.float32),device=dev);lab=[torch.full((len(target),),v,dtype=torch.long,device=dev) for v in labs];_,parts=m.loss_parts(target,source,*lab);obj.append(float(parts['cfm']));weights.append(len(target))
            r2=lambda v,w: 1-float(np.sum((v-w)**2))/float(np.sum((v-v.mean())**2)) if np.sum((v-v.mean())**2)>1e-12 else np.nan
            rows.append({**q,'model':model,'seed':s,'step':step,'W1_DEG50_exact':float(np.mean([wasserstein_distance(real[:,i],y[:,i]) for i in deg])),'response_pcc_deg50':b.corr(delta[deg],pred[deg]),'R2_mean_all':r2(r,p),'R2_var_all':r2(real.var(0,dtype=np.float64),y.var(0,dtype=np.float64)),'response_rmse_deg50':float(np.sqrt(np.mean((pred[deg]-delta[deg])**2))),'zero_fraction':float((y==0).mean()),'val_main_loss':float(np.average(obj,weights=weights))})
    m.train();return rows

def train(a,data,l,idx,vq,smoke=False):
    torch,dev,make=model_context(a);report,vocab,frames,_,genes,x,_=data;obs=frames['train'];fit=idx['fit'];labels=obs[['drug_idx','dose_idx','cell_line_idx']].to_numpy(np.int64);lines=obs.cell_line.to_numpy();ctrl=str(report['control_drug'])
    pools={line:fit[(lines[fit]==line)&(obs.iloc[fit].drug.str.casefold()==ctrl.casefold()).to_numpy()] for line in set(lines)}
    if any(len(p)<2 for p in pools.values()):raise ValueError('Insufficient controls')
    for model in (a.internal_models if smoke else a.models):
        if model not in a.internal_models:raise ValueError('Use professional runner')
        for s in ([a.seeds[0]] if smoke else a.selected_seeds):
            b.reset(torch,s);r=np.random.default_rng(b.seedof('train-v2',s));m=make(model,vocab,len(genes),a.width);m.train();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4)
            d=a.out/'models'/model/f'seed_{s}';stages=[2] if smoke else [t for t in a.stages if a.until is None or t<=a.until];start=0;hist=[];buf=[]
            if not stages or (not smoke and a.until is not None and a.until not in a.stages):raise ValueError('Until must be a locked stage')
            if not smoke:
                d.mkdir(parents=True,exist_ok=True)
                complete=[t for t in stages if (d/f'step_{t}.pt').exists() and (d/f'validation_{t}.csv').exists() and (d/f'history_{t}.csv').exists()]
                if complete:
                    start=max(complete);ck=torch.load(d/f'step_{start}.pt',map_location='cpu',weights_only=True)
                    if ck['config']['lock_sha256']!=b.digest(a.out/'lock.json') or ck['config']['model']!=model or ck['config']['seed']!=s:raise ValueError('Resume checkpoint mismatch')
                    m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);r.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng']);
                    if dev.type=='cuda' and ck['cuda_rng']:torch.cuda.set_rng_state_all(ck['cuda_rng'])
                    print('RESUME',model,s,start,flush=True)
            for step in range(start+1,max(stages)+1):
                tix=r.choice(fit,a.batch_size,replace=len(fit)<a.batch_size);srcix=np.array([r.choice(pools[lines[i]]) for i in tix]);
                for j in np.flatnonzero(tix==srcix):srcix[j]=r.choice(pools[lines[tix[j]]][pools[lines[tix[j]]]!=tix[j]])
                target=torch.tensor(np.asarray(x[tix],np.float32),device=dev);source=torch.tensor(np.asarray(x[srcix],np.float32),device=dev);lab=torch.tensor(labels[tix],device=dev)
                opt.zero_grad(set_to_none=True);loss,parts=m.loss_parts(target,source,*lab.unbind(1));loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                if not torch.isfinite(loss) or not torch.isfinite(gn):raise ValueError('Nonfinite loss/gradient')
                opt.step();buf.append((float(loss.detach()),float(parts['cfm'].detach()),float(parts['kl'].detach()),float(gn)))
                if step==1 or step%200==0 or step in stages:
                    av=np.mean(buf,axis=0);hist.append(dict(step=step,loss=av[0],main_loss=av[1],kl=av[2],grad_norm=av[3],window=len(buf)));buf=[];print(model,s,step,'loss',av[0],flush=True)
                if step in stages:
                    # Validation seed is isolated from training RNG, including CUDA RNG.
                    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if dev.type=='cuda' else []):
                        torch.manual_seed(b.seedof('validation',s));val=validate(a,torch,dev,m,data,idx,vq[:1] if smoke else vq,model,s,step)
                    if smoke:assert len(val)==1;continue
                    cfg=dict(model=model,seed=s,step=step,lock_sha256=b.digest(a.out/'lock.json'),n_params=sum(p.numel() for p in m.parameters() if p.requires_grad),train_samples=len(fit),train_draws=step*a.batch_size,device=str(dev),source='core fit only',objective='one-shot conditional residual MSE + source KL' if model=='NonFlow' else 'cosine DDPM epsilon MSE + source KL; DDIM inference' if model=='MatchedDiffusion' else 'source-anchored CFM + source KL')
                    pd.DataFrame(val).to_csv(d/f'validation_{step}.csv',index=False);pd.DataFrame(hist).to_csv(d/f'history_{step}.csv',index=False);hist=[]
                    cp={'model':m.state_dict(),'optimizer':opt.state_dict(),'config':cfg,'numpy_rng':r.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if dev.type=='cuda' else []};tmp=d/f'step_{step}.tmp';torch.save(cp,tmp);os.replace(tmp,d/f'step_{step}.pt');b.dump(d/f'config_{step}.json',cfg)
                    print('VALIDATED',model,s,step,pd.DataFrame(val).W1_DEG50_exact.mean(),flush=True)
    if smoke:print('SMOKE PASS locked internal models, inner generation + validation',flush=True)

def select(a,l,vq,inspect_only=False):
    d=a.out/'selection.json'
    if not inspect_only and d.exists():raise FileExistsError('Selection already fixed')
    available=[t for t in a.stages if all((a.out/'models'/m/f'seed_{s}'/f'validation_{t}.csv').exists() and (a.out/'models'/m/f'seed_{s}'/f'step_{t}.pt').exists() for m in a.internal_models for s in a.seeds)]
    if not available:raise ValueError('No common complete validation stage')
    required=a.stages[:-1] if a.stages==[2000,10000,30000,50000] else a.stages
    if not inspect_only and not set(required)<=set(available):raise ValueError('Required stages incomplete')
    rows=[];hashes={}
    for m in a.internal_models:
        for s in a.seeds:
            for t in available:
                p=a.out/'models'/m/f'seed_{s}'/f'validation_{t}.csv';cp=p.with_name(f'step_{t}.pt');z=pd.read_csv(p)
                if set(z.query_id)!={q['query_id'] for q in vq} or len(z)!=len(vq) or not (z.model==m).all() or not (z.seed==s).all() or not (z.step==t).all() or not np.isfinite(z.W1_DEG50_exact).all():raise ValueError('Incomplete validation')
                hashes[str(p.relative_to(a.out))]=b.digest(p);hashes[str(cp.relative_to(a.out))]=b.digest(cp);rows.append(z)
    v=pd.concat(rows,ignore_index=True);v.to_csv(a.out/'validation_all.csv',index=False);sm=v.groupby(['model','step','split'])[['W1_DEG50_exact','response_pcc_deg50','R2_mean_all','R2_var_all','response_rmse_deg50','val_main_loss']].mean().reset_index();sm.to_csv(a.out/'validation_summary.csv',index=False)
    print(sm.to_string(index=False),flush=True)
    if inspect_only:return
    if a.stages==[2000,10000,30000,50000]:
        zz=sm[sm.split=='val_OOD'].pivot(index='model',columns='step',values='W1_DEG50_exact');need=bool((zz[30000] < .99*zz[[2000,10000]].min(axis=1)).any())
        if need and 50000 not in available:raise RuntimeError('Predeclared extension triggered: train --until 50000 on all internal models and seeds before select')
        if not need:available=[t for t in available if t!=50000]
    z=sm[(sm.model=='E1')&(sm.split=='val_OOD')&sm.step.isin(available)].set_index('step').W1_DEG50_exact;best=int(z.idxmin())
    own={};deviations={};curve=sm[(sm.split=='val_OOD')&sm.step.isin(available)].pivot(index='model',columns='step',values='W1_DEG50_exact')
    for model,v in curve.iterrows():own[model]=int(min(v[v<=float(v.min())*1.01].index))
    chosen=max(own.values())
    for model,v in curve.iterrows():deviations[model]=float(v[chosen]/max(float(v.min()),1e-12)-1)
    pd.DataFrame([dict(model=model,own_near_best_step=step,common_step=chosen,relative_W1_increase_at_common=deviations[model],best_exact_step=int(curve.loc[model].idxmin()),boundary_optimum=int(curve.loc[model].idxmin())==max(available)) for model,step in own.items()]).to_csv(a.out/'training_sufficiency_decisions.csv',index=False)
    b.dump(d,dict(lock_sha256=b.digest(a.out/'lock.json'),chosen_step=chosen,eligible_stages=available,per_model_near_best=own,relative_W1_increase_at_common=deviations,best_exact_step=best,boundary_optimum=best==max(available),selection_rule=l['selection_rule'],validation_curve={str(k):float(v) for k,v in z.items()},hashes=hashes,convergence_claim=False));print('SELECTED COMMON STEP',chosen,'boundary optimum:',best==max(available),flush=True)

def prediction(a,data,l,idx,pq):
    torch,dev,make=model_context(a);_,vocab,_,_,genes,x,_=data;sel=b.read(a.out/'selection.json');assert sel['lock_sha256']==b.digest(a.out/'lock.json');step=sel['chosen_step']
    for m in a.models:
        if m not in a.internal_models:raise ValueError('External runner required or model not locked')
        for s in a.selected_seeds:
            cp=a.out/'models'/m/f'seed_{s}'/f'step_{step}.pt';cp_hash=b.digest(cp)
            if cp_hash!=sel['hashes'][str(cp.relative_to(a.out))]:raise ValueError('Selected checkpoint changed')
            ck=torch.load(cp,map_location='cpu',weights_only=True);cfg=ck['config'];assert cfg['lock_sha256']==sel['lock_sha256'] and cfg['model']==m and cfg['seed']==s and cfg['step']==step
            dest=b.pred_dir(a,m,s);dest.mkdir(parents=True,exist_ok=True)
            if (dest/'manifest.json').exists():raise FileExistsError(dest/'manifest.json')
            model=make(m,vocab,len(genes),a.width);model.load_state_dict(ck['model'],strict=True);model.eval();files={}
            for q in pq:
                k=q['query_id'];torch.manual_seed(b.seedof('prediction',s,k));y=gen(torch,dev,model,x,idx[k+'_source'],q,vocab,a.ode_steps);assert y.shape==(a.n_pred,len(genes));np.save(dest/f'{k}.npy',y);files[k]=b.digest(dest/f'{k}.npy')
            b.dump(dest/'manifest.json',dict(model=m,seed=s,lock_sha256=sel['lock_sha256'],selection_sha256=b.digest(a.out/'selection.json'),checkpoint_sha256=cp_hash,target_expression_as_inference_input=False,files=files));print('PREDICTED',m,s,flush=True)

def export(a,data,l,idx,pq):
    import anndata as ad
    report,_,frames,_,genes,x,_=data;obs=frames['train'];fit=idx['fit'];d=a.out/'external_input';d.mkdir(parents=True,exist_ok=True)
    if (d/'manifest.json').exists():raise FileExistsError('Export already exists')
    mat=sparse.vstack([sparse.csr_matrix(np.asarray(x[fit[i:i+512]],np.float32)) for i in range(0,len(fit),512)],format='csr');o=obs.iloc[fit].copy();o.index=o.source_cell_id.astype(str);o['dose_numeric']=o.dose.astype(float)
    ad.AnnData(mat,obs=o,var=pd.DataFrame(index=genes)).write_h5ad(d/'fit.h5ad',compression='gzip');del mat
    for q in pq:
        k=q['query_id'];src=idx[k+'_source'];np.save(d/f'{k}_source.npy',np.asarray(x[src],np.float32));pd.DataFrame({'source_cell_id':obs.iloc[src].source_cell_id.astype(str).to_numpy()}).to_csv(d/f'{k}_source_ids.csv',index=False)
    pd.DataFrame(pq).to_csv(d/'queries.csv',index=False);pd.DataFrame({'gene':genes}).to_csv(d/'genes.csv',index=False)
    fs={p.name:b.digest(p) for p in d.iterdir() if p.is_file() and p.name!='manifest.json'};b.dump(d/'manifest.json',dict(lock_sha256=b.digest(a.out/'lock.json'),files=fs,n_fit=len(fit),n_pred=a.n_pred,control_drug=str(report['control_drug']),seeds=a.seeds,target_expression_included=False,professional_policy=l['professional_policy']));print('EXPORTED core fit + fixed control sources only',flush=True)

def verify_submission(a):
    sel=b.read(a.out/'selection.json');lock_hash=b.digest(a.out/'lock.json')
    if sel['lock_sha256']!=lock_hash:raise ValueError('Selection belongs to different lock')
    for m in a.models:
        for s in a.seeds:
            mf=b.read(b.pred_dir(a,m,s)/'manifest.json')
            if m in a.internal_models:
                cp=a.out/'models'/m/f'seed_{s}'/f"step_{sel['chosen_step']}.pt"
                if mf.get('selection_sha256')!=b.digest(a.out/'selection.json') or mf.get('checkpoint_sha256')!=b.digest(cp):raise ValueError('Internal checkpoint provenance mismatch')
            else:
                native='CPA' if m=='CPA_mean' else m;d=a.out/'professional_models'/native/f'seed_{s}';cfg=b.read(d/'train_config.json')
                if cfg.get('smoke') is not False or cfg.get('lock_sha256')!=lock_hash or mf.get('train_config_sha256')!=b.digest(d/'train_config.json') or mf.get('export_manifest_sha256')!=b.digest(a.out/'external_input'/'manifest.json'):raise ValueError('Professional provenance mismatch')
                for f,h in mf['model_artifacts'].items():
                    if not (d/f).resolve().is_relative_to(d.resolve()) or b.digest(d/f)!=h:raise ValueError('Professional weight artifact changed')

def extra_metrics(a,data,idx,pq):
    # Preserve original scorer's extreme R², add explicit denominator, MAE/RMSE and valid-count summaries.
    _,_,_,_,_,x,_=data;out=a.out/('evaluation_'+a.eval_tag);rows=[]
    for q in pq:
        k=q['query_id'];real=np.asarray(x[idx[k]],np.float64);ctl=np.asarray(x[idx[k+'_control']],np.float64);delta=real.mean(0)-ctl.mean(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');pv=ttest_ind(real,ctl,axis=0,equal_var=False).pvalue
        deg=np.lexsort((-np.abs(delta),np.where(np.isfinite(pv),pv,1.)))[:50]
        preds=[('Vehicle',-1,'native',x[idx[k+'_source']])]
        for suf,n in [('_lower','NearestLower'),('_same','ObservedCondition')]:
            if k+suf in idx:preds.append((n,-1,'native',x[idx[k+suf]]))
        for m in a.models:
            for s in a.seeds:
                y=np.load(b.pred_dir(a,m,s)/f'{k}.npy');preds.append((m,s,'raw',y));z=np.maximum(y,0);z[z<.01]=0;preds.append((m,s,'S4',z))
        for m,s,post,y in preds:
            rec={**q,'model':m,'seed':s,'output':post}
            for name,fn in [('mean',lambda t:t.mean(0,dtype=np.float64)),('var',lambda t:t.var(0,dtype=np.float64))]:
                u=fn(real)[deg];v=fn(y)[deg];ss=float(np.sum((u-u.mean())**2));err=float(np.sum((u-v)**2));rec.update({f'{name}_SST_DEG50':ss,f'{name}_SSE_DEG50':err,f'{name}_MAE_DEG50':float(np.mean(np.abs(u-v))),f'{name}_RMSE_DEG50':float(np.sqrt(err/len(u))),f'{name}_PCC_DEG50':b.corr(u,v),f'{name}_R2_DEG50_eps1e-8':1-err/ss if ss>=1e-8 else np.nan})
            rows.append(rec)
    d=pd.DataFrame(rows);d.to_csv(out/'r2_denominator_audit.csv',index=False);d.groupby(['split','model','output'])[list(d.columns[-12:])].agg(['median','mean','count']).to_csv(out/'r2_robust_summary.csv')

def main():
    a=parse();requested=a.seeds;data=b.readmeta(a);pl,pi,pq=parent(a,data);a.selected_seeds=a.seeds if requested is None else requested
    if not set(a.selected_seeds)<=set(a.seeds) or len(set(a.selected_seeds))!=len(a.selected_seeds):raise ValueError('Seed not in original lock')
    if a.mode=='prepare':
        if not set(b.MODELS)<=set(a.models)<=set(MODELS):raise ValueError('Prepare must lock E1 E4 NonFlow and optionally MatchedDiffusion')
        prepare(a,data,pl,pi,pq);return
    l,idx,vq=load(a,data,pl,pi,pq)
    if a.mode in ('train','smoke') and not (a.out/'audit.json').exists():raise ValueError('Run audit first')
    if a.mode=='audit':audit(a,data,l,idx,vq,pq)
    elif a.mode in ('smoke','train'):train(a,data,l,idx,vq,a.mode=='smoke')
    elif a.mode in ('select','inspect'):select(a,l,vq,a.mode=='inspect')
    elif a.mode=='predict':prediction(a,data,l,idx,pq)
    elif a.mode=='export':export(a,data,l,idx,pq)
    elif a.mode=='score':
        if not (a.out/'selection.json').exists():raise ValueError('Select before scoring')
        # base scorer verifies hashes and shape before accessing targets. All 3 seeds required.
        verify_submission(a);b.score(a,data,l,idx,pq);extra_metrics(a,data,idx,pq)
    else:b.summarize(a)
if __name__=='__main__':main()
