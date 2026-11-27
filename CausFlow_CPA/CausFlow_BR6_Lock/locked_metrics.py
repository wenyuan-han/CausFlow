#!/usr/bin/env python3
"""Read-only audits for CausFlow native drug OOD and TrainingSufficiency_v2.
No model training. No imports of project code. All new files go to a fresh output folder.
"""
import argparse, ast, hashlib, itertools, json, os, re, sys, traceback, warnings, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, wasserstein_distance

MODELS = ('E1','E4','NonFlow','MatchedDiffusion')
SEEDS = (0,1,2)
PROFILE = dict(n_genes=5000,n_pred=256,n_queries=48,n_drugs=8,n_lines=3,n_val=24)
METRICS = {'W1_DEG50_exact':-1,'response_pcc_deg50':1,'response_rmse_deg50':-1,
           'response_pcc_all':1,'response_rmse_all':-1,'R2_mean_all':1,'R2_var_all':1}
EXPECTED_OBJECTIVES = {'E1':'source-anchored CFM + source KL','E4':'source-anchored CFM + source KL',
    'NonFlow':'one-shot conditional residual MSE + source KL',
    'MatchedDiffusion':'cosine DDPM epsilon MSE + source KL; DDIM inference'}

def req(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(z) for z in x]
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
def dump(p,x): Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def csv(p): return pd.read_csv(p,dtype={'dose':str,'source_cell_id':str,'drug':str,'cell_line':str},keep_default_na=False)
def verify(p,h): req(isinstance(h,str) and sha(p)==h,'Hash mismatch: '+str(p))
def safe_child(base,name):
    p=(base/name).resolve();req(p.is_relative_to(base.resolve()),'Unsafe path: '+str(name));return p
def load_indices(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def lines(p): return p.read_text(encoding='utf-8-sig').splitlines()
def func(p,name):
    src=p.read_text(encoding='utf-8-sig'); tree=ast.parse(src); xs=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    req(len(xs)==1, f'{p}: expected one {name}, got {len(xs)}')
    n=xs[0];return n,ast.get_source_segment(src,n)
def flat(s): return re.sub(r'\s+','',s)

def load_meta(cache):
    report=read(cache.parent/'audit.json');vocab=read(cache/'vocab.json')
    frames={s:csv(cache/f'{s}_obs.csv') for s in ('train','val','test','ood')}
    for s,d in frames.items():
        req({'source_cell_id','drug','dose','cell_line','drug_idx','dose_idx','cell_line_idx'}<=set(d),s+' columns')
        req(np.isfinite(d.dose.astype(float)).all(),s+' nonfinite dose')
        for k in ('drug','dose','cell_line'):
            v=list(map(str,vocab[k]));req(len(v)==len(set(v)),k+' duplicate vocabulary')
            req(np.array_equal(d[k].astype(str).map(dict(zip(v,range(len(v))))).to_numpy(),d[k+'_idx'].to_numpy()),s+' '+k+' vocab mismatch')
    allobs=pd.concat(frames.values(),keys=frames,names=['partition','row']).reset_index(level=0)
    req(not allobs.source_cell_id.duplicated().any(),'Cell IDs overlap across original partitions')
    genes=pd.read_csv(cache/'genes.csv',dtype=str,keep_default_na=False).gene.tolist()
    req(len(set(genes))==len(genes),'Duplicate genes')
    req(hashlib.sha256('\n'.join(genes).encode()).hexdigest()==report['gene_sha256'],'Gene order fingerprint mismatch')
    hashes={f:sha(cache/f) for f in ['genes.csv','vocab.json',*(f'{s}_obs.csv' for s in frames)]}
    hashes['audit.json']=sha(cache.parent/'audit.json');hashes['train_x.npy_size']=(cache/'train_x.npy').stat().st_size
    return report,vocab,frames,allobs,genes,hashes

def evidence(p,name,needles):
    n,src=func(p,name);f=flat(src)
    return {'path':str(p),'sha256':sha(p),'function':name,'start':n.lineno,'end':n.end_lineno,
            'checks':{k:flat(v) in f for k,v in needles.items()},'source':src}

def c0(a,dest):
    # The two private local roots can differ; never run the user's training entrypoint.
    gen=a.native_root/'CausFlow'/'Generation_utils.py';cf=a.native_root/'CausFlow'/'CausFlow.py'
    base=a.source/'factor_base_pilot.py';suff=a.source/'run_sufficiency.py'
    specs=[('pool',gen,'factor_value_pool',{'observed_values':'factor_values = np.unique(ori_data.obs[factor])','pool_index':'concept_embs[ori_data.obs[factor] == factor_value]'}),
           ('replace',gen,'target_embs_generation',{'strict_target_lookup':'factor_dict[target_factor][tgt_factor_value]','fallback_lookup':'factor_dict[target_factor][alt_key]'}),
           ('counterfactual',cf,'counterfactual_generation',{'same_data_pool':'factor_value_pool(concept_embs, concept_list, data_pwd)','encode_same_data':'extract_concept_embs(data_pwd, self.model','replace_call':'target_embs_generation(factor_dict, concept_list, multi_target_list, data_pwd, ref_concept_embs)'}),
           ('mapping',cf,'data_transformation',{'unknown_label_minus_one':'val2idx.get(i, -1)'}),
           ('reconstruction',cf,'sampling_cells',{'target_test_expression':'DisentanglementEncoder.eval()(x_batch, labels_batch)'}),
           ('adapter',base,'components',{'categorical_drug':'drug_e=self.drug_emb(drug)','source_encoder':'self.source_to_hidden(source)','dose_numeric':'self.dose_mlp(torch.log1p(self.dose_values[dose])'}),
           ('adapter_init',base,'__init__',{'lookup_table':'self.drug_emb=nn.Embedding(n_drugs,latent)'}),
           ('adapter_gen',suff,'gen',{'fixed_vocab_index':'list(map(str,vocab[c])).index(str(q[c]))'})]
    # __init__ in this exact module belongs to FactorSeparatedCausFlow; variations require review.
    ev={};errors=[]
    for key,p,n,needles in specs:
        try: ev[key]=evidence(p,n,needles)
        except Exception as e: errors.append(f'{key}: {e}')
    (dest/'source_evidence.txt').write_text('\n\n'.join(f'[{k}] {v["path"]}:{v["start"]}-{v["end"]}\nSHA256 {v["sha256"]}\n{v["source"]}' for k,v in ev.items()),encoding='utf8')
    native_ok=all(k in ev and all(ev[k]['checks'].values()) for k in ('pool','replace','counterfactual'))
    adapter_ok=all(k in ev and all(ev[k]['checks'].values()) for k in ('adapter','adapter_init','adapter_gen'))
    # Probe the exact local functions on in-memory synthetic observations, only if the inspected signatures match.
    # This executes trusted local function bodies, never project imports/training; opt-out available.
    probe={'status':'SKIPPED'}
    if native_ok and not a.no_native_probe:
        from types import SimpleNamespace
        obs=pd.DataFrame({'drug':[0,0,1,1]}); emb=np.arange(16,dtype=np.float32).reshape(2,4,2)
        scope={'np':np,'sc':SimpleNamespace(read_h5ad=lambda _ :SimpleNamespace(obs=obs))}
        try:
            for n in ('factor_value_pool','target_embs_generation'):
                node,_=func(gen,n);exec(compile(ast.Module(body=[node],type_ignores=[]),str(gen),'exec'),scope)
            pool=scope['factor_value_pool'](emb,['drug'],'synthetic'); ref=emb[:,:2,:].copy()
            np.random.seed(42); z=scope['target_embs_generation'](pool,['drug'],[{'target_factor':'drug','tgt_factor_value':1}],'synthetic',ref.copy())
            req(z.shape==ref.shape and np.array_equal(z[1],ref[1]) and all(any(np.array_equal(v,w) for w in emb[0,2:]) for v in z[0]),'Seen replacement invalid')
            unknown=[]
            for tgt in (2,'2','never_seen_drug'):
                try: scope['target_embs_generation'](pool,['drug'],[{'target_factor':'drug','tgt_factor_value':tgt}],'synthetic',ref.copy());unknown.append({'requested':tgt,'result':'unexpected_success'})
                except (KeyError,ValueError) as e: unknown.append({'requested':tgt,'result':type(e).__name__,'detail':str(e)})
            probe={'status':'PASS' if all(r['result']!='unexpected_success' for r in unknown) else 'REVIEW','seen_replacement':'PASS','unseen':unknown}
        except Exception as e: probe={'status':'REVIEW','error':str(e)}
    report,vocab,frames,allobs,genes,hashes=load_meta(a.cache)
    lock=read(a.bench/'lock.json');req(lock['cache_hashes']==hashes,'Cache metadata differs from lock');verify(a.bench/'indices.npz',lock['indices_sha256']);idx=load_indices(a.bench/'indices.npz');obs=frames['train'];fit=idx['fit'];idx_check(fit,len(obs),'fit',unique=True)
    ctrl=str(report['control_drug']); f=obs.iloc[fit];coverage=[]
    for drug in sorted(set(map(str,vocab['drug']))|set(allobs.drug)):
        if drug.casefold()==ctrl.casefold():continue
        g=allobs[allobs.drug==drug];tr=f[f.drug==drug];old=obs[obs.drug==drug];n_doses=old.dose.astype(float).nunique()
        row=dict(drug=drug,in_fixed_vocab=drug in list(map(str,vocab['drug'])),n_core_fit=len(tr),n_original_train=len(old),n_all_partitions=len(g),n_train_lines=old.cell_line.nunique(),n_train_doses=n_doses,**{f'n_{s}':int((df.drug==drug).sum()) for s,df in frames.items()})
        row['fixed_vocab_target_specific_fit_support']=bool(len(tr));row['rows_removed_if_drug_holdout']=len(old)
        coverage.append(row)
    pd.DataFrame(coverage).to_csv(dest/'drug_coverage.csv',index=False)
    counts=obs[obs.drug.str.casefold()!=ctrl.casefold()].groupby(['drug','cell_line','dose'],as_index=False).size();counts.to_csv(dest/'condition_coverage_original_train.csv',index=False)
    result={'scope':'audited original target-pool path and existing source-conditioned adapter, not all imaginable extensions','native_path_static':native_ok,'native_actual_function_probe':probe,
       'native_verdict':'NO_GO_NATIVE_DRUG_LEVEL_OOD' if native_ok and probe['status']=='PASS' else 'STATIC_NO_GO_CANDIDATE_REVIEW' if native_ok else 'NEEDS_SOURCE_REVIEW',
       'adapter_path_static':adapter_ok,'adapter_verdict':'NO_GO_UNSEEN_IDENTITY_WITH_CURRENT_CATEGORICAL_CONDITION' if adapter_ok else 'NEEDS_SOURCE_REVIEW',
       'adapter_explanation':'Absent from vocabulary: index lookup fails. If held-out drug is retained in a global vocabulary, a lookup row can exist but no target-specific supervision has been obtained from that drug; no chemical/MoA feature path in inspected components. An arbitrary/untrained row is not validated zero-shot generalization.',
       'n_candidate_drugs':len(coverage),'n_vocab_drugs_without_core_fit':sum(not r['n_core_fit'] and r['in_fixed_vocab'] for r in coverage),
       'cache_scope':'metadata only; no expression read or holdout trained; inherited genes not certified train-only','evidence':{k:{h:v[h] for h in ('path','sha256','function','start','end','checks')} for k,v in ev.items()},'source_review_errors':errors}
    dump(dest/'c0_report.json',result);print('C0',result['native_verdict'],result['adapter_verdict'],flush=True);return result

def idx_check(x,n,label,unique=False):
    req(x.ndim==1 and np.issubdtype(x.dtype,np.integer) and len(x)>0 and np.min(x)>=0 and np.max(x)<n,'Invalid indices '+label)
    if unique:req(len(x)==len(np.unique(x)),'Duplicate indices '+label)

def check_queries(q,obs,idx,fit,ctrl,n_pred,validation=False):
    req(not q.query_id.duplicated().any(),'Duplicate query IDs')
    pool=obs.iloc[fit];fit_mask=np.zeros(len(obs),bool);fit_mask[fit]=True
    held=[];rows=[]
    for row in q.itertuples(index=False):
        k=row.query_id;req(re.fullmatch(r'[qv]\d+',k) is not None,'Unsafe query ID')
        valid_splits=('val_ID','val_OOD') if validation else ('ID','dose_OOD');req(row.split in valid_splits,'Invalid split')
        ix=idx[k];idx_check(ix,len(obs),k,True);z=obs.iloc[ix];held.append(ix)
        req(len(ix)==int(row.n_real) and (z.drug==row.drug).all() and (z.cell_line==row.cell_line).all() and (z.dose.astype(float)==float(row.dose)).all(),'Query metadata mismatch '+k)
        req(not fit_mask[ix].any(),'Target in core fit '+k)
        elig=pool[(pool.drug==row.drug)&(pool.cell_line==row.cell_line)];target=float(row.dose);ds=elig.dose.astype(float);same=int((ds==target).sum());higher=int((ds>target).sum());lower=int((ds<target).sum())
        req((same==0 and higher==0 and lower>0) if 'OOD' in row.split else same>0,'Dose split support mismatch '+k)
        for suffix in ('_source','_control','_lower','_same'):
            if k+suffix not in idx:
                req(suffix not in ('_source','_control'),'Missing control/source '+k);continue
            r=idx[k+suffix];idx_check(r,len(obs),k+suffix);req(fit_mask[r].all(),'Reference outside fit '+k+suffix);t=obs.iloc[r]
            req((t.cell_line==row.cell_line).all(),'Reference cell line mismatch')
            if suffix in ('_source','_control'):
                req((t.drug.str.casefold()==ctrl.casefold()).all(),'Non-control reference '+k+suffix)
                if suffix=='_source':req(len(r)==n_pred,'Wrong source size '+k)
            else:
                req((t.drug==row.drug).all(),'Reference drug mismatch')
                if suffix=='_lower':req(lower>0 and (t.dose.astype(float)==ds[ds<target].max()).all(),'Wrong lower dose '+k)
                else:req((t.dose.astype(float)==target).all(),'Wrong same dose '+k)
        rows.append(dict(query_id=k,split=row.split,drug=row.drug,dose=row.dose,cell_line=row.cell_line,n_real=len(ix),fit_same=same,fit_higher=higher,fit_lower=lower))
    return held,rows

def normalized_hashes(mapping):
    # Original Windows runner saved str(relative_path), hence backslashes in JSON.
    result={str(k).replace(chr(92),'/'):v for k,v in mapping.items()}
    req(len(result)==len(mapping),'Duplicate normalized manifest path');return result

def check_selection(bench,sel,lock,vq,lh):
    req(sel['lock_sha256']==lh,'Selection lock mismatch'); hashes=normalized_hashes(sel['hashes']);stages=sorted({int(m.group(1)) for key in hashes if (m:=re.search(r'/validation_(\d+)\.csv$',key))})
    req(bool(stages) and set(stages)<=set(lock['stages']),'Selection stages mismatch')
    eligible=sorted(map(int,sel['eligible_stages']));req(set(eligible)<=set(stages),'Eligible validation missing')
    required=lock['stages'][:-1] if lock['stages']==[2000,10000,30000,50000] else lock['stages'];req(set(required)<=set(stages),'Required validation stages absent')
    rows=[];qkeys=set(vq.query_id);splitmap=dict(zip(vq.query_id,vq.split))
    for m,s,t in itertools.product(MODELS,SEEDS,stages):
        rel=f'models/{m}/seed_{s}/validation_{t}.csv';p=bench/rel;verify(p,hashes[rel]);v=pd.read_csv(p)
        req(len(v)==len(vq) and set(v.query_id)==qkeys and not v.query_id.duplicated().any() and (v.model==m).all() and (v.seed==s).all() and (v.step==t).all() and (v.query_id.map(splitmap)==v.split).all() and np.isfinite(v.W1_DEG50_exact).all() and (v.W1_DEG50_exact>=0).all(),'Validation table invalid '+rel)
        rows.append(v)
    val=pd.concat(rows,ignore_index=True);curve=val[val.split=='val_OOD'].groupby(['model','step']).W1_DEG50_exact.mean().unstack()
    if lock['stages']==[2000,10000,30000,50000]:
        need=bool((curve[30000]<.99*curve[[2000,10000]].min(axis=1)).any());req(not need or 50000 in stages,'50k extension triggered but missing')
        expected=[t for t in stages if need or t!=50000]
        req(expected==eligible,'Eligible stages differ from fixed rule')
    else:req(stages==eligible,'Eligible stages differ')
    curve=curve[eligible];own={m:int(min(v[v<=v.min()*1.01].index)) for m,v in curve.iterrows()};step=max(own.values())
    req(own==sel['per_model_near_best'] and step==sel['chosen_step'],'Selected budget does not reproduce')
    return step, pd.DataFrame([dict(model=m,own_near_best_step=own[m],best_exact_step=int(curve.loc[m].idxmin()),common_step=step,relative_W1_increase_at_common=float(curve.loc[m,step]/max(float(curve.loc[m].min()),1e-12)-1),boundary_optimum=int(curve.loc[m].idxmin())==max(eligible)) for m in MODELS]),val

COND_PREFIX=('source_to_hidden.','drug_emb.','dose_mlp.','line_emb.','core.DisentanglementEncoder.exogenous_encoder_m_v.')
def inspect_checkpoints(jobs,skip=False):
    if skip:return [{'condition_sha256':None,'dag_checked':False,'checkpoint_config_checked':False} for _ in jobs]
    worker=Path(__file__).resolve().with_name('checkpoint_worker.py')
    req(worker.is_file(),'Missing checkpoint_worker.py')
    env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[k]='1'
    result=subprocess.run([sys.executable,str(worker)],input=json.dumps(jobs).encode('utf8'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
    try: data=json.loads(result.stdout.decode('utf8'))
    except Exception: raise ValueError('Checkpoint worker failed (exit %s): %s'%(result.returncode,result.stderr.decode('utf8',errors='replace')))
    req(result.returncode==0 and 'results' in data,'Checkpoint worker failed: '+str(data.get('error',result.stderr.decode('utf8',errors='replace'))))
    req(len(data['results'])==len(jobs),'Incomplete worker results')
    return data['results']

def inspect_checkpoint(cp,cfg,m,skip=False):
    return inspect_checkpoints([{'path':str(cp),'cfg':cfg,'model':m}],skip)[0]

def b_audit(a,dest,profile=PROFILE):
    report,vocab,frames,allobs,genes,cache_hashes=load_meta(a.cache);obs=frames['train'];n=len(obs);bench=a.bench;parent=a.parent
    lock=read(bench/'lock.json');pl=read(parent/'lock.json');lh=sha(bench/'lock.json');sel=read(bench/'selection.json')
    req(lock['schema']=='training_sufficiency_v2' and pl['schema']=='locked_retrospective_v1','Unexpected lock schema')
    req(lock['cache_hashes']==cache_hashes==pl['cache_hashes'],'Cache metadata fingerprint differs')
    verify(parent/'lock.json',lock['parent_lock_sha256']);verify(parent/'indices.npz',pl['indices_sha256']);req(lock['parent_indices_sha256']==pl['indices_sha256'],'Parent index mismatch');verify(parent/'queries.csv',pl['queries_sha256'])
    for nm,key in [('indices.npz','indices_sha256'),('queries.csv','queries_sha256'),('validation_queries.csv','validation_queries_sha256')]:verify(bench/nm,lock[key])
    req(lock['queries_sha256']==pl['queries_sha256'],'Parent queries changed')
    req(set(lock['source_hashes'])=={'run_sufficiency.py','run_professional.py','run_locked_benchmark.py','response_model.py','factor_ablation_model.py','factor_base_pilot.py','matched_diffusion.py'},'Unexpected locked source set')
    for nm,h in lock['source_hashes'].items():
        verify(safe_child(a.source,nm),h)
        ref=Path(__file__).resolve().parent/'locked_source_reference'/nm
        req(ref.read_text(encoding='utf-8-sig')==safe_child(a.source,nm).read_text(encoding='utf-8-sig'),'Locked local source differs from reviewed archive (apart from newline/BOM): '+nm)
        # Raw local SHA is checked above; text equality tolerates Windows line endings.
    expected_core={'Modules.py','utils.py','Dataset.py','cpa_pipeline.py','cpa_adapter.py','__init__.py'}
    expected_src={'run_locked_benchmark.py','response_model.py','factor_ablation_model.py','factor_base_pilot.py'}
    req(set(pl['code_hashes'])==expected_core|expected_src,'Parent source set mismatch')
    for nm,h in pl['code_hashes'].items():verify(safe_child(a.source if nm in expected_src else a.root/'CausFlow',nm),h)
    if lock.get('array_sha256'):verify(a.cache/'train_x.npy',lock['array_sha256'])
    req(set(lock['internal_models'])==set(MODELS) and lock['seeds']==list(SEEDS),'Model/seed set mismatch')
    req(len(genes)==profile['n_genes'] and lock['n_pred']==profile['n_pred'],'Unexpected feature/prediction shape')
    q=csv(bench/'queries.csv');vq=csv(bench/'validation_queries.csv');idx=load_indices(bench/'indices.npz');pi=load_indices(parent/'indices.npz');fit=idx['fit'];idx_check(fit,n,'fit',True)
    req(len(fit)==lock['n_fit'] and len(q)==profile['n_queries'] and len(vq)==profile['n_val'],'Locked counts mismatch')
    req(q.drug.nunique()==profile['n_drugs'] and q.cell_line.nunique()==profile['n_lines'] and q.split.value_counts().to_dict()=={'ID':len(q)//2,'dose_OOD':len(q)//2},'Outer profile mismatch')
    req(not q.duplicated(['split','drug','cell_line']).any() and all(len(v)==profile['n_lines'] for _,v in q.groupby(['split','drug'])),'Drug/line coverage mismatch')
    req(not vq.drug.isin(q.drug).any(),'Inner and outer drug identity overlap')
    req(vq.split.value_counts().to_dict()=={'val_ID':len(vq)//2,'val_OOD':len(vq)//2} and not vq.duplicated(['split','drug','cell_line']).any() and all(len(v)==profile['n_lines'] for _,v in vq.groupby(['split','drug'])),'Inner validation condition profile mismatch')
    ctrl=str(report['control_drug']); outer,rows=check_queries(q,obs,idx,fit,ctrl,lock['n_pred']);inner,vr=check_queries(vq,obs,idx,fit,ctrl,lock['val_pred'],True)
    oi=np.concatenate(outer);vi=np.concatenate(inner);allidx=np.concatenate([fit,oi,vi]);req(len(allidx)==n and len(np.unique(allidx))==n,'core/inner/outer not disjoint and exhaustive')
    req(len(vi)==lock['n_inner_val'],'Inner val cell count mismatch');req(np.array_equal(np.sort(pi['fit']),np.sort(np.concatenate([fit,vi]))),'Parent fit differs from child core+inner')
    for k in pi:
        if k!='fit':req(k in idx and np.array_equal(pi[k],idx[k]),'Parent indices/reference changed '+k)
    # np.memmap opens shape/header; no target matrix slice until all provenance checks pass.
    x=np.load(a.cache/'train_x.npy',mmap_mode='r',allow_pickle=False);req(x.shape==(n,len(genes)),'Cache matrix shape mismatch')
    step,decisions,val=check_selection(bench,sel,lock,vq,lh);decisions.to_csv(dest/'training_sufficiency_decisions.csv',index=False);val.to_csv(dest/'validation_curves.csv',index=False)
    # Verify all recorded checkpoint bytes if explicitly requested; selected checkpoints always verified.
    selected_hashes=normalized_hashes(sel['hashes'])
    if a.all_checkpoint_hashes:
        for k,h in selected_hashes.items():
            if k.endswith('.pt'):verify(safe_child(bench,k),h)
    pred={};configs=[];jobs=[];qidset=set(q.query_id);sh=sha(bench/'selection.json')
    for m,s in itertools.product(MODELS,SEEDS):
        mdir=bench/'models'/m/f'seed_{s}';rel=f'models/{m}/seed_{s}/step_{step}.pt';cp=bench/rel;cfg=read(mdir/f'config_{step}.json');ch=selected_hashes[rel];verify(cp,ch)
        req(all(cfg[k]==v for k,v in dict(model=m,seed=s,step=step,lock_sha256=lh,train_samples=len(fit),train_draws=step*lock['batch_size'],source='core fit only',objective=EXPECTED_OBJECTIVES[m]).items()),'Checkpoint config mismatch '+m)
        d=bench/'predictions'/m/f'seed_{s}';mf=read(d/'manifest.json');req(all(mf[k]==v for k,v in dict(model=m,seed=s,lock_sha256=lh,selection_sha256=sh,checkpoint_sha256=ch,target_expression_as_inference_input=False).items()),'Prediction manifest mismatch '+str(d));req(mf['target_expression_as_inference_input'] is False,'Target input flag invalid');req(set(mf['files'])==qidset,'Prediction set incomplete')
        for k,h in mf['files'].items():
            p=d/f'{k}.npy';verify(p,h);y=np.load(p,mmap_mode='r',allow_pickle=False)
            req(y.shape==(profile['n_pred'],profile['n_genes']) and np.issubdtype(y.dtype,np.floating) and np.isfinite(y).all(),'Invalid prediction '+str(p));pred[m,s,k]=p
        jobs.append({'path':str(cp),'cfg':cfg,'model':m})
        configs.append({**cfg,'checkpoint_sha256':ch,'manifest_sha256':sha(d/'manifest.json')})
    print('CHECKPOINT WORKER: isolated CPU process',flush=True)
    for config,info in zip(configs,inspect_checkpoints(jobs,a.skip_checkpoint_tensors)):
        config.update(info);print('VERIFIED',config['model'],config['seed'],flush=True)
    c=pd.DataFrame(configs);req(c.n_params.nunique()==1,'Trainable parameter counts differ')
    if not a.skip_checkpoint_tensors:req(c.state_shape_sha256.nunique()==1,'State architecture mismatch')
    compare=[]
    for s in SEEDS:
        for m in MODELS[1:]:
            sub=c.set_index(['model','seed']);v=sub.loc[('E1',s),'condition_sha256'];w=sub.loc[(m,s),'condition_sha256']
            compare.append(dict(seed=s,comparison='E1-'+m,condition_parameters_identical=None if v is None or w is None else bool(v==w)))
    c['generation_network_evaluations_per_batch']=c.model.map({m:(1 if m=='NonFlow' else lock.get('ode_steps')) for m in MODELS});c.to_csv(dest/'checkpoint_and_prediction_audit.csv',index=False);pd.DataFrame(compare).to_csv(dest/'representation_match.csv',index=False);pd.DataFrame(rows+vr).to_csv(dest/'split_reference_audit.csv',index=False)
    status={'status':'PASS_WITH_LIMITATIONS' if not a.skip_checkpoint_tensors else 'CONDITIONAL_PASS_TENSORS_NOT_CHECKED','n_predictions':len(pred),'lock_sha256':lh,'chosen_step':step,'n_fit':len(fit),'n_inner_val':len(vi),'n_outer':len(oi),'n_queries':len(q),'n_genes':len(genes),'n_drugs':q.drug.nunique(),'source_hashes_verified':True,'checkpoint_dags_checked':not a.skip_checkpoint_tensors,'checkpoint_worker_sha256':sha(Path(__file__).resolve().with_name('checkpoint_worker.py')) if not a.skip_checkpoint_tensors else None,'array_hash_locked':bool(lock.get('array_sha256')),'all_historical_checkpoint_hashes_verified':bool(a.all_checkpoint_hashes),'scope':lock['scope'],'feature_policy':lock['feature_policy'],
        'limitations':['Retrospective outer queries already evaluated; not a new independent test.','Global inherited gene selection not certified train-only.','Hashes establish integrity of recorded artifacts, not independently proven absence of all preprocessing leakage.','Matched architecture with independently optimized condition parameters; not shared frozen representation.','E1/E4 use source-anchored flow, MatchedDiffusion uses Gaussian-noise DDIM, NonFlow uses one-shot residual MSE; compare complete generator formulations, not a pure solver substitution.','No validated biological causal graph or drug-level OOD claim.','No timing claim unless separately measured.'], 'score_ready':not a.skip_checkpoint_tensors}
    dump(dest/'b_audit_report.json',status);return status,q,idx,x,pred

def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else np.nan
def r2(a,b):
    den=float(np.sum((a-a.mean())**2));return 1-float(np.sum((a-b)**2))/den if den>1e-12 else np.nan

def rank_deg(real,ctl):
    delta=real.mean(0)-ctl.mean(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');p=ttest_ind(real,ctl,axis=0,equal_var=False).pvalue
    return np.lexsort((-np.abs(delta),np.where(np.isfinite(p),p,1.)))[:min(50,len(delta))]
def metrics(real,ctl,y,deg):
    true=real.mean(0)-ctl.mean(0);pr=y.mean(0)-ctl.mean(0)
    return dict(W1_DEG50_exact=float(np.mean([wasserstein_distance(real[:,i],y[:,i]) for i in deg])),response_pcc_deg50=corr(true[deg],pr[deg]),response_rmse_deg50=float(np.sqrt(np.mean((pr[deg]-true[deg])**2))),response_pcc_all=corr(true,pr),response_rmse_all=float(np.sqrt(np.mean((pr-true)**2))),R2_mean_all=r2(real.mean(0),y.mean(0)),R2_var_all=r2(real.var(0),y.var(0)))
def bh(p):
    p=np.asarray(p,float);o=np.argsort(p,kind='stable');q=np.empty_like(p);q[o]=np.minimum(1,np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return q

def signflip(x):
    x=np.asarray(x,float);req(len(x)<=18,'Too many drugs for exact enumeration');s=np.asarray(list(itertools.product([-1,1],repeat=len(x))),dtype=float);return float(np.mean(np.abs(s@x)>=abs(x.sum())-1e-12))

def stats(df,q,dest):
    keys=['split','query_id','drug','cell_line'];req(not df.duplicated(keys+['model','seed','output']).any(),'Duplicate score rows')
    primary=df[df.output=='S4'].copy();req(len(primary)==len(q)*len(MODELS)*len(SEEDS),'Incomplete S4 scores')
    # Pair same query and seed; every finite seed is required before a condition is admitted.
    def finite_mean(v):return float(v.mean()) if len(v)==len(SEEDS) and np.isfinite(v).all() else np.nan
    per=primary.groupby(keys+['model'])[list(METRICS)].agg(finite_mean).reset_index();per.to_csv(dest/'condition_means.csv',index=False)
    primary.groupby(['split','model','seed'])[list(METRICS)].mean().reset_index().to_csv(dest/'seed_summary.csv',index=False)
    summary=per.groupby(['split','model'])[list(METRICS)].agg(['mean','count']);summary.to_csv(dest/'model_summary.csv')
    results=[];drugrows=[];left=per[per.model=='E1'].drop(columns='model')
    for m in MODELS[1:]:
        j=left.merge(per[per.model==m].drop(columns='model'),on=keys,suffixes=('_a','_b'),validate='one_to_one')
        req(len(j)==len(q),'Missing matched queries '+m)
        for split,z in j.groupby('split'):
            for metric,sign in METRICS.items():
                v=(z[metric+'_a']-z[metric+'_b'])*sign;g=pd.DataFrame({'drug':z.drug,'difference':v});good=g.groupby('drug').difference.agg(['mean','count','size']);good=good[good['count']==good['size']];drug=good['mean'].to_numpy(); complete=len(good)==g.drug.nunique() and len(drug)>=2
                for nam,value in good['mean'].items():drugrows.append(dict(split=split,comparison='E1-'+m,metric=metric,drug=nam,benefit=value))
                boot=np.random.default_rng(20261123).choice(drug,(10000,len(drug)),replace=True).mean(1) if len(drug)>=2 else None
                results.append(dict(split=split,comparison='E1-'+m,metric=metric,n_conditions_finite=int(np.isfinite(v).sum()),n_conditions_expected=len(v),n_drugs_complete=len(drug),n_drugs_expected=g.drug.nunique(),complete=complete,mean_benefit=float(drug.mean()) if len(drug) else np.nan,ci_low=np.quantile(boot,.025) if boot is not None else np.nan,ci_high=np.quantile(boot,.975) if boot is not None else np.nan,p_signflip=signflip(drug) if complete else np.nan,primary_family=split=='dose_OOD' and metric in ('W1_DEG50_exact','response_pcc_deg50')))
    r=pd.DataFrame(results);r['p_BH_primary']=np.nan;ix=r.index[r.primary_family];req(len(ix)==6,'Primary family must have six tests');adj=bh(r.loc[ix,'p_signflip'].fillna(1));r.loc[ix,'p_BH_primary']=np.where(r.loc[ix,'complete'],adj,np.nan)
    r.to_csv(dest/'paired_drug_stats.csv',index=False);pd.DataFrame(drugrows).to_csv(dest/'paired_by_drug.csv',index=False)
    return r

def b_score(a,dest,profile=PROFILE):
    status,q,idx,x,pred=b_audit(a,dest,profile);req(status['score_ready'],'Full tensor audit required for scoring')
    rows=[]
    for row in q.to_dict('records'):
        k=row['query_id'];real=np.asarray(x[idx[k]],dtype=np.float64);ctl=np.asarray(x[idx[k+'_control']],dtype=np.float64);req(np.isfinite(real).all() and np.isfinite(ctl).all() and (real>=0).all() and (ctl>=0).all(),'Invalid scored real/control');deg=rank_deg(real,ctl)
        for m,s in itertools.product(MODELS,SEEDS):
            y=np.asarray(np.load(pred[m,s,k],allow_pickle=False),dtype=np.float64)
            for output in ('raw','S4'):
                z=y if output=='raw' else np.where(y>=.01,y,0.)
                rows.append({**row,'model':m,'seed':s,'output':output,**metrics(real,ctl,z,deg)})
        print('SCORED',k,flush=True)
    df=pd.DataFrame(rows);df.to_csv(dest/'metrics_by_condition.csv',index=False);r=stats(df,q,dest)
    dump(dest/'score_provenance.json',{'lock_sha256':status['lock_sha256'],'audit_status':status['status'],'checkpoint_worker_sha256':status['checkpoint_worker_sha256'],'primary_output':'S4','DEG50':'Welch p-value then descending absolute response, evaluation only','primary_family':'3 E1 contrasts x 2 dose_OOD metrics, exploratory retrospective BH; missing slots p=1','unit':'average seeds within query, average conditions within drug; drug-cluster bootstrap and two-sided exact sign flip','n_bootstrap':10000,'bootstrap_seed':20261123,'metric_rows':len(df),'qualification':'No new training, no test-based model selection; no superiority guaranteed.'});print('B_SCORE COMPLETE',flush=True)

def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');p.add_argument('--phase',choices=['c0','b-audit','b-score'],required=True);p.add_argument('--native-root');p.add_argument('--source');p.add_argument('--cache');p.add_argument('--parent');p.add_argument('--bench');p.add_argument('--out',required=True);p.add_argument('--no-native-probe',action='store_true');p.add_argument('--skip-checkpoint-tensors',action='store_true',help='Inventory-only; never use with b-score');p.add_argument('--all-checkpoint-hashes',action='store_true');a=p.parse_args();a.root=Path(a.root).resolve()
    for field,default in [('native_root','.'),('source','CausFlow_TrainingSufficiency'),('cache','Output/CPA_Fig2_4split/cache'),('parent','Output/LockedBenchmark_v1'),('bench','Output/TrainingSufficiency_v2')]:
        v=getattr(a,field);setattr(a,field,Path(v).resolve() if v else (a.root/default).resolve())
    a.out=Path(a.out).resolve();req(not any(a.out.is_relative_to(p) for p in (a.source,a.cache.parent,a.parent,a.bench,a.native_root/'CausFlow',a.root/'CausFlow')),'Output must not be inside original scripts or results');req(not(a.phase=='b-score' and a.skip_checkpoint_tensors),'Do not score without tensor checks');return a

def main():
    a=parse();a.out.mkdir(parents=True,exist_ok=False)
    try:
        if a.phase=='c0':c0(a,a.out)
        elif a.phase=='b-audit':status,*_=b_audit(a,a.out);print(status['status'],flush=True)
        else:b_score(a,a.out)
        dump(a.out/'run_status.json',{'status':'COMPLETE','phase':a.phase,'tool_script_sha256':sha(Path(__file__))});return 0
    except Exception as e:
        dump(a.out/'run_status.json',{'status':'BLOCKED','phase':a.phase,'error':str(e),'traceback':traceback.format_exc()});traceback.print_exc();return 1
if __name__=='__main__':sys.exit(main())
