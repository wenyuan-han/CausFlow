"""CPA Figure-2-like preparation and evaluation, with OOD isolation.

Requires a verified *preprocessed* h5ad (aligned 5k HVGs, log1p values).
Never guesses the 36 held-out drugs: supply an audited split or manifest.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from numpy.lib.format import open_memmap

FACTORS = ('drug','dose','cell_line')
SPLITS = ('train','val','test','ood')


def json_dump(path,obj):
    Path(path).parent.mkdir(exist_ok=True,parents=True)
    with open(path,'w',encoding='utf8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def normalize_dose(value,unit='um'):
    """Return canonical micromolar string; supports numeric nM/uM input."""
    s=str(value).strip().lower().replace('μ','u').replace('µ','u').replace(' ','')
    if s in {'nan','none','', 'vehicle','dmso'}: return '0'
    m=re.fullmatch(r'([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?)\s*(nm|um|mm)?',s)
    if not m:
        raise ValueError(f'Unrecognized dose {value!r}; clean metadata before use')
    number=float(m.group(1))
    if number<0 or not np.isfinite(number):raise ValueError(f'Invalid dose: {value}')
    measured_unit=m.group(2) or unit
    factor={'nm':1e-3,'um':1.,'mm':1e3}[measured_unit]
    return f'{number*factor:.10g}'


def _metadata(adata,args):
    for col in (args.drug_col,args.dose_col,args.line_col):
        if col not in adata.obs:
            raise ValueError(f'Missing obs column {col!r}; available: {list(adata.obs.columns)}')
    obs=pd.DataFrame(index=np.arange(adata.n_obs))
    if adata.obs[[args.drug_col,args.line_col]].isna().any().any():
        raise ValueError('Missing drug or cell-line labels in input obs')
    obs['source_cell_id']=adata.obs_names.astype(str).to_numpy()
    obs['drug']=adata.obs[args.drug_col].astype(str).to_numpy()
    obs['dose']=[normalize_dose(x,args.dose_unit) for x in adata.obs[args.dose_col]]
    obs['cell_line']=adata.obs[args.line_col].astype(str).to_numpy()
    missing_drug_dose=(obs.drug.str.casefold()!=args.control_drug.casefold()) & (obs.dose=='0')
    if missing_drug_dose.any():
        raise ValueError(f'Non-control drug cells with absent/zero dose: {int(missing_drug_dose.sum())}; clean metadata before training.')
    if obs[['drug','dose','cell_line']].isna().any().any():
        raise ValueError('NaN drug/dose/cell_line labels')
    return obs


def _assign_splits(obs,adata,args):
    if args.split_col in adata.obs.columns:
        mapping={'train':'train','training':'train','val':'val','validation':'val',
                 'test':'test','ood':'ood','out-of-distribution':'ood',
                 'out_of_distribution':'ood'}
        raw=adata.obs[args.split_col].astype(str).str.strip().str.lower()
        obs['split']=raw.map(mapping).to_numpy()
        if obs.split.isna().any():
            raise ValueError(f'Unknown split values: {raw[obs.split.isna()].unique()}; allowed labels are train/val/test/ood.')
        split_source=f'obs.{args.split_col}'
    else:
        if not args.ood_drugs:
            raise ValueError('STOP: No audited split and no --ood-drugs manifest. Never infer held-out drugs from OOD expression.')
        path=Path(args.ood_drugs)
        if not path.is_file():raise FileNotFoundError(path)
        drugs=set(x.strip() for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip() and not x.startswith('#'))
        if len(drugs)!=36 and not args.allow_protocol_deviation:
            raise ValueError(f'CPA Figure 2 expects 36 OOD drugs; manifest has {len(drugs)}. Use --allow-protocol-deviation for a pilot.')
        if not drugs.issubset(set(obs.drug)):
            raise ValueError(f'OOD drug names not found in data: {sorted(drugs-set(obs.drug))[:20]}')
        is_ood=obs.drug.isin(drugs)&(obs.dose.astype(float)==10.)
        obs['split']=np.where(is_ood,'ood','train')
        # When no pre-existing split is supplied, make deterministic
        # cell-level train/val/test partitions within each observed ID
        # drug-dose-cell-line condition. OOD conditions remain untouched.
        rng=np.random.default_rng(args.split_seed)
        groups=obs.index[obs.split=='train']
        train_df=obs.loc[groups]
        for _,indices in train_df.groupby(list(FACTORS),sort=True).indices.items():
            actual=np.asarray(train_df.index.to_numpy()[np.asarray(indices)],dtype=object)
            if len(actual)<3:
                continue
            perm=rng.permutation(actual)
            nval=max(1,int(np.floor(len(actual)*args.val_fraction)))
            ntest=max(1,int(np.floor(len(actual)*args.test_fraction)))
            while nval+ntest>=len(actual):
                if ntest>1: ntest-=1
                elif nval>1: nval-=1
                else: break
            obs.loc[perm[:nval],'split']='val'
            obs.loc[perm[nval:nval+ntest],'split']='test'
        split_source=f'manifest:{path} (condition-stratified cell split; seed {args.split_seed})'
    return split_source


def _audit(obs,adata,args,split_source):
    if any((obs.split==s).sum()==0 for s in SPLITS):
        raise ValueError(f'All train, val, test and ood must be nonempty: {obs.split.value_counts().to_dict()}')
    if obs.source_cell_id.duplicated().any():
        raise ValueError('Duplicate cell IDs: cannot certify split isolation')
    conds={s:set(map(tuple,obs.loc[obs.split==s,list(FACTORS)].itertuples(index=False,name=None))) for s in SPLITS}
    if any(conds['ood'] & conds[s] for s in ('train','val','test')):
        raise ValueError('OOD condition overlaps with train, validation, or test; STOP')
    train_conditions=conds['train']
    for split_name in ('val','test'):
        missing_id_conditions=conds[split_name]-train_conditions
        if missing_id_conditions:
            raise ValueError(f'{split_name} contains ID conditions absent from training; this is not a cell-held-out ID split. Examples: {sorted(missing_id_conditions)[:5]}')
    cn=obs[obs.split=='ood'].drop_duplicates(list(FACTORS))
    ood_drugs=set(cn.drug)
    train_drugs=set(obs.loc[obs.split=='train','drug'])
    train_doses=set(obs.loc[obs.split=='train','dose'])
    if not ood_drugs <= train_drugs:
        raise ValueError('OOD has entirely unseen drug(s); this is not CPA Figure 2 known-drug dose generalization.')
    if not set(cn.dose)<=train_doses:
        raise ValueError('OOD dose category is absent from training. A categorical dose embedding cannot predict it. Provide training examples at that dose from other drugs.')
    ctrl=obs.drug.str.casefold()==args.control_drug.casefold()
    counts=obs.loc[(obs.split=='train')&ctrl].groupby('cell_line').size()
    no_controls=sorted(set(cn.cell_line)-set(counts.index))
    if no_controls:
        raise ValueError(f'No matched train controls in cell lines {no_controls}. Set --control-drug correctly.')
    if (counts.reindex(sorted(set(cn.cell_line)),fill_value=0)<2).any():
        raise ValueError('Need at least 2 training control cells per OOD cell line.')
    per_condition=obs.groupby(['split',*FACTORS],sort=True).size().rename('n_cells').reset_index()
    assert not per_condition.duplicated(['split',*FACTORS]).any()
    id_condition_groups=conds['train']|conds['val']|conds['test']
    id_conditions_without_test=id_condition_groups-conds['test']
    test_group_sizes=(obs.loc[obs.split=='test'].groupby(list(FACTORS),observed=True).size())
    test_groups_below_min=int((test_group_sizes<args.min_ood_cells).sum())
    strict_issues=[]
    if adata.n_vars!=5000:strict_issues.append(f'genes={adata.n_vars} instead of 5000')
    if adata.n_obs not in (290888,290889):strict_issues.append(f'cells={adata.n_obs} instead of 290888/290889')
    if set(cn.cell_line)!={'A549','MCF7','K562'}:strict_issues.append(f'OOD cell lines={sorted(set(cn.cell_line))}')
    if len(cn)!=108:strict_issues.append(f'OOD conditions={len(cn)} not 108')
    if len(ood_drugs)!=36:strict_issues.append(f'OOD drugs={len(ood_drugs)} not 36')
    if set(cn.dose)!={'10'}:strict_issues.append(f'OOD doses={sorted(set(cn.dose))} not 10 uM')
    for d in ood_drugs:
        if len(set(cn.loc[cn.drug==d,'cell_line']))!=3:
            strict_issues.append(f'Drug {d}: not all three cell lines have OOD observations')
    if strict_issues and not args.allow_protocol_deviation:
        raise ValueError('STOP: CPA Figure 2 strict-audit failed: '+'; '.join(strict_issues[:12])+'; use --allow-protocol-deviation only for a labeled pilot')
    report={
        'status':'STRICT_AUDIT_PASS' if not strict_issues else 'PILOT_PROTOCOL_DEVIATION',
        'strict_issues':strict_issues,'split_origin':split_source,'n_cells':int(adata.n_obs),
        'n_genes':int(adata.n_vars),
        'n_by_split':{s:int((obs.split==s).sum()) for s in SPLITS},
        'id_condition_groups_without_test':int(len(id_conditions_without_test)),
        'test_condition_groups_below_min_cells':test_groups_below_min,
        'ood_drugs':sorted(ood_drugs), 'ood_conditions':int(len(cn)),
        'ood_control_policy':'TRAIN controls matched by cell line only',
        'warning':'Exact paper half-subsampling, HVG selection and validation split must be independently verified; matching counts does not certify exact CPA preprocessing.',
        'note':'CPA Methods states 290,888 cells; Figure 2 caption says 290,889.'}
    return per_condition,report


def prepare_h5ad(args):
    try:import anndata as ad
    except ImportError as e:raise RuntimeError('pip install anndata') from e
    cache=Path(args.cache_dir) if getattr(args,'cache_dir',None) else Path(args.out)/'cache'
    cache.mkdir(parents=True,exist_ok=True)
    a=ad.read_h5ad(args.data,backed='r')
    if a.n_vars>args.max_input_genes:
        raise ValueError(f'STOP: input has {a.n_vars} genes; build VERIFIED CPA 5000-HVG log1p h5ad first. Do not silently take first 5000 columns.')
    obs=_metadata(a,args)
    source=_assign_splits(obs,a,args)
    per_condition,report=_audit(obs,a,args,source)
    var_names=list(map(str,a.var_names))
    if len(var_names)!=len(set(var_names)):
        raise ValueError('Duplicated gene names in preprocessed matrix')
    report['gene_sha256']=hashlib.sha256('\n'.join(var_names).encode()).hexdigest()
    report['metadata_sha256']=hashlib.sha256(obs.to_csv(index=False).encode()).hexdigest()
    report['input_data_file']=str(Path(args.data).resolve())
    report['input_space']='preprocessed log1p expression (assumed, must audit externally)'
    report['control_drug']=args.control_drug
    report['dose_unit']='uM after canonical conversion'
    vocab={k:sorted(set(obs.loc[obs.split=='train',k])) for k in FACTORS}
    # Use numeric sort for doses to keep map interpretable.
    vocab['dose']=sorted(vocab['dose'],key=float)
    maps={k:{val:i for i,val in enumerate(vocab[k])} for k in FACTORS}
    for name in FACTORS:
        unknown=set(obs.loc[obs.split.isin(['val','test','ood']),name])-set(vocab[name])
        if unknown:raise ValueError(f'Validation/OOD unknown {name} categories {unknown}; no valid embedding')
        obs[name+'_idx']=obs[name].map(maps[name]).astype('int64')
    obs['original_row']=np.arange(a.n_obs,dtype=np.int64)
    # Preserve source-cell ID to inspect possible overlap/duplicates.
    for s in SPLITS:
        obs[obs.split==s].to_csv(cache/f'{s}_obs.csv',index=False)
    per_condition.to_csv(Path(args.out)/'split_audit.csv',index=False)
    pd.DataFrame({'gene':var_names}).to_csv(cache/'genes.csv',index=False)
    json_dump(cache/'vocab.json',vocab)
    json_dump(Path(args.out)/'audit.json',report)

    # Chunk through original data: do not densify all 290k x 5000 at once.
    writers={s:open_memmap(cache/f'{s}_x.npy',mode='w+',dtype='float32',
                           shape=(int((obs.split==s).sum()),a.n_vars)) for s in SPLITS}
    pos={s:0 for s in SPLITS}
    for beg in range(0,a.n_obs,args.prepare_chunk):
        end=min(beg+args.prepare_chunk,a.n_obs)
        block=a.X[beg:end,:]
        if sparse.issparse(block):block=block.toarray()
        block=np.asarray(block,dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f'NaN/Inf expression between original rows {beg}:{end}')
        if block.min()< -1e-6:
            raise ValueError('Negative values found: expects log1p expression, not z-scored/centered matrix.')
        names=obs.split.iloc[beg:end].to_numpy()
        for s in SPLITS:
            take=(names==s)
            n=int(take.sum())
            if n:
                writers[s][pos[s]:pos[s]+n]=block[take]
                pos[s]+=n
        if (beg//args.prepare_chunk)%100==0:
            print(f'cached original cells: {end}/{a.n_obs}',flush=True)
    for s in SPLITS:
        if pos[s]!=writers[s].shape[0]:raise AssertionError('cache count mismatch')
        writers[s].flush()
    if hasattr(a,'file'): a.file.close()
    print('PREPARED:',report['status'],report['n_by_split'])
    return report


def read_cache(out, include_eval=True, cache_dir=None):
    c=Path(cache_dir) if cache_dir else Path(out)/'cache'
    audit_path=Path(out)/'audit.json'
    if not audit_path.is_file() and cache_dir:
        audit_path=c.parent/'audit.json'
    report=json.loads(audit_path.read_text(encoding='utf8'))
    vocab=json.loads((c/'vocab.json').read_text(encoding='utf8'))
    x={s:np.load(c/f'{s}_x.npy',mmap_mode='r') for s in (SPLITS if include_eval else ('train','val'))}
    obs={s:pd.read_csv(c/f'{s}_obs.csv',dtype={'dose':str},keep_default_na=False) for s in SPLITS}
    for s in SPLITS:
        if s in x and len(obs[s])!=len(x[s]):raise ValueError('Cache arrays do not match obs rows')
    return report,vocab,x,obs


def r2(a,b):
    if len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():return float('nan')
    return float(r2_score(a,b))


def _deg_indices(real,control,n=50):
    # Welch t-test, evaluation-only: OOD ground truth used ONLY here.
    with np.errstate(all='ignore'):
        stat,p=ttest_ind(real,control,axis=0,equal_var=False,nan_policy='omit')
    p=np.asarray(p,dtype=float)
    p[~np.isfinite(p)]=1.
    delta=np.abs(real.mean(axis=0)-control.mean(axis=0))
    # p-value primary; effect size tie-break. This is not guaranteed to match
    # CPA notebook's exact DEG ranking; explicit protocol caveat in README.
    ids=np.lexsort((-delta,p))[:min(n,real.shape[1])]
    return np.asarray(ids,dtype=int)


def _wasserstein_quantile_approx(real,pred,quantiles=65):
    q=np.linspace(0,1,quantiles)
    x=np.quantile(real,q,axis=0)
    y=np.quantile(pred,q,axis=0)
    return float(np.abs(x-y).mean())


def condition_metrics(real,pred,control,degs=None):
    from scipy.stats import wasserstein_distance
    if degs is None:degs=_deg_indices(real,control)
    rm,pm=real.mean(axis=0),pred.mean(axis=0)
    rv,pv=real.var(axis=0),pred.var(axis=0)
    out={
        'R2_mean_all':r2(rm,pm),'R2_var_all':r2(rv,pv),
        'R2_mean_DEG50':r2(rm[degs],pm[degs]),
        'R2_var_DEG50':r2(rv[degs],pv[degs]),
        'W1_quantile_all_approx':_wasserstein_quantile_approx(real,pred),
        'W1_DEG50_exact':float(np.mean([wasserstein_distance(real[:,g],pred[:,g]) for g in degs])),
        'n_DEG':int(len(degs)),
        'real_mean_global':float(rm.mean()),'pred_mean_global':float(pm.mean()),
        'real_variance_global':float(rv.mean()),'pred_variance_global':float(pv.mean())}
    return out


def guard_training_only_sources(train_obs,train_x,control_drug):
    idx={}
    for line in sorted(set(train_obs.cell_line)):
        rows=np.flatnonzero((train_obs.cell_line.to_numpy()==line)&
                            (train_obs.drug.astype(str).str.casefold().to_numpy()==control_drug.casefold()))
        if len(rows)<2:raise ValueError(f'Not enough training control cells for {line}: {len(rows)}')
        idx[line]=rows
    return idx
