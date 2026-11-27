"""CPU-only corrected gene mapping + fit-only candidate HVG panel. No target expression."""
import argparse, hashlib, json, re
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csr_matrix

def require(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def strings(d):return np.array([x.decode() if isinstance(x,bytes) else str(x) for x in d[()]])
def axis(f,a,k=None):
    g=f[a]; k=k or g.attrs['_index'];k=k.decode() if isinstance(k,bytes) else k; d=g[k]
    if isinstance(d,h5py.Group):
        c=d['codes'][()];require((c>=0).all(),'Missing axis data');return pd.Index(strings(d['categories'])[c])
    return pd.Index(strings(d))
def corrected_mapping(annotation,old_ids,old_names):
    with gzip.open(annotation,'rt',encoding='utf-8-sig') as f:
        d=pd.read_csv(f,sep=r'\s+',dtype=str,keep_default_na=False)
    require(list(d.columns)==['id','gene_short_name'],'Unexpected GEO annotation header')
    require(len(old_ids)==len(d)+1,'Unexpected annotation length; no automatic remapping')
    require(np.array_equal(np.asarray(old_ids),np.concatenate((['id'],d.id.to_numpy()))),'Old Ensembl ID sequence does not match header-included annotation')
    with gzip.open(annotation,'rt',encoding='utf-8-sig') as f:
        historical=pd.read_csv(f,sep=r'\s+',engine='python',header=None,comment='#').iloc[:,:2].astype(str)
    require(np.array_equal(np.asarray(old_names),historical.iloc[:,1].to_numpy()),'Old symbol sequence does not match historical parser')
    return d

def rank_panel(d,n,s,ss,nrows,min_cells=100,top=5000):
    ids=d.id.to_numpy();eligible=(n>=min_cells)&np.array([bool(re.fullmatch(r'ENSG\d+(?:\.\d+)?',i)) for i in ids])
    ix=np.flatnonzero(eligible); require(len(ix)>=top,f'Only {len(ix)} eligible genes; do not silently lower minimum support')
    require(pd.Index(ids[ix]).is_unique,'Duplicate eligible Ensembl IDs; manual review needed')
    m=s[ix]/nrows;v=np.maximum((ss[ix]-s[ix]**2/nrows)/(nrows-1),0)
    require((m>0).all(),'Eligible gene has zero mean')
    logdisp=np.log(np.maximum(v/m,np.finfo(float).tiny));lm=np.log1p(m)
    t=pd.DataFrame({'raw_col_0based':ix,'gene_id':ids[ix],'gene_name':d.gene_short_name.to_numpy()[ix],'fit_detected':n[ix],'mean_normalized_counts':m,'variance_normalized_counts':v,'log_dispersion':logdisp})
    bins=pd.cut(lm,bins=20,duplicates='drop');g=pd.Series(logdisp).groupby(bins,observed=True);avg=g.transform('mean').to_numpy();sd=g.transform('std').to_numpy();score=np.divide(logdisp-avg,sd,out=np.zeros_like(logdisp),where=np.isfinite(sd)&(sd>0));score[~np.isfinite(sd)]=1.;t['dispersion_z']=score
    t=t.sort_values(['dispersion_z','gene_id','raw_col_0based'],ascending=[False,True,True]);t['selected']=False;t.iloc[:top,t.columns.get_loc('selected')]=True
    return t

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.root).resolve();out=Path(a.out).resolve();require(not out.exists(),'Choose a new output directory');require(out.parent==root/'Output','Output must be a new child of project Output');out.mkdir()
    data=root.parent/'Data';raw=data/'sciplex3_standard/sciPlex3_raw_counts.h5ad';ann=data/'GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz';processed=data/'sciplex3_cpa_like/sciplex3_cpa_like_model_290889_5000hvg.h5ad';cache=root/'Output/CPA_Fig2_4split/cache';bench=root/'Output/TrainingSufficiency_v2';pilot=root/'Output/BR5_Pilot_seed0_v1'
    plan=read_json(pilot/'plan.json');lock=read_json(bench/'lock.json');require(sha(bench/'lock.json')==plan['lock_sha256'],'Lock mismatch');require(sha(bench/'indices.npz')==lock['indices_sha256'],'Indices hash mismatch');require(sha(pilot/'fit_index.npz')==plan['fit_sha256'],'Pilot index hash mismatch')
    with np.load(bench/'indices.npz',allow_pickle=False) as z:fit=z['fit'].copy()
    with np.load(pilot/'fit_index.npz',allow_pickle=False) as z:require(np.array_equal(fit,z['fit']),'Pilot fit mismatch')
    for k in ['train_obs.csv','genes.csv']:
        require(sha(cache/k)==lock['cache_hashes'][k],'Cache metadata changed: '+k)
    obs=pd.read_csv(cache/'train_obs.csv',dtype={'source_cell_id':str},keep_default_na=False);require(len(fit)==220998 and len(np.unique(fit))==len(fit) and fit.min()>=0 and fit.max()<len(obs),'Unexpected fit membership');ids=pd.Index(obs.iloc[fit].source_cell_id);require(ids.is_unique,'Duplicate fit cell IDs')
    manifest=read_json(raw.parent/'source_manifest.json');require(sha(ann)==manifest['inputs']['genes']['sha256'],'GEO gene annotation hash mismatch')
    with h5py.File(raw,'r') as f,h5py.File(processed,'r') as h:
        old=axis(f,'var');oid=axis(f,'var','gene_id');oname=axis(f,'var','gene_name');d=corrected_mapping(ann,oid,oname);require(old.is_unique,'Old symbol index not unique')
        old_panel=axis(h,'var');cols=old.get_indexer(old_panel);require(len(cols)==5000 and (cols>=0).all() and (cols<len(d)).all(),'Historical panel has unmapped columns')
        require(np.array_equal(oid[cols],axis(h,'var','gene_id')),'Processed gene IDs not aligned to raw labels')
        mapdf=pd.DataFrame({'raw_col_0based':cols,'old_index':old_panel,'old_gene_id':oid[cols],'corrected_gene_id':d.id.to_numpy()[cols],'corrected_gene_name':d.gene_short_name.to_numpy()[cols]});mapdf.to_csv(out/'historical_panel_mapping.csv',index=False)
        rows=axis(f,'obs').get_indexer(ids);require((rows>=0).all(),'Fit cells missing from raw');rows.sort();x=f['X'];require(x.attrs['encoding-type'] in ('csr_matrix',b'csr_matrix'),'Raw matrix must be CSR');ptr=x['indptr'][()];ng=len(old)
        def batches():
            for i in range(0,len(rows),1024):
                r=rows[i:i+1024];lo,hi=int(r[0]),int(r[-1])+1;aa,bb=int(ptr[lo]),int(ptr[hi]);z=csr_matrix((x['data'][aa:bb],x['indices'][aa:bb],ptr[lo:hi+1]-aa),shape=(hi-lo,ng))[r-lo];z.sum_duplicates();z.eliminate_zeros();require(np.isfinite(z.data).all() and (z.data>=0).all(),'Invalid raw fit counts');require(z[:,len(d):].nnz==0,'Extra historical column has counts; stop');yield z[:,:len(d)]
        nc=len(rows);tot=np.empty(nc);n=np.zeros(len(d),np.int64);pos=0
        for z in batches():
            size=z.shape[0];tot[pos:pos+size]=np.asarray(z.sum(1)).ravel();pos+=size;n+=np.bincount(z.indices,minlength=len(d))
        require((tot>0).all(),'Zero library cell in fit');target=float(np.median(tot));print('FIT MEDIAN',target,flush=True)
        s=np.zeros(len(d));ss=s.copy();pos=0
        for z in batches():
            size=z.shape[0];v=z.astype(np.float64).multiply((target/tot[pos:pos+size])[:,None]).tocsr();pos+=size;s+=np.asarray(v.sum(0)).ravel();ss+=np.asarray(v.multiply(v).sum(0)).ravel()
        ranked=rank_panel(d,n,s,ss,nc);ranked.to_csv(out/'eligible_gene_ranking.csv',index=False);panel=ranked[ranked.selected].sort_values('raw_col_0based');panel.to_csv(out/'strict_candidate_panel.csv',index=False)
        mapdf['fit_detected']=n[cols];mapdf.to_csv(out/'historical_panel_mapping.csv',index=False)
        result={'status':'CANDIDATE_PANEL_READY_NOT_BENCHMARK_LOCKED','annotation_header_confirmed_in_old_axis':True,'old_gene_columns':ng,'true_annotation_rows':len(d),'fit_cells':nc,'fit_median_full_library':target,'old_panel_never_detected':int((n[cols]==0).sum()),'new_panel_never_detected':int((panel.fit_detected==0).sum()),'new_panel_min_detected':int(panel.fit_detected.min()),'old_new_panel_overlap_columns':int(np.isin(panel.raw_col_0based,cols).sum()),'eligible_human_genes':len(ranked),'selection':'fit-only human ENSG, at least 100 detected cells; Seurat-style log dispersion z-score in 20 mean bins, deterministic ties by gene ID; not claimed exact Scanpy implementation','annotation_sha256':sha(ann),'fit_index_sha256':sha(bench/'indices.npz'),'panel_sha256':sha(out/'strict_candidate_panel.csv'),'target_expression_used_for_feature_selection':False,'note':'No raw H5AD rewritten, no model trained, no test or validation values used in selection. Candidate requires new benchmark lock before training.'}
        (out/'panel_status.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
