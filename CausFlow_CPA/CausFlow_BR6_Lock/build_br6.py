"""BR6.2: build separate strict-panel fit cache; BR6.3: inner-only baselines. CPU only."""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csr_matrix
import panel_utils as u
import locked_metrics as m

EXPECTED_PANEL='6f886ac609dca2bbd1f6809782a6a6201df26a53c8c185e56858bbf208b2dc72'
EXPECTED_INDEX='7ce6ca4f5d1ed5e88755eba3b0099317bb8267945786b69a0507dc449fb63ff1'
REQ=u.require
SHA=u.sha
READ=u.read_json

def dump(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def load_panel(folder, oldidx):
    p=folder/'strict_candidate_panel.csv';s=READ(folder/'panel_status.json')
    REQ(s['status']=='CANDIDATE_PANEL_READY_NOT_BENCHMARK_LOCKED','Unexpected panel status')
    REQ(SHA(p)==s['panel_sha256']==EXPECTED_PANEL,'Panel differs from reviewed BR6.1 panel')
    REQ(SHA(oldidx)==s['fit_index_sha256']==EXPECTED_INDEX,'Fit indices differ from reviewed BR6.1 panel')
    d=pd.read_csv(p,dtype={'gene_id':str,'gene_name':str},keep_default_na=False)
    REQ(len(d)==5000 and d.raw_col_0based.is_unique and d.gene_id.is_unique and d.raw_col_0based.is_monotonic_increasing,'Invalid gene panel')
    REQ(d.gene_id.str.fullmatch(r'ENSG\d+(?:\.\d+)?').all() and (d.fit_detected>=100).all(),'Invalid support or species')
    return d,s

class Raw:
    def __init__(self,f,cols,target):
        self.x=f['X'];self.ptr=self.x['indptr'][()];self.shape=tuple(self.x.attrs['shape']);self.cols=np.asarray(cols,dtype=np.int64);self.target=float(target)
        REQ(self.x.attrs['encoding-type'] in ('csr_matrix',b'csr_matrix'),'Expected CSR raw H5AD')
        REQ(self.cols.ndim==1 and len(np.unique(self.cols))==len(self.cols) and self.cols.min()>=0 and self.cols.max()<self.shape[1]-1,'Bad physical columns')
    def block(self, rows):
        rows=np.asarray(rows,dtype=np.int64);REQ(len(rows)>0 and rows.min()>=0 and rows.max()<self.shape[0],'Bad raw row indices')
        # Sort only for IO; restore requested order, including repeated draws.
        order=np.argsort(rows,kind='stable');r=rows[order];lo,hi=int(r[0]),int(r[-1])+1;aa,bb=int(self.ptr[lo]),int(self.ptr[hi]);x=self.x
        z=csr_matrix((x['data'][aa:bb],x['indices'][aa:bb],self.ptr[lo:hi+1]-aa),shape=(hi-lo,self.shape[1]))[r-lo]
        z.sum_duplicates();z.eliminate_zeros();REQ(np.isfinite(z.data).all() and (z.data>=0).all() and (z.data==np.floor(z.data)).all(),'Invalid UMI counts')
        REQ(z[:,-1:].nnz==0,'Historical extra column is nonzero; stop')
        total=np.asarray(z.sum(1,dtype=np.float64)).ravel();REQ((total>0).all(),'Zero library in requested rows')
        v=z[:,self.cols].toarray().astype(np.float64);v*=self.target/total[:,None]
        inv=np.argsort(order,kind='stable');return v[inv],total[inv]
    def batches(self,rows,chunk=512):
        # Globally sort raw positions to avoid repeatedly reading large CSR spans.
        rows=np.asarray(rows,dtype=np.int64);order=np.argsort(rows,kind='stable')
        sorted_rows=rows[order];start=0
        while start<len(order):
            stop=min(start+chunk,len(order));stop=min(stop,int(np.searchsorted(sorted_rows,sorted_rows[start]+4096,side='left')))
            ix=order[start:stop];v,t=self.block(rows[ix]);yield ix,v,t;start=stop
    def write(self,rows,path,expected=None):
        rows=np.asarray(rows,dtype=np.int64);n=len(rows);g=len(self.cols);mm=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,g));support=np.zeros(g,np.int64);sm=np.zeros(g);ss=np.zeros(g);tot=np.empty(n);s=sq=0.;mx=0.
        for ix,v,t in self.batches(rows):
            support+=(v>0).sum(0);sm+=v.sum(0);ss+=np.square(v).sum(0);tot[ix]=t
            y=np.log1p(v).astype(np.float32);REQ(np.isfinite(y).all() and (y>=0).all(),'Invalid normalized expression');mm[ix]=y;s+=y.sum(dtype=np.float64);sq+=np.square(y,dtype=np.float64).sum();mx=max(mx,float(y.max()))
        mm.flush();del mm
        if expected is not None:
            REQ(np.array_equal(support,expected.fit_detected.to_numpy()),'Fit support differs from reviewed panel')
            REQ(float(np.median(tot))==self.target,'Fit normalization median mismatch')
            np.testing.assert_allclose(sm/n,expected.mean_normalized_counts,rtol=1e-7,atol=1e-10)
            np.testing.assert_allclose(np.maximum((ss-sm**2/n)/(n-1),0),expected.variance_normalized_counts,rtol=1e-6,atol=1e-9)
        mean=s/(n*g);return dict(n_cells=n,n_genes=g,mean=mean,sd=float(np.sqrt(max(sq/(n*g)-mean**2,0))),zero_fraction=float(1-support.sum()/(n*g)),min_detected=int(support.min()),never_detected=int((support==0).sum()),max=mx,median_library=float(np.median(tot)))

def input_meta(root):
    cache=root/'Output/CPA_Fig2_4split/cache';bench=root/'Output/TrainingSufficiency_v2';old=READ(bench/'lock.json');report,vocab,frames,_,_,hashes=m.load_meta(cache)
    REQ(old['cache_hashes']==hashes,'Historical cache metadata changed')
    for k,f in [('indices_sha256','indices.npz'),('queries_sha256','queries.csv'),('validation_queries_sha256','validation_queries.csv')]:m.verify(bench/f,old[k])
    ix=m.load_indices(bench/'indices.npz');obs=frames['train'];fit=ix['fit'];m.idx_check(fit,len(obs),'fit',True)
    qi=m.csv(bench/'validation_queries.csv');qo=m.csv(bench/'queries.csv');ctrl=str(report['control_drug'])
    a,ar=m.check_queries(qi,obs,ix,fit,ctrl,old['val_pred'],True);b,br=m.check_queries(qo,obs,ix,fit,ctrl,old['n_pred'],False)
    ids=np.concatenate([fit,*a,*b]);REQ(len(ids)==len(obs)==len(np.unique(ids)),'Fit/inner/outer overlap or missing parent rows')
    REQ(len(fit)==220998 and len(qi)==24 and len(qo)==48,'Unexpected benchmark size')
    return bench,old,obs,ix,qi,qo,ctrl,vocab,ar+br

def comparator(q,obs,fit,ix):
    k=q.query_id;key=k+('_lower' if 'OOD' in q.split else '_same')
    if key in ix:return ix[key]
    o=obs.iloc[fit];dose=o.dose.astype(float).to_numpy();good=(o.drug==q.drug).to_numpy()&(o.cell_line==q.cell_line).to_numpy();target=float(q.dose)
    if 'OOD' in q.split:
        good &= dose<target;REQ(good.any(),'No nearest lower reference');good &= dose==dose[good].max()
    else:good &= dose==target
    pool=fit[good];REQ(len(pool)>0,'No matching reference');seed=int(hashlib.sha256(('BR6-fixed-nearest|'+k).encode()).hexdigest()[:8],16)
    return np.random.default_rng(seed).choice(pool,len(ix[k+'_source']),replace=True)

def fingerprint_folder(out):
    return {p.relative_to(out).as_posix():SHA(p) for p in sorted(out.rglob('*')) if p.is_file()}

def prepare(root,paneldir,out):
    REQ(not out.exists() and out.parent==root/'Output','Use a fresh direct child of Output')
    bench,old,obs,ix,qi,qo,ctrl,vocab,checks=input_meta(root);panel,status=load_panel(paneldir,bench/'indices.npz');fit=ix['fit'];data=root.parent/'Data';raw=data/'sciplex3_standard/sciPlex3_raw_counts.h5ad';ann=data/'GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz'
    manifest=READ(raw.parent/'source_manifest.json');REQ(SHA(ann)==manifest['inputs']['genes']['sha256']==status['annotation_sha256'],'Annotation changed')
    out.mkdir();print('Hashing raw H5AD for new lock',flush=True);rawhash=SHA(raw)
    with h5py.File(raw,'r') as f:
        corrected=u.corrected_mapping(ann,u.axis(f,'var','gene_id'),u.axis(f,'var','gene_name'));cols=panel.raw_col_0based.to_numpy(np.int64);REQ(np.array_equal(corrected.id.to_numpy()[cols],panel.gene_id.to_numpy()) and np.array_equal(corrected.gene_short_name.to_numpy()[cols],panel.gene_name.to_numpy()),'Panel annotation does not match physical columns')
        axis=u.axis(f,'obs');REQ(axis.is_unique,'Duplicate raw cell IDs');rowmap=axis.get_indexer(pd.Index(obs.source_cell_id));REQ((rowmap>=0).all() and len(np.unique(rowmap))==len(rowmap),'Missing or duplicate raw row mapping')
        rr=Raw(f,cols,status['fit_median_full_library']);print('Writing fit-only matrix',flush=True);stats=rr.write(rowmap[fit],out/'fit_x.npy',panel);print('Fit support and moments matched',flush=True)
    shutil.copyfile(paneldir/'strict_candidate_panel.csv',out/'panel.csv');shutil.copyfile(paneldir/'panel_status.json',out/'panel_status.json');shutil.copyfile(bench/'indices.npz',out/'historical_indices.npz')
    qi.to_csv(out/'inner_queries.csv',index=False);qo.to_csv(out/'outer_queries.csv',index=False);pd.DataFrame(checks).to_csv(out/'split_audit.csv',index=False)
    fobs=obs.iloc[fit].reset_index(drop=True);fobs.to_csv(out/'fit_obs.csv',index=False);pd.DataFrame({'gene':panel.gene_id,'gene_name':panel.gene_name,'raw_col_0based':cols}).to_csv(out/'genes.csv',index=False);dump(out/'vocab.json',vocab)
    np.savez(out/'fit_index.npz',fit=fit,labels=fobs[['drug_idx','dose_idx','cell_line_idx']].to_numpy(np.int64),control=fobs.drug.str.casefold().eq(ctrl.casefold()).to_numpy());dump(out/'fit_scale.json',stats)
    # Store row mappings and query target IDs, not target values. Training programs must not load score_indices or raw H5AD.
    lookup=np.full(len(obs),-1,np.int64);lookup[fit]=np.arange(len(fit));refs={};targets={};mm=np.load(out/'fit_x.npy',mmap_mode='r')
    for split,qs in [('inner',qi),('outer',qo)]:
        for kind in ('sources','controls','nearest'):(out/split/kind).mkdir(parents=True)
        for q in qs.itertuples(index=False):
            k=q.query_id;targets[k]=rowmap[ix[k]];refs[k+'_source']=ix[k+'_source'];refs[k+'_control']=ix[k+'_control'];refs[k+'_nearest']=comparator(q,obs,fit,ix)
            for kind,suf in [('sources','_source'),('controls','_control'),('nearest','_nearest')]:
                jj=lookup[refs[k+suf]];REQ((jj>=0).all(),'Reference outside fit');np.save(out/split/kind/(k+'.npy'),mm[jj],allow_pickle=False)
    del mm;np.savez_compressed(out/'score_indices.npz',**targets);np.savez_compressed(out/'reference_indices.npz',**refs)
    dump(out/'lock.json',dict(schema='BR6_strict_panel_v2',status='LOCKED_FIT_AND_REFERENCES_ONLY',raw_h5ad=str(raw),raw_sha256=rawhash,annotation_sha256=SHA(ann),parent_lock_sha256=SHA(bench/'lock.json'),panel_sha256=SHA(out/'panel.csv'),normalization_target=status['fit_median_full_library'],control_drug=ctrl,n_fit=len(fit),n_genes=len(panel),n_inner=len(qi),n_outer=len(qo),target_expression_exported=False,feature_policy=status['selection'],scope='retrospective historical queries, NOT independent outer test; only fit used for panel/normalization; no target values exported',code={n:SHA(Path(__file__).parent/n) for n in ('build_br6.py','panel_utils.py','locked_metrics.py')},files=fingerprint_folder(out)))
    dump(out/'build_status.json',dict(status='COMPLETE',lock_sha256=SHA(out/'lock.json'),fit_scale=stats));print('BR6 LOCK COMPLETE',flush=True)

def verify(out):
    lock=READ(out/'lock.json');REQ(lock['schema']=='BR6_strict_panel_v2','Wrong schema');REQ(SHA(out/'lock.json')==READ(out/'build_status.json')['lock_sha256'],'Lock changed')
    for n,h in lock['code'].items():m.verify(Path(__file__).parent/n,h)
    for n,h in lock['files'].items():m.verify(m.safe_child(out,n),h)
    return lock

def baselines(out):
    lock=verify(out);dest=out/'inner_baseline_score';REQ(not dest.exists(),'Score folder exists; preserve historical output');raw=Path(lock['raw_h5ad']);m.verify(raw,lock['raw_sha256']);panel=pd.read_csv(out/'panel.csv');qs=m.csv(out/'inner_queries.csv');dest.mkdir();(dest/'targets').mkdir();scores=[]
    with h5py.File(raw,'r') as f,np.load(out/'score_indices.npz',allow_pickle=False) as ix:
        rr=Raw(f,panel.raw_col_0based.to_numpy(),lock['normalization_target'])
        for q in qs.to_dict('records'):
            k=q['query_id'];path=dest/'targets'/(k+'.npy');rr.write(ix[k],path);real=np.load(path).astype(np.float64);ctl=np.load(out/'inner/controls'/(k+'.npy')).astype(np.float64);deg=m.rank_deg(real,ctl)
            for name,folder in [('No-change','sources'),('Same/nearest-dose','nearest')]:
                y=np.load(out/'inner'/folder/(k+'.npy')).astype(np.float64);scores.append({**q,'model':name,'output':'raw',**m.metrics(real,ctl,y,deg)})
    d=pd.DataFrame(scores);REQ(len(d)==2*len(qs),'Missing scores');d.to_csv(dest/'metrics_by_condition.csv',index=False);d.groupby(['split','model'])[list(m.METRICS)].agg(['mean','count']).to_csv(dest/'summary.csv');dump(dest/'score_status.json',dict(status='COMPLETE',lock_sha256=SHA(out/'lock.json'),n_queries=len(qs),n_scores=len(d),scope='inner only; target values read here solely for scoring',target_hashes={p.name:SHA(p) for p in (dest/'targets').glob('*.npy')}));print('BR6 INNER BASELINES COMPLETE',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=['prepare','verify','inner-baselines']);p.add_argument('--root',default='.');p.add_argument('--panel',default='Output/BR6_StrictPanel_v1');p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.root).resolve();out=Path(a.out).resolve();panel=(root/a.panel).resolve()
    if a.phase=='prepare':prepare(root,panel,out)
    elif a.phase=='verify':verify(out);print('BR6 LOCK VERIFIED',flush=True)
    else:baselines(out)
if __name__=='__main__':main()
