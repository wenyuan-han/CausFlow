"""Small synthetic fixture, no real-data training. Optional AnnData round-trip."""
import argparse, ast, tempfile,sys
from pathlib import Path
import pandas as pd
import numpy as np
from run_afinal import HERE,original_code,make_obs,mat_csr,reference_rows,paired,req,sha,check
from e1_outer import select_step

def must_fail(fn):
    try:fn()
    except (RuntimeError, ValueError, FileExistsError):return
    raise AssertionError('Expected rejection')
def main():
    p=argparse.ArgumentParser();p.add_argument('--require-anndata',action='store_true');args=p.parse_args();original_code();print('PASS original adapter and actual CPA save guard hashes')
    for n in ('run_afinal.py','e1_outer.py','professional_worker.py','collect_audit.py','test_afinal.py'):
        for v in (9,10):ast.parse((HERE/n).read_text(),feature_version=(3,v))
    print('PASS Python 3.9/3.10 syntax')
    rng=np.random.default_rng(7);x=rng.integers(0,3,(10,53)).astype(np.float32);x[x==1]=0;ix=np.arange(10)*3+10;genes=pd.DataFrame({'gene':[f'ENSG{i}' for i in range(53)]});obs=pd.DataFrame({'source_cell_id':[f'cell{i}' for i in range(10)],'drug':['control']*5+['drug']*5,'dose':['0']*5+['1']*5,'cell_line':['A549']*10});o=make_obs(obs,genes,x,ix);csr=mat_csr(x);np.testing.assert_array_equal(csr.toarray(),x);q=dict(drug='drug',dose='10',cell_line='A549',split='dose_OOD');refs=rng.choice(ix[:5],256,replace=True);ii=pd.Index(ix).get_indexer(refs);y=x[ii];ids=reference_rows(ix,refs,q,obs,x,y,'control');assert ids==obs.iloc[ii].source_cell_id.tolist();must_fail(lambda:reference_rows(ix,refs,q,obs,x,y+1,'control'));must_fail(lambda:reference_rows(ix,refs,{**q,'dose':'1'},obs,x,y,'control'));must_fail(lambda:reference_rows(ix,refs,{**q,'cell_line':'K562'},obs,x,y,'control'));must_fail(lambda:make_obs(obs,genes,x,np.zeros(10)));print('PASS fit CSR equality, exact repeated control IDs, source mismatch, invalid OOD and cell-line rejection')
    d=pd.DataFrame([dict(seed=s,query_id=f'v{q:03d}',step=t,W1_DEG50_exact=.05 if t==2000 else .0498) for s in range(3) for q in range(12) for t in (2000,10000)]);assert select_step(d)[0]==2000;d.loc[d.step==10000,'W1_DEG50_exact']=.04;assert select_step(d)[0]==10000;must_fail(lambda:select_step(d.iloc[:-1]));print('PASS inner 1% earliest checkpoint rule and missing-row rejection')
    df=pd.DataFrame([dict(split=split,seed=s,query_id=f'{split}{j}',drug=f'd{j//3}',model=m,output='raw',W1_DEG50_exact=0.1+(m!='E1')*.02,response_pcc_deg50=.7-(m!='E1')*.1,R2_var_all=.8-(m!='E1')*.2) for split in ['ID','dose_OOD'] for s in range(3) for j in range(24) for m in ['E1','CPA','scGen_LowerDose','No-change','Same/nearest-dose']]);t,g=paired(df,['W1_DEG50_exact','response_pcc_deg50','R2_var_all']);assert len(t)==24 and len(g)==192 and (t.n_positive==8).all() and np.allclose(t.p_BH,.0078125);must_fail(lambda:paired(df.iloc[:-1],['W1_DEG50_exact','response_pcc_deg50','R2_var_all']));print('PASS paired drug sign-flip/bootstrap/BH and incomplete-pair rejection')
    sys.path.insert(0,str(HERE/'reference'));from cpa_save_repair import install_save_guard
    class Fake:
        def save(self,dir_path,overwrite=False,save_anndata=False,**kwargs):
            path=Path(dir_path);path.mkdir();assert overwrite;(path/'model.pt').write_bytes(b'fake test only')
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'model';install_save_guard(Fake);Fake().save(path);h=sha(path/'model.pt');check(path/'model.pt',h);must_fail(lambda:Fake().save(path));(path/'model.pt').write_bytes(b'changed');must_fail(lambda:check(path/'model.pt',h));print('PASS save guard mock and file tamper rejection (not real CPA)')
        try:import anndata as ad
        except ImportError:
            if args.require_anndata:raise
            print('SKIP AnnData round trip: not installed');return
        ad.AnnData(csr,obs=o,var=pd.DataFrame(index=genes.gene.tolist())).write_h5ad(Path(temp)/'fixture.h5ad');a=ad.read_h5ad(Path(temp)/'fixture.h5ad');np.testing.assert_array_equal(a.X.toarray(),x);assert list(a.obs_names)==list(o.index) and list(a.var_names)==genes.gene.tolist();print('PASS AnnData H5AD round-trip')
if __name__=='__main__':main()
