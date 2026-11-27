"""Use original successful CPA/scGen adapters, including the verified CPA save guard."""
import argparse,sys,importlib.util
from pathlib import Path
from run_afinal import HERE,verify,req,sha,check,read,dump,pred_receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--model',required=True,choices=['CPA','scGen_LowerDose']);p.add_argument('--mode',required=True,choices=['smoke','run']);p.add_argument('--seeds',nargs='+',type=int,choices=[0,1,2],default=[0]);a=p.parse_args();out=Path(a.out).resolve();l,m,_,_=verify(out);req(len(set(a.seeds))==len(a.seeds),'Duplicate seeds')
    sys.path.insert(0,str(HERE/'reference/original_locked_source'));import run_professional as r
    v=r.versions();req(v['torch']=='2.0.0+cu118' and v.get('scvi-tools')=='0.20.3' and v.get('anndata')=='0.9.2' and v.get('numpy')=='1.23.5' and all(v.get(k)==w for k,w in {'scanpy':'1.9.8','pytorch-lightning':'1.9.5','lightning':'2.2.5','torchmetrics':'0.11.4'}.items()) and v['cuda_available'],'Use original verified GPU environment')
    cls=r.check_upstream(a.model)
    if a.model=='CPA':
        sys.path.insert(0,str(HERE/'reference'));from cpa_save_repair import install_save_guard;install_save_guard(cls)
    # Smoke runs actual native train / save / prediction, without formal outputs.
    if a.mode=='smoke':
        req(a.seeds==[0],'Smoke uses seed 0');fn='fit_cpa' if a.model=='CPA' else 'fit_scgen';original=getattr(r,fn)
        def wrapped(*args,**kw):
            model=original(*args,**kw);dest=args[-2];model.save(str(dest/'model'),overwrite=False,save_anndata=False)
            import torch
            weights=dest/'model/model.pt';req(weights.is_file() and weights.stat().st_size>0,'Missing smoke weights');ck=torch.load(weights,map_location='cpu');req(isinstance(ck,dict) and bool(ck.get('model_state_dict')),'No model_state_dict in smoke file');return model
        setattr(r,fn,wrapped)
    else:
        sm=out/'professional_smoke'/a.model/'seed_0';rec=read(sm/'receipt.json');req(rec['export_sha256']==sha(out/'external_input/manifest.json') and rec['versions']==v and rec['worker_sha256']==sha(Path(__file__)),'Smoke receipt mismatch');check(sm/'model/model.pt',rec['weights_sha256']);check(sm/'train_config.json',rec['config_sha256'])
    queries=__import__('pandas').read_csv(out/'original_queries.csv').query_id.tolist()
    for seed in a.seeds:
        if a.mode=='run':
            names=['CPA','CPA_mean'] if a.model=='CPA' else [a.model];dest=out/'professional_models'/a.model/f'seed_{seed}';paths=[out/'predictions'/n/f'seed_{seed}' for n in names]
            if all((d/'manifest.json').is_file() for d in paths):
                for n in names:pred_receipt(out,n,seed,queries)
                print('SKIP VERIFIED',a.model,seed,flush=True);continue
            req(not dest.exists() and not any(d.exists() for d in paths),'Partial run exists: preserve before retry')
        sys.argv=[str(Path(r.__file__)),'--mode',a.mode,'--out',str(out),'--model',a.model,'--device','cuda','--seeds',str(seed)];r.main()
        if a.mode=='smoke':
            sm=out/'professional_smoke'/a.model/'seed_0';dump(sm/'receipt.json',dict(export_sha256=sha(out/'external_input/manifest.json'),versions=v,worker_sha256=sha(Path(__file__)),weights_sha256=sha(sm/'model/model.pt'),config_sha256=sha(sm/'train_config.json')));print('A-FINAL NATIVE SAVE SMOKE PASS',a.model,flush=True)
        else:
            for n in names:pred_receipt(out,n,seed,queries)
            print('A-FINAL NATIVE COMPLETE',a.model,seed,flush=True)
if __name__=='__main__':main()
