"""Matched MERFISH-Brain reconstruction ablations for revised Table 5.
Run from any directory using --root pointing at CausCell-main.
No input/reference checkpoints are modified. Use --resume for the same V2 run.
"""
import argparse
import gc
import hashlib
import importlib.metadata as md
import inspect
import json
import logging
import math
import platform
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import t

from causcell import CausCell as OriginalCausCell
from causcell_matched import CausCell as MatchedCausCell
from causcell_matched.Modules import VARIANTS, MeanConditionProjection
from causcell_matched.resume import (atomic_json, atomic_write, checked_checkpoint, run_lock, TrainingPaused, sync_dir)
from reference_evaluator import (concept_list, concept_counts, concept_cdag,
                                 METRICS, evaluate_one_setting, set_seed,
                                 sha256, state_fingerprint)

HERE = Path(__file__).resolve().parent
REF = HERE / 'reference'
SETTINGS = {'train_steps': 100000, 'train_lr': 5e-6, 'batch_size': 64,
            'accumulation': 2, 'eval_seed': 2026, 'profile_size': 374,
            'sampling_counts': 10, 'inference_steps': 100, 'sample_batch_size': 128,
            'checkpoint_key': 'model', 'kmeans_random_state': 0}
VARIANT_LABELS = {
    'full': 'Full', 'wo_causal': 'w/o SCM',
    'wo_dis': 'w/o concept-supervised disentanglement',
    'wo_attn': 'w/o cross-attention', 'wo_adv': 'w/o adversarial',
    'wo_sparse': 'w/o sparsity',
}


def dump(path, obj):
    atomic_json(path,obj)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def source_lock():
    files = [HERE/'run_matched_ablation.py', HERE/'reference_evaluator.py']
    for folder in ['causcell', 'causcell_matched', 'reference']:
        files += [f for f in (HERE/folder).rglob('*') if f.is_file() and f.suffix in ('.py', '.json', '.csv')]
    return {str(f.relative_to(HERE)): sha256(f) for f in sorted(files)}


def check_sources():
    expected = load(REF/'source_hashes.json')
    for rel, digest in expected.items():
        if sha256(HERE/rel) != digest:
            raise RuntimeError(f'Pristine Full source changed: {rel}')
    actual_sha = sha256(Path(inspect.getfile(OriginalCausCell)))
    if actual_sha != load(REF/'table6_manifest.json')['source_sha256']:
        raise RuntimeError('Full source does not match Table 6 manifest')
    if Path(inspect.getfile(MatchedCausCell)).resolve() != HERE/'causcell_matched'/'CausCell.py':
        raise RuntimeError('Unexpected matched package imported')


def check_data(root):
    manifest = load(REF/'table6_manifest.json')
    names = {'train':'ID_train','ID':'ID_test','OOD':'OOD_test'}
    paths = {k:root/'Data'/f'transformed_MERFISH_{v}.h5ad' for k,v in names.items()}
    info = {}
    genes = None
    for k, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        if digest != manifest['data'][k]['sha256']:
            raise RuntimeError(f'{k} H5AD SHA256 mismatch: {digest}; expected {manifest["data"][k]["sha256"]}. Do not retrain or transform these splits.')
        a = sc.read_h5ad(path)
        if list(a.shape) != manifest['data'][k]['shape']:
            raise RuntimeError(f'{k}: shape mismatch {a.shape}')
        for c, n in zip(concept_list, concept_counts):
            v = pd.to_numeric(a.obs[c], errors='raise').to_numpy()
            if not np.isfinite(v).all() or not ((v >= 0) & (v < n) & (v == np.floor(v))).all():
                raise RuntimeError(f'{k}: invalid encoded label {c}')
        if genes is None:
            genes = tuple(a.var_names)
        elif tuple(a.var_names) != genes:
            raise RuntimeError(f'{k}: gene order differs from train')
        info[k] = {'path':str(path),'sha256':digest,'shape':list(a.shape)}
        del a
    return paths, info


def full_reference(full_dir, seed, device, load_model=True):
    # Verify actual final weights against the previously uploaded audit, not just directory names.
    audit = pd.read_csv(REF/'table6_checkpoint_audit.csv')
    a = audit[audit.Seed == seed]
    if len(a) != 1:
        raise RuntimeError(f'Missing unique reference checkpoint for seed {seed}')
    a = a.iloc[0]
    ckpt = full_dir/f'seed_{seed}'/'model-10.pt'
    digest = sha256(ckpt)
    if digest != a.FileSHA256:
        raise RuntimeError(f'Full checkpoint SHA256 mismatch: {ckpt}')
    record = {'Variant':'full', 'Seed':seed, 'Checkpoint':str(ckpt), 'FileSHA256':digest,
              'ModelStateSHA256':a.ModelStateSHA256, 'CheckpointStep':100000}
    if not load_model:
        return None, record
    model = OriginalCausCell(device=device)
    model.load_trained(concept_list=concept_list, concept_counts=concept_counts,
                       concept_cdag=concept_cdag, results_folder=str(ckpt.parent),
                       trained_profile_size=374, milestone=10, timesteps=100)
    got = state_fingerprint(model.model)
    if got != a.ModelStateSHA256:
        raise RuntimeError(f'Loaded Full state hash mismatch for seed {seed}')
    model.model.eval()
    return model, record


def verify_variant(model, variant):
    net = model.model.flow_fn
    enc = net.DisentanglementEncoder
    expected = torch.zeros(4,4) if variant == 'wo_causal' else torch.tensor(concept_cdag).float()
    if not torch.equal(enc.causal_dag.detach().cpu(), expected):
        raise RuntimeError(f'{variant}: unexpected DAG')
    if enc.causal_dag.requires_grad:
        raise RuntimeError('DAG must be fixed')
    if getattr(net, 'variant', 'full') != variant:
        raise RuntimeError('Variant flag mismatch')
    if variant != 'full' and getattr(enc, 'variant', None) != variant:
        raise RuntimeError('Encoder variant flag mismatch')
    actual = [isinstance(b.attn1, MeanConditionProjection) for b in net.Cross_attention_module]
    if actual != [variant == 'wo_attn']*len(actual):
        raise RuntimeError('Attention replacement mismatch')
    if not all(torch.isfinite(p).all().item() for p in model.model.parameters()):
        raise FloatingPointError('Nonfinite model weights')
    return {'dag':enc.causal_dag.detach().cpu().tolist(),
            'condition_tokens':4, 'token_width':32,
            'concept_loss_weight':0 if variant=='wo_dis' else 20,
            'adversarial_weight':0 if variant=='wo_adv' else 1,
            'scm_residual_weight':0 if variant=='wo_causal' else 1,
            'sparsity_weight':0 if variant=='wo_sparse' else 0.0001,
            'mean_projection_blocks':sum(actual),
            'trainable_parameters':sum(p.numel() for p in net.parameters() if p.requires_grad)}


def probe_losses(model, train_path, variant):
    # Actual forward pass on a small original-data batch; only diagnostics.
    # RNG is reset again at evaluation, so this does not change the formal evaluator.
    a = sc.read_h5ad(train_path, backed='r')
    try:
        raw = a.X[:16, :374]
        if hasattr(raw, 'toarray'): raw = raw.toarray()
        x = torch.as_tensor(np.asarray(raw, dtype=np.float32), device=model.device)
        y = torch.as_tensor(a.obs.iloc[:16][concept_list].to_numpy(dtype=np.int64), device=model.device)
    finally:
        a.file.close()
    set_seed(2026)
    with torch.no_grad():
        losses = model.model(x, y, torch.ones(len(x), device=x.device))
    scalars = dict(zip(['cfm','scm_residual','concept','adversarial','kl'],[float(q) for q in losses[:5]]))
    if not all(math.isfinite(q) for q in scalars.values()) or not torch.isfinite(losses[5]).all():
        raise FloatingPointError(f'{variant}: nonfinite diagnostic losses')
    key = {'wo_causal':'scm_residual','wo_dis':'concept','wo_adv':'adversarial'}.get(variant)
    if key and scalars[key] != 0: raise RuntimeError(f'{variant}: disabled loss is nonzero')
    scalars['velocity_l1'] = float(torch.norm(losses[5],p=1,dim=1).mean())
    return scalars


def cleanup():
    # The original trainer adds handlers to the root logger. Close them between
    # variants to avoid writing subsequent runs into earlier training logs.
    logger = logging.getLogger()
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fit(root_out, train_path, variant, seed, steps, euler_steps, device, resume_every=1000):
    dest = root_out/variant/f'seed_{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    interval = 10000 if steps == 100000 else steps
    milestone = steps//interval
    set_seed(seed)
    # All variants use identical shared architecture initialization order. Only
    # the new projection parameters in wo_attn are created after original init.
    model = MatchedCausCell(device=device, save_and_sample_every=interval,
                            gradient_accumulate_every=2)
    config = dict(variant=variant, seed=seed, training_num_steps=steps,
                  training_batch_size=64, training_lr=5e-6, timesteps=euler_steps,
                  concept_list=concept_list, concept_counts=concept_counts,
                  concept_cdag=concept_cdag)
    identity=hashlib.sha256(json.dumps({'config':config,'manifest':sha256(root_out/'run_manifest.json'),
                     'data':sha256(train_path),'resume_every':resume_every},sort_keys=True).encode()).hexdigest()
    cfg={**config,'resume_identity':identity,'resume_every':resume_every}
    if (dest/'train_config.json').exists() and load(dest/'train_config.json')!=cfg:
        raise RuntimeError(f'Training config changed: {dest}')
    dump(dest/'train_config.json', cfg)
    model.train(training_data_pwd=str(train_path),model_save_pwd=str(dest),
                resume_identity=identity,resume_every=resume_every,**config)
    cleanup()
    ckpt = dest/f'model-{milestone}.pt'
    raw = checked_checkpoint(ckpt)
    if int(raw['step']) != steps:
        raise RuntimeError(f'Unexpected saved step: {raw["step"]}')
    del raw
    model.model = None
    model.load_trained(concept_list=concept_list, concept_counts=concept_counts,
                       concept_cdag=concept_cdag, results_folder=str(dest),
                       trained_profile_size=374, milestone=milestone, timesteps=euler_steps,
                       variant=variant)
    model.model.eval()
    checks = verify_variant(model, variant)
    checks['post_training_probe'] = probe_losses(model, train_path, variant)
    record = {'Variant':variant,'Seed':seed,'Checkpoint':str(ckpt),'CheckpointStep':steps,
              'FileSHA256':sha256(ckpt),'ModelStateSHA256':state_fingerprint(model.model)}
    dump(dest/'mechanism_audit.json',checks)
    dump(dest/'training_complete.json',record)
    return model, record, dest


def smoke(paths, out, info, lock, args):
    # Intentionally different *diagnostic* subsets; never report these as Table 5.
    subsets = {}
    for k,n in [('train',1024),('ID',256)]:
        a = sc.read_h5ad(paths[k]); rng = np.random.default_rng(3000 if k=='train' else 3001)
        indices = np.sort(rng.choice(a.n_obs, size=n, replace=False))
        small = a[indices].copy()
        subset = out/f'smoke_{k}.h5ad'; small.write_h5ad(subset)
        subsets[k] = subset
        np.save(out/f'smoke_{k}_indices.npy', indices)
        del a,small
    rows = []
    # Test Full plus all five mechanisms. Smoke uses 4 integration steps, 2 draws.
    for v in VARIANTS:
        print(f'\nSMOKE {v}',flush=True)
        model, rec, dest = fit(out,subsets['train'],v,0,args.smoke_steps,4,args.device,resume_every=2)
        set_seed(2026)
        y = model.get_generated_cells(testing_data_pwd=str(subsets['ID']), saved_pwd=str(dest),
                    concept_list=concept_list,concept_counts=concept_counts,concept_cdag=concept_cdag,
                    profile_size=374,sampling_counts=2,sample_batch_size=128)
        y = np.asarray(y)
        if y.shape != (256,374) or not np.isfinite(y).all() or (y<0).any():
            raise RuntimeError(f'{v}: smoke inference failed, shape={y.shape}')
        rows.append({**rec,'Shape':list(y.shape),'Finite':True,'Nonnegative':True,
                     'NonzeroFraction':float(np.count_nonzero(y)/y.size),'Status':'PASS_SMOKE'})
        pd.DataFrame(rows).to_csv(out/'smoke_checkpoint_audit.csv',index=False)
        del model,y
        cleanup()
    if len({r['ModelStateSHA256'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate model state across smoke variants')
    # A zero/nonzero fraction is diagnostic only; short training is not a quality test.
    resume_test=smoke_resume_test(out,subsets['train'],args.device,args.smoke_steps)
    dump(out/'smoke_report.json',{'resume_test':resume_test,'status':'PASS','mode':'smoke','data':info,'source_lock':lock,
                                'steps':args.smoke_steps,'inference_steps':4,'sampling_counts':2,'device':args.device,'torch_version':torch.__version__,'results':rows})
    print('SMOKE PASS. Do not use these short-run results in manuscript.')


def smoke_resume_test(out, train_path, device, steps):
    # One continuous and one intentionally paused run, same data and seed.
    # Compare model+EMA+optimizer+RNG+loader states, not just reported scores.
    from causcell_matched.resume import ResumeStore
    steps=max(3,steps); base=out/'resume_test'
    def train(name,pause=None):
        dest=base/name; dest.mkdir(parents=True,exist_ok=True)
        model=MatchedCausCell(device=device,save_and_sample_every=steps,gradient_accumulate_every=2)
        model.train(training_data_pwd=str(train_path),model_save_pwd=str(dest),concept_list=concept_list,
                    concept_counts=concept_counts,concept_cdag=concept_cdag,variant='wo_adv',seed=0,
                    training_num_steps=steps,training_batch_size=64,training_lr=5e-6,timesteps=4,
                    train_log=False,resume_identity='smoke-resume-'+name,resume_every=2,pause_after_updates=pause)
        return ResumeStore(dest,'smoke-resume-'+name).load()
    a=train('continuous')
    try: train('interrupted',max(1,steps//2))
    except TrainingPaused: pass
    else: raise RuntimeError('Intentional pause not triggered')
    b=train('interrupted')
    def equal(a,b):
        if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    keys=['model','ema','optimizer','scaler','rng','loader','next_step']
    matches={k:equal(a[k],b[k]) for k in keys}
    # GPUs may have nondeterministic kernels. Exact loader/RNG/cursor must match;
    # require exact model/optimizer on this local smoke as conservative gate.
    report={'status':'PASS' if all(matches.values()) else 'FAIL','comparisons':matches,'steps_argument':steps,
            'note':'Same environment, one controlled pause at a completed optimizer update.'}
    dump(base/'resume_test_report.json',report)
    if report['status']!='PASS': raise RuntimeError('Resume smoke mismatch; inspect resume_test_report.json; do not bypass')
    return report

def evaluate(model, path, dest, seed, variant, setting, record=None):
    ed=dest/setting
    staging=dest/(setting+'.partial')
    identity={'checkpoint':record['FileSHA256'],'data':sha256(path),'seed':seed,'variant':variant,'setting':setting} if record else None
    if ed.exists():
        commit=load(ed/'evaluation_complete.json')
        if commit['identity']!=identity or commit['metrics_sha256']!=sha256(ed/'metrics.json'):
            raise RuntimeError(f'Invalid completed evaluation {ed}')
        row=load(ed/'metrics.json')
        if row['Variant']!=variant or row['Seed']!=seed or row['Scenario']!=setting or not all(math.isfinite(float(row[m])) for m in METRICS):
            raise RuntimeError(f'Invalid metrics at {ed}')
        print(f'SKIP completed evaluation {variant}/{seed}/{setting}',flush=True)
        return row
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True)
    model.model.eval(); set_seed(2026)
    row=evaluate_one_setting(model,str(path),f'{variant}_seed{seed}_{setting}',staging)
    if not all(math.isfinite(float(row[m])) for m in METRICS):
        raise RuntimeError(f'Nonfinite formal metric {variant}/{seed}/{setting}: {row}')
    row={k:float(v) for k,v in row.items()}
    row.update(Variant=variant,VariantLabel=VARIANT_LABELS[variant],Seed=seed,Scenario=setting)
    dump(staging/'metrics.json',row)
    dump(staging/'evaluation_complete.json',{'identity':identity,'metrics_sha256':sha256(staging/'metrics.json')})
    staging.rename(ed)
    sync_dir(dest)
    return row


def write_csv(df,path):
    atomic_write(path,lambda f:df.to_csv(f,index=False))


def completed_training(out, variant, seed, device):
    dest=out/variant/f'seed_{seed}'; marker=dest/'training_complete.json'
    if not marker.exists(): return None
    r=load(marker); ckpt=dest/'model-10.pt'
    try:
        d=checked_checkpoint(ckpt)
        if d['step']!=100000 or r['FileSHA256']!=sha256(ckpt) or r['Variant']!=variant or r['Seed']!=seed: raise RuntimeError('Final checkpoint metadata mismatch')
    except Exception as e:
        # A completed evaluation is bound to that precise checkpoint. Never silently replace it.
        if any((dest/k/'evaluation_complete.json').exists() for k in ('ID','OOD')):
            raise RuntimeError(f'Completed evaluation has damaged weight file {ckpt}; restore exact backup') from e
        print(f'Final export invalid; recover from resume slots: {e}',flush=True)
        return None
    del d
    model=MatchedCausCell(device=device)
    model.load_trained(concept_list=concept_list,concept_counts=concept_counts,concept_cdag=concept_cdag,
                       results_folder=str(dest),trained_profile_size=374,milestone=10,timesteps=100,variant=variant)
    model.model.eval(); verify_variant(model,variant)
    if state_fingerprint(model.model)!=r['ModelStateSHA256']: raise RuntimeError('Completed model state mismatch')
    print(f'SKIP completed training {variant}/{seed}',flush=True)
    return model,r,dest

def check_anchor(row, reference):
    r=reference[(reference.Seed==row['Seed']) & (reference.Scenario==row['Scenario'])]
    if len(r)!=1: raise RuntimeError('Missing Table 6 reference row')
    return {m:{'reference':float(r.iloc[0][m]),'recomputed':row[m],
               'delta':row[m]-float(r.iloc[0][m]),
               'match_4dp':f'{row[m]:.4f}'==f'{float(r.iloc[0][m]):.4f}'} for m in METRICS}


def summarize(rows, out, seeds, settings, variants):
    df=pd.DataFrame(rows)
    write_csv(df,out/'metrics_by_seed.csv')
    records=[]; deltas=[]
    for (setting, variant),g in df.groupby(['Scenario','Variant'],sort=False):
        for m in METRICS:
            x=g[m].astype(float); n=len(x); sd=x.std(ddof=1)
            ci=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else np.nan
            records.append(dict(Scenario=setting,Variant=variant,VariantLabel=VARIANT_LABELS[variant],Metric=m,N=n,Mean=x.mean(),SD=sd,CI95_halfwidth=ci))
            if variant!='full':
                full=df[(df.Scenario==setting)&(df.Variant=='full')][['Seed',m]]
                p=g[['Seed',m]].merge(full,on='Seed',suffixes=('_variant','_full'),validate='one_to_one')
                diff=p[m+'_variant']-p[m+'_full']; n2=len(diff); sd2=diff.std(ddof=1)
                deltas.append(dict(Scenario=setting,Variant=variant,Metric=m,N=n2,MeanVariantMinusFull=diff.mean(),SD=sd2,CI95_halfwidth=float(t.ppf(.975,n2-1)*sd2/np.sqrt(n2)) if n2>1 else np.nan))
    s=pd.DataFrame(records); write_csv(s,out/'metrics_summary.csv')
    write_csv(pd.DataFrame(deltas),out/'paired_differences.csv')
    needed={(v,i,k) for v in ('full',*variants) for i in seeds for k in settings}
    observed=set(zip(df.Variant,df.Seed,df.Scenario))
    if len(df)==len(needed) and observed==needed:
        table=[]
        for k in settings:
            for v in ('full',*variants):
                rr=s[(s.Scenario==k)&(s.Variant==v)].set_index('Metric')
                table.append({'Setting':k,'Variant':VARIANT_LABELS[v],**{m:f'{rr.loc[m,"Mean"]:.4f}' if len(seeds)==1 else f'{rr.loc[m,"Mean"]:.4f} ± {rr.loc[m,"SD"]:.4f}' for m in METRICS}})
        write_csv(pd.DataFrame(table),out/('table5_candidate.csv' if set(variants)==set(VARIANTS[1:]) and 'ID' in settings else 'ablation_subset_summary.csv'))
        return True
    return False


def formal(paths, out, info, args, full_dir):
    rows=[]; audits=[]; comparisons=[]
    ref=pd.read_csv(REF/'table6_seedwise.csv')
    variants=args.variants
    total=(1+len(variants))*len(args.seeds)*len(args.settings)
    def current(variant,seed,phase,setting=None):
        dump(out/'current_task.json',dict(variant=variant,seed=seed,phase=phase,setting=setting,\
            processed_evaluations_this_launch=len(rows),total_evaluations=total))
    # Re-evaluate all selected Full anchors before spending compute on ablations.
    for seed in args.seeds:
        current('full',seed,'verify_reference')
        model, rec=full_reference(full_dir,seed,args.device)
        verify_variant(model,'full'); audits.append(rec)
        dest=out/'full'/f'seed_{seed}'; dest.mkdir(parents=True,exist_ok=True)
        for setting in args.settings:
            current('full',seed,'evaluate',setting)
            row=evaluate(model,paths[setting],dest,seed,'full',setting,rec)
            comp=check_anchor(row,ref)
            comparisons.append({'Seed':seed,'Scenario':setting,'metrics':comp})
            dump(out/'full_anchor_comparison.json',comparisons)
            if not all(v['match_4dp'] for v in comp.values()):
                raise RuntimeError('Full re-evaluation differs from Table 6 at four-decimal reporting precision. See full_anchor_comparison.json. No ablation training started. Investigate environment or update both tables under a common evaluator, not just Table 5.')
            rows.append(row)
        del model; cleanup()
        write_csv(pd.DataFrame(audits),out/'checkpoint_audit.csv')
    summarize(rows,out,args.seeds,args.settings,variants)
    for v in variants:
        for seed in args.seeds:
            print(f'\nFORMAL {v} seed={seed}', flush=True)
            current(v,seed,'verify_or_train')
            model, record, dest=completed_training(out,v,seed,args.device) or fit(out,paths['train'],v,seed,100000,100,args.device,args.resume_every)
            if record['ModelStateSHA256'] in {a['ModelStateSHA256'] for a in audits}:
                raise RuntimeError(f'Duplicate trained model state: {v}/{seed}')
            audits.append(record)
            write_csv(pd.DataFrame(audits),out/'checkpoint_audit.csv')
            for setting in args.settings:
                current(v,seed,'evaluate',setting)
                rows.append(evaluate(model,paths[setting],dest,seed,v,setting,record))
                dump(out/'progress.json',{'status':'RUNNING','completed_evaluations':len(rows),'total_evaluations':(1+len(variants))*len(args.seeds)*len(args.settings),'last':{'variant':v,'seed':seed,'setting':setting}})
                summarize(rows,out,args.seeds,args.settings,variants)
            del model; cleanup()
    complete=summarize(rows,out,args.seeds,args.settings,variants)
    dump(out/'formal_status.json',{'status':'COMPLETE' if complete else 'INCOMPLETE',
         'all_five_variants':set(variants)==set(VARIANTS[1:]),'seeds':args.seeds,
         'settings':args.settings,'rows':len(rows),'note':'Reconstruction only; no drug-response inference claim.'})
    if not complete: raise RuntimeError('Missing formal rows')
    dump(out/'progress.json',{'status':'COMPLETE','completed_evaluations':len(rows),'total_evaluations':len(rows)})
    dump(out/'current_task.json',{'phase':'COMPLETE','total_evaluations':len(rows)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, help='Original CausCell-main project root')
    p.add_argument('--mode', required=True, choices=['preflight','smoke','formal'])
    p.add_argument('--out', required=True, help='New output dir, absolute or relative to root')
    p.add_argument('--full-dir',default='Output/SeedFix_Formal')
    p.add_argument('--smoke-report',help='Required for formal; valid PASS smoke_report.json')
    p.add_argument('--smoke-steps',type=int,default=10)
    p.add_argument('--seeds',type=int,nargs='+',default=[0])
    p.add_argument('--settings',nargs='+',choices=['ID','OOD'],default=['ID'])
    p.add_argument('--variants',nargs='+',choices=VARIANTS[1:],default=list(VARIANTS[1:]))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--resume',action='store_true',help='Resume the same locked V2 formal run')
    p.add_argument('--resume-every',type=int,default=1000,help='Completed optimizer updates between rotating saves')
    args=p.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or any(s not in range(5) for s in args.seeds): p.error('Seeds must be distinct and in 0..4')
    if len(set(args.settings))!=len(args.settings) or len(set(args.variants))!=len(args.variants): p.error('Duplicate settings or variants')
    if not 1<=args.smoke_steps<=100: p.error('Use 1..100 smoke steps')
    if args.mode=='formal' and args.device!='cuda': p.error('Formal reproduction requires CUDA')
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    root=Path(args.root).resolve(); out=(root/args.out).resolve(); full_dir=(root/args.full_dir).resolve()
    if args.resume and args.mode!='formal': p.error('--resume is for formal runs only; use a new smoke directory')
    if args.resume_every<1: p.error('--resume-every must be positive')
    if args.resume and not (out/'run_manifest.json').is_file(): raise RuntimeError('No V2 manifest to resume')
    if not args.resume and out.exists(): raise FileExistsError(f'Refusing to overwrite {out}; use --resume for this V2 formal run')
    check_sources(); lock=source_lock(); paths,info=check_data(root)
    for seed in args.seeds:
        full_reference(full_dir,seed,args.device,load_model=False)
    if args.mode=='formal':
        if not args.smoke_report: p.error('Formal requires --smoke-report')
        report=load((root/args.smoke_report).resolve())
        if report.get('status')!='PASS' or report.get('resume_test',{}).get('status')!='PASS' or report.get('source_lock')!=lock or {r['Variant'] for r in report.get('results',[])}!=set(VARIANTS):
            raise RuntimeError('Smoke report is missing, stale or incomplete; run smoke again')
        if any(report['data'][k]['sha256']!=info[k]['sha256'] for k in info): raise RuntimeError('Smoke data differ from formal')
        if report.get('device')!='cuda' or report.get('torch_version')!=torch.__version__: raise RuntimeError('Run real CUDA smoke in current PyTorch environment before formal training')
    out.mkdir(parents=True,exist_ok=args.resume)
    versions={}
    for name in ['torch','numpy','scipy','scikit-learn','scanpy','anndata','pandas','einops','joblib']:
        try: versions[name]=md.version(name)
        except md.PackageNotFoundError: versions[name]='not found'
    manifest={'schema':2,'resume_every':args.resume_every,'mode':args.mode,'root':str(root),'full_dir':str(full_dir),'data':info,
          'seeds':args.seeds,'settings':args.settings,'variants':args.variants,'protocol':SETTINGS,
          'source_lock':lock,'versions':versions,'python':sys.version,'platform':platform.platform(),
          'device':args.device,'gpu':torch.cuda.get_device_name(0) if args.device=='cuda' else None,
          'runtime':{'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'cuda_devices':torch.cuda.device_count(),
                     'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,
                     'cudnn_benchmark':torch.backends.cudnn.benchmark,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                     'cudnn_tf32':torch.backends.cudnn.allow_tf32,'num_threads':torch.get_num_threads()},
          'smoke_report_sha256':sha256((root/args.smoke_report).resolve()) if args.mode=='formal' else None,
          'concept_list':concept_list,'concept_counts':concept_counts,'concept_cdag':concept_cdag,
          'notes':['Original trainer saves checkpoint step 100000 after its update at step 100000 (loop starts 0). Retained for exact protocol matching.', 'wo_dis removes concept supervision only, not the full encoder; DAG, residual adversarial loss, KL and conditioning retained.', 'wo_attn replaces attention in each block while retaining FF and norms.','Identical original ODE sparsification and outer threshold are used for all variants.']}
    with run_lock(out):
        if args.resume:
            if load(out/'run_manifest.json')!=manifest: raise RuntimeError('Run manifest differs; same data, sources, settings and environment required')
        else: dump(out/'run_manifest.json',manifest)
        print(f'PREFLIGHT PASS: data and reference checkpoint file hashes match. Output: {out}',flush=True)
        if args.mode=='preflight':
            for seed in args.seeds:
                model,rec=full_reference(full_dir,seed,args.device); verify_variant(model,'full'); del model; cleanup()
            dump(out/'preflight_status.json',{'status':'PASS'})
        elif args.mode=='smoke': smoke(paths,out,info,lock,args)
        else:
            try:
                dump(out/'formal_status.json',{'status':'RUNNING','seeds':args.seeds,'settings':args.settings})
                formal(paths,out,info,args,full_dir)
            except KeyboardInterrupt:
                dump(out/'formal_status.json',{'status':'PAUSED','note':'Use same command plus --resume. Only committed states are reused.'})
                print('Paused. Restart the same formal command with --resume.',flush=True)
                raise SystemExit(130)
            except Exception as e:
                dump(out/'formal_status.json',{'status':'INCOMPLETE','error':str(e),'note':'Inspect traceback. Resume after fixing external failure, not by changing experiment code.'})
                raise

if __name__=='__main__':
    main()
