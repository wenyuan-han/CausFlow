"""Synthetic interface test: mock historical torch factory, not the actual CausFlow backbone."""
import tempfile, shutil, subprocess, sys, ast
from pathlib import Path
import numpy as np
from run_br64 import dump,sha,read,HERE

def run(*args):subprocess.run([sys.executable,*map(str,args)],check=True)
with tempfile.TemporaryDirectory() as td:
    root=Path(td);src=root/'CausFlow_TrainingSufficiency';src.mkdir();metrics=root/'CausFlow_BR6_Lock';metrics.mkdir();shutil.copyfile(HERE.parent/'CausFlow_BR6_Lock/locked_metrics.py',metrics/'locked_metrics.py');o=root/'out';o.mkdir();l=root/'lock';l.mkdir();(l/'inner/sources').mkdir(parents=True);(l/'inner/controls').mkdir();(l/'inner/nearest').mkdir()
    (src/'run_locked_benchmark.py').write_text('''import torch,random,numpy as np,hashlib
class M(torch.nn.Module):
 def __init__(self,n):super().__init__();self.f=torch.nn.Linear(n,n)
 def loss_parts(self,y,s,*lab):
  v=(self.f(s)-(y-s)).square().mean();return v,{'cfm':v,'kl':v*0}
 def generate(self,s,*lab,steps):return s+self.f(s)
def reset(t,s):random.seed(s);np.random.seed(s);t.manual_seed(s)
def seedof(*v):return 42
def torch_context(a):return torch,torch.device(a.device),lambda name,vocab,G,width:M(G)
''')
    rng=np.random.default_rng(7);x=rng.random((8,6),dtype=np.float32);np.save(l/'fit_x.npy',x);np.savez(l/'fit_index.npz',fit=np.arange(8),labels=np.array([[0,0,0]]*4+[[1,1,0]]*4),control=np.array([True]*4+[False]*4));vocab={'drug':['control','A'],'dose':['0','1','10'],'cell_line':['L']};dump(l/'vocab.json',vocab);qs=[dict(query_id=f'v00{i}',split=s,drug='A',dose=d,cell_line='L',n_real=4) for i,(s,d) in enumerate([('val_ID','1'),('val_OOD','10')])]
    for q in qs:
        k=q['query_id'];np.save(l/'inner/sources'/f'{k}.npy',x[rng.choice(4,256)]);np.save(l/'inner/controls'/f'{k}.npy',x[:4]);np.save(l/'inner/nearest'/f'{k}.npy',x[4:][rng.choice(4,256)])
    dump(l/'lock.json',{'files':{p.relative_to(l).as_posix():sha(p) for p in l.rglob('*') if p.is_file()}});dump(o/'plan.json',dict(schema='br64_e1_v1',root=str(root),lock_dir=str(l),lock_sha256=sha(l/'lock.json'),code={str(src/'run_locked_benchmark.py'):sha(src/'run_locked_benchmark.py')},seed=0,width=8,batch=4,lr=.0001,stages=[2],ode_steps=2,smoke=True,queries=qs,n_genes=6,n_fit=8,vocab=vocab));dump(o/'plan_status.json',{'sha256':sha(o/'plan.json')})
    run(HERE/'worker.py',o,'train','cpu');run(HERE/'worker.py',o,'predict','cpu');assert not (l/'inner_baseline_score').exists();d=l/'inner_baseline_score';(d/'targets').mkdir(parents=True)
    for q in qs:np.save(d/'targets'/f"{q['query_id']}.npy",x[4:])
    dump(d/'score_status.json',dict(status='COMPLETE',lock_sha256=sha(l/'lock.json'),target_hashes={p.name:sha(p) for p in (d/'targets').glob('*.npy')}));run(HERE/'run_br64.py','--phase','score','--out',o);assert read(o/'score/decision.json')['score_count']==8;assert read(o/'score/decision.json')['verdict']=='SMOKE_ONLY';print('PASS mock factory train/predict without targets, score only after targets exist');(o/'plan.json').write_text('{}');p=subprocess.run([sys.executable,str(HERE/'worker.py'),str(o),'predict','cpu'],capture_output=True);assert p.returncode!=0;print('PASS modified plan rejected')
for f in HERE.glob('*.py'):
    for v in (9,10):ast.parse(f.read_text(encoding='utf8'),feature_version=v)
print('PASS Python 3.9 and 3.10 syntax')
