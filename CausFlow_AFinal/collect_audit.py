"""Package small final audit records, not expression, weights, large predictions or environments."""
import argparse,zipfile
from pathlib import Path
from run_afinal import verify,req,sha,read,check,MODELS

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--zip',required=True);a=p.parse_args();out=Path(a.out).resolve();zpath=Path(a.zip).resolve();verify(out);s=read(out/'score/score_status.json');req(s['status']=='COMPLETE' and s['rows']==1440,'Score not complete');req(not zpath.exists(),'Result archive already exists');
    for n,h in s['files'].items():check(out/'score'/n,h)
    picks=[out/n for n in ('lock.json','export_status.json','original_queries.csv','e1_selection.json','e1_selection_status.json','e1_inner_selection_metrics.csv','external_input/manifest.json','external_input/genes.csv','external_input/queries.csv')];picks+=list((out/'score').glob('*'))
    for model in ('E1',*MODELS):
        for seed in (0,1,2):picks.append(out/'predictions'/model/f'seed_{seed}'/'manifest.json')
    for model in ('CPA','scGen_LowerDose'):
        smoke=out/'professional_smoke'/model/'seed_0';picks.extend([smoke/'receipt.json',smoke/'train_config.json'])
        for seed in (0,1,2):
            d=out/'professional_models'/model/f'seed_{seed}'
            picks.extend(p for p in d.iterdir() if p.is_file() and p.suffix in ('.csv','.json','.txt') and 'conditional_variance' not in p.name)
    with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(picks)):z.write(p,p.relative_to(out).as_posix())
        for seed,x in read(out/'e1_selection.json')['runs'].items():
            d=Path(x['path']);
            for n in ('plan.json','plan_status.json','train_status.json','predict_status.json','training_history.json','score/decision.json','score/metrics_by_condition.csv'):
                z.write(d/n,f'E1_inner/seed_{seed}/{n}')
    print('A-FINAL AUDIT ZIP',zpath)
if __name__=='__main__':main()
