"""Audit all raw blocks and evaluate calibration-frozen deployment plans."""
import argparse,hashlib,itertools,json,statistics
from datetime import datetime
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interval(xs):
    n=len(xs);mean=statistics.mean(xs)
    width=2.7764451051977987*statistics.stdev(xs)/(n**.5) if n==5 else None
    observed=abs(mean);null=[abs(statistics.mean(s*x for s,x in zip(signs,xs))) for signs in itertools.product([-1,1],repeat=n)]
    return {'mean':mean,'n_processes':n,'sample_sd':statistics.stdev(xs),'descriptive_t95':[mean-width,mean+width] if width is not None else None,
            'exact_two_sided_signflip_p':sum(x>=observed-1e-12 for x in null)/len(null),'values':xs}
def main():
    a=argparse.ArgumentParser();a.add_argument('--raw',type=Path,required=True);a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    plan=json.loads(args.plan.read_text());source=plan['collector_sha256'];plan_hash=sha(args.plan);rows=[]
    for model in ['resnet18','resnet50']:
      for batch in [1,4]:
       for run in range(1,9):
        p=args.raw/f'{model}-b{batch}-r{run}.json';d=json.loads(p.read_text());assert d['status']=='completed'
        assert d['source_sha256']==source and d['model']==model and d['batch']==batch and d['run']==run
        assert d['partition']==('calibration' if run<=3 else 'evaluation')
        assert len(d['pairs'])==80 and len(d['masks'])==4
        if run>=4:
          assert d['frozen_plan_sha256']==plan_hash
          assert datetime.fromisoformat(d['started_utc'])>datetime.fromisoformat(plan['frozen_utc'])
        for m in range(4):
          rr=[r for r in d['pairs'] if r['mask']==m];assert len(rr)==20
          assert sum(r['order']=='AB' for r in rr)==10
          av=statistics.mean(r['A']['latency_ms'] for r in rr);bv=statistics.mean(r['B']['latency_ms'] for r in rr)
          gain=100*(av-bv)/av;assert abs(gain-d['masks'][str(m)]['gain_percent'])<1e-9
          assert d['masks'][str(m)]['all_timed_outputs_equivalent']==all(r['allclose'] for r in rr)
        rows.append({'path':p.name,'sha256':sha(p),'data':d})
    pids=[r['data']['pid'] for r in rows];assert len(set(pids))==32
    cal_ids=set(rows[0]['data']['sample_ids']);eval_ids=set(rows[3]['data']['sample_ids']);assert len(cal_ids)==len(eval_ids)==100 and not cal_ids&eval_ids
    results=[]
    for model in ['resnet18','resnet50']:
      for batch in [1,4]:
       block=[r['data'] for r in rows if r['data']['model']==model and r['data']['batch']==batch and r['data']['run']>=4]
       selected=plan['selections'][f'{model}-b{batch}']
       masks=[]
       for m in range(4):
        ms=[d['masks'][str(m)] for d in block]
        masks.append({'mask':m,'actions':ms[0]['actions'],'gain_percent':interval([v['gain_percent'] for v in ms]),
          'baseline_mean_ms':statistics.mean(v['baseline_mean_ms'] for v in ms),'candidate_mean_ms':statistics.mean(v['candidate_mean_ms'] for v in ms),
          'all_quality_and_numerical_gates_pass':all(v['accepted'] for v in ms),
          'min_quality_retention':min(v['quality_retention'] for v in ms)})
       k1=selected['k1_mask'];k2=selected['k2_fixed_combo_mask']
       def gs(d,m):return d['masks'][str(m)]['gain_percent'] if m else 0.
       results.append({'model':model,'batch':batch,'frozen_selections':selected,'all_masks':masks,
          'k1_frozen_gain_percent':interval([gs(d,k1) for d in block]),
          'k2_frozen_gain_percent':interval([gs(d,k2) for d in block]),
          'k1_minus_uniform_F_L_pp':interval([gs(d,k1)-.5*(gs(d,1)+gs(d,2)) for d in block]),
          'method_scope':'calibration-selected deployment actions, not a COPA-v-latency comparison'})
    out={'kind':'new-physical-layout-fold-action-validation','protocol_source_sha256':source,'frozen_plan_sha256':plan_hash,
      'counts':{'processes':32,'calibration_processes':12,'evaluation_processes':20,'coalition_blocks':128,'AB_BA_pairs':2560,'timed_model_calls':5120,'unique_quality_images':200},
      'all_32_process_ids_unique':True,'calibration_evaluation_images_disjoint':True,
      'hardware':rows[0]['data']['hardware'],'results':results,
      'raw_files':[{'path':r['path'],'sha256':r['sha256']} for r in rows],
      'interpretation':'New held-out deployment-validation case; no historical FITAS15.6/66 authentication; process CIs are descriptive, four tasks from two pretrained architectures; all masks and control noise retained.'}
    args.output.write_text(json.dumps(out,indent=2));print(json.dumps({'counts':out['counts'],'contexts':[{'model':r['model'],'batch':r['batch'],'k2':r['k2_frozen_gain_percent']['mean']} for r in results]}))
if __name__=='__main__':main()
