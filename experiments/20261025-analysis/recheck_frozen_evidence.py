"""Recompute paired fixed-plan effects from restored measured artifacts, without hardware."""
from pathlib import Path
import argparse,json,hashlib,statistics,math,csv
TCRIT_DF4=2.7764451051977987

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(values):
 assert len(values)==5
 m=statistics.mean(values);sd=statistics.stdev(values);e=TCRIT_DF4*sd/math.sqrt(5)
 return {'n_processes':5,'values':values,'mean':m,'sample_sd':sd,'descriptive_t95':[m-e,m+e]}
def resolve(data_root,name,digest):
 found=[p for p in data_root.rglob(name) if p.is_file() and sha(p)==digest]
 if not found:raise FileNotFoundError(f'Missing expected content {name} SHA256={digest}')
 return found[0]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--release-dir',type=Path,required=True);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 audit_path=a.release_dir/'process/methods/planner-matrix-independent-audit.json';audit=json.loads(audit_path.read_text());contexts=[]
 for c in audit['contexts']:
  task=resolve(a.data_root,Path(c['input_path']).name,c['input_sha256']);rp=resolve(a.data_root,Path(c['result_path']).name,c['result_sha256']);x=json.loads(task.read_text());result=json.loads(rp.read_text());gains=[]
  for run in x['eval']:
   actual=resolve(a.data_root,Path(run['artifact_path']).name,run['artifact_sha256']);raw=json.loads(actual.read_text());coal={r['mask']:r for r in run['coalitions']}
   for r in coal.values():assert math.isclose(statistics.mean(r['latency_samples_ms']),r['mean_ms'],rel_tol=1e-12)
   baseline=coal[0]['mean_ms'];gains.append({m:100*(baseline-v['mean_ms'])/baseline for m,v in coal.items()})
  plans={}
  for k in ['copa_dag','dag_quadratic_runtime','latency','same_evidence_exhaustive_search','measured_calibration_shapley']:
   mask=result['fixed_strategies'][k]['mask'];values=[g[mask] for g in gains]
   assert all(math.isclose(v,e['raw_benefit_percent'],rel_tol=1e-12,abs_tol=1e-12) for v,e in zip(values,result['fixed_strategies'][k]['evaluation']))
   plans[k]={'mask':mask,'raw_latency_reduction_percent':stats(values)}
  masks=[m for m in range(16) if m.bit_count()<=2]
  plans['uniform_random_at_most_2']={'mask':'uniform11subsets','raw_latency_reduction_percent':stats([statistics.mean(g[m] for m in masks) for g in gains])}
  policies=[p for p in result['ppo'] if p['configuration']=='ppo_with_heldout_embedding'];assert len(policies)==5
  plans['ppo_with_heldout_embedding']={'masks':[p['plan']['mask'] for p in policies],'raw_latency_reduction_percent':stats([statistics.mean(g[p['plan']['mask']] for p in policies) for g in gains])}
  contrasts={}
  for k in ['copa_dag','dag_quadratic_runtime','ppo_with_heldout_embedding']:
   for b in ['latency','uniform_random_at_most_2','same_evidence_exhaustive_search']:
    contrasts[k+' minus '+b]=stats([v-w for v,w in zip(plans[k]['raw_latency_reduction_percent']['values'],plans[b]['raw_latency_reduction_percent']['values'])])
  contexts.append({'model':x['model_id'],'batch':x['input_shape'][0],'device':x['device_id'],'task_sha256':c['input_sha256'],'result_sha256':c['result_sha256'],'plans':plans,'paired_contrasts_percentage_points':contrasts})
 out={'source_audit_sha256':sha(audit_path),'metric':'raw latency reduction100*(baseline-candidate)/baseline; no clipping','scope':'Five-process descriptive conditional t intervals on fixed tasks. Contrasts are within-process paired. No population or familywise superiority claim; all eight contexts and controls retained. Random is exact uniform expectation over11measured subsets, not independent random hardware runs.','contexts':contexts,'new_hardware_executions':0}
 (a.output_dir/'paired-fixed-plan-contrasts.json').write_text(json.dumps(out,indent=2)+'\n')
 summary_path=a.release_dir/'results/primary-prediction-summary.json';s=json.loads(summary_path.read_text());rows=[]
 for r in s['aggregate_results']:
  if r['feature_mode']=='probe' and r['protocol']=='model_holdout':rows.append({'method':r['method'],'micro_mre_five_seed_mean':r['micro_mre_seed_mean'],'micro_mre_seed_sample_sd':r['micro_mre_seed_sample_sd']})
 ab={'source_sha256':sha(summary_path),'scope':'All same-information baseline and six relation controls, probe model holdout; selected single-control contrast does not establish all relations help.','rows':rows,'paired_comparisons':[r for r in s['paired_comparisons'] if r['feature_mode']=='probe' and r['protocol']=='model_holdout']}
 (a.output_dir/'all-relation-controls.json').write_text(json.dumps(ab,indent=2)+'\n')
 print(json.dumps({'contexts_verified':len(contexts),'relation_baseline_controls':len(rows),'new_hardware_executions':0}))
if __name__=='__main__':main()
