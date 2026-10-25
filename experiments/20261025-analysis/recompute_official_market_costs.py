from pathlib import Path
import json,hashlib,datetime,csv
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--baseline-summary',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
ROOT=a.output_dir;ROOT.mkdir(parents=True,exist_ok=True)
p=a.baseline_summary;old=json.loads(p.read_text())
providers=[{'name':'Runpod Pods A100 SXM80GB','usd_gpu_h':1.79,'min_gpus':1,'url':'https://www.runpod.io/pricing'}, {'name':'Lambda 8x A100 SXM80GB','usd_gpu_h':2.79,'min_gpus':8,'url':'https://lambda.ai/instances'}]
npu={'name':'Tianyi PAK3 pak3.4xlarge.8 Ascend910B64GB','cny_hour':38.45,'cny_month':18454,'url':'https://www.ctyun.cn/document/10029787/10047957'}
# One baseline per context; all five original evaluation process means are retained upstream.
base={}
for row in old['scenarios']:
 key=(row['model_id'],row['batch_size'])
 base[key]={'a100_mig_ms':row['a100_mig_baseline_mean_ms'],'npu_ms':row['npu_baseline_mean_ms']}
rows=[]
for (model,batch),lat in base.items():
 for provider in providers:
  for fx in [6.5,7.0,7.5]:
   for alpha in [.25,2/7,1.0]:
    plans=[('official_hourly',npu['cny_hour'],None)]+[('official_monthly',npu['cny_month']/hours,hours) for hours in [672,720,744]]
    for plan,rate,hours in plans:
     pa=provider['usd_gpu_h']*fx*alpha;ca=pa*lat['a100_mig_ms']/(3600*batch);cn=rate*lat['npu_ms']/(3600*batch)
     rows.append({'model':model,'batch':batch,'provider':provider['name'],'min_order_gpus':provider['min_gpus'],'fx_scenario_cny_usd':fx,'MIG_cost_allocation_alpha_scenario':alpha,'npu_plan':plan,'month_hours_scenario':hours,'a100_mig_mean_ms':lat['a100_mig_ms'],'npu_mean_ms':lat['npu_ms'],'a100_allocated_cny_h':pa,'npu_cny_h':rate,'a100_cny_1000_samples':ca,'npu_cny_1000_samples':cn,'cost_reduction_percent':100*(1-cn/ca),'supports_66_percent_or_better':cn<=.34*ca,'required_whole_A100_CNY_h_for_66':rate*lat['npu_ms']/(.34*lat['a100_mig_ms']*alpha),'required_NPU_max_ms_for_66':.34*pa*lat['a100_mig_ms']/rate})
out={'kind':'official_quote_and_observed_baseline_capacity_normalized_cost_scenarios','retrieved_date':'2026-10-25','sources':providers+[npu],'baseline_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':rows,'count':len(rows),'supports_66_count':sum(r['supports_66_percent_or_better'] for r in rows),'formula':'per1000samples=allocated_hourly_price*batch_latency_ms/(3600*batch); reduction=1-C_N/C_A','necessary_condition_for_66':'(p_N/p_A)*(latency_N/latency_A)*(batch_A/batch_N)<=0.34','assumptions':['FX values are explicit sensitivity assumptions, not observed exchange rates.','A100 physical timing is MIG2g.20GB, not full-card performance. alpha is an allocation scenario, not a public MIG price.','Lambda2.79 is a per-GPU allocation of an8-GPU minimum order. Entire unused-node reservation is not charged to one benchmark query.','One source and target instance is a cost-allocation scenario; geography, tax, engineering, migration effort and utilization are not measured.','These current quotes cannot establish a historical negotiated rate or authenticate the old500/170 inputs.'],'new_hardware_executions':0}
(ROOT/'official-market-cost-recheck.json').write_text(json.dumps(out,indent=2)+'\n')
with (ROOT/'official-market-cost-recheck.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'scenarios':len(rows),'supports_66':out['supports_66_count'],'reduction_range_percent':[min(r['cost_reduction_percent'] for r in rows),max(r['cost_reduction_percent'] for r in rows)],'illustrative_hourly_quote_ratio_at_FX7':38.45/(1.79*7)}))
