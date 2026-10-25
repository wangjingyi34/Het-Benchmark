"""Fixed-input arithmetic only: no hardware runs, live quotes or MIG tariff.

Example:
 python calculate_market_cost_scenarios.py --task-dir restored/planner-inputs/tasks \
   --output-prefix reproduced/market-cost-scenarios
"""
import argparse
import csv
import datetime
import hashlib
import itertools
import json
import statistics
from pathlib import Path

FXS=[6.5,7.,7.5];ALPHAS=[.25,2/7,1.];HOURS=[672,720,744]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def generate(task_dir):
    tasks={};sources=[]
    for p in sorted(task_dir.glob('*.json'),key=lambda p:(not p.name.startswith('npu-'),p.name)):
        d=json.loads(p.read_text())
        if not isinstance(d,dict) or d.get('schema_version')!='het.measured.coalitions.v2':continue
        if d['device_id'].startswith('npu_'):backend='npu'
        elif d['device_id'].startswith('cuda_'):backend='a100_mig'
        else:continue
        values=[]
        assert len(d['eval'])==5
        for block in d['eval']:
            assert block['provenance']=='physical_measurement'
            row=next(r for r in block['coalitions'] if r['mask']==0)
            assert row['status']=='ok' and len(row['latency_samples_ms'])==30
            values.append(statistics.mean(row['latency_samples_ms']))
        key=(d['model_id'],d['input_shape'][0],backend)
        if key in tasks:raise ValueError('duplicate model/batch/backend; specify exact eight-task directory')
        tasks[key]={'mean_ms':statistics.mean(values),'process_means_ms':values,'process_count':5,
                    'sample_sd_ms':statistics.stdev(values),'device_id':d['device_id']}
        sources.append({'task_path':str(p.resolve()),'task_sha256':sha(p),'model_id':key[0],'batch':key[1],'backend':backend})
    assert len(tasks)==8
    rows=[]
    for model,batch in [('resnet18',1),('resnet18',4),('resnet50',1),('resnet50',4)]:
        ts=tasks[(model,batch,'a100_mig')]['mean_ms'];tn=tasks[(model,batch,'npu')]['mean_ms']
        for fx,alpha,hours,billing in itertools.product(FXS,ALPHAS,HOURS,['hourly_38_CNY','monthly_18500_CNY']):
            ra=1.79*fx*alpha;rn=38. if billing=='hourly_38_CNY' else 18500./hours
            ca=ra*ts/(3600*batch);cn=rn*tn/(3600*batch);parity=ra*ts/tn
            rows.append({'model_id':model,'batch_size':batch,'a100_mig_baseline_mean_ms':ts,'npu_baseline_mean_ms':tn,
                'physical_process_blocks_per_backend':5,'fx_CNY_per_USD_scenario':fx,'a100_opportunity_allocation_alpha':alpha,
                'month_hours_scenario':hours,'npu_billing_scenario':billing,'runpod_full_card_USD_per_hour_fixed_scenario':1.79,
                'a100_allocated_opportunity_CNY_per_hour':ra,'npu_equivalent_CNY_per_hour':rn,
                'a100_CNY_per_1000_sample_requests':ca,'npu_CNY_per_1000_sample_requests':cn,
                'npu_minus_a100_CNY_per_1000':cn-ca,'npu_cost_reduction_percent_vs_a100_scenario':100*(1-cn/ca),
                'npu_cheaper_in_this_scenario':cn<ca,'a100_monthly_opportunity_allocation_CNY':ra*hours,
                'npu_monthly_allocation_CNY':rn*hours,'parity_npu_max_CNY_per_hour':parity,
                'parity_npu_max_CNY_per_month_at_this_H':parity*hours,
                'parity_min_a100_alpha_at_this_FX':rn*tn/(1.79*fx*ts),
                'parity_CNY_per_USD_at_this_alpha':rn*tn/(1.79*alpha*ts),
                'parity_max_npu_latency_ms_at_these_prices':ra*ts/rn})
    return tasks,sources,rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task-dir',type=Path,required=True);ap.add_argument('--output-prefix',type=Path,required=True)
    a=ap.parse_args();jp=a.output_prefix.with_suffix('.json');cp=a.output_prefix.with_suffix('.csv')
    if jp.exists() or cp.exists():raise FileExistsError('preserve earlier cost scenario outputs')
    tasks,sources,rows=generate(a.task_dir)
    out={'kind':'fixed_input_capacity_normalized_cost_scenarios','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'generator_sha256':sha(__file__),'new_physical_tests':0,'request_unit':'one sample inference, batch wall time allocated evenly',
        'cost_unit':'CNY per1000 sample requests','latency_selection':'unchanged mean of five original eval mask0 process means',
        'rates_are_fixed_scenarios_not_new_live_quotes':True,'source_rates':{'npu_hourly_CNY':38,'npu_monthly_CNY':18500,
            'source':'direct author-confirmed scenario inputs; public process/market-cost-assumptions.json',
            'runpod_full_A100_SXM80GB_USD_per_hour':1.79,'FX_CNY_per_USD':FXS,'alpha':ALPHAS,'month_hours':HOURS},
        'formulae':{'per1000':'CNY_per_hour * batch_latency_ms / (3600 * batch_size)',
            'monthly_hourly_equivalent':'18500 / assumed_month_hours',
            'npu_hourly_parity':'a100_opportunity_rate * measured_A100_MIG_latency / measured_NPU_latency'},
        'assumptions':['100% capacity-normalized allocation utilization; no tax/storage/network/engineering costs inferred.',
            'Owned A100 actual cash rental outlay is not the scenario opportunity cost.',
            'Runpod reference is whole-card price. Measured performance is MIG2g20; alpha is illustrative allocation, not a quoted MIG tariff or fabricated full-card performance.',
            'FX values are scenario parameters, not current quotations. Monthly utilization is assumed, not measured.',
            'Original hardware/software stacks and FP32 settings retained. Five process blocks characterize conditional run variability.',
            'Negative savings retained; these scenarios do not authenticate historical66% savings.'],
        'sources':sources,'baselines':[{'model_id':m,'batch_size':b,'backend':k,**v} for (m,b,k),v in tasks.items()],
        'scenario_count':len(rows),'npu_cheaper_scenario_count':sum(r['npu_cheaper_in_this_scenario'] for r in rows),'scenarios':rows}
    jp.parent.mkdir(parents=True,exist_ok=True);jp.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    with cp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'json':str(jp),'csv':str(cp),'scenarios':len(rows),'new_physical_tests':0}))

if __name__=='__main__':main()
