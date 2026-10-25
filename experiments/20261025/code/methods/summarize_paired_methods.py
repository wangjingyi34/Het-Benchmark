"""Frozen-plan evaluation on independent AB/BA process confirmations.

Uses ratio of means. No-op is zero by definition; mask0 A/B noise is separately
reported unaltered. Physical oracle is posthoc baseline-normalized, never a plan.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from migration_bridge import measured_quality
from summarize_bridge import vector

# e88 differs only in the terminal print after the final persistent save.
# Independently reverse-reconstructed to 2f; see bridge-early-results receipt.
COLLECTORS={'2f33287e7d6afffb9303724ac61186c5712cbaaa0784795196d7b1f4ec6c22bb',
            'e88bce1b159f018e5dfa5793c726d362b70bcf0ed9795bf2a881d0ac5c1408c5'}
MASKS={m for m in range(16) if m.bit_count()<=2}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def quality(q):
    # Reuse the independently recomputed labeled-ledger arithmetic.
    return measured_quality({**q,'status':'ok'})

def paired_row(row):
    if row['status']!='ok' or len(row['pairs'])!=20:raise ValueError('complete successful confirmation row required')
    pairs=row['pairs']
    if any(p['status']!='ok' for p in pairs) or {p['pair_index'] for p in pairs}!=set(range(20)):
        raise ValueError('timing pair missing/duplicated/failed')
    if sum(p['order']=='AB' for p in pairs)!=10 or sum(p['order']=='BA' for p in pairs)!=10:raise ValueError('AB/BA allocation differs')
    a=[];b=[]
    for p in pairs:
        x=p['A']['latency_ms'];y=p['B']['latency_ms']
        if any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<=0 for t in [x,y]):raise ValueError('invalid positive physical timing')
        if abs(p['baseline_ms']-x)>1e-9 or abs(p['candidate_ms']-y)>1e-9 or abs(p['paired_delta_ms']-(x-y))>1e-9:raise ValueError('pair arithmetic disagrees')
        a.append(x);b.append(y)
    am=statistics.mean(a);bm=statistics.mean(b);gain=100*(am-bm)/am
    if abs(row['summary']['ratio_of_means_gain_percent']-gain)>1e-9:raise ValueError('stored ratio-of-means differs')
    aq,amap=quality(row['baseline_quality']);bq,bmap=quality(row['candidate_quality'])
    if aq<=0 or set(amap)!=set(bmap) or any(amap[k][1]!=bmap[k][1] for k in amap):raise ValueError('A/B quality identity differs')
    feasible=row['candidate_valid'] is True and all(p['output_equivalent'] is True for p in pairs) and aq>0 and bq/aq>=.99
    if feasible!=row['final_candidate_feasible']:raise ValueError('final feasibility arithmetic differs')
    byorder={o:{'baseline_mean_ms':statistics.mean(p['A']['latency_ms'] for p in pairs if p['order']==o),
                'candidate_mean_ms':statistics.mean(p['B']['latency_ms'] for p in pairs if p['order']==o)} for o in ['AB','BA']}
    for v in byorder.values():v['ratio_of_means_gain_percent']=100*(v['baseline_mean_ms']-v['candidate_mean_ms'])/v['baseline_mean_ms']
    return {'mask':row['mask'],'baseline_mean_ms':am,'candidate_mean_ms':bm,'raw_gain_percent':gain,
        'raw_delta_ms':am-bm,'mean_pair_gain_auxiliary_percent':statistics.mean(p['paired_gain_percent'] for p in pairs),
        'quality_baseline':aq,'quality_candidate':bq,'local_retention':bq/aq,'quality_map':bmap,
        'candidate_feasible':feasible,'order_breakdown':byorder}

def source_quality(source):
    states=[]
    for b in source['eval']:
        r=next(r for r in b['coalitions'] if r['mask']==0);q,m=measured_quality(r)
        states.append((q,m))
    if any(s!=states[0] for s in states):raise ValueError('source evaluation quality/predictions vary; requires explicit matched-source study')
    return states[0]

def evaluate_context(raws,trace,source,planner_results=None):
    if len(raws)!=5 or {r['run_index'] for r in raws}!=set(range(101,106)):raise ValueError('five fixed fresh-process blocks required')
    if len({r['runtime']['pid'] for r in raws})!=5:raise ValueError('fresh process PIDs not distinct')
    sq,smap=source_quality(source);source_eval=source['eval'][0]
    plans={'cross_dag':trace['action_priority']['selected_mask'],**{k:v['selected_mask'] for k,v in trace['matched_control_plans'].items()}}
    planner_provenance=[]
    for item in planner_results or []:
        p=item['payload']
        if p['schema_version']!='het.planner.results.v2' or p['task_id']!=trace['context']['task_id'] or p['device_id']!=trace['context']['target_device_id']:
            raise ValueError('planner result context differs')
        if set(p['training_runs'])!=set(trace['context']['target_baseline_run_ids']) or p['budget']!=2:
            raise ValueError('planner did not use the same frozen calibration runs/budget')
        tag=item.get('tag','planner')
        for name,r in p['fixed_strategies'].items():
            if r['status']=='evaluated' and r['mask'] is not None:plans[f'{tag}/fixed/{name}']=r['mask']
        for r in p['ppo']:
            if r['eval_access_during_training'] is not False or r['unit_fixture_data'] is not False or set(r['training_run_ids'])!=set(p['training_runs']):
                raise ValueError('PPO training/label isolation provenance differs')
            plans[f'{tag}/ppo/{r["configuration"]}/seed{r["seed"]}']=r['plan']['mask']
        planner_provenance.append({k:v for k,v in item.items() if k!='payload'})
    if any(m not in MASKS for m in plans.values()):raise ValueError('frozen plan exceeds action budget')
    results={k:[] for k in plans};noise=[];oracle=[];randoms=[]
    for raw in sorted(raws,key=lambda r:r['run_index']):
        if raw['source_sha256'] not in COLLECTORS or raw['status']!='completed' or raw['provenance']!='physical_measurement':raise ValueError('collector protocol/provenance not frozen complete physical run')
        if raw['task_id']!=trace['context']['task_id'] or raw['device_id']!=trace['context']['target_device_id'] or raw['input_shape']!=trace['context']['input_shape']:raise ValueError('confirmation context differs')
        if raw['action_ids']!=[r['action_id'] for r in trace['action_priority']['actions']]:raise ValueError('action mask identity/order differs')
        if raw['dataset_manifest_sha256']!=trace['context']['dataset_manifest_sha256']:raise ValueError('dataset manifest differs')
        # The original collector also stored an explanatory prose 'reason'.
        # Compare the actual configuration, while retaining both frozen sources.
        if raw['model_configuration']['inplace_disabled_modules']!=trace['context']['model_configuration']['inplace_disabled_modules']:
            raise ValueError('in-place module allocation configuration differs')
        precision=raw['runtime']['precision']
        if precision['dtype']!='float32' or precision['cuda_matmul_tf32'] is not False or precision['cudnn_tf32'] is not False:raise ValueError('FP32 precision protocol differs')
        if raw['device_id'].startswith('npu_') and (precision.get('conv_hf32') is not False or precision.get('matmul_hf32') is not False):raise ValueError('NPU HF32 precision differs')
        if raw['weights_state_sha256']!=source_eval['weights_state_sha256'] or raw['input_tensor_sha256']!=source_eval['input_tensor_sha256']:raise ValueError('source/confirmation weights or eval input differ')
        if set(raw['quality_sample_ids'])!=set(smap) or {r['mask'] for r in raw['coalitions']}!=MASKS:raise ValueError('quality IDs or complete <=2 support differs')
        rows={r['mask']:paired_row(r) for r in raw['coalitions']}
        for row in rows.values():
            if set(row['quality_map'])!=set(smap) or any(row['quality_map'][k][1]!=smap[k][1] for k in smap):raise ValueError('source/target labels differ')
            row['source_retention']=row['quality_candidate']/sq
            row['migration_quality_feasible']=row['candidate_feasible'] and row['source_retention']>=.99
            row.pop('quality_map')
        noise.append({'run_id':raw['run_id'],**rows[0]})
        # A no-op deploys A unchanged. Baseline-copy B noise remains in the
        # negative-control table, never converted into optimization benefit.
        feasible={m:r['raw_gain_percent'] for m,r in rows.items() if r['migration_quality_feasible']}
        if 0 in feasible:feasible[0]=0.
        best=max([(g,-m) for m,g in feasible.items()]) if feasible else None
        oracle_gain=best[0] if best else None
        oracle.append({'run_id':raw['run_id'],'oracle_mask':-best[1] if best else None,'best_baseline_normalized_gain_percent':oracle_gain,
                       'scope':'posthoc feasible paired-normalized target-action oracle; not used in selection'})
        outcomes={}
        for m,row in rows.items():
            rawgain=0. if m==0 else row['raw_gain_percent']
            accepted=row['migration_quality_feasible'] and (m==0 or rawgain>=0)
            outcomes[m]={'mask':m,'raw_gain_percent':rawgain,'accepted_gain_percent':rawgain if accepted else 0.,
                'accepted':accepted,'quality_retention_to_source':row['source_retention'],'quality_retention_to_target':row['local_retention'],
                'raw_delta_ms':0. if m==0 else row['raw_delta_ms'],
                'oracle_regret_percentage_points':oracle_gain-(rawgain if accepted else 0.) if oracle_gain is not None else None}
        for name,m in plans.items():results[name].append({'run_id':raw['run_id'],'run_index':raw['run_index'],**outcomes[m]})
        randoms.append({'run_id':raw['run_id'],'uniform_at_most_k_raw_gain_percent':statistics.mean(o['raw_gain_percent'] for o in outcomes.values()),
            'uniform_at_most_k_accepted_gain_percent':statistics.mean(o['accepted_gain_percent'] for o in outcomes.values()),
            'uniform_exact_k_raw_gain_percent':statistics.mean(o['raw_gain_percent'] for m,o in outcomes.items() if m.bit_count()==2),
            'uniform_exact_k_accepted_gain_percent':statistics.mean(o['accepted_gain_percent'] for m,o in outcomes.items() if m.bit_count()==2)})
    summaries={name:{'frozen_mask':plans[name],'raw_gain_percent':vector([r['raw_gain_percent'] for r in rows]),
        'accepted_gain_percent':vector([r['accepted_gain_percent'] for r in rows]),
        'oracle_regret_pp':vector([r['oracle_regret_percentage_points'] for r in rows if r['oracle_regret_percentage_points'] is not None]),
        'quality_retentions_to_source':[r['quality_retention_to_source'] for r in rows],'accepted_count':sum(r['accepted'] for r in rows),'rows':rows} for name,rows in results.items()}
    ppo_groups={}
    for name in summaries:
        if '/ppo/' in name:ppo_groups.setdefault(name.rsplit('/seed',1)[0],[]).append(name)
    ppo_summary={}
    for configuration,names in ppo_groups.items():
        # Average policy seeds within each common process block before forming
        # a hardware interval. Never count five seeds times five blocks as 25.
        raw_means=[statistics.mean(results[n][i]['raw_gain_percent'] for n in names) for i in range(5)]
        accepted_means=[statistics.mean(results[n][i]['accepted_gain_percent'] for n in names) for i in range(5)]
        seed_means=[summaries[n]['raw_gain_percent']['mean'] for n in names]
        ppo_summary[configuration]={'policy_count':len(names),'members':names,'raw_gain_seed_averaged_within_process_percent':vector(raw_means),
            'accepted_gain_seed_averaged_within_process_percent':vector(accepted_means),'policy_seed_mean_raw_gains_percent':seed_means,
            'policy_seed_conditional_sample_sd':statistics.stdev(seed_means) if len(seed_means)>1 else None,
            'scope':'five physical process blocks; policy seeds characterize training randomness on the same physical tables'}
    return {'model_id':raws[0]['model_id'],'input_shape':raws[0]['input_shape'],'device_id':raws[0]['device_id'],
        'plans':summaries,'mask0_negative_control':{'raw_gain_percent':vector([r['raw_gain_percent'] for r in noise]),'raw_delta_ms':vector([r['raw_delta_ms'] for r in noise]),'rows':noise},
        'ppo_configuration_summary':ppo_summary,'planner_provenance':planner_provenance,
        'posthoc_paired_normalized_oracle':oracle,'random_expected_controls':randoms,
        'source_quality_reference':{'quality':sq,'sample_count':len(smap),'run_ids':[b['run_id'] for b in source['eval']],
            'scope':'same fixed evaluation sample IDs, labels and deterministic source predictions verified across old source runs; not new source timings'},
        'interpretation':'Independent confirmation of original calibration-selected plans; all negative gains, AB/BA noise and original results retained. Five process blocks are uncertainty units.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',type=Path,required=True);ap.add_argument('--traces',nargs='+',type=Path,required=True)
    ap.add_argument('--planner-results',nargs='*',type=Path,default=[])
    ap.add_argument('--task-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError('do not overwrite earlier confirmation analysis')
    tasks={sha(p):json.loads(p.read_text()) for p in a.task_dir.glob('*.json') if isinstance(json.loads(p.read_text()),dict) and json.loads(p.read_text()).get('schema_version')=='het.measured.coalitions.v2'}
    rawfiles=[];allraw=[]
    for p in sorted(a.raw_dir.glob('*.json')):
        r=json.loads(p.read_text())
        if isinstance(r,dict) and r.get('schema_version')=='het.fullmodel.paired-confirmation.v1':allraw.append(r);rawfiles.append({'path':str(p),'sha256':sha(p)})
    out={'kind':'independent_frozen_bridge_plan_paired_confirmation','contexts':[],
        'raw_artifacts':rawfiles,'traces':[{'path':str(p),'sha256':sha(p)} for p in a.traces],'analyzer_sha256':sha(__file__)}
    for path in a.traces:
        trace=json.loads(path.read_text());source=tasks[trace['source_task_sha256']]
        rs=[r for r in allraw if r['task_id']==trace['context']['task_id'] and r['device_id']==trace['context']['target_device_id']]
        planners=[]
        for i,p in enumerate(a.planner_results):
            data=json.loads(p.read_text())
            if data['task_id']==trace['context']['task_id'] and data['device_id']==trace['context']['target_device_id']:
                planners.append({'payload':data,'path':str(p),'sha256':sha(p),'tag':f'planner{i}'})
        out['contexts'].append(evaluate_context(rs,trace,source,planners))
    a.output.write_text(json.dumps(out,indent=2));print(json.dumps({'output':str(a.output),'contexts':len(out['contexts'])}))

if __name__=='__main__':main()
