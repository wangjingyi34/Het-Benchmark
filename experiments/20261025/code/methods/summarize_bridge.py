"""Independent post-selection arithmetic for a physical cross-hardware trace.

Does not alter the diagnostic game, fit a policy, or choose a calibration plan.
Heldout oracle and exact random-subset expectation are evaluation controls.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from migration_bridge import latency,measured_quality

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def vector(values):
    if not values:return {'n_process_blocks':0,'mean':None,'sample_sd':None,'descriptive_t95':None}
    mu=statistics.mean(values);sd=statistics.stdev(values) if len(values)>1 else None
    # The formal physical study has five heldout process blocks. Do not invent
    # a general t-quantile approximation for different sample counts.
    interval=[mu-2.7764451051977987*sd/(len(values)**.5),mu+2.7764451051977987*sd/(len(values)**.5)] if len(values)==5 else None
    return {'n_process_blocks':len(values),'values':values,'mean':mu,'sample_sd':sd,
            'descriptive_t95':interval,'uncertainty_scope':'run-to-run conditional variability on this fixed model/input/sample set; not a dataset/population or hardware-family interval'}

def analyze(trace,source,target):
    if digest(source)!=trace['source_task_sha256'] or digest(target)!=trace['target_task_sha256']:
        raise ValueError('source/target assembled task differs from executed trace')
    src=json.loads(Path(source).read_text());tgt=json.loads(Path(target).read_text())
    sblocks={b['run_id']:b for b in src['eval']};tblocks={b['run_id']:b for b in tgt['eval']}
    threshold=tgt['quality_definition']['threshold'];budget=trace['action_priority']['budget_at_most']
    md_by_id={md['id']:md for md in tgt['action_metadata']}
    support=[md_by_id[aid]['hardware_supported'] is True for aid in tgt['action_ids']]
    def record(block,mask):return next(r for r in block['coalitions'] if r['mask']==mask)
    def gate(sb,tb,mask):
        br=record(tb,0);r=record(tb,mask);sr=record(sb,0)
        base=latency(br);sq,_=measured_quality(sr);bq,_=measured_quality(br)
        supported=all(not(mask&(1<<i)) or supported for i,supported in enumerate(support))
        if r['status']!='ok':return {'accepted':False,'raw_gain_percent':None,'deployed_latency_ms':base if bq/sq>=threshold else None}
        q,cmap=measured_quality(r);_,bmap=measured_quality(br)
        if set(cmap)!=set(bmap) or any(cmap[k][1]!=bmap[k][1] for k in bmap):raise ValueError('candidate oracle quality sample/label identity differs')
        tm=latency(r)
        accepted=supported and q/bq>=threshold and q/sq>=threshold and tm<=base
        return {'accepted':accepted,'raw_gain_percent':100*(base-tm)/base,
                'deployed_latency_ms':tm if accepted else (base if bq/sq>=threshold else None),
                'retention_to_source':q/sq,'retention_to_target':q/bq}
    byplan={};oracles=[];randoms={'random_uniform_at_most_k':[],'random_uniform_exact_k':[]}
    for paired in trace['physical_eval']:
        sb=sblocks[paired['source_eval_run_id']];tb=tblocks[paired['target_eval_run_id']]
        sq,smap=measured_quality(record(sb,0));bq,tmap=measured_quality(record(tb,0))
        if set(smap)!=set(tmap) or any(smap[k][1]!=tmap[k][1] for k in smap):raise ValueError('independent paired labels differ')
        base=latency(record(tb,0));possible=[];valid=[]
        for mask in range(1<<len(tgt['action_ids'])):
            if mask.bit_count()>budget:continue
            if any(mask&(1<<i) and not supported for i,supported in enumerate(support)):continue
            g=gate(sb,tb,mask);possible.append((mask,g))
            if g['accepted']:valid.append((g['deployed_latency_ms'],mask))
        best=min(valid) if valid else None
        oracles.append({'run_id':tb['run_id'],'best_feasible_mask':best[1] if best else None,
                        'best_feasible_latency_ms':best[0] if best else None,'feasible_plan_count':len(valid),
                        'allowed_mask_count':len(possible),'scope':'posthoc physically measured target oracle, never used to select plan'})
        for kind,subset in [('random_uniform_at_most_k',possible),('random_uniform_exact_k',[(m,g) for m,g in possible if m.bit_count()==budget])]:
            deploy=[g['deployed_latency_ms'] for _,g in subset]
            raw=[g['raw_gain_percent'] for _,g in subset]
            mean_deploy=statistics.mean(deploy) if subset and all(v is not None for v in deploy) else None
            randoms[kind].append({'run_id':tb['run_id'],'subset_count':len(subset),
                'raw_gain_percent':statistics.mean(raw) if subset and all(v is not None for v in raw) else None,
                'accepted_gain_percent':100*(base-mean_deploy)/base if mean_deploy is not None else None,
                'target_oracle_regret_ms':mean_deploy-best[0] if mean_deploy is not None and best else None,
                'rejection_fraction':statistics.mean(not g['accepted'] for _,g in subset),
                'scope':'exact expected value over physical subset table; not extra independent hardware runs'})
        for name,rows in trace['physical_eval_by_plan'].items():
            matching=[r for r in rows if r['source_eval_run_id']==sb['run_id'] and r['target_eval_run_id']==tb['run_id']]
            if len(matching)!=1:raise ValueError('plan pairing duplicated/missing')
            r=matching[0];g=gate(sb,tb,r['requested_target_mask'])
            if g['accepted']!=r['migration_accepted']:raise ValueError('independent accepted/reject arithmetic disagrees')
            if abs(g['raw_gain_percent']-r['raw_target_optimization_benefit_percent'])>1e-9:raise ValueError('independent raw gain differs')
            byplan.setdefault(name,[]).append({**r,'accepted_target_gain_percent':100*(base-g['deployed_latency_ms'])/base if g['deployed_latency_ms'] is not None else None,
                'target_oracle_regret_ms':g['deployed_latency_ms']-best[0] if g['deployed_latency_ms'] is not None and best else None,
                'oracle_mask':best[1] if best else None})
    summaries={}
    for name,rows in byplan.items():
        summaries[name]={'selected_mask':rows[0]['requested_target_mask'],
            'raw_target_gain_percent':vector([r['raw_target_optimization_benefit_percent'] for r in rows]),
            'accepted_target_gain_percent':vector([r['accepted_target_gain_percent'] for r in rows if r['accepted_target_gain_percent'] is not None]),
            'target_oracle_regret_ms':vector([r['target_oracle_regret_ms'] for r in rows if r['target_oracle_regret_ms'] is not None]),
            'migration_acceptances':sum(r['migration_accepted'] for r in rows),'quality_retention_to_source':[r['candidate_retention_to_source'] for r in rows],
            'quality_retention_to_target':[r['candidate_retention_to_target'] for r in rows],'rows':rows}
    paired_margins={}
    for name,rows in byplan.items():
        if name=='cross_dag':continue
        paired_margins[name]=vector([a['raw_target_optimization_benefit_percent']-b['raw_target_optimization_benefit_percent'] for a,b in zip(byplan['cross_dag'],rows)])
    phi=trace['operator_estimate']['values'];plain=trace['context']['plain_target_minus_source_ms']
    order=sorted(phi,key=lambda p:(-phi[p],p));abs_mass=sum(abs(v) for v in phi.values())
    return {'kind':'independent_physical_migration_trace_summary','model_id':trace['context']['model_id'],'input_shape':trace['context']['input_shape'],
        'source_device_id':trace['context']['source_device_id'],'target_device_id':trace['context']['target_device_id'],
        'operator_count':len(phi),'operator_mass_ms':{'signed':sum(phi.values()),'positive':sum(max(v,0) for v in phi.values()),
            'negative':sum(min(v,0) for v in phi.values()),'absolute':abs_mass,'plain_absolute':sum(abs(v) for v in plain.values()),
            'dag_vs_plain_L1':sum(abs(phi[p]-plain[p]) for p in phi),
            'dag_vs_plain_L1_over_plain_L1':sum(abs(phi[p]-plain[p]) for p in phi)/max(sum(abs(v) for v in plain.values()),1e-12)},
        'physical_calibration_gap_ms':trace['context']['physical_migration_gap_ms'],
        'operator_surrogate_gap_ms':trace['context']['operator_surrogate_gap_ms'],
        'unallocated_physical_runtime_residual_ms':trace['context']['unattributed_physical_gap_residual_ms'],
        'operator_rank':order,'top10_operators':[{'node_id':p,'phi_ms':phi[p],'pair_mean_se_ms':trace['operator_estimate']['standard_errors'][p],'plain_delta_ms':plain[p]} for p in order[:10]],
        'same_surrogate_reference_comparison':trace['reference']['estimate_error'] if trace['reference'] else None,
        'action_priority':trace['action_priority'],'plan_results':summaries,'cross_dag_minus_control_paired_gain_pp':paired_margins,
        'heldout_feasible_oracle':oracles,'random_controls':randoms,
        'quality_sample_interpretation':'100 fixed evaluation images repeated across runs; not 500 independent quality samples',
        'evidence_scope':'Operator attribution is a diagnostic profile game. Physical utilities and regret use actually executed complete target-model action coalitions.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trace',type=Path,required=True);ap.add_argument('--source-task',type=Path,required=True)
    ap.add_argument('--target-task',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError('preserve earlier summary')
    d=analyze(json.loads(a.trace.read_text()),a.source_task,a.target_task)
    d['input_sha256']={'trace':digest(a.trace),'source_task':digest(a.source_task),'target_task':digest(a.target_task),'summarizer':digest(__file__)}
    a.output.write_text(json.dumps(d,indent=2));print(json.dumps({'output':str(a.output),'operator_count':d['operator_count'],'plans':list(d['plan_results'])}))

if __name__=='__main__':main()
