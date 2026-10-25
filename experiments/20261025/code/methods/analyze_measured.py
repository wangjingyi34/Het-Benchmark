"""Train on physical calibration tables and score frozen plans on heldout runs."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from measured_planner import (MeasuredTable,train_ppo,select_plan,evaluate_plan,
                              fixed_rank_mask,attribution_rankings,uniform_random_expectation,
                              feasible_oracle,calibration_search)
from runtime_calibration import evaluate_runtime_variants


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--budget',type=int,default=2);ap.add_argument('--steps',type=int,default=4096)
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55])
    ap.add_argument('--with-embedding',action='store_true')
    args=ap.parse_args()
    if args.output.exists():raise ValueError('refuse to overwrite an earlier result')
    torch.set_num_threads(1)
    table=MeasuredTable(json.loads(args.input.read_text()))
    results={'schema_version':'het.planner.results.v2','task_id':table.payload['task_id'],
             'device_id':table.payload['device_id'],'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
             'budget':args.budget,'scope':'New offline PPO on measured calibration coalitions; heldout runs scored after plan selection.',
             'source_study_scope':table.payload.get('study_scope',table.payload.get('status','see source artifacts')),
             'quality_definition':table.payload['quality_definition'],
             'quality_sample_disjointness_checked':table.quality_sample_disjointness_checked,
             'training_runs':[b['run_id'] for b in table.splits['train']],
             'evaluation_runs':[b['run_id'] for b in table.splits['eval']],
             'fixed_strategies':{},'ppo':[]}
    partial=args.output.with_name(args.output.stem+'.partial.json')
    def save_partial():
        # Durable progress only; reward, selection and evaluation are unchanged.
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temp=partial.with_suffix('.tmp');temp.write_text(json.dumps(results,indent=2));temp.replace(partial)
    for name,field,pos in [('latency','operator_latency_ms',False),('local_profile_difference','local_benefit_ms',True),
                           ('source_target_difference','source_target_delta_ms',True)]:
        mask=fixed_rank_mask(table,field,args.budget,pos)
        results['fixed_strategies'][name]={'status':'not_available_missing_profile' if mask is None else 'evaluated',
                                           'mask':mask,'evaluation':None if mask is None else evaluate_plan(table,mask)}
    attributions=attribution_rankings(table,args.budget)
    for name,values in attributions.items():
        results['fixed_strategies'][name]={'status':'evaluated','attribution':values,
                                           'mask':values['mask'],'evaluation':evaluate_plan(table,values['mask'])}
    for name,greedy in [('same_evidence_greedy',True),('same_evidence_exhaustive_search',False)]:
        mask=calibration_search(table,args.budget,greedy)
        results['fixed_strategies'][name]={'status':'evaluated' if mask is not None else 'no_feasible_calibration_plan',
                                           'mask':mask,'evaluation':None if mask is None else evaluate_plan(table,mask)}
    results['uniform_random_exact_k_expectation']=uniform_random_expectation(table,args.budget,False)
    results['uniform_random_at_most_k_expectation']=uniform_random_expectation(table,args.budget,True)
    results['heldout_feasible_oracle']=feasible_oracle(table,args.budget)
    if table.payload.get('copa_game_uncalibrated') or table.payload.get('copa_game'):
        runtime=evaluate_runtime_variants(table,args.budget)
        results['runtime_ablation']=runtime
        for name,v in runtime['variants'].items():
            mask=v.get('selected_mask')
            results['fixed_strategies'][name]={'status':'evaluated' if mask is not None else 'invalid_unclipped_proxy',
                                               'mask':mask,'evaluation':None if mask is None else evaluate_plan(table,mask)}
    configurations=[('ppo_hardware_mask',True,False,False),('ppo_without_hardware_mask',False,False,False)]
    if args.with_embedding:
        configurations += [('ppo_with_heldout_embedding',True,True,False),
                           ('ppo_embedding_zeroed_same_capacity',True,True,True)]
    else:results['embedding_ablation_status']='not_run_without_frozen_nonzero_heldout_embedding'
    checkpoints=args.output.parent/(args.output.stem+'-checkpoints');checkpoints.mkdir(parents=True,exist_ok=True)
    results['status']='running';save_partial()
    for name,mask_flag,embedding,drop_embedding in configurations:
        for seed in args.seeds:
            model,log=train_ppo(table,args.budget,seed,args.steps,mask_flag,embedding,drop_embedding=drop_embedding)
            plan=select_plan(model,table,args.budget,mask_flag,embedding,drop_embedding)
            evaluation=evaluate_plan(table,plan['mask'])
            path=checkpoints/f'{name}-seed{seed}.pt';torch.save(model.state_dict(),path)
            results['ppo'].append({'configuration':name,**log,'plan':plan,'evaluation':evaluation,
                                   'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            save_partial()
    results['interpretation']=[
        'Eval measurements are never used as rewards, observations or plan-selection features.',
        'All strategies share the final quality and latency-nonregression rejection gate; raw adverse effects remain in records.',
        'Action budget is at most k with stop/no-op; both uniform exact-k and uniform at-most-k random expectations are reported.',
        'PPO state and reward both use the same measured calibration-run means; there is no hidden sampled-run context.',
        'Different PPO seeds are policy-training repeats, not independent hardware or model samples.',
        'Measured-calibration Shapley uses the full calibration table; it is not a low-query surrogate result.',
        'Greedy and exhaustive calibration comparators receive the same complete measured table as PPO.',
        'Embedding-zeroed control requires real frozen embedding provenance and uses exactly the same policy dimensions as embedding-enabled PPO.',
        'Runtime calibration is fixed second-order and reads only train masks of order<=2; high-order masks and eval blocks never enter fitting.',
        'Oracle is posthoc and random baseline is exact subset expectation, not independent physical repeats.',
        'If all primary actions are supported, hardware-mask ablation is uninformative about constraint benefits.',
        'This implements new v2 methods and does not retroactively authenticate submitted historical results.'
    ]
    results['status']='completed'
    results['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(results,indent=2))
    print(json.dumps({'output':str(args.output),'task':results['task_id'],'ppo_runs':len(results['ppo']),
                      'fixed_strategies':list(results['fixed_strategies'])}))


if __name__=='__main__':main()
