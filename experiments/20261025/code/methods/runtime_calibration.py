"""Preregistered low-order runtime calibration, never fit on heldout outcomes.

For n=4 actions, only masks of cardinality <=2 (11/16) identify intercept,
single and pair residual terms. Other five coalitions and all eval blocks are
held out. No clipping hides nonpositive extrapolations.
"""
import copy
import itertools
import math
import statistics

from copa_v2 import ActionDAGGame,exact_shapley,estimate_shapley,improvement_game,finite,signed_l1_and_ranks
from measured_planner import mean_latency


def low_order_masks(n):return [m for m in range(1<<n) if m.bit_count()<=2]


def truncated_quadratic_residual(base_game,calibration_latency,n):
    required=low_order_masks(n)
    if any(m not in calibration_latency for m in required):raise ValueError('low-order calibration rows missing')
    residual={m:finite(calibration_latency[m],f'calibration[{m}]')-base_game.raw_runtime(m) for m in required}
    intercept=residual[0]
    main={i:residual[1<<i]-intercept for i in range(n)}
    pairs={(i,j):residual[(1<<i)|(1<<j)]-intercept-main[i]-main[j] for i in range(n) for j in range(i+1,n)}
    def predict(mask):
        return intercept+sum(v for i,v in main.items() if mask&(1<<i))+sum(v for (i,j),v in pairs.items() if mask&(1<<i) and mask&(1<<j))
    return predict,{'kind':'second_order_binary_mobius_residual','intercept_ms':intercept,
                    'main_ms':{str(i):v for i,v in main.items()},
                    'pair_ms':{f'{i},{j}':v for (i,j),v in pairs.items()},
                    'fit_masks':required,'withheld_masks':[m for m in range(1<<n) if m not in required],
                    'hyperparameters':'fixed order2, no tuning, no clipping',
                    'constant_fit_ms':statistics.mean(residual.values())}


def build_runtime_variants(table):
    payload=table.payload.get('copa_game_uncalibrated',table.payload.get('copa_game'))
    if not payload:raise ValueError('measured uncalibrated DAG node profiles required')
    profile_runs=set(payload.get('profile_run_ids',[]));train_runs={b['run_id'] for b in table.splits['train']}
    if not profile_runs or profile_runs-train_runs:raise ValueError('DAG profiles must identify train-only run IDs')
    actions=list(table.actions);n=table.n
    def make(aggregation,overhead):
        return ActionDAGGame(actions,payload['base_costs_ms'],payload['edges'],
                             payload['action_cost_updates_ms'],overhead,aggregation)
    # Only required low-order masks are read while fitting. High-order masks
    # are deliberately read later by evaluate_runtime_variants.
    calibration={m:statistics.mean(mean_latency(b['records'][m]) for b in table.splits['train']) for m in low_order_masks(n)}
    dag=make('critical_path',0.0);serial=make('sum',0.0)
    dag_overhead,dag_fit=truncated_quadratic_residual(dag,calibration,n)
    sum_overhead,sum_fit=truncated_quadratic_residual(serial,calibration,n)
    variants={'dag_quadratic_runtime':make('critical_path',dag_overhead),
              'dag_constant_runtime':make('critical_path',dag_fit['constant_fit_ms']),
              'dag_no_runtime':dag,
              'sum_quadratic_runtime':make('sum',sum_overhead)}
    trace={'fit_run_ids':sorted(train_runs),'fit_masks':low_order_masks(n),
           'fit_coalition_count':len(calibration),'withheld_training_masks':dag_fit['withheld_masks'],
           'evaluation_access_during_fit':False,'dag_fit':dag_fit,'sum_fit':sum_fit,
           'comparison_budget':'all variants offered the same low-order end-to-end measurements and node profiles',
           'clipping_policy':'none; nonpositive predictions retained and variant attribution marked invalid'}
    return variants,trace


def phi_error(reference,predicted):
    result=signed_l1_and_ranks(reference,predicted)
    denominator=sum(abs(v) for v in reference.values())
    result['l1_over_reference_abs_sum']=sum(abs(reference[k]-predicted[k]) for k in reference)/denominator if denominator>1e-12 else None
    result['reference_l1_denominator_ms']=denominator
    return result


def evaluate_runtime_variants(table,budget=2,total_permutations=500,sampling_seeds=(11,22,33,44,55)):
    variants,trace=build_runtime_variants(table);n=table.n
    result={'fit':trace,'variants':{},'scope':'new v2 coalition-dependent runtime correction, not reproduction of historical 0.98%'}
    heldout_train=trace['withheld_training_masks']
    train_truth={m:statistics.mean(mean_latency(b['records'][m]) for b in table.splits['train']) for m in heldout_train}
    for name,game in variants.items():
        raw={m:game.raw_runtime(m) for m in range(1<<n)}
        invalid=[m for m,v in raw.items() if not math.isfinite(v) or v<=0]
        record={'predictions_ms':raw,'nonpositive_or_nonfinite_masks':invalid,'clamped_predictions':0,
                'heldout_train_masks':heldout_train,
                'heldout_train_coalition_mae_ms':statistics.mean(abs(raw[m]-train_truth[m]) for m in heldout_train) if heldout_train else None,
                'eval':[],'sampler':[]}
        exact=None
        if not invalid:
            exact=exact_shapley(table.actions,improvement_game(game))
            record['exact_surrogate_shapley']=exact
            order=sorted(range(n),key=lambda i:(-exact['values'][table.actions[i]],i))
            eligible=[i for i in order if table.supported[i] and exact['values'][table.actions[i]]>0][:budget]
            record['selected_mask']=sum(1<<i for i in eligible)
            for seed in sampling_seeds:
                estimate=estimate_shapley(table.actions,improvement_game(game),total_permutations,seed,True)
                record['sampler'].append({'seed':seed,'comparison':'same frozen surrogate exact reference',
                                          'sampling_error':phi_error(exact['values'],estimate['values']),**estimate})
        else:record['attribution_status']='invalid_proxy_requires_new_measurement_no_silent_clamp'
        for block in table.splits['eval']:
            truth={m:mean_latency(block['records'][m]) for m in range(1<<n)}
            physical=exact_shapley(table.actions,lambda m:truth[0]-truth[m])
            row={'run_id':block['run_id'],'physical_exact_shapley':physical,
                 'all_coalition_mae_ms':statistics.mean(abs(raw[m]-truth[m]) for m in range(1<<n)),
                 'all_coalition_mre':statistics.mean(abs(raw[m]-truth[m])/truth[m] for m in range(1<<n)),
                 'unfitted_high_order_mae_ms':statistics.mean(abs(raw[m]-truth[m]) for m in heldout_train) if heldout_train else None}
            if exact is not None:
                row['physical_attribution_error']=phi_error(physical['values'],exact['values'])
                row['frozen_plan_evaluation']=table.final_gate(block,record['selected_mask'])
            record['eval'].append(row)
        result['variants'][name]=record
    result['interpretation']=[
        'Lower-order fit rows are calibration construction, not validation accuracy.',
        'Five high-order masks and every eval block are excluded from runtime fitting.',
        'Sampler error compares antithetic estimate to exact SAME SURROGATE Shapley.',
        'Physical attribution error compares frozen surrogate Shapley to independently timed eval coalitions.',
        'A DAG critical path remains a surrogate, not proof of simultaneous hardware execution.',
        'This small complete table does not establish query efficiency for thousands of operators.'
    ]
    return result
