"""Cross-hardware operator diagnosis -> actionable target intervention trace.

Source/target profiles are separate physical measurements. Their hybrid DAG
game is a diagnostic surrogate, never claimed to execute across two devices.
Unknown runtime is not forced to reconcile with physical endpoint measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from copa_v2 import ActionDAGGame,estimate_shapley,exact_shapley,finite,signed_l1_and_ranks


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latency(record):
    if record.get('status')!='ok':raise ValueError('non-ok physical measurement')
    values=record.get('latency_samples_ms')
    if not isinstance(values,list) or not values:raise ValueError('raw timing vector missing')
    values=[finite(v,'timing sample') for v in values]
    if min(values)<=0:raise ValueError('nonpositive physical timing')
    mean=statistics.mean(values)
    if 'mean_ms' in record and abs(finite(record['mean_ms'],'stored mean')-mean)>max(1e-9,mean*1e-9):
        raise ValueError('stored mean disagrees with raw timing')
    return mean


def canonical_graph(graph):
    """Remove process addresses from function reprs, never change graph names."""
    nodes=[];ids=set()
    for node in graph['nodes']:
        nid=node['id']
        if nid in ids:raise ValueError('duplicate FX node ID')
        ids.add(nid);target=str(node['target'])
        if node['operation'] in ['call_function','call_method']:
            target=re.sub(r'0x[0-9a-fA-F]+','<process-address>',target)
        nodes.append({'id':nid,'operation':node['operation'],'target':target,
                      'shape':node.get('shape'),'dtype':node.get('dtype')})
    edges=sorted({(e['source'],e['target']) for e in graph['edges']})
    if any(a not in ids or b not in ids for a,b in edges):raise ValueError('FX edge references unknown node')
    return {'nodes':nodes,'edges':edges}


def validate_raw_group(raws,allow_unit_fixtures=False):
    if not raws:raise ValueError('no physical raw blocks')
    reference=raws[0];required=['task_id','model_id','input_shape','weights_state_sha256',
        'input_tensor_sha256','dataset_manifest_sha256','dataset_id','model_configuration']
    run_ids=set();graph=canonical_graph(reference['graph'])
    for d in raws:
        if d.get('schema_version')!='het.fullmodel.run.v2':raise ValueError('not a full-model raw measurement')
        if d.get('provenance')!='physical_measurement' and not(allow_unit_fixtures and d.get('provenance')=='UNIT_FIXTURE'):
            raise ValueError('physical provenance required')
        if d.get('phase')!='train' or d.get('quality_partition')!='calibration' or not 1<=d.get('run_index',0)<=3:
            raise ValueError('only calibration raw blocks may build migration profiles')
        if not d.get('run_id') or d['run_id'] in run_ids:raise ValueError('duplicate/missing calibration run')
        run_ids.add(d['run_id'])
        for field in required:
            if field not in d or d[field]!=reference[field]:raise ValueError(f'raw identity mismatch: {field}')
        for field in ['weights_state_sha256','input_tensor_sha256','dataset_manifest_sha256']:
            if not isinstance(d[field],str) or not re.fullmatch(r'[0-9a-f]{64}',d[field]):raise ValueError(f'missing real SHA256: {field}')
        if canonical_graph(d['graph'])!=graph:raise ValueError('raw dependency graph/shape/dtype mismatch')
        if d.get('device_id')!=reference.get('device_id') or not d.get('device_id'):raise ValueError('mixed device group')
        protocol=d.get('protocol',{})
        if protocol.get('dtype')!='float32' or protocol.get('tf32') is not False:
            raise ValueError('cross-hardware v2 protocol requires explicit FP32 and TF32 off')
        if protocol.get('mps_fallback_env')=='1':raise ValueError('silent MPS CPU fallback forbidden')
        if not protocol.get('source_sha256'):raise ValueError('raw executing source hash missing')
        baseline=[r for r in d['coalitions'] if r['mask']==0]
        if len(baseline)!=1:raise ValueError('one physical baseline required')
        latency(baseline[0])
    return reference,graph


def node_costs(raws,graph):
    profiled=[d for d in raws if d.get('node_profiles')]
    if not profiled:raise ValueError('no actual calibration FX node profiles')
    costs={};statuses={};per_run={}
    for d in profiled:
        rows={r['node_id']:r for r in d['node_profiles']}
        if len(rows)!=len(d['node_profiles']):raise ValueError('duplicate node profile ID')
        if set(rows)!={n['id'] for n in graph['nodes']}:raise ValueError('node profile ledger incomplete')
        per_run[d['run_id']]=rows
    for node in graph['nodes']:
        nid=node['id'];rows=[per_run[d['run_id']][nid] for d in profiled]
        state={r['status'] for r in rows}
        if state=={'structural_zero_cost'}:
            if node['operation'] in ['call_module','call_function','call_method']:
                raise ValueError('executed operator cannot silently get structural zero cost')
            if any(r.get('mean_ms')!=0 for r in rows):raise ValueError('nonzero structural cost')
            costs[nid]=0.;statuses[nid]='structural_zero_cost'
        elif state=={'ok'}:costs[nid]=statistics.mean(latency(r) for r in rows);statuses[nid]='ok'
        else:raise ValueError(f'missing/failed/inconsistent node profiling: {nid}')
    return costs,statuses,[d['run_id'] for d in profiled]


def build_operator_migration_game(source_raws,target_raws,allow_unit_fixtures=False):
    src,graph=validate_raw_group(source_raws,allow_unit_fixtures)
    tgt,tgraph=validate_raw_group(target_raws,allow_unit_fixtures)
    if src['device_id']==tgt['device_id']:raise ValueError('migration source and target must differ')
    for field in ['task_id','model_id','input_shape','weights_state_sha256','input_tensor_sha256',
                  'dataset_manifest_sha256','dataset_id','model_configuration']:
        if src[field]!=tgt[field]:raise ValueError(f'source-target mismatch: {field}')
    if graph!=tgraph:raise ValueError('source-target dependency DAG mismatch')
    sc,ss,sruns=node_costs(source_raws,graph);tc,ts,truns=node_costs(target_raws,graph)
    if ss!=ts:raise ValueError('source-target execution-status semantics differ')
    players=[n['id'] for n in graph['nodes'] if ss[n['id']]=='ok']
    if not players:raise ValueError('no measured executed operators')
    updates={n:{n:tc[n]} for n in players}
    game=ActionDAGGame(players,sc,graph['edges'],updates,runtime_overhead=0.,aggregation='critical_path')
    full=(1<<len(players))-1
    sfull=statistics.mean(latency(next(r for r in d['coalitions'] if r['mask']==0)) for d in source_raws)
    tfull=statistics.mean(latency(next(r for r in d['coalitions'] if r['mask']==0)) for d in target_raws)
    context={'source_device_id':src['device_id'],'target_device_id':tgt['device_id'],
        'model_id':src['model_id'],'task_id':src['task_id'],'input_shape':src['input_shape'],
        'weights_state_sha256':src['weights_state_sha256'],'input_tensor_sha256':src['input_tensor_sha256'],
        'dataset_manifest_sha256':src['dataset_manifest_sha256'],'dataset_id':src['dataset_id'],
        'model_configuration':src['model_configuration'],'graph':graph,'players':players,
        'source_node_costs_ms':sc,'target_node_costs_ms':tc,
        'plain_target_minus_source_ms':{n:tc[n]-sc[n] for n in players},
        'source_profile_run_ids':sruns,'target_profile_run_ids':truns,
        'source_baseline_run_ids':[d['run_id'] for d in source_raws],
        'target_baseline_run_ids':[d['run_id'] for d in target_raws],
        'source_physical_mean_ms':sfull,'target_physical_mean_ms':tfull,
        'physical_migration_gap_ms':tfull-sfull,
        'source_operator_path_ms':game(0),'target_operator_path_ms':game(full),
        'operator_surrogate_gap_ms':game(full)-game(0),
        'unattributed_physical_gap_residual_ms':(tfull-sfull)-(game(full)-game(0)),
        'runtime_term':'unknown across-hardware hybrid; omitted rather than anchored or rescaled',
        'endpoint_anchor_applied':False,'profile_value_status':'mixed physical profiles in a non-executable diagnostic surrogate',
        'source_runtime':[d.get('runtime',{}) for d in source_raws],
        'target_runtime':[d.get('runtime',{}) for d in target_raws]}
    return game,context


def grouped_action_priority(target_task,context,estimate,budget=2):
    if isinstance(budget,bool) or not isinstance(budget,int) or not 0<budget<=len(target_task['action_ids']):
        raise ValueError('invalid at-most action budget')
    if target_task['task_id']!=context['task_id'] or target_task['device_id']!=context['target_device_id']:
        raise ValueError('target action task identity mismatch')
    if target_task.get('model_id')!=context['model_id'] or target_task.get('input_shape')!=context['input_shape']:
        raise ValueError('target action model/shape mismatch')
    target_train={b['run_id'] for b in target_task['train']}
    if set(context['target_profile_run_ids'])-target_train:raise ValueError('migration target profiles outside calibration')
    players=context['players'];index={p:i for i,p in enumerate(players)}
    module_nodes={}
    for n in context['graph']['nodes']:
        if n['operation']=='call_module':module_nodes.setdefault(n['target'],[]).append(n['id'])
    rows=[];units=estimate['unit_values']
    for md in target_task['action_metadata']:
        if not md.get('module_paths'):raise ValueError('action module identities missing')
        if not md.get('profile_run_ids') or set(md['profile_run_ids'])-target_train:
            raise ValueError('action headroom profiles must be calibration-only')
        nodes=[]
        for path in md['module_paths']:
            if path not in module_nodes:raise ValueError(f'action module missing from actual DAG: {path}')
            nodes.extend(module_nodes[path])
        if len(set(nodes))!=len(nodes) or any(n not in index for n in nodes):raise ValueError('ambiguous/unmeasured action node mapping')
        grouped_units=[sum(u[index[n]] for n in nodes) for u in units]
        phi=sum(estimate['values'][n] for n in nodes)
        se=statistics.stdev(grouped_units)/math.sqrt(len(grouped_units)) if len(grouped_units)>1 else None
        base=finite(md.get('operator_latency_ms'),'target baseline local profile')
        candidate=finite(md.get('candidate_latency_ms'),'target candidate local profile')
        if base<=0 or candidate<=0:raise ValueError('positive measured local profiles required')
        headroom=base-candidate
        if 'local_benefit_ms' in md and abs(finite(md['local_benefit_ms'],'local headroom')-headroom)>max(1e-9,base*1e-8):
            raise ValueError('local profile headroom arithmetic mismatch')
        ratio=max(headroom,0.)/base
        precision=abs(phi)/(abs(phi)+se) if se is not None and abs(phi)+se>0 else 0.
        rows.append({'action_id':md['id'],'module_paths':md['module_paths'],'operator_node_ids':nodes,
            'grouped_migration_phi_ms':phi,'grouped_pair_mean_se_ms':se,
            'grouped_plain_target_minus_source_ms':sum(context['plain_target_minus_source_ms'][n] for n in nodes),
            'target_local_baseline_ms':base,'target_local_candidate_ms':candidate,
            'raw_target_headroom_ms':headroom,'positive_headroom_ratio':ratio,
            'mc_precision_weight':precision,'hardware_supported':md.get('hardware_supported') is True,
            'headroom_profile_run_ids':md['profile_run_ids']})
    positive=sum(max(r['grouped_migration_phi_ms'],0.) for r in rows)
    fallback=positive<=1e-12
    for r in rows:
        r['positive_migration_burden_share']=max(r['grouped_migration_phi_ms'],0.)/positive if not fallback else 0.
        r['priority_score']=(r['raw_target_headroom_ms'] if fallback else
            r['positive_migration_burden_share']*r['positive_headroom_ratio']*r['mc_precision_weight']) if r['hardware_supported'] else None
    order=sorted([r for r in rows if r['hardware_supported'] and r['priority_score']>0],key=lambda r:(-r['priority_score'],r['action_id']))
    selected=[r['action_id'] for r in order[:budget]]
    mask=sum(1<<target_task['action_ids'].index(a) for a in selected)
    return {'actions':rows,'ordered_action_ids':[r['action_id'] for r in order],'selected_action_ids':selected,
        'selected_mask':mask,'budget_at_most':budget,
        'selection_source':'source/target calibration node profiles and target calibration headroom only',
        'headroom_fallback_used':fallback,
        'fallback_reason':'no positive grouped migration regression; target-local headroom optimization' if fallback else None,
        'priority_definition':'positive grouped burden share * positive target headroom ratio * MC precision weight, hardware feasibility required',
        'uncertainty_scope':'MC precision conditional on fixed surrogate only; not a calibrated bound on hybrid-surrogate bias',
        'grouping_scope':'linear aggregation of operator attributions, not redefined group-as-player Shapley'}


def matched_control_plans(target_task,priority,budget):
    """Same evidence and <=k budget; no evaluation observations select plans."""
    rows=priority['actions'];plans={}
    plain_positive=sum(max(r['grouped_plain_target_minus_source_ms'],0.) for r in rows)
    for name in ['cross_plain_sum','target_headroom','target_operator_latency']:
        scores={}
        for r in rows:
            if not r['hardware_supported']:continue
            if name=='cross_plain_sum':
                # Analytic sum-game Shapley has no sampling variance; give it
                # precision one. Preserve the same no-regression fallback.
                score=(max(r['grouped_plain_target_minus_source_ms'],0.)/plain_positive*r['positive_headroom_ratio']
                       if plain_positive>1e-12 else r['raw_target_headroom_ms'])
            elif name=='target_headroom':score=r['raw_target_headroom_ms']
            else:score=r['target_local_baseline_ms']
            scores[r['action_id']]=score
        order=sorted([a for a,s in scores.items() if s>0],key=lambda a:(-scores[a],a))
        selected=order[:budget]
        plans[name]={'selected_mask':sum(1<<target_task['action_ids'].index(a) for a in selected),
            'selected_action_ids':selected,'ordered_action_ids':order,'scores':scores,
            'budget_at_most':budget,'selection_source':'same source/target calibration profiles only',
            'headroom_fallback_used':name=='cross_plain_sum' and plain_positive<=1e-12}
    return plans


def measured_quality(record):
    if record.get('status')!='ok' or record.get('quality_evaluated') is not True:raise ValueError('quality was not successfully measured')
    ids=record.get('sample_ids');pred=record.get('predictions');labels=record.get('labels')
    if not ids or not isinstance(pred,list) or not isinstance(labels,list) or len(ids)!=len(pred) or len(ids)!=len(labels):
        raise ValueError('raw labeled prediction ledger missing')
    if len(set(ids))!=len(ids):raise ValueError('duplicate quality sample identity')
    for vector in [pred,labels]:
        if any(isinstance(x,bool) or not isinstance(x,int) or not 0<=x<1000 for x in vector):raise ValueError('invalid ImageNet1000 prediction/label')
    q=sum(a==b for a,b in zip(pred,labels))/len(ids)
    if abs(q-finite(record.get('quality_score'),'reported quality'))>1e-12:raise ValueError('reported accuracy differs from raw predictions')
    return q,{i:(p,l) for i,p,l in zip(ids,pred,labels)}


def paired_migration_quality(source_block,target_block,target_mask,local_gate,threshold=.99):
    """After plan selection only: paired source baseline and target candidate."""
    if local_gate.get('requested_mask')!=target_mask:raise ValueError('local gate belongs to another plan')
    source=source_block['records'][0];baseline=target_block['records'][0];candidate=target_block['records'][target_mask]
    sq,smap=measured_quality(source);tq,tmap=measured_quality(baseline)
    if sq<=0 or tq<=0:raise ValueError('positive paired quality baselines required')
    cq,cmap=measured_quality(candidate)
    if set(smap)!=set(tmap) or set(smap)!=set(cmap):raise ValueError('source/target quality samples differ')
    if any(smap[i][1]!=tmap[i][1] or smap[i][1]!=cmap[i][1] for i in smap):raise ValueError('source/target labels disagree')
    global_retention=cq/sq;local_retention=cq/tq
    accepted=bool(local_gate.get('accepted')) and global_retention>=threshold
    physical_source=latency(source);physical_target=latency(baseline);physical_candidate=latency(candidate)
    return {'source_eval_run_id':source_block['run_id'],'target_eval_run_id':target_block['run_id'],
        'requested_target_mask':target_mask,'quality_sample_count':len(smap),
        'source_baseline_quality':sq,'target_baseline_quality':tq,'target_candidate_quality':cq,
        'candidate_retention_to_source':global_retention,'candidate_retention_to_target':local_retention,
        'source_target_baseline_retention':tq/sq,'source_candidate_prediction_agreement':statistics.mean(smap[i][0]==cmap[i][0] for i in smap),
        'quality_threshold':threshold,'local_target_gate':local_gate,'migration_accepted':accepted,
        'migration_rejection_reason':None if accepted else ('cross_device_quality_below_source_threshold' if global_retention<threshold else local_gate.get('reason')),
        'fallback':'none' if accepted else ('target_baseline' if tq/sq>=threshold else 'original_source_baseline'),
        'source_physical_baseline_ms':physical_source,'target_physical_baseline_ms':physical_target,
        'target_measured_candidate_ms':physical_candidate,
        'physical_gap_before_ms':physical_target-physical_source,
        'physical_gap_after_requested_plan_ms':physical_candidate-physical_source,
        'raw_target_optimization_benefit_percent':100*(physical_target-physical_candidate)/physical_target,
        'interpretation':'paired observed top1 retention, not a population guarantee or executable mixed-hardware coalition'}


def load_raw_artifacts(task,raw_map=None):
    rows=[];manifest=[];raw_map=raw_map or {};seen=set()
    blocks={b['run_id']:(phase,b) for phase in ['train','eval'] for b in task[phase]}
    for item in task.get('source_artifacts',[]):
        original=item['path'];path=Path(raw_map.get(original,original))
        if not path.exists():raise ValueError('raw artifact absent; supply explicit original-path map: '+original)
        if sha(path)!=item['sha256']:raise ValueError('raw artifact hash mismatch: '+str(path))
        d=json.loads(path.read_text())
        if d.get('schema_version')!='het.fullmodel.run.v2' or d.get('provenance')!='physical_measurement':
            raise ValueError('raw artifact not a full-model physical measurement')
        for key in ['task_id','device_id','model_id','input_shape']:
            if d.get(key)!=task.get(key):raise ValueError('raw/assembled identity mismatch: '+key)
        rid=d['run_id']
        if rid in seen or rid not in blocks:raise ValueError('raw run duplicated or absent from assembled split')
        seen.add(rid);phase,block=blocks[rid]
        if d['phase']!=phase or block['coalitions']!=d['coalitions']:
            raise ValueError('assembled split/coalitions disagree with hashed original raw block')
        for key in ['input_tensor_sha256','weights_state_sha256','baseline_quality']:
            if block.get(key)!=d.get(key):raise ValueError('assembled block identity mismatch: '+key)
        if d['phase']=='train':rows.append(d)
        manifest.append({'original_path':original,'resolved_path':str(path.resolve()),'sha256':item['sha256'],
                         'run_id':d['run_id'],'run_index':d['run_index'],'phase':d['phase']})
    if seen!=set(blocks):raise ValueError('not every assembled block has a hashed raw artifact')
    return rows,manifest


def run_bridge(source_task,target_task,source_raws,target_raws,budget=2,total_permutations=500,seed=42,
               reference_permutations=0,reference_seed=90210,allow_unit_fixtures=False):
    game,context=build_operator_migration_game(source_raws,target_raws,allow_unit_fixtures)
    for task,label,device in [(source_task,'source',context['source_device_id']),
                              (target_task,'target',context['target_device_id'])]:
        if task['device_id']!=device or task['task_id']!=context['task_id']:raise ValueError(label+' assembled-task identity mismatch')
    estimate=estimate_shapley(context['players'],game,total_permutations,seed,True)
    priority=grouped_action_priority(target_task,context,estimate,budget)
    plain=context['plain_target_minus_source_ms']
    result={'kind':'new_cross_hardware_operator_diagnosis','context':context,'operator_estimate':estimate,
        'sign_convention':'positive target-minus-source latency regression; negative acceleration',
        'sum_ablation_exact_shapley_ms':plain,
        'dag_vs_plain':signed_l1_and_ranks(plain,estimate['values']),
        'dag_vs_sum_l1_ms':sum(abs(plain[n]-estimate['values'][n]) for n in plain),
        'action_priority':priority,'physical_eval':[],'reference':None,
        'source_quality_definition':source_task['quality_definition'],'target_quality_definition':target_task['quality_definition']}
    result['matched_control_plans']=matched_control_plans(target_task,priority,budget)
    # Internal efficiency is algebraic. Physical mismatch is independently reported.
    result['sample_sum_vs_physical_gap_residual_ms']=sum(estimate['values'].values())-context['physical_migration_gap_ms']
    if len(context['players'])<=16:
        ref=exact_shapley(context['players'],game)
        result['reference']={'kind':'exact same diagnostic surrogate, not physical attribution truth',
                             **ref,'estimate_error':signed_l1_and_ranks(ref['values'],estimate['values'])}
    elif reference_permutations:
        if reference_permutations<=total_permutations:raise ValueError('reference budget must exceed estimate budget')
        # Fresh game avoids treating cache reuse as independent query latency.
        refgame,_=build_operator_migration_game(source_raws,target_raws,allow_unit_fixtures)
        ref=estimate_shapley(context['players'],refgame,reference_permutations,reference_seed,True)
        result['reference']={'kind':'higher-budget independent sampling of SAME diagnostic surrogate, not physical truth',
                             **ref,'estimate_error':signed_l1_and_ranks(ref['values'],estimate['values'])}
    result['interpretation']=[
        'Each operator source/target cost is physically profiled, but hybrid profiles are not executable interventions.',
        'Unknown cross-hardware runtime is left unallocated; no endpoint anchoring or attribution rescaling.',
        'Sum ablation is analytically the plain per-operator target-source difference.',
        'Pair-mean SE measures sampling uncertainty conditional on a fixed proxy, not physical causal uncertainty.',
        'Action grouping/headroom uses calibration data only. Heldout target intervention and paired task quality validate its practical decision after selection.',
        'Same-surrogate higher-budget reference does not validate attribution against hardware truth.'
    ]
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-task',type=Path,required=True)
    ap.add_argument('--target-task',type=Path,required=True);ap.add_argument('--raw-path-map',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--budget',type=int,default=2)
    ap.add_argument('--permutations',type=int,default=500);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--reference-permutations',type=int,default=0);ap.add_argument('--reference-seed',type=int,default=90210)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('preserve previous cross-hardware trace')
    source=json.loads(args.source_task.read_text());target=json.loads(args.target_task.read_text())
    mapping=json.loads(args.raw_path_map.read_text()) if args.raw_path_map else None
    sraw,smanifest=load_raw_artifacts(source,mapping);traw,tmanifest=load_raw_artifacts(target,mapping)
    result=run_bridge(source,target,sraw,traw,args.budget,args.permutations,args.seed,
                      args.reference_permutations,args.reference_seed)
    # Torch/planner import is delayed: pure math factory/unit tests stay lightweight.
    from measured_planner import MeasuredTable
    stable=MeasuredTable(source);ttable=MeasuredTable(target)
    if source['quality_definition']!=target['quality_definition']:raise ValueError('source-target metric/dataset/threshold definition differs')
    source_indices={m['run_id']:m['run_index'] for m in smanifest if m['phase']=='eval'}
    target_indices={m['run_id']:m['run_index'] for m in tmanifest if m['phase']=='eval'}
    source_eval={source_indices[b['run_id']]:b for b in stable.splits['eval']}
    if len(source_eval)!=len(stable.splits['eval']):raise ValueError('duplicate source eval run index')
    mask=result['action_priority']['selected_mask']
    plans={'cross_dag':result['action_priority'],**result['matched_control_plans']}
    result['physical_eval_by_plan']={k:[] for k in plans}
    for target_block in ttable.splits['eval']:
        # Pair using explicit original run_index and immutable sample IDs; do
        # not parse names, guess aliases, or create all-pairs pseudo-replicates.
        idx=target_indices[target_block['run_id']]
        if idx not in source_eval:raise ValueError('no source eval block with matching explicit run_index')
        sb=source_eval[idx]
        if set(sb['quality_sample_ids'])!=set(target_block['quality_sample_ids']):raise ValueError('paired eval quality samples differ')
        for name,plan in plans.items():
            chosen=plan['selected_mask']
            row=paired_migration_quality(sb,target_block,chosen,
                ttable.final_gate(target_block,chosen),target['quality_definition']['threshold'])
            row['paired_run_index']=idx;result['physical_eval_by_plan'][name].append(row)
        result['physical_eval'].append(result['physical_eval_by_plan']['cross_dag'][-1])
    result['eval_pairing']='one-to-one original run_index and exact sample identities; repeated pictures are not new independent accuracy samples'
    result['source_artifacts']=smanifest;result['target_artifacts']=tmanifest
    result['source_task_sha256']=sha(args.source_task);result['target_task_sha256']=sha(args.target_task)
    result['source_sha256']={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'output':str(args.output),'source':result['context']['source_device_id'],
                     'target':result['context']['target_device_id'],'operators':len(result['context']['players']),
                     'selected_mask':mask,'paired_eval_comparisons':len(result['physical_eval'])}))


if __name__=='__main__':main()
