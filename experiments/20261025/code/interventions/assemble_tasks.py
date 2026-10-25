"""Freeze planner input from actually completed independent physical run blocks.

No missing block is synthesized. Ranking metadata uses training profiles only.
"""
import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    groups=defaultdict(list)
    for path in sorted(args.raw_dir.glob('*.json')):
        d=json.loads(path.read_text())
        if d.get('schema_version')!='het.fullmodel.run.v2':continue
        groups[(d['task_id'],d['device_id'])].append((path,d))
    if not groups:raise ValueError('No measured full-model blocks found')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    index=[]
    for (task,device),entries in groups.items():
        entries.sort(key=lambda p:p[1]['run_index'])
        runs=[d['run_index'] for _,d in entries]
        if len(runs)!=len(set(runs)):raise ValueError('Duplicate run indices')
        if set(runs[:3])!={1,2,3} or not any(r>=4 for r in runs):raise ValueError('Need real calibration1-3 and heldout eval blocks')
        assert len({d['weights_state_sha256'] for _,d in entries})==1
        assert len({d['dataset_manifest_sha256'] for _,d in entries})==1
        actions=entries[0][1]['action_ids'];metadata=[]
        train=[d for _,d in entries if d['phase']=='train']
        for aid in actions:
            profiles=[next(a for a in d['action_profiles'] if a['id']==aid) for d in train]
            metadata.append({'id':aid,'module_paths':profiles[0]['module_paths'],
                'hardware_supported':all(a['hardware_supported'] and a['numerical_equivalence'] for a in profiles),
                'operator_latency_ms':statistics.mean(a['baseline']['mean_ms'] for a in profiles),
                'candidate_latency_ms':statistics.mean(a['candidate']['mean_ms'] for a in profiles),
                'local_benefit_ms':statistics.mean(a['local_benefit_ms'] for a in profiles),
                'source_target_delta_ms':None,'quality_delta_prior':0.0,
                'quality_prior_basis':'ConvBN analytic eval-mode identity, independently tested per coalition',
                'profile_run_ids':[d['run_id'] for d in train]})
        payload={'schema_version':'het.measured.coalitions.v2','task_id':task,'device_id':device,
            'model_id':entries[0][1]['model_id'],'input_shape':entries[0][1]['input_shape'],
            'runtime':entries[0][1]['runtime'],'action_ids':actions,'action_metadata':metadata,
            'quality_definition':{'metric':'ImageNet1000_top1_accuracy','dataset_id':entries[0][1]['dataset_id'],
                'threshold':.99,'mode':'retention_ratio'},'train':[],'eval':[],
            'source_artifacts':[{'path':str(p.resolve()),'sha256':sha(p)} for p,_ in entries],
            'graph':entries[0][1]['graph'],'status':'new physical deployment study; not historical FITAS reproduction'}
        # Genuine FX-node costs, including residual add/flatten, measured only in train.
        profiled_train=[d for d in train if d.get('node_profiles')]
        if profiled_train:
            graph=entries[0][1]['graph'];costs={};usable=True
            for node in graph['nodes']:
                rows=[next((r for r in d['node_profiles'] if r['node_id']==node['id']),None) for d in profiled_train]
                if any(r is None or r['status'] not in ['ok','structural_zero_cost'] for r in rows):usable=False;break
                costs[node['id']]=statistics.mean(r['mean_ms'] for r in rows)
            by_module={n['target']:n['id'] for n in graph['nodes'] if n['operation']=='call_module'}
            if usable:
                updates={}
                for md in metadata:
                    conv,bn=md['module_paths'];updates[md['id']]={by_module[conv]:md['candidate_latency_ms'],by_module[bn]:0.0}
                payload['copa_game']={'action_ids':actions,'base_costs_ms':costs,
                    'edges':[[e['source'],e['target']] for e in graph['edges']],
                    'action_cost_updates_ms':updates,'runtime_overhead_ms':0.0,
                    'profile_run_ids':[d['run_id'] for d in profiled_train],
                    'aggregation':'critical_path',
                    'interpretation':'Uncalibrated isolated-node-cost dependency-path proxy. Runtime calibration and held-out coalition audits are separate, not inferred from DAG edges.'}
            else:payload['copa_game_status']='unavailable_due_to_missing_or_failed_real_node_profile'
        else:payload['copa_game_status']='unavailable_no_real_node_profiles'
        for path,d in entries:
            assert d['provenance']=='physical_measurement'
            block={'run_id':d['run_id'],'provenance':d['provenance'],'quality_partition':d['quality_partition'],
                'baseline_quality':d['baseline_quality'],'quality_sample_ids':d['coalitions'][0].get('sample_ids',[]),
                'input_tensor_sha256':d['input_tensor_sha256'],'weights_state_sha256':d['weights_state_sha256'],
                'coalitions':d['coalitions'],'artifact_path':str(path.resolve()),'artifact_sha256':sha(path)}
            # The first shuffled coalition can have failed; use explicit mask0.
            mask0=next(c for c in d['coalitions'] if c['mask']==0)
            block['quality_sample_ids']=mask0.get('sample_ids',[])
            assert block['baseline_quality']==mask0['quality_score']
            payload[d['phase']].append(block)
        train_samples={i for b in payload['train'] for i in b['quality_sample_ids']}
        eval_samples={i for b in payload['eval'] for i in b['quality_sample_ids']}
        assert train_samples and eval_samples and not train_samples&eval_samples
        name=task+'_'+hashlib.sha256(device.encode()).hexdigest()[:10]+'.json'
        out=args.output_dir/name
        if out.exists():raise FileExistsError('Use a new assembly directory to preserve prior freeze')
        out.write_text(json.dumps(payload,indent=2))
        index.append({'file':str(out),'task':task,'device':device,'train_runs':len(payload['train']),
                      'eval_runs':len(payload['eval']),'sha256':sha(out)})
    (args.output_dir/'index.json').write_text(json.dumps(index,indent=2));print(json.dumps(index,indent=2))

if __name__=='__main__':main()
