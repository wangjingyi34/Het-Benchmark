"""Freeze a validation-selected policy, then actually measure selected queries.

The run subcommand rejects files containing benchmark graphs/labels as its
hardware input. Every decision is written before any selected-query timing.
Offline replay metrics and fresh active measurements remain separate artifacts.
"""
import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
from torch.fx import GraphModule

from collect_profiles import build_model,tensor_hash,SafeTracer,Capture,family_for,features,callable_for,timed,sync
from evaluate_profiles import RGAT,MLP,GCN,RELATIONS,build_graph,collate,predict
from structural_attributes import attributes_for_node

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()

def extract_hardware(args):
    if args.output.exists():raise ValueError('do not overwrite hardware descriptor')
    raw=json.loads(args.profile.read_text())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'kind':'hardware_descriptor_only','hardware':raw['hardware'],
        'source_profile_sha256':sha(args.profile),'source_collector_code_sha256':raw['provenance']['code_sha256'],
        'note':'Only independent hardware probes and non-latency device identity; no model/profile targets'},indent=2))

def export_policy(args):
    if args.output.exists():raise ValueError('do not overwrite a frozen policy')
    summary=json.loads(args.summary.read_text())
    matches=[r for r in summary['results'] if r['method']==args.method and r['feature_mode']==args.feature_mode and r['split_id']==args.split_id]
    if len(matches)!=1:raise ValueError('policy selector must identify exactly one validated result entry')
    row=matches[0]
    if not row['fallback']['threshold_selected_only_on_validation']:raise ValueError('policy threshold was not validation-only')
    checkpoints=[]
    for fit in row['seed_fits']:
        path=args.checkpoint_dir/Path(fit['checkpoint']).name if args.checkpoint_dir else Path(fit['checkpoint'])
        if sha(path)!=fit['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch')
        checkpoints.append({'path':str(path.resolve()),'sha256':fit['checkpoint_sha256'],'seed':fit['seed']})
    out={'kind':'validation_frozen_active_fallback_policy','created_utc':utc(),'source_summary_sha256':sha(args.summary),
         'method':args.method,'feature_mode':args.feature_mode,'split_id':args.split_id,
         'protocol':row['protocol'],'train_model_ids':row['train_model_ids'],
         'validation_model_ids':row['validation_model_ids'],'test_model_ids':row['test_model_ids'],
         'shape_ids':[0,1,2],'threshold':row['fallback']['threshold'],
         'threshold_infinite':row['fallback']['threshold_infinite'],
         'validation_budget_fraction':row['fallback']['validation_budget_fraction'],
         'threshold_selected_only_on_validation':True,'checkpoints':checkpoints,
         'evaluation_code_sha256':summary['code_sha256'],
         'source_physical_hashes':[m['sha256'] for m in summary['source_manifest']],
         'source_metadata_hashes':[m['metadata_sidecar_sha256'] for m in summary['source_manifest']],
         'target_latencies_or_test_errors_in_policy':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({'policy':str(args.output),'checkpoints':len(checkpoints),'threshold':out['threshold'],
        'threshold_infinite':out['threshold_infinite'],'test_latencies_used_to_choose_threshold':False}))

def feature_graph(model_id,shape_id,device,hardware):
    model,inputs,meta=build_model(model_id,shape_id);ih=tensor_hash(model,inputs)
    gm=GraphModule(model,SafeTracer().trace(model)).eval().to(device);inputs=[x.to(device) for x in inputs]
    cap=Capture(gm);cap.run(*inputs);sync(device)
    gid=f'{model_id}:s{shape_id}';fxnodes=list(gm.graph.nodes)
    graph={'graph_id':gid,'model_id':model_id,'shape_id':shape_id,**meta,
           'input_weights_sha256':ih,'nodes':[],'edges':[],'records':[]}
    for i,n in enumerate(fxnodes):
        info=cap.info[n.name];family=family_for(gm,n) if n.name in cap.calls else 'Context'
        node={'node_id':f'{gid}/{n.name}','fx_op':n.op,'target':str(n.target),
              'module_path':str(n.target) if n.op=='call_module' else '',
              'family':family,**info,'dependency_ids':[f'{gid}/{p.name}' for p in n.all_input_nodes],
              'features':features(gm,n,info,i,len(fxnodes))}
        node['operator_attributes']=attributes_for_node(gm,node,model_id);graph['nodes'].append(node)
        for pred in n.all_input_nodes:graph['edges'].append({'source':f'{gid}/{pred.name}','target':node['node_id'],'relation':'dependency'})
    return graph,gm,cap

def load_fit(path,expected_sha,dim,policy):
    if sha(path)!=expected_sha:raise ValueError('checkpoint changed after policy freezing')
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['feature_mode']!=policy['feature_mode'] or ck['method']!=policy['method']:raise ValueError('checkpoint policy mismatch')
    if ck['training_model_ids']!=policy['train_model_ids'] or ck['validation_model_ids']!=policy['validation_model_ids']:raise ValueError('training provenance mismatch')
    method=ck['method']
    fit={'kind':method,'scaler':ck['scaler'],'target_scale':ck['target_scale']}
    if method=='ridge':fit['weight']=ck['ridge_weight'];return fit
    if method=='mlp':net=MLP(dim)
    elif method=='gcn':net=GCN(dim)
    else:net=RGAT(dim,drop=RELATIONS.index(method[5:]) if method.startswith('drop_') else None)
    net.load_state_dict(ck['state_dict']);net.eval();fit['net']=net
    return fit

def validate_runtime(h,device):
    if h['backend']!=device.type:raise ValueError('active device backend differs from hardware descriptor')
    if bool(h['descriptor_strict']['host_arm64'])!=(platform.machine() in ['arm64','aarch64']):raise ValueError('host architecture differs')
    if device.type=='cuda':
        if not torch.cuda.is_available():raise ValueError('CUDA unavailable; no CPU fallback')
        prop=torch.cuda.get_device_properties(device)
        if prop.name!=h['hardware_name'] or prop.total_memory!=h['visible_device_memory_bytes']:raise ValueError('GPU model/visible partition differs')
        return {'hardware_name':prop.name,'visible_device_memory_bytes':prop.total_memory,
                'compute_capability':list(torch.cuda.get_device_capability(device))}
    if device.type=='npu':
        from collect_profiles_npu import hf32_flags
        if not torch.npu.is_available():raise ValueError('NPU unavailable; no CPU fallback')
        name=str(torch.npu.get_device_name(device));memory=None
        if hasattr(torch.npu,'get_device_properties'):memory=getattr(torch.npu.get_device_properties(device),'total_memory',None)
        if not memory and hasattr(torch.npu,'mem_get_info'):_,memory=torch.npu.mem_get_info(device)
        if name!=h['hardware_name'] or memory!=h['visible_device_memory_bytes']:raise ValueError('NPU chip/visible memory differs')
        return {'hardware_name':name,'visible_device_memory_bytes':memory,'hf32_flags':hf32_flags()}
    if device.type=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS unavailable; no CPU fallback')
    return {'host_machine':platform.machine(),'host_processor':platform.processor(),'backend':device.type}

def active_run(args):
    if args.output.exists() or args.output.with_suffix('.decisions.json').exists():raise ValueError('do not overwrite an active measurement')
    policy=json.loads(args.policy.read_text())
    if policy['kind']!='validation_frozen_active_fallback_policy' or policy['target_latencies_or_test_errors_in_policy']:
        raise ValueError('invalid active policy')
    desc=json.loads(args.hardware_only.read_text())
    if desc.get('kind')!='hardware_descriptor_only' or 'graphs' in desc or 'records' in desc:
        raise ValueError('run accepts hardware-only features; benchmark target records are forbidden')
    h=desc['hardware']
    if args.device=='npu':import torch_npu  # noqa: register device alias before constructing torch.device
    device=torch.device(args.device)
    torch.set_num_threads(h['threads']);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    observed=validate_runtime(h,device);started=time.monotonic();decisions=[];graph_info=[];failures=[];fits=None
    with torch.inference_mode():
        # Stage1 does shape propagation and frozen CPU predictor inference only.
        # It records no target operator latency and does not load a raw benchmark.
        for model_id in policy['test_model_ids']:
            for shape_id in policy['shape_ids']:
                try:
                    graph,gm,cap=feature_graph(model_id,shape_id,device,h)
                    built=build_graph(graph,h,policy['feature_mode'],20,unlabelled=True)
                    batch=collate([built],'cpu');assert torch.isnan(batch['y']).all()
                    if fits is None:
                        fits=[load_fit(Path(c['path']),c['sha256'],batch['x'].shape[1],policy) for c in policy['checkpoints']]
                    predictions=np.stack([predict(f,batch) for f in fits]);mu=predictions.mean(0)
                    sd=predictions.std(0,ddof=1) if len(fits)>1 else np.zeros_like(mu)
                    threshold=math.inf if policy['threshold_infinite'] else policy['threshold']
                    for meta,p,s in zip(batch['meta'],mu,sd):
                        uncertainty=float(s/max(float(p),1e-6))
                        decisions.append({'record_id':f"{h['device_id']}/{meta['node_id']}",
                            'node_id':meta['node_id'],'model_id':model_id,'shape_id':shape_id,
                            'prediction_ms':float(p),'uncertainty':uncertainty,
                            'trigger_measurement':bool(uncertainty>threshold),
                            'input_weights_sha256':graph['input_weights_sha256']})
                    graph_info.append({'graph_id':graph['graph_id'],'model_id':model_id,'shape_id':shape_id,
                        'input_weights_sha256':graph['input_weights_sha256'],
                        'unlabelled_structural_features_sha256':hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest()})
                    del gm,cap,batch,graph,built
                    if device.type=='cuda':torch.cuda.empty_cache()
                except Exception as e:failures.append({'stage':'feature_construction','model_id':model_id,'shape_id':shape_id,'error':type(e).__name__+': '+str(e)[:500]})
        freeze={'kind':'active_fallback_decisions_frozen_before_selected_query_timing','frozen_utc':utc(),
            'policy_sha256':sha(args.policy),'hardware_only_sha256':sha(args.hardware_only),
            'source_code_sha256':sha(__file__),'collector_code_sha256':sha(Path(__file__).with_name('collect_profiles.py')),
            'device_id':h['device_id'],'runtime':observed,'python':sys.version,'torch':torch.__version__,
            'graphs':graph_info,'decisions':decisions,'failures':failures,
            'decision_phase_wall_seconds':time.monotonic()-started,'operator_latency_labels_read':False}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        frozen_path=args.output.with_suffix('.decisions.json');frozen_path.write_text(json.dumps(freeze,indent=2))
        frozen_sha=sha(frozen_path);measure_started=time.monotonic();records=[]
        selected={(r['model_id'],r['shape_id'],r['node_id']):r for r in decisions if r['trigger_measurement']}
        # Stage2 rebuilds one selected graph at a time to avoid retaining all
        # model activations. It may now measure only the already frozen queries.
        for item in graph_info:
            relevant=[r for key,r in selected.items() if key[:2]==(item['model_id'],item['shape_id'])]
            if not relevant:continue
            try:
                graph,gm,cap=feature_graph(item['model_id'],item['shape_id'],device,h)
                if graph['input_weights_sha256']!=item['input_weights_sha256']:raise ValueError('input/weights changed between decision and measurement')
                random.Random(1025+item['shape_id']).shuffle(relevant)
                for decision in relevant:
                    name=decision['node_id'].rsplit('/',1)[-1];node,nargs,nkwargs=cap.calls[name]
                    row={**decision,'status':'measured','policy_sha256':freeze['policy_sha256'],
                         'frozen_decisions_sha256':frozen_sha,'warmup':args.warmup,
                         'inner_repeats':args.inner,'timing_units':'fresh synchronized isolated-call block means, ms'}
                    before=time.monotonic()
                    try:row.update(timed(callable_for(gm,node,nargs,nkwargs),device,args.warmup,args.repeats,args.inner))
                    except Exception as e:row['status']='failed';row['error']=type(e).__name__+': '+str(e)[:500]
                    row['measurement_wall_seconds_including_warmup']=time.monotonic()-before;records.append(row)
                del graph,gm,cap
                if device.type=='cuda':torch.cuda.empty_cache()
            except Exception as e:failures.append({'stage':'selected_graph_measurement','graph_id':item['graph_id'],'error':type(e).__name__+': '+str(e)[:500]})
        out={'kind':'physically_executed_active_fallback','completed_utc':utc(),
             'policy_sha256':freeze['policy_sha256'],'frozen_decisions_sha256':frozen_sha,
             'device_id':h['device_id'],'runtime':observed,'query_count':len(decisions),
             'triggered_count':len(selected),'successful_measurements':sum(r['status']=='measured' for r in records),
             'failed_measurements':sum(r['status']!='measured' for r in records),'failures':failures,'records':records,
             'decision_phase_wall_seconds':freeze['decision_phase_wall_seconds'],
             'measurement_phase_wall_seconds_including_graph_setup':time.monotonic()-measure_started,
             'individual_measurement_wall_seconds_including_warmup':sum(r['measurement_wall_seconds_including_warmup'] for r in records),
             'scope':'Fresh selected-query execution. No global pure/hybrid MRE is inferred from unmeasured query labels; offline replay is a separate experiment.',
             'unmeasured_labels_used_or_invented':False}
        args.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['records','failures']},ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    h=sub.add_parser('hardware');h.add_argument('--profile',required=True,type=Path);h.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('policy');p.add_argument('--summary',required=True,type=Path);p.add_argument('--method',default='rgat')
    p.add_argument('--feature-mode',default='probe');p.add_argument('--split-id',default='model-fold0')
    p.add_argument('--checkpoint-dir',type=Path);p.add_argument('--output',required=True,type=Path)
    r=sub.add_parser('run');r.add_argument('--policy',required=True,type=Path);r.add_argument('--hardware-only',required=True,type=Path)
    r.add_argument('--device',required=True,choices=['cpu','mps','cuda','npu']);r.add_argument('--output',required=True,type=Path)
    r.add_argument('--repeats',type=int,default=20);r.add_argument('--warmup',type=int,default=10);r.add_argument('--inner',type=int,default=8)
    args=ap.parse_args()
    if args.command=='hardware':extract_hardware(args)
    elif args.command=='policy':export_policy(args)
    else:active_run(args)

if __name__=='__main__':main()
