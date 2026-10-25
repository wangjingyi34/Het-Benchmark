"""Infer action embeddings from a frozen checkpoint and an unlabelled FX graph.

No model timing or task-quality label is measured or passed to the predictor.
The architecture's random weights only permit shape propagation; parameter
values are not predictor inputs. This is not an intervention evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.fx import GraphModule
import numpy as np
from collect_profiles import build_model, portable_init, exact_uniform, SafeTracer, Capture, family_for, features, shapeinfo, tensor_hash
from evaluate_profiles import RGAT, build_graph, collate, predict
from structural_attributes import attributes_for_node

ACTION_PATHS=[['conv1','bn1'],['layer1.0.conv1','layer1.0.bn1'],
              ['layer2.0.conv1','layer2.0.bn1'],['layer2.0.downsample.0','layer2.0.downsample.1']]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--hardware-from',required=True,type=Path)
    ap.add_argument('--models',nargs='+',default=['resnet18','resnet50'])
    ap.add_argument('--batches',nargs='+',type=int,default=[1,4]);ap.add_argument('--side',type=int,default=224)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if ck['method']!='rgat':raise ValueError('Action state requires full RGAT checkpoint')
    if set(args.models)&set(ck['training_model_ids']):raise ValueError('Target model was in predictor training; use the model-heldout fold')
    if not set(args.models).issubset(set(ck['test_model_ids'])):
        raise ValueError('Requested action models must be test-held-out, not validation/early-stopping models')
    raw=json.loads(args.hardware_from.read_text());hardware=raw.get('hardware',raw)
    mode=ck['feature_mode'];rows=[]
    for model_id in args.models:
        for batch in args.batches:
            model,_,meta=build_model(model_id,0)
            rng=np.random.Generator(np.random.PCG64(111222+batch));inputs=[exact_uniform((batch,3,args.side,args.side),rng)]
            gm=GraphModule(model,SafeTracer().trace(model)).eval()
            with torch.inference_mode():
                cap=Capture(gm);cap.run(*inputs)
            gid=f'{model_id}:unlabelled-b{batch}-s{args.side}'
            graph={'graph_id':gid,'model_id':model_id,'shape_id':f'b{batch}-s{args.side}',
                   **meta,'input_weights_sha256':tensor_hash(model,inputs),'nodes':[],'edges':[],'records':[]}
            fxnodes=list(gm.graph.nodes)
            for i,n in enumerate(fxnodes):
                info=cap.info[n.name];family=family_for(gm,n) if n.name in cap.calls else 'Context'
                nid=f'{gid}/{n.name}'
                graph['nodes'].append({'node_id':nid,'fx_op':n.op,'target':str(n.target),
                    'module_path':str(n.target) if n.op=='call_module' else '',
                    'family':family,**info,'dependency_ids':[f'{gid}/{p.name}' for p in n.all_input_nodes],
                    'features':features(gm,n,info,i,len(fxnodes))})
                graph['nodes'][-1]['operator_attributes']=attributes_for_node(gm,graph['nodes'][-1],model_id)
                for pred in n.all_input_nodes:graph['edges'].append({'source':f'{gid}/{pred.name}','target':nid,'relation':'dependency'})
            g=build_graph(graph,hardware,mode,minimum_repeats=20,unlabelled=True)
            b=collate([g],'cpu');assert torch.isnan(b['y']).all()
            net=RGAT(b['x'].shape[1]);net.load_state_dict(ck['state_dict']);net.eval()
            fit={'kind':'rgat','net':net,'scaler':ck['scaler'],'target_scale':ck['target_scale']}
            _,embedding=predict(fit,b,embeddings=True)
            bypath={m['module_path']:z for m,z in zip(b['meta'],embedding) if m['module_path']}
            for index,paths in enumerate(ACTION_PATHS):
                if not all(p in bypath for p in paths):raise ValueError('Missing action module path: '+str(paths))
                z=np.stack([bypath[p] for p in paths]).mean(0)
                assert len(z)==32 and np.isfinite(z).all() and np.linalg.norm(z)>0
                rows.append({'model_id':model_id,'input_shape':[batch,3,args.side,args.side],
                    'device_id':hardware['device_id'],'dtype':'float32','action_index':index,'module_paths':paths,
                    'graph_embedding':z.astype(float).tolist(),
                    'embedding_provenance':{'checkpoint_sha256':sha(args.checkpoint),
                       'training_model_ids':ck['training_model_ids'],'split_id':ck['split_id'],
                       'validation_model_ids':ck['validation_model_ids'],'test_model_ids':ck['test_model_ids'],
                       'feature_mode':mode,'unlabelled_fx_graph':True,
                       'target_latency_used_at_inference':False,'task_quality_used_at_inference':False,
                       'used_inference_latency_labels':False,
                       'hardware_source_sha256':sha(args.hardware_from),
                       'hardware_name':hardware['hardware_name'],'backend':hardware['backend'],
                       'hardware_threads':hardware['threads'],
                       'architecture_source':meta['architecture_source'],
                       'feature_graph_weights':'portable random initialization for shape propagation, not task inference',
                       'used_target_labels':False}})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'kind':'frozen_inductive_action_embeddings','records':rows},indent=2))
    print(json.dumps({'output':str(args.output),'records':len(rows),'dimension':32,'target_labels_used':False}))

if __name__=='__main__':main()
