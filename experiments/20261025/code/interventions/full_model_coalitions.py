"""Real pretrained full-model interventions; one invocation is one fresh run block.

Conv-BN fusion is a predefined function-preserving deployment action, not COPA.
All scores are measured; unsupported, numerical and quality failures are retained.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval

ACTIONS=[('stem','conv1','bn1'),
 ('main_early','layer1.0.conv1','layer1.0.bn1'),
 ('main_transition','layer2.0.conv1','layer2.0.bn1'),
 ('shortcut_transition','layer2.0.downsample.0','layer2.0.downsample.1')]

def sha(b):return hashlib.sha256(b).hexdigest()
def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)
    elif device.type=='mps':torch.mps.synchronize()
    elif device.type=='npu':torch.npu.synchronize()

def assign(model,path,value):
    parent,_,name=path.rpartition('.')
    setattr(model.get_submodule(parent) if parent else model,name,value)

def state_hash(model):
    h=hashlib.sha256()
    for key,t in sorted(model.state_dict().items()):h.update(key.encode());h.update(t.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def disable_inplace_modules(model):
    """Keep replay inputs immutable; record this mathematically equivalent config."""
    disabled=[]
    for name,module in model.named_modules():
        if isinstance(getattr(module,'inplace',None),bool) and module.inplace:
            module.inplace=False;disabled.append(name)
    return disabled

def hardware(device):
    if device.type=='cuda':return {'hardware':torch.cuda.get_device_name(0),'memory_bytes':torch.cuda.get_device_properties(0).total_memory,'cuda':torch.version.cuda}
    if device.type=='mps':return {'hardware':'Apple MPS on '+platform.machine(),'mps_available':torch.backends.mps.is_available()}
    if device.type=='npu':return {'hardware':str(torch.npu.get_device_name(0))}
    return {'hardware':'CPU '+platform.processor()+' '+platform.machine()}

def timed(fn,device,warmup,repeats):
    for _ in range(warmup):fn()
    sync(device);values=[]
    for _ in range(repeats):
        sync(device);start=time.perf_counter_ns();fn();sync(device)
        values.append((time.perf_counter_ns()-start)/1e6)
    return {'latency_samples_ms':values,'mean_ms':statistics.mean(values),
            'median_ms':statistics.median(values),'sample_sd_ms':statistics.stdev(values)}

def fx_graph(model,x):
    from torch.fx import symbolic_trace
    from torch.fx.passes.shape_prop import ShapeProp
    gm=symbolic_trace(model);ShapeProp(gm).propagate(x.cpu())
    nodes=[];edges=[]
    for n in gm.graph.nodes:
        tm=n.meta.get('tensor_meta')
        nodes.append({'id':n.name,'operation':n.op,'target':str(n.target),
            'shape':list(tm.shape) if hasattr(tm,'shape') else None,
            'dtype':str(tm.dtype) if hasattr(tm,'dtype') else None})
        edges.extend({'source':p.name,'target':n.name,'relation':'actual_fx_dependency'} for p in n.all_input_nodes)
    return {'nodes':nodes,'edges':edges,'semantics':'Actual torch.fx data-dependency DAG. PyTorch dispatch is not assumed to execute independent branches in parallel.'}

def measured_fx_nodes(base_cpu,x,device,warmup,repeats):
    """Replay actual baseline-node inputs, including functional residual adds.

    These are isolated node costs, not sums asserted to equal full wall clock.
    Only calibration-run profiles may be used to build a prediction game.
    """
    from torch.fx import symbolic_trace,Interpreter
    from torch.utils._pytree import tree_map
    gm=symbolic_trace(copy.deepcopy(base_cpu)).to(device).eval()
    captured={}
    class Capture(Interpreter):
        def run_node(self,n):
            args,kwargs=self.fetch_args_kwargs_from_env(n)
            if n.op in ['call_module','call_function','call_method']:
                captured[n.name]=tree_map(lambda a:a.detach().clone() if isinstance(a,torch.Tensor) else a,(args,kwargs))
            return super().run_node(n)
    Capture(gm).run(x);sync(device)
    rows=[]
    for node in gm.graph.nodes:
        if node.name not in captured:
            rows.append({'node_id':node.name,'operation':node.op,'target':str(node.target),
                         'status':'structural_zero_cost','mean_ms':0.0})
            continue
        args,kwargs=captured[node.name]
        def fn(n=node,a=args,k=kwargs):
            if n.op=='call_module':return gm.get_submodule(str(n.target))(*a,**k)
            if n.op=='call_function':return n.target(*a,**k)
            return getattr(a[0],str(n.target))(*a[1:],**k)
        try:
            t=timed(fn,device,warmup,repeats)
            rows.append({'node_id':node.name,'operation':node.op,'target':str(node.target),
                         'status':'ok',**t})
        except Exception as e:
            rows.append({'node_id':node.name,'operation':node.op,'target':str(node.target),
                         'status':'failed','error':str(e)})
    del gm
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['resnet18','resnet50'],required=True)
    ap.add_argument('--device',choices=['cpu','mps','cuda','npu'],required=True)
    ap.add_argument('--run',type=int,choices=range(1,9),required=True)
    ap.add_argument('--batch',type=int,choices=[1,4],required=True)
    ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--warmup',type=int,default=10);ap.add_argument('--repeats',type=int,default=30)
    ap.add_argument('--quality-batch',type=int,default=16)
    ap.add_argument('--weights',default='IMAGENET1K_V1')
    ap.add_argument('--profile-all-nodes',action='store_true')
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite original run')
    if args.repeats<2:raise ValueError('repeats >=2')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1' and args.device=='mps':raise RuntimeError('MPS CPU fallback must not be silently enabled')
    if args.device=='npu':import torch_npu
    import torchvision
    from torchvision import models
    device=torch.device(args.device)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(42)
    if hasattr(torch.backends,'cuda'):torch.backends.cuda.matmul.allow_tf32=False
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    weights_cls=getattr(models,'ResNet18_Weights' if args.model=='resnet18' else 'ResNet50_Weights')
    weights=weights_cls[args.weights]
    base_cpu=getattr(models,args.model)(weights=weights,progress=True).eval()
    disabled_inplace=disable_inplace_modules(base_cpu)
    for p in base_cpu.parameters():p.requires_grad_(False)
    w_hash=state_hash(base_cpu)
    manifest=json.loads((args.assets/'manifest.json').read_text())
    assert sha((args.assets/'quality_inputs.npz').read_bytes())==manifest['quality_asset_sha256']
    partition='calibration' if args.run<=3 else 'evaluation'
    in_meta=manifest['timing_inputs'][partition+f'_b{args.batch}']
    xnp=np.load(args.assets/in_meta['file'],allow_pickle=False)
    assert sha(xnp.tobytes())==in_meta['tensor_sha256']
    q=np.load(args.assets/'quality_inputs.npz',allow_pickle=False)
    ids=[i for i,s in enumerate(manifest['samples']) if s['partition']==partition]
    q_images=torch.from_numpy(q['images'][ids].copy())
    labels=torch.from_numpy(q['labels'][ids].copy())
    sample_ids=[manifest['samples'][i]['sample_id'] for i in ids]
    task_id=f'{args.model}_b{args.batch}_imagenet_val10class_v2'
    run_id=f'{args.device}_{task_id}_run{args.run}'
    # Trace before moving: this retains real residual edges including shortcut.
    graph=fx_graph(base_cpu,torch.from_numpy(xnp))
    model=copy.deepcopy(base_cpu).to(device).eval();x=torch.from_numpy(xnp).to(device)
    captured={};handles=[]
    def capture(k):
        def hook(module,inputs):
            if inputs and isinstance(inputs[0],torch.Tensor):captured[k]=inputs[0].detach().clone()
        return hook
    for aid,conv,bn in ACTIONS:handles.append(model.get_submodule(conv).register_forward_pre_hook(capture(aid)))
    with torch.inference_mode():model(x);sync(device)
    for h in handles:h.remove()
    def quality(m):
        outputs=[]
        with torch.inference_mode():
            for start in range(0,len(ids),args.quality_batch):
                outputs.append(m(q_images[start:start+args.quality_batch].to(device)).cpu())
        logits=torch.cat(outputs);pred=logits.argmax(dim=1)
        return logits,{'quality_score':int((pred==labels).sum())/len(ids),'quality_evaluated':True,
            'correct_count':int((pred==labels).sum()),
            'sample_ids':sample_ids,'predictions':pred.tolist(),'labels':labels.tolist(),
            'correct':(pred==labels).tolist(),'sample_count':len(ids),
            'logits_sha256':sha(logits.contiguous().numpy().tobytes())}
    started=datetime.now(timezone.utc).isoformat()
    with torch.inference_mode():
        baseline_logits,baseline_q=quality(model)
        if baseline_q['quality_score']<=0:raise RuntimeError('Zero baseline quality cannot define retention')
        standalone=[]
        for aid,conv_path,bn_path in ACTIONS:
            conv=model.get_submodule(conv_path);bn=model.get_submodule(bn_path)
            plain=nn.Sequential(copy.deepcopy(conv),copy.deepcopy(bn)).eval()
            candidate=fuse_conv_bn_eval(copy.deepcopy(conv),copy.deepcopy(bn)).eval()
            z=captured[aid]
            before=plain(z);after=candidate(z);sync(device)
            equivalent=torch.allclose(before,after,atol=1e-4,rtol=1e-4)
            order=[0,1];random.Random(args.run+int(sha(aid.encode())[:8],16)).shuffle(order)
            t={}
            for which in order:t[str(which)]=timed(lambda:(candidate if which else plain)(z),device,args.warmup,args.repeats)
            standalone.append({'id':aid,'module_paths':[conv_path,bn_path],
                'input_shape':list(z.shape),'hardware_supported':True,'numerical_equivalence':bool(equivalent),
                'baseline':t['0'],'candidate':t['1'],'local_benefit_ms':t['0']['mean_ms']-t['1']['mean_ms'],
                'quality_delta_prior':0.0,'prior_basis':'eval ConvBN folding is analytically function preserving; actual quality separately evaluated',
                'source_target_delta_ms':None})
        node_profiles=measured_fx_nodes(base_cpu,x,device,args.warmup,args.repeats) if args.profile_all_nodes else []
        frozen_at=datetime.now(timezone.utc).isoformat()
        # No coalition outcomes have influenced any standalone rank or input choice.
        order=list(range(16));random.Random(int(sha(run_id.encode())[:8],16)).shuffle(order)
        coalitions=[]
        del model
        for mask in order:
            m=copy.deepcopy(base_cpu)
            for i,(aid,cpath,bpath) in enumerate(ACTIONS):
                if mask&(1<<i):
                    fused=fuse_conv_bn_eval(m.get_submodule(cpath),m.get_submodule(bpath))
                    assign(m,cpath,fused);assign(m,bpath,nn.Identity())
            m=m.to(device).eval()
            row={'mask':mask,'action_ids':[ACTIONS[i][0] for i in range(4) if mask&(1<<i)],
                 'artifact_id':run_id+f'_mask{mask}','quality_partition':partition,
                 'provenance':'physical_measurement','status':'ok'}
            try:
                logits,qrow=quality(m);row.update(qrow)
                eq=torch.allclose(logits,baseline_logits,atol=1e-4,rtol=1e-4)
                delta=logits-baseline_logits
                row['equivalence']={'passed':bool(eq),'atol':1e-4,'rtol':1e-4,
                    'max_abs_error':float(delta.abs().max()),'relative_l2_error':float(delta.norm()/baseline_logits.norm().clamp_min(1e-12))}
                row['retention_ratio']=row['quality_score']/baseline_q['quality_score']
                row['quality_threshold_passed']=row['retention_ratio']>=.99
                if not eq:row['status']='failed';row['failure_reason']='numerical_equivalence_failed'
                else:row.update(timed(lambda:m(x),device,args.warmup,args.repeats))
            except Exception as e:
                row.update({'status':'oom' if 'out of memory' in str(e).lower() else 'failed','error':str(e),'quality_evaluated':False})
            coalitions.append(row);del m
            if device.type=='cuda':torch.cuda.empty_cache()
    out={'schema_version':'het.fullmodel.run.v2','task_id':task_id,'run_id':run_id,'run_index':args.run,
        'phase':'train' if args.run<=3 else 'eval','quality_partition':partition,'provenance':'physical_measurement',
        'device_id':args.device+'_'+hardware(device)['hardware'].replace(' ','_'),
        'model_id':args.model,'model_source':'torchvision.models','weights':args.weights,'weights_url':weights.url,
        'model_configuration':{'inplace_disabled_modules':disabled_inplace,
           'reason':'Immutable node-profile replay inputs and consistent out-of-place baseline/candidate inference; not torchvision default allocation configuration'},
        'weights_state_sha256':w_hash,'input_shape':list(xnp.shape),'input_tensor_sha256':in_meta['tensor_sha256'],
        'dataset_manifest_sha256':sha((args.assets/'manifest.json').read_bytes()),
        'dataset_id':manifest['dataset_id'],'quality_metric':manifest['quality_metric'],
        'baseline_quality':baseline_q['quality_score'],'quality_sample_count':len(ids),
        'action_ids':[a[0] for a in ACTIONS],'action_profiles':standalone,'node_profiles':node_profiles,
        'graph':graph,'coalition_order':order,'coalitions':coalitions,
        'proxy_profile_frozen_at':frozen_at,'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'runtime':{'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,
                   'platform':platform.platform(),'pid':os.getpid(),'threads':torch.get_num_threads(),**hardware(device)},
        'protocol':{'dtype':'float32','tf32':False,'warmup':args.warmup,'repeats':args.repeats,
           'timing':'per-call synchronized wall clock including dispatch; full model forwards',
           'train_runs':[1,2,3],'planned_eval_runs':[4,5,6,7,8],'quality_threshold':.99,'quality_mode':'retention_ratio',
           'source_sha256':sha(Path(__file__).read_bytes()),'mps_fallback_env':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')},
        'limitations':['New full-model intervention study, not reconstruction of historical FITAS.',
                       'Finite 10-class original ImageNet validation subset, not full ImageNet.',
                       'Actual residual dependency DAG does not imply parallel branch scheduling.',
                       'Device-specific same-backend fusion effects do not alone establish cross-chip causal attribution.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({'output':str(args.output),'run_id':run_id,'quality_baseline':out['baseline_quality'],
        'coalitions_ok':sum(c['status']=='ok' for c in coalitions),'equivalence_pass':sum(c.get('equivalence',{}).get('passed',False) for c in coalitions),
        'weight_hash':w_hash,'input_hash':in_meta['tensor_sha256']}),flush=True)

if __name__=='__main__':main()
