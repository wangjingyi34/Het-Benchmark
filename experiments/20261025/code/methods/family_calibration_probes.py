"""Fourteen fixed family calibration workloads, auxiliary to the main corpus.

Budget is 3 existing hardware probes + 14 new probes = 17 query types.
No model latency/quality labels are read. Failed probes remain explicit.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

PREDICTION=Path(__file__).resolve().parent.parent/'prediction'
sys.path.insert(0,str(PREDICTION))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models.densenet import _DenseLayer
from torchvision.ops import stochastic_depth
import collect_profiles as core

CORE_SHA='7050d276d82bead7f6b936be35f93059be5d24b0073eb7045434174bec618f26'
SPECS=[
 {'family':'Conv2D','operator':'torch.nn.Conv2d','shape':[1,8,16,16],'out_channels':8,'kernel':3,'padding':1},
 {'family':'Conv1D','operator':'torch.nn.Conv1d','shape':[1,8,64],'out_channels':8,'kernel':3,'padding':1},
 {'family':'MatMul','operator':'torch.matmul','shape':[128,128]},
 {'family':'Attention','operator':'torch.nn.functional.scaled_dot_product_attention','shape':[1,4,32,16],'dropout_p':0.,'is_causal':False},
 {'family':'Normalization','operator':'torch.nn.LayerNorm','shape':[4,32],'normalized_shape':[32]},
 {'family':'Activation','operator':'torch.nn.GELU','shape':[32,32],'approximate':'none'},
 {'family':'Pooling','operator':'torch.nn.AdaptiveAvgPool2d','shape':[1,8,8,8],'output_size':[1,1]},
 {'family':'Elementwise','operator':'torch.add','shape':[32,32]},
 {'family':'View','operator':'torch.Tensor.transpose','shape':[2,16,32],'dimensions':[1,2]},
 {'family':'Embedding','operator':'torch.nn.Embedding','shape':[2,16],'vocabulary':128,'embedding_width':16,'index_dtype':'int64'},
 {'family':'Dropout','operator':'torch.nn.Dropout','shape':[32,32],'p':.1,'training':False,'semantics':'eval identity'},
 {'family':'Reduction','operator':'torch.Tensor.mean','shape':[4,64],'dim':1},
 {'family':'Composite','operator':'torchvision.models.densenet._DenseLayer','shape':[1,32,8,8],'growth_rate':8,'bn_size':2,'drop_rate':0.,'memory_efficient':False},
 {'family':'Other','operator':'torchvision.ops.stochastic_depth','shape':[1,8,8,8],'p':.2,'mode':'row','training':False,'semantics':'eval identity'},
]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()

def build(index):
    s=SPECS[index];family=s['family'];rng=np.random.Generator(np.random.PCG64(892000+index));x=core.exact_uniform(tuple(s['shape']),rng)
    module=nn.Identity()
    if family=='Conv2D':module=nn.Conv2d(8,8,3,padding=1)
    elif family=='Conv1D':module=nn.Conv1d(8,8,3,padding=1)
    elif family=='Normalization':module=nn.LayerNorm(32)
    elif family=='Activation':module=nn.GELU(approximate='none')
    elif family=='Pooling':module=nn.AdaptiveAvgPool2d(1)
    elif family=='Embedding':
        module=nn.Embedding(128,16);x=torch.from_numpy(rng.integers(0,128,size=tuple(s['shape']),dtype=np.int64))
    elif family=='Dropout':module=nn.Dropout(.1)
    elif family=='Composite':module=_DenseLayer(32,8,2,0.,memory_efficient=False)
    module.eval();disabled=[]
    for name,m in module.named_modules():
        if isinstance(getattr(m,'inplace',None),bool) and m.inplace:m.inplace=False;disabled.append(name)
    core.portable_init(module,891000+index)
    for p in module.parameters():p.requires_grad_(False)
    inputs=[x]
    if family in ['MatMul','Elementwise']:inputs.append(core.exact_uniform(tuple(s['shape']),rng))
    if family=='Attention':inputs += [core.exact_uniform(tuple(s['shape']),rng),core.exact_uniform(tuple(s['shape']),rng)]
    return module,inputs,{'input_seed':892000+index,'weight_seed':891000+index,'inplace_disabled_modules':disabled}

def callable_for(index,module,inputs):
    family=SPECS[index]['family']
    if family=='MatMul':return lambda:torch.matmul(*inputs)
    if family=='Attention':return lambda:F.scaled_dot_product_attention(*inputs,dropout_p=0.,is_causal=False)
    if family=='Elementwise':return lambda:torch.add(*inputs)
    if family=='View':return lambda:inputs[0].transpose(1,2)
    if family=='Reduction':return lambda:inputs[0].mean(dim=1)
    if family=='Other':return lambda:stochastic_depth(inputs[0],p=.2,mode='row',training=False)
    return lambda:module(*inputs)

def identity(index,module,inputs,meta):
    return {'probe_id':'family14-'+SPECS[index]['family'],'specification':SPECS[index],**meta,
        'actual_module_class':module.__class__.__module__+'.'+module.__class__.__name__,
        'module_repr':repr(module),'input_shapes':core.shapeinfo(inputs),
        'input_weights_sha256':core.tensor_hash(module,inputs),
        'input_sha256':core.tensor_hash(nn.Identity(),inputs),'weights_sha256':core.tensor_hash(module,[])}

def freeze(path):
    if path.exists():raise FileExistsError('preserve fixed probe manifest')
    assert sha(core.__file__)==CORE_SHA
    rows=[]
    for i in range(14):
        module,inputs,meta=build(i);rows.append(identity(i,module,inputs,meta))
    d={'kind':'frozen_family14_calibration_manifest','frozen_utc':utc(),'collector_sha256':sha(__file__),
       'frozen_core_sha256':CORE_SHA,'probe_count':14,'existing_probe_count':3,'total_query_type_budget':17,
       'warmup':10,'repeats':20,'inner_repeats':8,'untimed_output_check_calls_after_timing':1,
       'family_mapping':'one exact family to its same named probe; fixed before new measurements',
       'failure_rule':'retain failure; only fall back to previously frozen original-three-probe mapping; do not invent ratios or retune k/temperature/gamma',
       'scope':'auxiliary hardware calibration only, excluded from the 13905 main query labels and from task-quality claims',
       'probes':rows,'freeze_runtime':{'torch':torch.__version__,'torchvision':torchvision.__version__,'platform':platform.platform()},
       'timing_observations_at_freeze':0}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,indent=2));print(json.dumps({'freeze':str(path),'probes':14,'observations':0,'sha256':sha(path),'collector_sha256':sha(__file__)}))

def collect(a):
    if a.output.exists():raise FileExistsError('preserve previous physical probe result')
    frozen=json.loads(a.manifest.read_text());assert frozen['kind']=='frozen_family14_calibration_manifest'
    assert frozen['collector_sha256']==sha(__file__) and frozen['frozen_core_sha256']==sha(core.__file__)==CORE_SHA
    descriptor=json.loads(a.hardware_from.read_text())
    if descriptor.get('kind')!='hardware_descriptor_only' or 'graphs' in descriptor or 'records' in descriptor:
        raise ValueError('hardware-only descriptor required; use active_fallback extract-hardware')
    h=descriptor['hardware'];assert len(h['probes'])==3
    if a.device=='npu':import torch_npu
    device=torch.device(a.device);torch.set_num_threads(h['threads']);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if a.device=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('no silent MPS CPU fallback')
    from active_fallback import validate_runtime
    started=time.monotonic();observed=validate_runtime(h,device)
    if a.device=='npu' and observed['hf32_flags']!={'conv':False,'matmul':False}:raise ValueError('NPU FP32 flags not fixed')
    out={'kind':'physical_family14_auxiliary_calibration','schema_version':'het.family-calibration-probes.v1','started_utc':utc(),
        'collector_sha256':sha(__file__),'frozen_core_sha256':CORE_SHA,'manifest_sha256':sha(a.manifest),
        'hardware_descriptor_sha256':sha(a.hardware_from),'hardware_source_profile_sha256':descriptor['source_profile_sha256'],
        'hardware':h,'observed_runtime':observed,'runtime':{'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'platform':platform.platform()},
        'warmup':10,'repeats':20,'inner_repeats':8,'existing_probe_count':3,'new_probe_count':14,'total_query_type_budget':17,
        'scope':frozen['scope'],'model_latency_or_task_quality_labels_used':False,'records':[],'failures':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        p=a.output.with_suffix('.partial');p.write_text(json.dumps(out,indent=2));p.replace(a.output)
    with torch.inference_mode():
        for i,spec in enumerate(SPECS):
            before=time.monotonic();row={'probe_id':'family14-'+spec['family'],'family':spec['family'],'status':'running','started_utc':utc()}
            try:
                module,inputs,meta=build(i);description=identity(i,module,inputs,meta)
                if description!=frozen['probes'][i]:raise ValueError('input/weights/configuration differ from frozen cross-stack bytes')
                row.update(description);module=module.to(device);inputs=[x.to(device) for x in inputs]
                fn=callable_for(i,module,inputs)
                row.update(core.timed(fn,device,10,20,8))
                output=fn();core.sync(device)
                row['output_shapes']=core.shapeinfo(output);row['output_sha256']=core.tensor_hash(nn.Identity(),core.tensors(output))
                row['output_all_finite']=all(bool(torch.isfinite(t).all()) for t in core.tensors(output))
                row['status']='measured' if row['output_all_finite'] else 'invalid_output'
                row['untimed_output_check_calls_after_timing']=1
            except Exception as error:
                row.update({'status':'oom' if 'out of memory' in str(error).lower() else 'failed','error_type':type(error).__name__,'error':str(error)})
                out['failures'].append({'probe_id':row['probe_id'],'family':spec['family'],'status':row['status'],'error':row['error']})
            row['wall_seconds_including_build_transfer_warmup_and_output_check']=time.monotonic()-before;row['completed_utc']=utc()
            out['records'].append(row);save();print(json.dumps({'probe':row['probe_id'],'status':row['status'],'wall_seconds':row['wall_seconds_including_build_transfer_warmup_and_output_check']}),flush=True)
    out['completed_utc']=utc();out['wall_seconds_including_runtime_validation']=time.monotonic()-started
    out['successful_probes']=sum(r['status']=='measured' for r in out['records']);save()
    print(json.dumps({'output':str(a.output),'successful':out['successful_probes'],'failed':14-out['successful_probes'],'budget':17}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('collect');c.add_argument('--manifest',type=Path,required=True);c.add_argument('--hardware-from',type=Path,required=True)
    c.add_argument('--device',choices=['cpu','mps','cuda','npu'],required=True);c.add_argument('--output',type=Path,required=True)
    a=p.parse_args();freeze(a.output) if a.command=='freeze' else collect(a)
