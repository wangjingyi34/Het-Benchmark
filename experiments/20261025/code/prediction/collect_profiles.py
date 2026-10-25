"""New physical FX operator profiles; no imported historical latency labels.

Run each backend separately. Timing labels are synchronized isolated operator
block means, not full-model latency or task accuracy. We retain the actual FX
dependencies and executable inputs so this distinction is auditable.
"""
import argparse
import hashlib
import json
import math
import operator
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.fx import GraphModule, Interpreter, Tracer

SCHEMA = 'het-physical-profile-v2.1'
MODEL_IDS = ['resnet18','resnet50','mobilenet_v2','densenet121',
             'efficientnet_b0','squeezenet1_1','text_encoder_ref',
             'text_cnn_ref','audio_conv_ref','audio_dilated_ref',
             'causal_lm_ref','dual_encoder_ref']

def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)
    elif device.type == 'mps': torch.mps.synchronize()
    elif device.type == 'npu': torch.npu.synchronize()

def timed(fn, device, warmup, repeats, inner):
    for _ in range(warmup): fn()
    sync(device)
    samples=[]
    for _ in range(repeats):
        sync(device); start=time.perf_counter_ns()
        for _ in range(inner): fn()
        sync(device)
        samples.append((time.perf_counter_ns()-start)/1e6/inner)
    return {'latency_samples_ms':samples,'latency_mean_ms':statistics.mean(samples),
            'latency_median_ms':statistics.median(samples),
            'latency_sd_ms':statistics.stdev(samples) if len(samples)>1 else None}

def tensors(x):
    if isinstance(x,torch.Tensor):return [x]
    if isinstance(x,(tuple,list)):return [t for y in x for t in tensors(y)]
    if isinstance(x,dict):return [t for y in x.values() for t in tensors(y)]
    return []

def shapeinfo(x):
    return [{'shape':list(t.shape),'dtype':str(t.dtype),'stride':list(t.stride()),
             'numel':t.numel(),'element_size':t.element_size()} for t in tensors(x)]

def tensor_hash(model, inputs):
    h=hashlib.sha256()
    for name,t in list(model.state_dict().items())+[(f'input_{i}',t) for i,t in enumerate(inputs)]:
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def exact_uniform(shape,rng):
    # Integer PCG64 stream and power-of-two arithmetic are portable across hosts.
    raw=rng.integers(0,65536,size=shape,dtype=np.uint32)
    return torch.from_numpy((raw.astype(np.float32)-32768.0)/32768.0)

def portable_init(model,seed):
    rng=np.random.Generator(np.random.PCG64(seed))
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.ndim>=2:
                fan=max(1,int(np.prod(p.shape[1:])))
                p.copy_(exact_uniform(tuple(p.shape),rng)*(1/math.sqrt(fan)))
            elif name.endswith('weight'):p.fill_(1.)
            else:p.zero_()
        for name,b in model.named_buffers():
            if name.endswith('running_var'):b.fill_(1.)
            elif name.endswith('running_mean') or name.endswith('num_batches_tracked'):b.zero_()
    return model

class Attention(nn.Module):
    def __init__(self,d=64,h=4,causal=False):
        super().__init__();self.h=h;self.d=d;self.causal=causal
        self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.out=nn.Linear(d,d)
        self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))
    def forward(self,x):
        z=self.n1(x);b=z.size(0);s=z.size(1)
        q=self.q(z).reshape(b,s,self.h,self.d//self.h).transpose(1,2)
        k=self.k(z).reshape(b,s,self.h,self.d//self.h).transpose(1,2)
        v=self.v(z).reshape(b,s,self.h,self.d//self.h).transpose(1,2)
        a=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=self.causal)
        a=a.transpose(1,2).contiguous().reshape(b,s,self.d)
        x=x+self.out(a)
        return x+self.ff(self.n2(x))

class TextEncoder(nn.Module):
    def __init__(self,causal=False):
        super().__init__();self.emb=nn.Embedding(2048,64)
        self.blocks=nn.Sequential(Attention(causal=causal),Attention(causal=causal))
        self.head=nn.Linear(64,128 if causal else 10)
    def forward(self,x):
        x=self.blocks(self.emb(x))
        return self.head(x if self.blocks[0].causal else x.mean(dim=1))

class TextCNN(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(2048,64)
        self.c1=nn.Conv1d(64,64,3,padding=1);self.c2=nn.Conv1d(64,64,5,padding=2)
        self.pool=nn.AdaptiveMaxPool1d(1);self.fc=nn.Linear(128,10)
    def forward(self,x):
        z=self.emb(x).transpose(1,2)
        return self.fc(torch.cat([self.pool(F.gelu(self.c1(z))).flatten(1),
                                  self.pool(F.gelu(self.c2(z))).flatten(1)],dim=1))

class Audio(nn.Module):
    def __init__(self,dilated=False):
        super().__init__();self.c1=nn.Conv1d(1,16,9,stride=4,padding=4)
        self.c2=nn.Conv1d(16,32,5,stride=2,padding=4 if dilated else 2,dilation=2 if dilated else 1)
        self.c3=nn.Conv1d(32,32,3,padding=4 if dilated else 1,dilation=4 if dilated else 1)
        self.pool=nn.AdaptiveAvgPool1d(1);self.fc=nn.Linear(32,12)
    def forward(self,x):
        x=F.gelu(self.c1(x));x=F.gelu(self.c2(x));x=x+F.gelu(self.c3(x))
        return self.fc(self.pool(x).flatten(1))

class DualEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.visual=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(16,64))
        self.text=nn.Embedding(2048,64);self.proj=nn.Linear(128,64);self.fc=nn.Linear(64,10)
    def forward(self,image,tokens):
        v=self.visual(image);t=self.text(tokens).mean(dim=1)
        return self.fc(F.gelu(self.proj(torch.cat([v,t],dim=1))))

def build_model(model_id,shape_id):
    # Model creation may use torch RNG internally; every parameter is then replaced
    # by the explicit portable stream. No pretrained-weight quality claim is made.
    idx=MODEL_IDS.index(model_id);seed=270000+idx
    if idx<6:
        from torchvision import models
        model=getattr(models,model_id)(weights=None)
        workload='CV';source=f'torchvision.models.{model_id}(weights=None)'
    elif model_id=='text_encoder_ref':model=TextEncoder();workload='NLP';source='self-contained two-block pre-norm SDPA encoder reference'
    elif model_id=='text_cnn_ref':model=TextCNN();workload='NLP';source='self-contained two-branch text CNN reference'
    elif model_id=='audio_conv_ref':model=Audio();workload='Audio';source='self-contained three-stage 1D residual convolution reference'
    elif model_id=='audio_dilated_ref':model=Audio(True);workload='Audio';source='self-contained dilated 1D residual convolution reference'
    elif model_id=='causal_lm_ref':model=TextEncoder(True);workload='LLM';source='self-contained causal decoder reference, not a pretrained LLM'
    else:model=DualEncoder();workload='Multimodal';source='self-contained executable image/text late-fusion reference'
    model.eval()
    disabled=[]
    for name,m in model.named_modules():
        if hasattr(m,'inplace') and isinstance(m.inplace,bool) and m.inplace:
            m.inplace=False;disabled.append(name)
    portable_init(model,seed)
    rng=np.random.Generator(np.random.PCG64(370000+idx*100+shape_id))
    batch=[1,2,4][shape_id]
    if workload=='CV':
        side=[96,128,160][shape_id];inputs=[exact_uniform((batch,3,side,side),rng)]
    elif workload in ['NLP','LLM']:
        seq=[32,64,128][shape_id]
        inputs=[torch.from_numpy(rng.integers(0,2048,size=(batch,seq),dtype=np.int64))]
    elif workload=='Audio':inputs=[exact_uniform((batch,1,[2048,4096,8192][shape_id]),rng)]
    else:
        side=[32,48,64][shape_id];seq=[32,64,128][shape_id]
        inputs=[exact_uniform((batch,3,side,side),rng),torch.from_numpy(rng.integers(0,2048,size=(batch,seq),dtype=np.int64))]
    return model,inputs,{'workload':workload,'architecture_source':source,
                         'inplace_disabled_modules':disabled,'initialization_seed':seed,
                         'initialization':'PCG64 uint16 uniform -> exact power-of-two FP32; fan-scaled random weights; normalization weight1/bias0'}

class SafeTracer(Tracer):
    def is_leaf_module(self,m,qualname):
        # DenseLayer contains Python list/control flow; retaining it as one
        # composite operator avoids inventing a dependency chain or failed trace.
        if m.__class__.__name__=='_DenseLayer':return True
        return super().is_leaf_module(m,qualname)

class Capture(Interpreter):
    def __init__(self,gm):super().__init__(gm);self.calls={};self.info={}
    def run_node(self,n):
        args,kwargs=self.fetch_args_kwargs_from_env(n)
        out=super().run_node(n)
        if n.op in ['call_module','call_function','call_method'] and tensors(out):
            self.calls[n.name]=(n,args,kwargs)
        self.info[n.name]={'input_shapes':shapeinfo((args,kwargs)),'output_shapes':shapeinfo(out)}
        return out

def callable_for(gm,n,args,kwargs):
    if n.op=='call_module':return lambda:gm.get_submodule(n.target)(*args,**kwargs)
    if n.op=='call_function':return lambda:n.target(*args,**kwargs)
    if n.op=='call_method':return lambda:getattr(args[0],n.target)(*args[1:],**kwargs)
    raise ValueError(n.op)

def family_for(gm,n):
    if n.op=='call_module':name=gm.get_submodule(n.target).__class__.__name__.lower()
    else:name=str(n.target).lower()
    for tokens,family in [(['conv2d'],'Conv2D'),(['conv1d'],'Conv1D'),(['linear','matmul','bmm'],'MatMul'),
        (['scaled_dot_product','multiheadattention'],'Attention'),(['layernorm','batchnorm','groupnorm'],'Normalization'),
        (['softmax'],'Softmax'),(['embedding'],'Embedding'),(['pool'],'Pooling'),(['dropout'],'Dropout'),
        (['gelu','relu','silu','tanh','sigmoid'],'Activation'),(['reshape','flatten','transpose','contiguous','getitem','view'],'View'),
        (['mean','sum'],'Reduction'),(['add','mul','cat','sub','div'],'Elementwise')]:
        if any(t in name for t in tokens):return family
    return 'Composite' if n.op=='call_module' else 'Other'

def features(gm,n,info,index,total):
    ins=info['input_shapes'];outs=info['output_shapes']
    inp=sum(x['numel'] for x in ins);out=sum(x['numel'] for x in outs)
    p=0
    if n.op=='call_module':p=sum(t.numel() for t in gm.get_submodule(n.target).parameters())
    shapes=(ins[0]['shape'] if ins else [])[:4];shapes=shapes+[0]*(4-len(shapes))
    return {'input_numel':inp,'output_numel':out,'parameter_numel':p,
            'input_bytes':sum(x['numel']*x['element_size'] for x in ins),
            'output_bytes':sum(x['numel']*x['element_size'] for x in outs),
            'rank':len(ins[0]['shape']) if ins else 0,
            'dim0':shapes[0],'dim1':shapes[1],'dim2':shapes[2],'dim3':shapes[3],
            'position_fraction':index/max(total-1,1),'fanin':len(n.all_input_nodes),
            'fanout':len(n.users)}

def hardware(device,args):
    if device.type=='cuda':
        prop=torch.cuda.get_device_properties(device)
        name=prop.name;memory=prop.total_memory;compute=list(torch.cuda.get_device_capability(device))
    elif device.type=='mps':name=platform.processor()+' Apple MPS';memory=0;compute=[0,0]
    else:name=platform.processor()+' CPU';memory=0;compute=[0,0]
    if not name.strip() or name.strip()=='CPU':name=platform.machine()+' '+device.type
    h={'device_id':args.device_id or (device.type+'-'+name.replace(' ','_')),
       'backend':device.type,'hardware_name':name,'threads':torch.get_num_threads(),
       'visible_device_memory_bytes':memory,'compute_capability':compute,
       'descriptor_strict':{'backend_cpu':int(device.type=='cpu'),'backend_cuda':int(device.type=='cuda'),
           'backend_mps':int(device.type=='mps'),'backend_npu':int(device.type=='npu'),
           'host_arm64':int(platform.machine() in ['arm64','aarch64']),
           'host_logical_cpus':os.cpu_count() or 1,'threads':torch.get_num_threads(),
           'device_memory_bytes':memory,'compute_major':compute[0],'compute_minor':compute[1]},
       'probes':[]}
    rng=np.random.Generator(np.random.PCG64(9137))
    a=exact_uniform((256,256),rng).to(device);b=exact_uniform((256,256),rng).to(device)
    x=exact_uniform((1,16,32,32),rng).to(device);w=exact_uniform((16,16,3,3),rng).to(device)
    z=exact_uniform((128,128),rng).to(device)
    for name,fn in [('matmul256',lambda:a@b),('conv2d16',lambda:F.conv2d(x,w,padding=1)),('layernorm128',lambda:F.layer_norm(z,(128,)))]:
        h['probes'].append({'name':name,'inputs_seed':9137,'purpose':'independent hardware descriptor only; no model target labels',
            'warmup':10,'inner_repeats':8,**timed(fn,device,10,20,8)})
    return h

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',required=True,choices=['cpu','mps','cuda','npu'])
    ap.add_argument('--device-id');ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--models',nargs='+',default=MODEL_IDS);ap.add_argument('--shapes',nargs='+',type=int,default=[0,1,2])
    ap.add_argument('--repeats',type=int,default=20);ap.add_argument('--warmup',type=int,default=10)
    ap.add_argument('--inner',type=int,default=8);ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    if args.output.exists():raise RuntimeError('Never overwrite prior physical profiles: '+str(args.output))
    if args.device=='npu':import torch_npu  # noqa
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    device=torch.device(args.device);codehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bundle={'schema_version':SCHEMA,'provenance':{'experiment_id':f'het-v2-{args.device}-{time.time_ns()}',
        'utc':datetime.now(timezone.utc).isoformat(),'code_sha256':codehash,'python':sys.version,
        'torch':torch.__version__,'cuda':torch.version.cuda,'platform':platform.platform(),
        'mps_fallback_environment':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        'timing_protocol':'isolated FX executable call; each sample synchronized block wall clock / inner; includes dispatch; no full-model/task-accuracy interpretation',
        'repeats':args.repeats,'warmup':args.warmup,'inner_repeats':args.inner},'hardware':None,'graphs':[],'failures':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temp=args.output.with_suffix(args.output.suffix+'.partial');temp.write_text(json.dumps(bundle,indent=2));temp.replace(args.output)
    with torch.inference_mode():
        bundle['hardware']=hardware(device,args)
        for model_id in args.models:
            for shape_id in args.shapes:
                gid=f'{model_id}:s{shape_id}'
                try:
                    model,inputs,meta=build_model(model_id,shape_id)
                    ih=tensor_hash(model,inputs)
                    tracer=SafeTracer();gm=GraphModule(model,tracer.trace(model)).eval().to(device)
                    inputs=[x.to(device) for x in inputs]
                    capture=Capture(gm);capture.run(*inputs);sync(device)
                    executable=[n for n in gm.graph.nodes if n.name in capture.calls]
                    graph={'graph_id':gid,'model_id':model_id,'shape_id':shape_id,**meta,
                           'dtype':'float32','input_shapes':shapeinfo(inputs),'input_weights_sha256':ih,
                           'nodes':[],'edges':[],'records':[]}
                    for i,n in enumerate(gm.graph.nodes):
                        inf=capture.info[n.name];nid=f'{gid}/{n.name}';family=family_for(gm,n) if n.name in capture.calls else 'Context'
                        node={'node_id':nid,'fx_op':n.op,'target':str(n.target),
                              'module_path':str(n.target) if n.op=='call_module' else '',
                              'family':family,**inf,'parameters':features(gm,n,inf,i,len(list(gm.graph.nodes)))['parameter_numel'],
                              'dependency_ids':[f'{gid}/{p.name}' for p in n.all_input_nodes],
                              'features':features(gm,n,inf,i,len(list(gm.graph.nodes)))}
                        graph['nodes'].append(node)
                        for pred in node['dependency_ids']:graph['edges'].append({'source':pred,'target':nid,'relation':'dependency'})
                    order=list(executable);random.Random(8141+shape_id+MODEL_IDS.index(model_id)*10).shuffle(order)
                    for n in order:
                        _,nargs,nkwargs=capture.calls[n.name]
                        record={'record_id':f"{bundle['hardware']['device_id']}/{gid}/{n.name}",
                                'run_id':bundle['provenance']['experiment_id'],'op_instance_id':f'{gid}/{n.name}',
                                'status':'measured','warmup':args.warmup,'inner_repeats':args.inner,
                                'timing_units':'ms per call; synchronized block mean',
                                'latency_samples_ms':[],'latency_mean_ms':None,'latency_median_ms':None,'latency_sd_ms':None}
                        try:record.update(timed(callable_for(gm,n,nargs,nkwargs),device,args.warmup,args.repeats,args.inner))
                        except Exception as e:
                            record['status']='oom' if 'out of memory' in str(e).lower() else 'unsupported'
                            record['error']=type(e).__name__+': '+str(e)[:500]
                        graph['records'].append(record)
                    graph['full_model_timing']=timed(lambda:gm(*inputs),device,args.warmup,args.repeats,1)
                    bundle['graphs'].append(graph);save()
                    print(json.dumps({'graph':gid,'backend':args.device,'operators':len(executable),
                        'measured':sum(r['status']=='measured' for r in graph['records']),
                        'failed':sum(r['status']!='measured' for r in graph['records']),
                        'model_mean_ms':graph['full_model_timing']['latency_mean_ms']},ensure_ascii=False),flush=True)
                    del capture,gm,model,inputs
                    if device.type=='cuda':torch.cuda.empty_cache()
                except Exception as e:
                    bundle['failures'].append({'graph_id':gid,'model_id':model_id,'shape_id':shape_id,
                        'status':'failed','error':type(e).__name__+': '+str(e)[:1000]});save()
                    print(json.dumps({'graph':gid,'status':'failed','error':str(e)[:300]}),flush=True)
    bundle['provenance']['completed_utc']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps({'complete':True,'graphs':len(bundle['graphs']),'failed_graphs':len(bundle['failures']),
        'output':str(args.output),'code_sha256':codehash}),flush=True)

if __name__=='__main__':main()
