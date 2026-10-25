"""Ascend identity/storage adapter around the frozen physical collector core.

No historical profile is modified. Model construction, portable inputs and
timing remain collect_profiles.py; both source hashes are retained.
"""
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
import torch_npu
from torch.nn import functional as F
import collect_profiles as core

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def hf32_flags():
    out={}
    for part in ['conv','matmul']:
        owner=getattr(torch.npu,part,None)
        if owner is not None and hasattr(owner,'allow_hf32'):
            setattr(owner,'allow_hf32',False);out[part]=bool(owner.allow_hf32)
        else:out[part]='API_unavailable'
    return out

def npu_hardware(device,args):
    if device.type!='npu':raise ValueError('this adapter is only for physical NPU collection')
    if not torch.npu.is_available():raise RuntimeError('NPU unavailable; no CPU substitute')
    name=str(torch.npu.get_device_name(device))
    memory=None;property_source=None
    if hasattr(torch.npu,'get_device_properties'):
        prop=torch.npu.get_device_properties(device)
        memory=getattr(prop,'total_memory',None);property_source='torch.npu.get_device_properties'
    if not memory and hasattr(torch.npu,'mem_get_info'):
        _,memory=torch.npu.mem_get_info(device);property_source='torch.npu.mem_get_info'
    if not name or not memory:raise RuntimeError('Cannot verify NPU identity and memory from the actual runtime')
    flags=hf32_flags()
    h={'device_id':args.device_id or ('NPU-'+name),'backend':'npu','hardware_name':name,
       'threads':torch.get_num_threads(),'visible_device_memory_bytes':int(memory),'compute_capability':[0,0],
       'descriptor_strict':{'backend_cpu':0,'backend_cuda':0,'backend_mps':0,'backend_npu':1,
           'host_arm64':int(platform.machine() in ['arm64','aarch64']),
           'host_logical_cpus':os.cpu_count() or 1,'threads':torch.get_num_threads(),
           'device_memory_bytes':int(memory),'compute_major':0,'compute_minor':0},
       'adapter_provenance':{'adapter_code_sha256':sha(__file__),
           'frozen_core_code_sha256':sha(Path(core.__file__)),
           'torch_npu':getattr(torch_npu,'__version__','unknown'),
           'identity_memory_source':property_source,'hf32_flags':flags,
           'model_input_and_timing_core_changed':False},'probes':[]}
    rng=np.random.Generator(np.random.PCG64(9137))
    a=core.exact_uniform((256,256),rng).to(device);b=core.exact_uniform((256,256),rng).to(device)
    x=core.exact_uniform((1,16,32,32),rng).to(device);w=core.exact_uniform((16,16,3,3),rng).to(device)
    z=core.exact_uniform((128,128),rng).to(device)
    for pname,fn in [('matmul256',lambda:a@b),('conv2d16',lambda:F.conv2d(x,w,padding=1)),('layernorm128',lambda:F.layer_norm(z,(128,)))]:
        h['probes'].append({'name':pname,'inputs_seed':9137,
            'purpose':'independent hardware descriptor only; no model target labels',
            'warmup':10,'inner_repeats':8,**core.timed(fn,device,10,20,8)})
    return h

original_shapeinfo=core.shapeinfo
def shapeinfo_with_storage(x):
    out=original_shapeinfo(x)
    for item,t in zip(out,core.tensors(x)):
        if t.device.type=='npu' and hasattr(torch_npu,'get_npu_format'):
            try:item['backend_storage_format']=int(torch_npu.get_npu_format(t))
            except Exception as exc:
                item['backend_storage_format']='API_unavailable_for_tensor'
                item['backend_storage_format_error']=type(exc).__name__+': '+str(exc)[:200]
    return out

if __name__=='__main__':
    core.hardware=npu_hardware;core.shapeinfo=shapeinfo_with_storage
    core.main()
