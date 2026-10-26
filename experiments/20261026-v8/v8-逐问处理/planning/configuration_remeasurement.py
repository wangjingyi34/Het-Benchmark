"""Fresh, identity-bound timing of six source configurations and two models.

This is a new FP16 inference experiment, never a reconstruction of old timers.
Weights and inputs come from portable CPU PCG64 generators, not training.
"""
import argparse,hashlib,json,os,platform,random,statistics,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
from torchvision import models
import torchvision
P=Path(__file__).resolve().parent
CONFIGS=[
 {'id':'matmul_4096','input':[4096,4096],'operation':'x @ x.T','layout':'contiguous input; transposed second operand'},
 {'id':'conv2d_224','input':[1,64,224,224],'operation':'Conv2d(64,128,3,padding=1,bias=True)','layout':'NCHW contiguous'},
 {'id':'attention_512','input':[1,512,768],'operation':'MultiheadAttention(embed_dim=768,num_heads=12,batch_first=True,dropout=0,need_weights=True)','layout':'contiguous BSE'},
 {'id':'layernorm_768','input':[32,512,768],'operation':'LayerNorm(768,eps=1e-5,elementwise_affine=True)','layout':'contiguous BSE'},
 {'id':'softmax_512x512','input':[8,12,512,512],'operation':'Softmax(dim=-1)','layout':'contiguous BHSS'},
 {'id':'linear_768_3072','legacy_label':'linear_4096','input':[32,512,768],'operation':'Linear(768,3072,bias=True)','layout':'contiguous BSE'},
 {'id':'mobilenet_v2','input':[1,3,224,224],'operation':'torchvision.models.mobilenet_v2(weights=None)','layout':'NCHW contiguous'},
 {'id':'resnet50','input':[1,3,224,224],'operation':'torchvision.models.resnet50(weights=None)','layout':'NCHW contiguous'},
]
def utc():return datetime.now(timezone.utc).isoformat()
def shabytes(x):return hashlib.sha256(x).hexdigest()
def filehash(p):return shabytes(Path(p).read_bytes())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def setup(index):
 torch.manual_seed(260800+index)
 constructors=[lambda:nn.Identity(),lambda:nn.Conv2d(64,128,3,padding=1),lambda:nn.MultiheadAttention(768,12,batch_first=True),lambda:nn.LayerNorm(768),lambda:nn.Softmax(dim=-1),lambda:nn.Linear(768,3072),lambda:models.mobilenet_v2(weights=None),lambda:models.resnet50(weights=None)]
 m=constructors[index]().eval();rng=np.random.Generator(np.random.PCG64(26081000+index))
 # Replace architecture initialization to make every platform consume identical
 # PCG64 FP16 parameter bytes; batch statistics retain their explicit defaults.
 with torch.no_grad():
  for name,t in m.named_parameters():
   if t.ndim>1:
    a=(rng.standard_normal(tuple(t.shape),dtype=np.float32)/np.sqrt(np.prod(t.shape[1:]))).astype(np.float16)
   else:a=np.ones(tuple(t.shape),dtype=np.float16) if name.endswith('weight') else np.zeros(tuple(t.shape),dtype=np.float16)
   t.copy_(torch.from_numpy(a).to(t.dtype))
 m=m.half()
 for t in m.parameters():t.requires_grad_(False)
 x=torch.from_numpy(np.random.Generator(np.random.PCG64(26082000+index)).standard_normal(CONFIGS[index]['input'],dtype=np.float32).astype(np.float16))
 h=hashlib.sha256()
 for name,t in m.state_dict().items():h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(t.contiguous().numpy().tobytes())
 identity={'id':CONFIGS[index]['id'],'input_sha256':shabytes(x.numpy().tobytes()),'weights_sha256':h.hexdigest(),'shape':list(x.shape),'strides':list(x.stride()),'dtype':str(x.dtype),'state_elements':sum(t.numel() for t in m.state_dict().values()),'parameter_seed':26081000+index,'input_seed':26082000+index}
 return m,x,identity
def sync(d):
 if d.type=='cuda':torch.cuda.synchronize()
 if d.type=='mps':torch.mps.synchronize()
def freeze(a):
 assert not a.output.exists()
 torch.set_num_threads(4);identities=[]
 for i in range(8):
  m,x,ident=setup(i);identities.append(ident);del m,x
 save(a.output,{'created_utc':utc(),'script_sha256':filehash(__file__),'configurations':CONFIGS,'identities':identities,'platforms':['A100 SXM4 MIG 2g.20GB','Apple M5 MPS'],'processes_per_platform':5,'warmup_per_workload':20,'calls_per_workload':100,'dtype':'float16','mode':'eval + inference_mode; MHA native fastpath disabled on both backends; TF32 disabled; cuDNN benchmark disabled','weights':'untrained portable PCG64 weights; no task-accuracy claims','configuration_source':'Original exp_cross_platform_eval.py candidate shape/operator configurations; linear renamed to actual 768-to-3072 implementation','scope':'New identity-bound controlled measurements; not the original full A100 PCIe, original five-platform LOPO or original 4.35/4.441 run authentication','failure_policy':'Every workload has status; retain unsupported/error/nonfinite without replacement; no number is inferred from another platform'})
def measure(a):
 assert not a.output.exists();p=json.loads(a.protocol.read_text());assert p['script_sha256']==filehash(__file__) and p['configurations']==CONFIGS
 assert torch.__version__.split('+')[0]=='2.9.1'
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.mha.set_fastpath_enabled(False)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 d=torch.device(a.device);assert (d.type!='cuda' or torch.cuda.is_available()) and (d.type!='mps' or torch.backends.mps.is_available())
 hw={'platform':platform.platform(),'machine':platform.machine(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'device':a.device,'threads':torch.get_num_threads()}
 if d.type=='cuda':hw.update(device_name=torch.cuda.get_device_name(0),memory_bytes=torch.cuda.get_device_properties(0).total_memory)
 if d.type=='mps':hw['chip']=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
 out={'status':'started','created_utc':utc(),'process':a.process,'pid':os.getpid(),'hardware':hw,'protocol_sha256':filehash(a.protocol),'rows':[]};save(a.output,out)
 order=list(range(8));random.Random(26083000+a.process).shuffle(order);out['order']=order
 for i in order:
  row={'id':CONFIGS[i]['id'],'status':'started'}
  try:
   m,x,ident=setup(i);assert ident==p['identities'][i];m=m.to(d);x=x.to(d);row.update(identity=ident)
   def call():
    if i==0:return torch.mm(x,x.T)
    if i==2:return m(x,x,x,need_weights=True)[0]
    return m(x)
   with torch.inference_mode():
    for _ in range(20):y=call()
    sync(d);row['timings']=[]
    for j in range(100):
     sync(d);start=time.perf_counter_ns();y=call();sync(d);end=time.perf_counter_ns();row['timings'].append({'start_ns':start,'end_ns':end,'latency_ms':(end-start)/1e6})
    yn=y.cpu().contiguous().numpy();row.update(output_shape=list(yn.shape),finite=bool(np.isfinite(yn).all()),output_sha256=shabytes(yn.tobytes()))
    row['status']='measured' if row['finite'] else 'nonfinite';row['mean_ms']=statistics.mean(t['latency_ms'] for t in row['timings']);row['sd_ms']=statistics.stdev(t['latency_ms'] for t in row['timings'])
   del m,x,y,yn
  except Exception as e:row.update(status='error',error=type(e).__name__+': '+str(e))
  out['rows'].append(row);save(a.output,out);print(json.dumps({'workload':row['id'],'status':row['status'],'mean_ms':row.get('mean_ms')}),flush=True)
 out.update(status='completed',completed_utc=utc());save(a.output,out)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','measure']);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--protocol',type=Path);ap.add_argument('--device',choices=['cuda','mps']);ap.add_argument('--process',type=int);a=ap.parse_args();globals()[a.action](a)
