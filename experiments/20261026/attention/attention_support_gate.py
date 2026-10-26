"""Follow-up support-gated full-MHA profiling with independent repeatability runs."""
import argparse,hashlib,json,random,time,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import attention_ood_challenge as core
from active_fallback import validate_runtime
ROOT=Path(__file__).resolve().parent
CONFIGS=[(b,s,e,h) for b in (1,2) for s in (320,640,960,1280) for e,h in ((256,8),(448,8),(896,14))]
core.CONFIGS=CONFIGS;core.MODEL_ID='attention_v7_support_followup'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):
 p=Path(p);assert not p.exists(),p;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,allow_nan=False))
def verify(a):
 p=read(a.protocol)
 for n,h in p['source_hashes'].items():assert sha(ROOT/n)==h,n
 assert p['configurations']==[list(x) for x in CONFIGS]
 return p
def freeze(a):
 c=read(ROOT/'training-attention-catalog.json');assert list(c['targets'])==['call_function <built-in function scaled_dot_product_attention>']
 files=['attention_support_gate.py','attention_ood_challenge.py','collect_profiles.py','structural_attributes.py','evaluate_profiles.py','active_fallback.py','training-attention-catalog.json']
 save(a.output,{'kind':'support_gate_followup_protocol','created_utc':utc(),'configurations':CONFIGS,'source_hashes':{n:sha(ROOT/n) for n in files},
   'design_history':'Designed after the first V7 three-anchor calibration experiment. All first-experiment outcomes are retained; this follow-up measures an operational fallback on different shapes.',
   'support_rule':'training catalog contains functional SDPA and no complete-MHA composite module; every complete-MHA query is outside this executable contract and triggers direct synchronized profiling',
   'decisions':[{'case':i,'trigger_profile':True,'reason':'unsupported full-MHA composite leaf'} for i in range(24)],
   'budget':'24 of 24 new queries per backend; 100 percent direct measurement cost; no three-probe efficiency claim',
   'procedure':'Process0 profiles all queries and freezes their measured output; processes1-5 independently remeasure all24 in fixed shuffled order. No refit or error-dependent selection.',
   'warmup':5,'blocks':10,'inner':4,'endpoints':['paired measurement reproducibility MRE','WAPE','MAE','all raw timings','measurement wall overhead'],
   'interpretation':'repeatability of direct measured fallback, not learned prediction accuracy or original five-platform LOPO',
   'devices':['a100mig','m5mps']})
def collect(a):
 verify(a);h=read(a.hardware)['hardware'];device=core.set_runtime(a.device,h['threads']);runtime=validate_runtime(h,device)
 ref=None
 if a.process:
  ref=read(a.reference);assert ref['process']==0 and ref['protocol_sha256']==sha(a.protocol)
 indices=list(range(24));random.Random(262700+a.process).shuffle(indices);rows=[];fail=[];start=time.monotonic()
 with torch.inference_mode():
  for i in indices:
   t=time.monotonic()
   try:
    graph,gm,cap=core.construct_graph(i,device)
    if ref:assert graph['input_weights_sha256']==next(r['input_weights_sha256'] for r in ref['rows'] if r['case']==i)
    _,(node,args,kwargs)=next(iter(cap.calls.items()))
    r=core.core.timed(core.core.callable_for(gm,node,args,kwargs),device,5,10,4)
    rows.append({'case':i,'input_weights_sha256':graph['input_weights_sha256'],**r,'wall_seconds':time.monotonic()-t});del graph,gm,cap
   except Exception as e:fail.append({'case':i,'error':type(e).__name__+': '+str(e)})
 save(a.output,{'kind':'support_gate_direct_measurement' if not a.process else 'independent_fallback_repeatability_run','created_utc':utc(),'process':a.process,
  'protocol_sha256':sha(a.protocol),'hardware_sha256':sha(a.hardware),'reference_sha256':sha(a.reference) if ref else None,
  'device_id':h['device_id'],'runtime':runtime,'torch_version':torch.__version__,'rows':rows,'failures':fail,'wall_seconds':time.monotonic()-start})
 print(json.dumps({'device':a.device,'process':a.process,'queries':len(rows),'failures':len(fail),'seconds':time.monotonic()-start}),flush=True)
def summarize(a):
 p=verify(a);ref=read(a.reference);files=sorted(a.results.glob('repeat-*.json'));assert len(files)==5
 raw=[read(f) for f in files];assert len(ref['rows'])==24 and not ref['failures'];assert all(not r['failures'] and len(r['rows'])==24 and r['reference_sha256']==sha(a.reference) for r in raw)
 rows=[]
 for i in range(24):
  r=next(x for x in ref['rows'] if x['case']==i);vals=[next(x['latency_mean_ms'] for x in m['rows'] if x['case']==i) for m in raw]
  truth=float(np.mean(vals));error=abs(r['latency_mean_ms']-truth)
  rows.append({'case':i,'configuration':CONFIGS[i],'fallback_measured_ms':r['latency_mean_ms'],'independent_process_means_ms':vals,'reference_mean_ms':truth,'ae_ms':error,'re_pct':100*error/truth})
 save(a.output,{'kind':'support_gate_reproducibility_summary','created_utc':utc(),'protocol_sha256':sha(a.protocol),'reference_sha256':sha(a.reference),'replicate_hashes':{f.name:sha(f) for f in files},
  'device_id':ref['device_id'],'query_count':24,'profile_fraction':1.0,'independent_processes':5,'mre_pct':float(np.mean([x['re_pct'] for x in rows])),
  'mae_ms':float(np.mean([x['ae_ms'] for x in rows])),'wape_pct':100*sum(x['ae_ms'] for x in rows)/sum(x['reference_mean_ms'] for x in rows),
  'profiling_wall_seconds':ref['wall_seconds'],'repeatability_evaluation_wall_seconds':sum(x['wall_seconds'] for x in raw),'rows':rows,'interpretation':p['interpretation']})
 print(json.dumps({k:v for k,v in read(a.output).items() if k not in ['rows','replicate_hashes']}),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','collect','summarize'])
 for n in ['protocol','hardware','reference','results','output']:ap.add_argument('--'+n,type=Path)
 ap.add_argument('--device');ap.add_argument('--process',type=int,default=0);a=ap.parse_args();globals()[a.action](a)
if __name__=='__main__':main()
