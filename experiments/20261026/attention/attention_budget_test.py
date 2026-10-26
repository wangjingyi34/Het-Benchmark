"""Preregistered new-shape MHA test with three measured calibration queries.

No target timing is loaded by prepare. Calibrate freezes predictions before any
evaluation process runs. Evaluation queries exclude the three measured anchors.
"""
import argparse, hashlib, json, math, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import attention_ood_challenge as old
from evaluate_profiles import build_graph, collate, predict
from active_fallback import load_fit, validate_runtime

ROOT=Path(__file__).resolve().parent
CONFIGS=[(b,s,e,h) for b in (1,2) for s in (256,384,768,1536) for e,h in ((192,6),(384,6),(640,10))]
old.CONFIGS=CONFIGS
old.MODEL_ID='attention_v7_newshape'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    p=Path(p)
    if p.exists():raise ValueError('immutable artifact already exists: '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False))
def check_protocol(p):
    z=read(p)
    assert z['configurations']==[list(x) for x in CONFIGS]
    for n,h in z['code_hashes'].items():assert sha(ROOT/n)==h,n
    return z
def freeze(a):
    fs=['attention_budget_test.py','attention_ood_challenge.py','collect_profiles.py','structural_attributes.py','evaluate_profiles.py','active_fallback.py']
    z={'created_utc':utc(),'kind':'preregistered_attention_budget_protocol','configurations':CONFIGS,
       'checkpoint_manifest_sha256':sha(ROOT/'frozen-source-predictors.json'),
       'code_hashes':{f:sha(ROOT/f) for f in fs},'methods':['rgat ensemble of seeds 11,22,33,44,55'],
       'selection':'descending sample SD of log predictions, case index ascending breaks ties; exactly 3 anchors',
       'correction':'ensemble prediction multiplied by exp(weighted mean anchor log(actual/pred)); weights exp(-0.5 squared Euclidean distance on log2(batch,sequence,width,heads)), normalized over all three anchors; no tuned coefficients',
       'evaluation':'all other 21 cases in 5 independent fresh processes; anchor process precedes correction freeze, correction precedes evaluation',
       'blocks':10,'inner':4,'warmup':5,'seeds':'PCG64 portable parameters 681000+i, input 781000+i',
       'endpoints':['query mean absolute relative error','WAPE','MAE ms','paired per-query relative error difference','process variation','measurement and decision wall times'],
       'platforms':['a100mig','m5mps'],'scope':'full-MHA new shapes, not 21 independent architectures or original five-hardware LOPO; old full-MHA results were known at design, all 24 configurations here are different',
       'failures':'retain all failures; do not change selection/hyperparameters after measuring labels'}
    save(a.output,z)
def prepare(a):
    p=check_protocol(a.protocol);f=read(ROOT/'frozen-source-predictors.json');assert sha(ROOT/'frozen-source-predictors.json')==p['checkpoint_manifest_sha256']
    h=read(a.hardware)['hardware'];old.set_runtime('cpu',1);start=time.monotonic();rows=[];fits=None
    with torch.inference_mode():
      for i in range(len(CONFIGS)):
        graph,gm,cap=old.construct_graph(i,torch.device('cpu'))
        batch=collate([build_graph(graph,h,'strict',10,unlabelled=True)],'cpu');assert torch.isnan(batch['y']).all()
        if fits is None:
          fits=[]
          for c in f['checkpoints']:
            if c['method']!='rgat':continue
            pol={'feature_mode':c['feature_mode'],'method':c['method'],'train_model_ids':c['training_model_ids'],'validation_model_ids':c['validation_model_ids']}
            fits.append(load_fit(ROOT/'checkpoints'/Path(c['path']).name,c['sha256'],batch['x'].shape[1],pol))
        pred=np.array([float(predict(x,batch)[0]) for x in fits]);assert len(pred)==5 and np.all(pred>0)
        rows.append({'case':i,'configuration':CONFIGS[i],'input_weights_sha256':graph['input_weights_sha256'],
                     'seed_predictions_ms':pred.tolist(),'prediction_ms':float(pred.mean()),'uncertainty':float(np.log(pred).std(ddof=1))})
        del gm,cap,graph,batch
    anchors=sorted(range(len(rows)),key=lambda i:(-rows[i]['uncertainty'],i))[:3]
    save(a.output,{'kind':'frozen_unlabelled_attention_decisions','created_utc':utc(),'protocol_sha256':sha(a.protocol),'hardware_sha256':sha(a.hardware),
       'device_id':h['device_id'],'anchors':anchors,'evaluation_cases':[i for i in range(24) if i not in anchors],
       'decisions':rows,'target_timings_read':False,'feature_capture_and_prediction_wall_seconds':time.monotonic()-start})
def measure(a):
    check_protocol(a.protocol);d=read(a.decisions);assert d['protocol_sha256']==sha(a.protocol)
    h=read(a.hardware)['hardware'];assert sha(a.hardware)==d['hardware_sha256']
    device=old.set_runtime(a.device,h['threads']);observed=validate_runtime(h,device)
    if a.stage=='evaluation':
        correction=read(a.correction);assert correction['decisions_sha256']==sha(a.decisions)
        indices=list(d['evaluation_cases']);correction_sha=sha(a.correction)
    else:indices=list(d['anchors']);correction_sha=None
    random.Random(260260+a.process).shuffle(indices);rows=[];errors=[];start=time.monotonic()
    with torch.inference_mode():
      for i in indices:
        t=time.monotonic()
        try:
          graph,gm,cap=old.construct_graph(i,device);assert graph['input_weights_sha256']==d['decisions'][i]['input_weights_sha256']
          _,(node,args,kwargs)=next(iter(cap.calls.items()))
          timing=old.core.timed(old.core.callable_for(gm,node,args,kwargs),device,5,10,4)
          rows.append({'case':i,'input_weights_sha256':graph['input_weights_sha256'],**timing,'wall_seconds':time.monotonic()-t})
          del graph,gm,cap
        except Exception as e:errors.append({'case':i,'error':type(e).__name__+': '+str(e)})
    save(a.output,{'kind':'fresh_physical_attention_'+a.stage,'stage':a.stage,'process':a.process,'created_utc':utc(),
       'protocol_sha256':sha(a.protocol),'decisions_sha256':sha(a.decisions),'correction_sha256':correction_sha,
       'torch_version':torch.__version__,'observed_runtime':observed,'rows':rows,'errors':errors,'wall_seconds':time.monotonic()-start})
    print(json.dumps({'stage':a.stage,'process':a.process,'rows':len(rows),'errors':len(errors),'seconds':time.monotonic()-start}),flush=True)
def calibrate(a):
    check_protocol(a.protocol);d=read(a.decisions);m=read(a.measurements)
    assert m['stage']=='anchors' and m['decisions_sha256']==sha(a.decisions) and not m['errors']
    assert sorted(x['case'] for x in m['rows'])==sorted(d['anchors'])
    actual={x['case']:x['latency_mean_ms'] for x in m['rows']}
    logs=np.array([math.log(actual[i]/d['decisions'][i]['prediction_ms']) for i in d['anchors']])
    coords=np.log2(np.array(CONFIGS,dtype=float));rows=[]
    for i in d['evaluation_cases']:
      distances=((coords[d['anchors']]-coords[i])**2).sum(1);w=np.exp(-.5*(distances-distances.min()));w=w/w.sum()
      raw=d['decisions'][i]['prediction_ms'];rows.append({'case':i,'pure_ms':raw,'calibrated_ms':float(raw*np.exp(w@logs)),'anchor_weights':w.tolist()})
    save(a.output,{'kind':'frozen_attention_three_query_correction','created_utc':utc(),'protocol_sha256':sha(a.protocol),'decisions_sha256':sha(a.decisions),
       'anchor_measurements_sha256':sha(a.measurements),'anchor_log_ratios':logs.tolist(),'predictions':rows,'evaluation_timings_read':False})
def summarize(a):
    p=check_protocol(a.protocol);d=read(a.decisions);c=read(a.correction);files=sorted(a.results.glob('eval-*.json'));assert len(files)==5
    raw=[read(f) for f in files];assert all(not x['errors'] and x['correction_sha256']==sha(a.correction) for x in raw)
    pred={x['case']:x for x in c['predictions']};rows=[];procs=[]
    for m in raw:
      assert sorted(x['case'] for x in m['rows'])==sorted(d['evaluation_cases'])
      for r in m['rows']:assert np.isclose(np.mean(r['latency_samples_ms']),r['latency_mean_ms'])
      tr={r['case']:r['latency_mean_ms'] for r in m['rows']}
      procs.append({'process':m['process'],**{k+'_mre_pct':float(np.mean([abs(pred[i][k+'_ms']-tr[i])/tr[i]*100 for i in tr])) for k in ['pure','calibrated']}})
    for i in d['evaluation_cases']:
      vals=[next(x['latency_mean_ms'] for x in m['rows'] if x['case']==i) for m in raw];true=float(np.mean(vals))
      rows.append({'case':i,'configuration':CONFIGS[i],'fresh_process_means_ms':vals,'physical_ms':true,**pred[i],
        'pure_re_pct':abs(pred[i]['pure_ms']-true)/true*100,'calibrated_re_pct':abs(pred[i]['calibrated_ms']-true)/true*100})
    metrics={}
    for k in ['pure','calibrated']:
      errors=np.array([abs(x[k+'_ms']-x['physical_ms']) for x in rows]);metrics[k]={'mre_pct':float(np.mean([x[k+'_re_pct'] for x in rows])),
        'mae_ms':float(errors.mean()),'wape_pct':float(100*errors.sum()/sum(x['physical_ms'] for x in rows))}
    save(a.output,{'kind':'attention_three_query_independent_evaluation','created_utc':utc(),'protocol_sha256':sha(a.protocol),'decisions_sha256':sha(a.decisions),
       'correction_sha256':sha(a.correction),'measurement_files':{f.name:sha(f) for f in files},'device_id':d['device_id'],
       'anchor_count':3,'evaluation_queries':21,'fresh_evaluation_processes':5,'metrics':metrics,'per_process':procs,'rows':rows,
       'decision_seconds':d['feature_capture_and_prediction_wall_seconds'],'evaluation_wall_seconds':sum(x['wall_seconds'] for x in raw),
       'scope':p['scope'],'test_queries_or_latencies_used_to_fit_correction':False})
    print(json.dumps(metrics),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','measure','calibrate','summarize'])
    for n in ['protocol','hardware','decisions','measurements','correction','results','output']:ap.add_argument('--'+n,type=Path)
    ap.add_argument('--device',choices=['cpu','mps','cuda','npu']);ap.add_argument('--stage',choices=['anchors','evaluation']);ap.add_argument('--process',type=int,default=0)
    a=ap.parse_args();globals()[a.action](a)
if __name__=='__main__':main()
