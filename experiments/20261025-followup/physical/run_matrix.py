"""One subprocess per preregistered block; freeze all plans before evaluation."""
import hashlib,json,statistics,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/work');OUT=ROOT/'layout-raw';OUT.mkdir(exist_ok=True)
STATE={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running','runs':[],
       'collector_sha256':hashlib.sha256((ROOT/'layout_validation.py').read_bytes()).hexdigest()}
def write_state():
 p=ROOT/'matrix-state.tmp';p.write_text(json.dumps(STATE,indent=2));p.replace(ROOT/'matrix-state.json')
def run_one(model,batch,run,plan=None):
 out=OUT/f'{model}-b{batch}-r{run}.json'
 cmd=[sys.executable,str(ROOT/'layout_validation.py'),'--model',model,'--batch',str(batch),'--run',str(run),'--assets','/work/assets/imagenet-val200','--output',str(out)]
 if plan:cmd+=['--frozen-plan',str(plan)]
 STATE['current']={'model':model,'batch':batch,'run':run};write_state()
 with (OUT/f'{model}-b{batch}-r{run}.log').open('w') as f:
  process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
  STATE['current']['pid']=process.pid;write_state();code=process.wait()
 result=json.loads(out.read_text()) if out.exists() else {'status':'no_raw_file'}
 STATE['runs'].append({'model':model,'batch':batch,'run':run,'pid':process.pid,'exit_code':code,
     'status':result['status'],'file':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() else None})
 write_state()
 if code or result['status']!='completed':raise RuntimeError('Failed block retained; do not overwrite or silently rerun')
 assert result['source_sha256']==STATE['collector_sha256']
 assert result['process_identity']['pid']==process.pid
def main():
 for model in ['resnet18','resnet50']:
  for batch in [1,4]:
   for run in [1,2,3]:run_one(model,batch,run)
 plan={'evaluation_started':False,'calibration_files':[],'selections':{},'collector_sha256':STATE['collector_sha256']}
 for model in ['resnet18','resnet50']:
  for batch in [1,4]:
   values=[]
   for run in [1,2,3]:
    f=OUT/f'{model}-b{batch}-r{run}.json';values.append(json.loads(f.read_text()))
    plan['calibration_files'].append({'name':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
   gains={m:statistics.mean(v['masks'][str(m)]['gain_percent'] for v in values) for m in range(4)}
   valid={m:all(v['masks'][str(m)]['accepted'] for v in values) for m in range(4)}
   eligible=[0]+[m for m in [1,2] if valid[m] and gains[m]>0]
   k1=sorted(eligible,key=lambda m:(-(gains[m] if m else 0),m))[0]
   k2=3 if valid[3] and gains[3]>0 else 0
   plan['selections'][f'{model}-b{batch}']={'k1_mask':k1,'k2_fixed_combo_mask':k2,'all_calibration_gain_percent':gains,'all_calibration_accepted':valid}
 plan['frozen_utc']=datetime.now(timezone.utc).isoformat();path=ROOT/'frozen-layout-plans.json';path.write_text(json.dumps(plan,indent=2))
 STATE['frozen_plan_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();write_state()
 for model in ['resnet18','resnet50']:
  for batch in [1,4]:
   for run in [4,5,6,7,8]:run_one(model,batch,run,path)
 STATE.update({'status':'completed','completed_utc':datetime.now(timezone.utc).isoformat()});write_state()
if __name__=='__main__':
 try:main()
 except Exception as error:
  STATE.update({'status':'failed','error':str(error)});write_state();raise
