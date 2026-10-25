"""One sequential CPU-only, nice19 real-table PPO matrix in research pod."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from cloud_early_bridge import remote,require,infra

HERE=Path(__file__).resolve().parent;INPUT=HERE/'planner-matrix-inputs';OUT=HERE/'planner-matrix-results';REMOTE='/work/planner-matrix-v2'
def sha(data):return hashlib.sha256(data).hexdigest()
RUNNER=r'''import pathlib,json,os,subprocess,sys,time,hashlib
root=pathlib.Path('/work/planner-matrix-v2');os.nice(19)
env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
index=json.loads((root/'index.json').read_text());status={'kind':'sequential_cpu_measured_PPO_matrix','start_epoch':time.time(),'nice':os.nice(0),'threads':1,'npu_used':False,'gpu_used':False,'runs':[]}
def save():
 p=root/'status.tmp';p.write_text(json.dumps(status,indent=2));p.replace(root/'status.json')
save()
for e in index:
 name=pathlib.Path(e['file']).stem;out=root/'results'/(name+'.json')
 cmd=[sys.executable,str(root/'methods/analyze_measured.py'),'--input',str(root/'tasks'/pathlib.Path(e['file']).name),'--output',str(out),'--budget','2','--steps','4096','--seeds','11','22','33','44','55','--with-embedding']
 started=time.time()
 with (root/'results'/(name+'.log')).open('x') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 status['runs'].append({'task_id':e['task_id'],'device_id':e['device_id'],'start_epoch':started,'finish_epoch':time.time(),'exit_code':r.returncode,'command':cmd,'output':str(out)})
 save()
 if r.returncode:raise SystemExit(r.returncode)
status['completed_epoch']=time.time();save()
'''

def upload():
    OUT.mkdir(exist_ok=True);index=json.loads((INPUT/'index.json').read_text());files={}
    for name in ['analyze_measured.py','measured_planner.py','copa_v2.py','runtime_calibration.py']:files['methods/'+name]=(HERE/name).read_bytes()
    mapping={}
    for entry in index:
        p=Path(entry['file']);data=p.read_bytes();assert sha(data)==entry['sha256'];files['tasks/'+p.name]=data
        for artifact in json.loads(data)['source_artifacts']:
            raw=Path(artifact['path']);blob=raw.read_bytes();assert sha(blob)==artifact['sha256'];files['raw/'+raw.name]=blob;mapping[artifact['path']]=REMOTE+'/raw/'+raw.name
    for p in INPUT.glob('device-mapping-*.json'):files[p.name]=p.read_bytes()
    for p in INPUT.glob('action-embeddings-*.json'):files[p.name]=p.read_bytes()
    files['index.json']=json.dumps(index,indent=2).encode();files['raw-path-map.json']=json.dumps(mapping,indent=2).encode();files['run.py']=RUNNER.encode()
    manifest={k:sha(v) for k,v in files.items()};files['input-hashes.json']=json.dumps(manifest,indent=2).encode()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tf:
        for name,data in files.items():
            info=tarfile.TarInfo('planner-matrix-v2/'+name);info.size=len(data);tf.addfile(info,io.BytesIO(data))
        info=tarfile.TarInfo('planner-matrix-v2/results');info.type=tarfile.DIRTYPE;info.mode=0o755;tf.addfile(info)
    data=buffer.getvalue();require(remote(['exec','-i','-n',infra.NS,infra.POD,'--','tar','-xzf','-','-C','/work'],data,timeout=90))
    (OUT/'input-hashes.json').write_text(json.dumps(manifest,indent=2));(OUT/'upload-receipt.json').write_text(json.dumps({'bytes':len(data),'sha256':sha(data),'raw_files':len(mapping),'tasks':len(index)},indent=2))
    code="import pathlib,json,hashlib; p=pathlib.Path('/work/planner-matrix-v2'); m=json.loads((p/'input-hashes.json').read_text()); bad=[n for n,h in m.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=h]; assert not bad,bad; print(json.dumps({'verified_files':len(m),'bad':bad}))"
    result=require(remote(['exec','-n',infra.NS,infra.POD,'--','python','-c',code]));(OUT/'remote-hash-verification.json').write_bytes(result);print(result.decode())

def launch():
    code="import pathlib,sys,subprocess,json; p=pathlib.Path('/work/planner-matrix-v2'); assert not (p/'status.json').exists(); f=(p/'runner.log').open('x'); r=subprocess.Popen([sys.executable,'-u',str(p/'run.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(json.dumps({'pid':r.pid,'remote':str(p)}))"
    data=require(remote(['exec','-n',infra.NS,infra.POD,'--','python','-c',code]));(OUT/'launch-receipt.json').write_bytes(data);print(data.decode())

def status():
    data=require(remote(['exec','-n',infra.NS,infra.POD,'--','cat',REMOTE+'/status.json']));(OUT/'status-snapshot.json').write_bytes(data);print(data.decode())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['upload','launch','status']);a=p.parse_args();{'upload':upload,'launch':launch,'status':status}[a.action]()
