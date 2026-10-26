"""Research-pod CPU-only bridge execution; never touches paid NPU or old runs."""
import argparse
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
V2=HERE.parent
sys.path.insert(0,str(V2))
import infra_controller as infra

REMOTE='/work/bridge-analysis-early'
OUT=HERE/'bridge-early-results'
INDEX=V2/'evidence/early-tasks/index.json'
MODELS=['resnet18']

def digest(data):return hashlib.sha256(data).hexdigest()

def remote(args,input=None,timeout=50):
    return subprocess.run(infra.SSH+[shlex.join(infra.K+args)],input=input,capture_output=True,timeout=timeout)

def require(result):
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
    return result.stdout

RUNNER=r'''import json,os,pathlib,subprocess,sys,time,platform
root=pathlib.Path('/work/bridge-analysis-early')
os.nice(19)
env=os.environ.copy()
env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
manifest=json.loads((root/'input-manifest.json').read_text())
status={'kind':'CPU_only_frozen_early_migration_bridge','started_epoch':time.time(),'python':sys.version,'platform':platform.platform(),'nice':os.nice(0),'threads':1,'paid_npu_used':False,'runs':[]}
def write():
 p=root/'status.tmp';p.write_text(json.dumps(status,indent=2));p.replace(root/'status.json')
write()
for row in manifest['pairs']:
 out=root/'results'/('a100-to-910b-'+row['context']+'.json')
 command=[sys.executable,str(root/'methods/migration_bridge.py'),'--source-task',str(root/row['source']),'--target-task',str(root/row['target']),'--output',str(out),'--raw-path-map',str(root/'raw-path-map.json'),'--budget','2','--permutations','500','--seed','42','--reference-permutations','5000','--reference-seed','90210']
 started=time.time();r=subprocess.run(command,env=env,capture_output=True,text=True)
 (root/'results'/('a100-to-910b-'+row['context']+'.stdout.txt')).write_text(r.stdout)
 (root/'results'/('a100-to-910b-'+row['context']+'.stderr.txt')).write_text(r.stderr)
 status['runs'].append({'context':row['context'],'command':command,'started_epoch':started,'finished_epoch':time.time(),'exit_code':r.returncode,'output':str(out),'stdout':r.stdout,'stderr':r.stderr})
 write()
 if r.returncode:raise SystemExit(r.returncode)
status['completed_epoch']=time.time();write()
'''

def upload():
    OUT.mkdir(parents=True,exist_ok=True)
    index=json.loads(INDEX.read_text())
    contents={};mapping={};entries=[];lookup={}
    for p in HERE.glob('*.py'):
        if p.name=='cloud_early_bridge.py':continue
        contents['methods/'+p.name]=p.read_bytes()
    for entry in index:
        task=Path(entry['file'])
        if not task.is_absolute():
            options=[V2.parent/task,INDEX.parent/task,INDEX.parent/task.name,HERE/task]
            task=next((p for p in options if p.exists()),options[0])
        data=task.read_bytes()
        if digest(data)!=entry['sha256']:raise ValueError('task SHA mismatch')
        contents['tasks/'+task.name]=data
        d=json.loads(data);lookup[(d['model_id'],d['input_shape'][0],d['device_id'].split('_')[0])]='tasks/'+task.name
        entries.append({'kind':'task','original_path':str(task),'remote_path':REMOTE+'/tasks/'+task.name,'sha256':digest(data)})
        for artifact in d['source_artifacts']:
            raw=Path(artifact['path']);blob=raw.read_bytes()
            if digest(blob)!=artifact['sha256']:raise ValueError('raw SHA mismatch')
            key='raw/'+raw.name
            if key in contents and contents[key]!=blob:raise ValueError('raw basename collision')
            contents[key]=blob;mapping[artifact['path']]=REMOTE+'/'+key
    pairs=[{'context':f'{model}-b{batch}','source':lookup[(model,batch,'cuda')],
            'target':lookup[(model,batch,'npu')]} for model in MODELS for batch in [1,4]]
    for original,resolved in mapping.items():
        entries.append({'kind':'raw','original_path':original,'remote_path':resolved,'sha256':digest(contents[resolved[len(REMOTE)+1:]])})
    manifest={'scope':'48 actual raw blocks and 6 assembled contexts; immutable copy, CPU-only offline analysis',
              'files':entries,'pairs':pairs,'method_sha256':{k:digest(v) for k,v in contents.items() if k.startswith('methods/')}}
    contents['input-manifest.json']=json.dumps(manifest,indent=2).encode()
    contents['raw-path-map.json']=json.dumps(mapping,indent=2).encode()
    contents['run.py']=RUNNER.replace('/work/bridge-analysis-early',REMOTE).encode()
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w:gz') as tf:
        for name,blob in contents.items():
            info=tarfile.TarInfo(Path(REMOTE).name+'/'+name);info.size=len(blob);tf.addfile(info,io.BytesIO(blob))
        info=tarfile.TarInfo(Path(REMOTE).name+'/results');info.type=tarfile.DIRTYPE;info.mode=0o755;tf.addfile(info)
    bundle=data.getvalue()
    require(remote(['exec','-i','-n',infra.NS,infra.POD,'--','tar','-xzf','-','-C','/work'],bundle,timeout=90))
    (OUT/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'raw-path-map.json').write_text(json.dumps(mapping,indent=2))
    (OUT/'upload-receipt.json').write_text(json.dumps({'archive_sha256':digest(bundle),'archive_bytes':len(bundle),'files':len(contents),'raw_files':len(mapping),'remote':REMOTE},indent=2))
    print(json.dumps({'uploaded_bytes':len(bundle),'raw_files':len(mapping),'task_files':len(index)}))

def launch():
    code="import pathlib,subprocess,sys,json; p=pathlib.Path('/work/bridge-analysis-early'); assert not (p/'status.json').exists(); f=(p/'runner.log').open('x'); q=subprocess.Popen([sys.executable,'-u',str(p/'run.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(json.dumps({'pid':q.pid,'remote':str(p)}))"
    code=code.replace('/work/bridge-analysis-early',REMOTE)
    data=require(remote(['exec','-n',infra.NS,infra.POD,'--','python','-c',code]))
    (OUT/'launch-receipt.json').write_bytes(data);print(data.decode())

def status():
    data=require(remote(['exec','-n',infra.NS,infra.POD,'--','cat',REMOTE+'/status.json']))
    d=json.loads(data);print(json.dumps(d));(OUT/'status-snapshot.json').write_text(json.dumps(d,indent=2))

def pull():
    data=require(remote(['exec','-n',infra.NS,infra.POD,'--','tar','-czf','-','-C',REMOTE,'results','status.json','runner.log','run.py','input-manifest.json','raw-path-map.json'],timeout=90))
    (OUT/'cloud-output.tar.gz').write_bytes(data)
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tf:
        tf.extractall(OUT,filter='data')
    print(json.dumps({'downloaded_bytes':len(data),'sha256':digest(data),'output':str(OUT)}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['upload','launch','status','pull'])
    ap.add_argument('--index',type=Path,default=INDEX);ap.add_argument('--remote-name',default='bridge-analysis-early')
    ap.add_argument('--output-name',default='bridge-early-results');ap.add_argument('--models',nargs='+',choices=['resnet18','resnet50'],default=MODELS)
    a=ap.parse_args()
    if any('/' in value or value in ['.','..'] for value in [a.remote_name,a.output_name]):raise ValueError('one directory name required')
    INDEX=a.index;REMOTE='/work/'+a.remote_name;OUT=HERE/a.output_name;MODELS=a.models
    {'upload':upload,'launch':launch,'status':status,'pull':pull}[a.action]()
