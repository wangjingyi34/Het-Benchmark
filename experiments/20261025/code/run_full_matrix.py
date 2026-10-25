"""Run the frozen full-model protocol serially on each requested backend."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--devices',nargs='+',required=True)
    ap.add_argument('--script',type=Path,required=True);ap.add_argument('--assets',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--python',default=sys.executable)
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    codehash=hashlib.sha256(args.script.read_bytes()).hexdigest()
    receipt={'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':codehash,
             'devices':args.devices,'planned_runs':len(args.devices)*32,'runs':[]}
    state=args.output_dir/('matrix-state-'+'-'.join(args.devices)+'.json')
    def save():
        t=state.with_suffix('.tmp');t.write_text(json.dumps(receipt,indent=2));t.replace(state)
    save()
    for device in args.devices:
        for model in ['resnet18','resnet50']:
            for batch in [1,4]:
                for run in range(1,9):
                    path=args.output_dir/f'{device}-{model}-b{batch}-r{run}.json'
                    if path.exists():
                        d=json.loads(path.read_text());assert d['protocol']['source_sha256']==codehash
                        assert d['run_index']==run and d['model_id']==model and d['input_shape'][0]==batch
                        assert len(d['coalitions'])==16
                        r={'file':str(path),'status':'existing_verified','run_id':d['run_id']}
                    else:
                        cmd=[args.python,str(args.script),'--model',model,'--device',device,'--run',str(run),
                             '--batch',str(batch),'--assets',str(args.assets),'--output',str(path)]
                        if run<=3:cmd.append('--profile-all-nodes')
                        result=subprocess.run(cmd,text=True,capture_output=True)
                        if result.returncode:
                            receipt['failed_command']={'args':cmd,'exit_code':result.returncode,
                                                       'stdout':result.stdout,'stderr':result.stderr};save()
                            raise RuntimeError(result.stderr)
                        d=json.loads(path.read_text());assert d['protocol']['source_sha256']==codehash
                        r={'file':str(path),'status':'executed','run_id':d['run_id'],
                           'stdout':result.stdout,'stderr':result.stderr,
                           'coalitions_ok':sum(x['status']=='ok' for x in d['coalitions']),
                           'quality_accepted':sum(x.get('quality_threshold_passed',False) for x in d['coalitions'])}
                    receipt['runs'].append(r);save();print(json.dumps(r),flush=True)
    receipt['completed_utc']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps({'complete':True,'runs':len(receipt['runs']),'manifest':str(state)}),flush=True)

if __name__=='__main__':main()
