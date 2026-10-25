"""Serial fresh-process confirmation; preserve all outcomes and prior evidence."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--python',required=True)
    ap.add_argument('--script',type=Path,required=True)
    ap.add_argument('--assets',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--device',default='npu')
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    codehash=hashlib.sha256(args.script.read_bytes()).hexdigest()
    assert codehash=='e88bce1b159f018e5dfa5793c726d362b70bcf0ed9795bf2a881d0ac5c1408c5'
    receipt={'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':codehash,
             'planned_blocks':20,'runs':[],
             'first_block_source_patch':{'old_sha256':'2f33287e7d6afffb9303724ac61186c5712cbaaa0784795196d7b1f4ec6c22bb',
                  'scope':'Only move flush=True from json.dumps to print after final persisted result; no measurement change'}}
    state=args.output_dir/'paired-matrix-state.json'
    if state.exists():raise FileExistsError('Inspect existing schedule before restarting')
    def save():
        p=state.with_suffix('.tmp');p.write_text(json.dumps(receipt,indent=2));p.replace(state)
    save()
    for model in ['resnet18','resnet50']:
        for batch in [1,4]:
            for run in range(101,106):
                output=args.output_dir/f'{args.device}-{model}-b{batch}-r{run}.json'
                if output.exists():
                    d=json.loads(output.read_text())
                    allowed=[codehash]
                    if model=='resnet18' and batch==1 and run==101:
                        allowed.append(receipt['first_block_source_patch']['old_sha256'])
                    assert d['source_sha256'] in allowed
                    assert d['status'] in ['completed','completed_with_failures']
                    assert d['model_id']==model and d['run_index']==run and d['input_shape'][0]==batch
                    status='existing_verified'
                else:
                    command=[args.python,str(args.script),'--device',args.device,'--model',model,
                             '--batch',str(batch),'--run',str(run),'--assets',str(args.assets),'--output',str(output)]
                    result=subprocess.run(command,capture_output=True,text=True,timeout=900)
                    if result.returncode:
                        receipt['failed_command']={'command':command,'exit_code':result.returncode,
                                                   'stdout':result.stdout,'stderr':result.stderr};save()
                        raise RuntimeError(result.stderr)
                    d=json.loads(output.read_text());assert d['source_sha256']==codehash
                    status='executed'
                item={'run_id':d['run_id'],'file':str(output),'status':status,
                      'collection_status':d['status'],'masks':len(d['coalitions']),
                      'pairs':sum(r.get('completed_pairs',0) for r in d['coalitions']),
                      'feasible':sum(r.get('final_candidate_feasible',False) for r in d['coalitions'])}
                receipt['runs'].append(item);save();print(json.dumps(item),flush=True)
    receipt['completed_utc']=datetime.now(timezone.utc).isoformat();save()

if __name__=='__main__':main()
