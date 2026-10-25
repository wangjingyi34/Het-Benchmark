"""Measure only the three independent hardware descriptors, no model labels."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import torch
import collect_profiles as core

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',required=True,choices=['cpu','mps','cuda','npu'])
    ap.add_argument('--device-id',required=True);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    if args.output.exists():raise ValueError('do not overwrite hardware probes')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.device=='npu':
        import collect_profiles_npu as adapter
        fn=adapter.npu_hardware
    else:fn=core.hardware
    device=torch.device(args.device)
    with torch.inference_mode():h=fn(device,args)
    out={'kind':'hardware_descriptor_only','created_utc':datetime.now(timezone.utc).isoformat(),
         'hardware':h,'source_profile_sha256':None,
         'source_collector_code_sha256':hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
         'probe_only_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'model_latency_labels_collected':0,'scope':'Exactly three independent hardware microprobes; no model target timing'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({'output':str(args.output),'device_id':h['device_id'],'backend':h['backend'],
                      'hardware_name':h['hardware_name'],'probes':len(h['probes']),'model_labels':0}))

if __name__=='__main__':main()
