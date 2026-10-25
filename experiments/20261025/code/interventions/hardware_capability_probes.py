"""Separate real backend/precision probes, never synthetic failed model actions."""
import argparse
import json
import time
from datetime import datetime,timezone
from pathlib import Path

import torch
from full_model_coalitions import sync,hardware,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','mps','cuda','npu'],required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.device=='npu':import torch_npu
    d=torch.device(args.device);rows=[]
    def fp32():
        x=torch.ones((32,64),device=d,dtype=torch.float32);y=x@x.t();sync(d);return list(y.shape)
    def fp64():
        x=torch.ones((32,64),device=d,dtype=torch.float64);y=x@x.t();sync(d);return list(y.shape)
    def fp8():
        x=torch.ones((32,64),device=d).to(torch.float8_e4m3fn)
        y=torch.ones((32,64),device=d).to(torch.float8_e4m3fn).t()
        scale=torch.tensor(1.0,device=d)
        z=torch._scaled_mm(x,y,scale_a=scale,scale_b=scale,out_dtype=torch.float32)
        sync(d);return list(z.shape)
    for name,fn in [('fp32_matmul',fp32),('fp64_matmul',fp64),('fp8_e4m3_scaled_mm',fp8)]:
        start=time.perf_counter()
        try:result=fn();row={'probe':name,'status':'supported','output_shape':result}
        except Exception as e:row={'probe':name,'status':'unsupported_or_failed','exception_type':type(e).__name__,'exception':str(e)}
        row.update({'elapsed_seconds':time.perf_counter()-start,'attempted':True,'provenance':'physical_backend_execution'})
        rows.append(row)
    out={'schema_version':'het.backend.probes.v2','utc':datetime.now(timezone.utc).isoformat(),
         'device':args.device,'torch':torch.__version__,**hardware(d),'probes':rows,
         'source_sha256':sha(Path(__file__).read_bytes()),
         'scope':'Independent precision/backend capability checks; not members of the main four ConvBN action set. All-supported main-set mask ablation should have zero effect.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
