"""New physical deployment validation; never authenticates historical FITAS.

F is all eligible Conv/BN folds; L is channels-last parameters with the input
conversion included in every timed call. Every F/L coalition and no-op control
is retained. The 3 calibration / 5 evaluation process split is frozen.
"""
import argparse,copy,hashlib,json,os,platform,random,statistics,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch,torchvision
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval
from torchvision import models
from full_model_coalitions import assign,disable_inplace_modules,hardware,sha,state_hash,sync

ACTIVE_STATE={}
ACTIVE_OUTPUT=None
def persist():
    if ACTIVE_OUTPUT is not None:
        ACTIVE_OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        temporary=ACTIVE_OUTPUT.with_suffix('.partial')
        temporary.write_text(json.dumps(ACTIVE_STATE,indent=2))
        temporary.replace(ACTIVE_OUTPUT)

class Candidate(nn.Module):
    def __init__(self,base,mask):
        super().__init__();self.model=copy.deepcopy(base).eval();self.layout=bool(mask&2)
        self.folded=[]
        if mask&1:
            for prefix,m in list(self.model.named_modules()):
                for cname,conv in list(m.named_children()):
                    bname='bn'+cname[4:] if cname.startswith('conv') else ('1' if cname=='0' else '')
                    bn=getattr(m,bname,None) if bname else None
                    if isinstance(conv,nn.Conv2d) and isinstance(bn,nn.BatchNorm2d):
                        setattr(m,cname,fuse_conv_bn_eval(conv,bn));setattr(m,bname,nn.Identity())
                        self.folded.append((prefix+'.'+cname).strip('.'))
        if self.layout:self.model.to(memory_format=torch.channels_last)
    def forward(self,x):
        if self.layout:x=x.contiguous(memory_format=torch.channels_last)
        return self.model(x)

def utc():return datetime.now(timezone.utc).isoformat()
def th(x):return sha(x.detach().cpu().contiguous().numpy().tobytes())
def quality(model,images,labels,device):
    logits=torch.cat([model(images[i:i+16].to(device)).cpu() for i in range(0,len(images),16)])
    pred=logits.argmax(1)
    return logits,{'accuracy':(pred==labels).double().mean().item(),'labels':labels.tolist(),
                   'predictions':pred.tolist(),'logits_sha256':th(logits)}
def call(model,x,device):
    sync(device);start=time.perf_counter_ns();y=model(x);sync(device);end=time.perf_counter_ns()
    return {'start_ns':start,'end_ns':end,'latency_ms':(end-start)/1e6},y

def main():
    global ACTIVE_OUTPUT
    a=argparse.ArgumentParser();a.add_argument('--model',choices=['resnet18','resnet50'],required=True)
    a.add_argument('--batch',type=int,choices=[1,4],required=True);a.add_argument('--run',type=int,choices=range(1,9),required=True)
    a.add_argument('--assets',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--frozen-plan',type=Path)
    a.add_argument('--device',default='cuda',choices=['cpu','cuda']);args=a.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    ACTIVE_OUTPUT=args.output
    ACTIVE_STATE.update({'status':'started','started_utc':utc(),'pid':os.getpid(),
        'process_identity':{'pid':os.getpid(),'ppid':os.getppid(),'python':sys.version,'platform':platform.platform(),'argv':sys.argv},
        'model':args.model,'batch':args.batch,'run':args.run,
        'source_sha256':sha(Path(__file__).read_bytes()),'pairs':[]})
    persist()
    if args.run>=4:
        if args.frozen_plan is None:raise ValueError('Evaluation requires the pre-evaluation frozen plan')
        plans=json.loads(args.frozen_plan.read_text())
        assert plans['evaluation_started'] is False and len(plans['calibration_files'])==12
        ACTIVE_STATE['frozen_plan_sha256']=sha(args.frozen_plan.read_bytes())
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    device=torch.device(args.device)
    partition='calibration' if args.run<=3 else 'evaluation'
    manifest=json.loads((args.assets/'manifest.json').read_text())
    assert sha((args.assets/'quality_inputs.npz').read_bytes())==manifest['quality_asset_sha256']
    meta=manifest['timing_inputs'][partition+f'_b{args.batch}']
    xn=np.load(args.assets/meta['file'],allow_pickle=False);assert sha(xn.tobytes())==meta['tensor_sha256']
    q=np.load(args.assets/'quality_inputs.npz',allow_pickle=False)
    ids=[i for i,s in enumerate(manifest['samples']) if s['partition']==partition]
    images=torch.from_numpy(q['images'][ids].copy());labels=torch.from_numpy(q['labels'][ids].copy())
    wc=getattr(models,'ResNet18_Weights' if args.model=='resnet18' else 'ResNet50_Weights')
    base=getattr(models,args.model)(weights=wc.IMAGENET1K_V1).eval()
    disabled=disable_inplace_modules(base)
    for p in base.parameters():p.requires_grad_(False)
    original_hash=state_hash(base);x=torch.from_numpy(xn).to(device);assert x.is_contiguous()
    candidates={m:Candidate(base,m).to(device).eval() for m in range(4)}
    baselines={m:Candidate(base,0).to(device).eval() for m in range(4)}
    ACTIVE_STATE.update({'protocol':'layout-validation-v4','source_sha256':sha(Path(__file__).read_bytes()),
          'helper_sha256':sha(Path(__file__).with_name('full_model_coalitions.py').read_bytes()),
          'model':args.model,'batch':args.batch,'run':args.run,'partition':partition,'device':args.device,
          'hardware':hardware(device),'torch':torch.__version__,'torchvision':torchvision.__version__,
          'weights_hash':original_hash,'input_hash':th(x),'input_shape':list(x.shape),'input_strides':list(x.stride()),
          'asset_manifest_sha256':sha((args.assets/'manifest.json').read_bytes()),
          'quality_asset_sha256':manifest['quality_asset_sha256'],
          'sample_ids':[manifest['samples'][i]['sample_id'] for i in ids],
          'precision':{'dtype':'float32','matmul_tf32':False,'cudnn_tf32':False,'benchmark':False},
          'disabled_inplace':disabled,'masks':{},'pairs':[],
          'scope':'Supplementary measured deployment actions, not original FITAS reconstruction or cross-hardware COPA proof.'})
    data=ACTIVE_STATE
    for m in range(4):
        first=next(candidates[m].model.parameters())
        data.setdefault('layout_readback',{})[str(m)]={'parameter_shape':list(first.shape),'parameter_stride':list(first.stride()),
            'parameter_channels_last':bool(first.is_contiguous(memory_format=torch.channels_last)),'folded_modules':candidates[m].folded}
    persist()
    with torch.inference_mode():
        for mask in range(4):
            bl,bq=quality(baselines[mask],images,labels,device);cl,cq=quality(candidates[mask],images,labels,device)
            eq=torch.allclose(bl,cl,rtol=1e-4,atol=1e-4)
            retention=cq['accuracy']/bq['accuracy'] if bq['accuracy'] else None
            data['masks'][str(mask)]={'mask':mask,'actions':(['all_conv_bn_fold'] if mask&1 else [])+(['channels_last_with_conversion'] if mask&2 else []),
                'folded_modules':candidates[mask].folded,'baseline_quality':bq,'candidate_quality':cq,
                'max_absolute_logit_difference':float((bl-cl).abs().max()),'allclose':bool(eq),
                'quality_retention':retention,'quality_gate':bool(retention is not None and retention>=.99 and eq)}
            persist()
        # Both sides have the same resident model count, number of warmups and
        # invocation frequency. Input conversion for L happens within forward.
        for m in range(4):
            for _ in range(30):baselines[m](x);candidates[m](x)
        sync(device)
        seed=42000+args.run+args.batch*100+(1000 if args.model=='resnet50' else 0)
        rng=random.Random(seed);orders={m:['AB']*10+['BA']*10 for m in range(4)}
        for o in orders.values():rng.shuffle(o)
        rounds=[]
        for i in range(20):
            masks=list(range(4));rng.shuffle(masks);rounds.append(masks)
            for m in masks:
                order=orders[m][i];calls={};ys={}
                for side in order:
                    calls[side],ys[side]=call(baselines[m] if side=='A' else candidates[m],x,device)
                close=torch.allclose(ys['A'],ys['B'],rtol=1e-4,atol=1e-4)
                data['pairs'].append({'round':i,'mask':m,'order':order,'A':calls['A'],'B':calls['B'],
                    'allclose':bool(close),'A_output_sha256':th(ys['A']),'B_output_sha256':th(ys['B'])})
            persist()
        data['randomization']={'seed':seed,'orders':orders,'round_masks':rounds}
    for m in range(4):
        rows=[r for r in data['pairs'] if r['mask']==m]
        av=statistics.mean(r['A']['latency_ms'] for r in rows);bv=statistics.mean(r['B']['latency_ms'] for r in rows)
        data['masks'][str(m)].update({'baseline_mean_ms':av,'candidate_mean_ms':bv,'gain_percent':100*(av-bv)/av,
            'all_timed_outputs_equivalent':all(r['allclose'] for r in rows),
            'accepted':data['masks'][str(m)]['quality_gate'] and all(r['allclose'] for r in rows)})
    data['status']='completed';data['completed_utc']=utc();persist()
    print(json.dumps({'output':str(args.output),'sha256':sha(args.output.read_bytes()),
                      'gains':{m:v['gain_percent'] for m,v in data['masks'].items()},
                      'accepted':all(v['accepted'] for v in data['masks'].values())}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as error:
        if ACTIVE_OUTPUT is not None:
            ACTIVE_STATE.update({'status':'failed','failed_utc':utc(),'exception':{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}})
            persist()
        raise
