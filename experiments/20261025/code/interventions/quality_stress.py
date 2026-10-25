"""Real labeled fault-injection check of final measured quality rejection.

The corrupt constant-logit model is intentionally invalid. It is never treated
as an optimization action or counted in main-study performance comparisons.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models

from full_model_coalitions import hardware,timed,sha,state_hash,disable_inplace_modules

class CorruptConstantModel(nn.Module):
    def forward(self,x):return torch.zeros((x.shape[0],1000),device=x.device,dtype=x.dtype)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','mps','cuda','npu'],required=True)
    ap.add_argument('--model',choices=['resnet18','resnet50'],default='resnet18')
    ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--quality-batch',type=int,default=16);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.device=='npu':import torch_npu
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    d=torch.device(args.device)
    weights=(models.ResNet18_Weights if args.model=='resnet18' else models.ResNet50_Weights).IMAGENET1K_V1
    baseline=getattr(models,args.model)(weights=weights).eval().to(d);fault=CorruptConstantModel().eval().to(d)
    disabled_inplace=disable_inplace_modules(baseline)
    manifest=json.loads((args.assets/'manifest.json').read_text());q=np.load(args.assets/'quality_inputs.npz',allow_pickle=False)
    assert sha((args.assets/'quality_inputs.npz').read_bytes())==manifest['quality_asset_sha256']
    payload={'schema_version':'het.measured.coalitions.v2','task_id':'FAULT_INJECTION_'+args.model,
        'device_id':args.device+'_'+hardware(d)['hardware'],'model_id':args.model,
        'action_ids':['intentional_constant_logit_corruption'],
        'action_metadata':[{'id':'intentional_constant_logit_corruption','hardware_supported':True,
                            'profile_run_ids':[],'purpose':'intentional invalid candidate; not a main optimization'}],
        'quality_definition':{'metric':'ImageNet1000_top1_accuracy','dataset_id':manifest['dataset_id'],
                              'mode':'retention_ratio','threshold':.99},'train':[],'eval':[]}
    with torch.inference_mode():
        for partition,split in [('calibration','train'),('evaluation','eval')]:
            ids=[i for i,s in enumerate(manifest['samples']) if s['partition']==partition]
            images=torch.from_numpy(q['images'][ids].copy());labels=torch.from_numpy(q['labels'][ids].copy())
            sample_ids=[manifest['samples'][i]['sample_id'] for i in ids]
            inp=torch.from_numpy(np.load(args.assets/f'{partition}_b1.npy',allow_pickle=False)).to(d)
            rows=[]
            for mask,model in [(0,baseline),(1,fault)]:
                chunks=[]
                for start in range(0,len(ids),args.quality_batch):
                    chunks.append(model(images[start:start+args.quality_batch].to(d)).cpu())
                logits=torch.cat(chunks);pred=logits.argmax(dim=1);correct=int((pred==labels).sum())
                if partition=='evaluation' and mask==0:baseline_eval_logits=logits.clone()
                rows.append({'mask':mask,'status':'ok','provenance':'physical_measurement',
                    'artifact_id':f'fault_{args.device}_{args.model}_{partition}_mask{mask}',
                    'quality_evaluated':True,'quality_score':correct/len(ids),'correct_count':correct,
                    'sample_ids':sample_ids,'predictions':pred.tolist(),'labels':labels.tolist(),
                    'logits_sha256':sha(logits.contiguous().numpy().tobytes()),
                    **timed(lambda:model(inp),d,10,30)})
            payload[split].append({'run_id':f'fault_{args.device}_{args.model}_{partition}',
                'provenance':'physical_measurement','quality_partition':partition,'quality_sample_ids':sample_ids,
                'baseline_quality':rows[0]['quality_score'],'coalitions':rows})
    # Execute the exact newly implemented final gate, not a hand-coded expected assertion.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'methods'))
    from measured_planner import MeasuredTable
    table=MeasuredTable(payload)
    result=table.final_gate(table.splits['eval'][0],1)
    # Execute the selected fallback model after rejection, rather than merely
    # returning a rollback flag or copying its earlier accuracy into the result.
    selected=baseline if result['accepted_mask']==0 else fault
    with torch.inference_mode():
        restored_chunks=[]
        for start in range(0,len(ids),args.quality_batch):
            restored_chunks.append(selected(images[start:start+args.quality_batch].to(d)).cpu())
        restored_logits=torch.cat(restored_chunks);restored=restored_logits.argmax(dim=1)
        rollback_correct=int((restored==labels).sum())
        rollback_timing=timed(lambda:selected(inp),d,10,30)
    initial_eval_baseline=payload['eval'][0]['coalitions'][0]
    rollback_accuracy=rollback_correct/len(ids)
    restored_retention=rollback_accuracy/initial_eval_baseline['quality_score']
    logits_difference=restored_logits-baseline_eval_logits
    rollback_logits_equivalent=torch.allclose(restored_logits,baseline_eval_logits,atol=1e-4,rtol=1e-4)
    executed_rollback={'executed_after_rejection':True,'model_selected':'original_pretrained_baseline',
        'provenance':'physical_measurement','sample_ids':sample_ids,'labels':labels.tolist(),
        'predictions':restored.tolist(),'correct_count':rollback_correct,'quality_score':rollback_accuracy,
        'retention_vs_initial_eval_baseline':restored_retention,
        'predictions_match_initial_baseline':restored.tolist()==initial_eval_baseline['predictions'],
        'logits_sha256':sha(restored_logits.contiguous().numpy().tobytes()),
        'logits_equivalent_to_initial_baseline':bool(rollback_logits_equivalent),
        'logits_allclose_atol':1e-4,'logits_allclose_rtol':1e-4,
        'logits_max_abs_error':float(logits_difference.abs().max()),
        'logits_relative_l2_error':float(logits_difference.norm()/baseline_eval_logits.norm().clamp_min(1e-12)),
        'weights_state_sha256':state_hash(selected),**rollback_timing}
    out={'kind':'physical_labeled_fault_injection','utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Intentional corrupt candidate, excluded from main optimization action set and reported benefit.',
        'runtime':{'torch':torch.__version__,'platform':platform.platform(),**hardware(d)},
        'weights_state_sha256':state_hash(baseline),'input_assets_manifest_sha256':sha((args.assets/'manifest.json').read_bytes()),
        'model_configuration':{'inplace_disabled_modules':disabled_inplace},
        'source_sha256':sha(Path(__file__).read_bytes()),
        'gate_source_sha256':{name:sha((Path(__file__).resolve().parents[1]/'methods'/name).read_bytes())
                              for name in ['measured_planner.py','copa_v2.py']},
        'task':payload,'actual_gate_result':result,'executed_rollback':executed_rollback,
        'stress_check_passed':result['accepted'] is False and result['rollback_to_baseline'] is True
              and result['reason']=='quality_below_threshold' and restored_retention>=.99 and bool(rollback_logits_equivalent)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({'gate':result,'executed_rollback':{k:v for k,v in executed_rollback.items()
          if k not in ['sample_ids','labels','predictions','latency_samples_ms']}},indent=2))
    if not out['stress_check_passed']:raise RuntimeError('Physical stress/rollback check failed; raw evidence was saved')

if __name__=='__main__':main()
