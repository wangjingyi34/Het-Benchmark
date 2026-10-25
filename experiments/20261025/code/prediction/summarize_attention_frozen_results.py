"""Reproduce and audit completed MHA OOD scores; never refit predictors."""
import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(c,m):
    if not c:raise ValueError(m)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v2',required=True,type=Path)
    ap.add_argument('--result-dir',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();require(not args.output.exists(),'do not overwrite OOD summary')
    root=args.v2;freeze_path=root/'prediction/attention-freeze-main-strict.json';freeze=json.loads(freeze_path.read_text())
    expected={(r['method'],r['seed']):r['sha256'] for r in freeze['checkpoints']}
    require(len(expected)==15,'need the exact15 frozen main strict checkpoints')
    groups=collections.defaultdict(list);stackgroups=collections.defaultdict(list);artifacts=[];reference_hashes=None
    for stack in ['m5cpu','m5mps','a100mig','cloudcpu','npub3']:
        profile=root/'evidence'/('npu-final-backup/attention-npub3.json' if stack=='npub3' else 'attention-'+stack+'.json')
        result_path=args.result_dir/(stack+'.json');raw=json.loads(profile.read_text());result=json.loads(result_path.read_text())
        require(result['profile_sha256']==sha(profile),'raw/result byte binding differs')
        require(result['prediction_freeze_sha256']==raw['provenance']['prediction_freeze_sha256']==sha(freeze_path),'predictor freeze byte binding differs')
        require(raw['provenance']['mha_native_fastpath_enabled'] is False,'MHA native execution path changed')
        require(len(raw['graphs'])==12 and len(result['rows'])==180 and not result['failures'] and not raw['failures'],'physical or prediction outcomes incomplete')
        hashes={g['shape_id']:g['input_weights_sha256'] for g in raw['graphs']}
        if reference_hashes is None:reference_hashes=hashes
        require(hashes==reference_hashes,'cross-stack MHA input/weight hashes differ')
        truth={}
        for graph in raw['graphs']:
            require(len(graph['records'])==1 and graph['records'][0]['status']=='measured','MHA physical query failed')
            record=graph['records'][0];samples=record['latency_samples_ms']
            require(len(samples)==10 and all(math.isfinite(x) and x>0 for x in samples),'invalid MHA timing blocks')
            require(record['warmup']==5 and record['inner_repeats']==4,'MHA timing budget changed')
            truth[graph['shape_id']]=statistics.mean(samples)
        seen=set()
        for row in result['rows']:
            key=(row['method'],row['seed']);query=(row['case_index'],*key)
            require(key in expected and row['checkpoint_sha256']==expected[key],'wrong checkpoint used')
            require(query not in seen,'duplicate MHA prediction');seen.add(query)
            require(row['inference_latency_labels_used'] is False,'inference used target latency')
            require(math.isclose(row['physical_latency_ms'],truth[row['case_index']],rel_tol=1e-10,abs_tol=1e-10),'MHA label mismatch')
            error=abs(row['prediction_ms']-truth[row['case_index']])/max(truth[row['case_index']],1e-6)*100
            require(math.isclose(error,row['relative_error_percent'],rel_tol=1e-10,abs_tol=1e-8),'MHA relative error mismatch')
            groups[key].append(row);stackgroups[(stack,*key)].append(row)
        artifacts.append({'stack':stack,'profile_sha256':sha(profile),'result_sha256':sha(result_path),'physical_cases':12,'prediction_rows':180})
    results=[];perstack=[]
    for method in ['mlp','gcn','rgat']:
        seeds=[]
        for seed in [11,22,33,44,55]:
            rows=groups[(method,seed)];require(len(rows)==60,'pooled MHA support differs')
            absolute=[abs(r['prediction_ms']-r['physical_latency_ms']) for r in rows]
            seeds.append({'seed':seed,'physical_cases':60,'micro_MRE_percent':statistics.mean(r['relative_error_percent'] for r in rows),
                          'MAE_ms':statistics.mean(absolute),'WAPE_percent':sum(absolute)/sum(r['physical_latency_ms'] for r in rows)*100})
        results.append({'method':method,'seed_metrics':seeds,'micro_MRE_seed_mean':statistics.mean(r['micro_MRE_percent'] for r in seeds),
            'micro_MRE_seed_sample_SD':statistics.stdev(r['micro_MRE_percent'] for r in seeds),
            'MAE_ms_seed_mean':statistics.mean(r['MAE_ms'] for r in seeds),'WAPE_percent_seed_mean':statistics.mean(r['WAPE_percent'] for r in seeds)})
        for stack in ['m5cpu','m5mps','a100mig','cloudcpu','npub3']:
            values=[statistics.mean(r['relative_error_percent'] for r in stackgroups[(stack,method,s)]) for s in [11,22,33,44,55]]
            perstack.append({'stack':stack,'method':method,'physical_cases':12,'micro_MRE_seed_mean':statistics.mean(values),'micro_MRE_seed_sample_SD':statistics.stdev(values)})
    out={'kind':'independently_verified_complete_frozen_MHA_OOD_scores','summarizer_code_sha256':sha(__file__),
         'source_challenge_code_sha256':sha(root/'prediction/attention_ood_challenge.py'),'freeze_sha256':sha(freeze_path),
         'physical_cases':60,'prediction_rows':900,'prediction_failures':0,'checkpoint_count':15,
         'all12_input_weight_hashes_match_across5stacks':True,'all_checks_passed':True,
         'results':results,'per_stack_results':perstack,'artifact_manifest':artifacts,
         'scope':'Complete MHA module (QKV/output projections included), larger shapes and short FX graph OOD. Separate from primitive SDPA main labels. No retraining, parameter tuning or test-query selection. Seed variation is not independent hardware replication.'}
    args.output.write_text(json.dumps(out,indent=2))
    with args.output.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(perstack[0]));writer.writeheader();writer.writerows(perstack)
    print(json.dumps({'all_checks_passed':True,'physical_cases':60,'predictions':900,'output':str(args.output)}))


if __name__=='__main__':main()
