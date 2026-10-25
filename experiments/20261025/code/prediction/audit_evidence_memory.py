"""Numerical support/configuration audit and sampled anchor trace export."""
import argparse
import collections
import gzip
import json
import math
from pathlib import Path

import numpy as np

import evaluate_evidence_memory as e


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result-dir',required=True,type=Path)
    ap.add_argument('--reference-summary',required=True,type=Path);ap.add_argument('--profiles-dir',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    e.require(not args.output.exists(),'do not overwrite memory audit')
    d=json.loads((args.result_dir/'summary.json').read_text());freeze=json.loads((args.result_dir/'method-freeze.json').read_text())
    baseline=json.loads(args.reference_summary.read_text());rows,manifest,_=e.load_records(baseline,args.profiles_dir)
    e.require(d['completed_utc'] and len(d['results'])==66 and len(d['aggregate_results'])==12,'incomplete memory matrix')
    e.require(freeze['code_sha256']==d['code_sha256']==e.sha(Path(e.__file__)),'enhancement source changed')
    e.require(d['reference_summary_sha256']==e.sha(args.reference_summary) and freeze['source_manifest']==manifest,'main evidence source differs')
    e.require(freeze['frozen_utc']<=d['created_utc']<d['completed_utc'],'method freeze did not precede run scoring')
    lookup={r['record_id']:i for i,r in enumerate(rows)};predictions=collections.defaultdict(dict)
    with gzip.open(args.result_dir/'predictions.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);key=(row['feature_mode'],row['method'],row['split_id']);i=lookup[row['record_id']]
            e.require(i not in predictions[key],'duplicate memory prediction')
            e.require(math.isclose(row['true_latency_ms'],rows[i]['latency_ms'],rel_tol=1e-10,abs_tol=1e-12),'physical truth mismatch')
            predictions[key][i]=row['prediction_ms']
    entries={(r['feature_mode'],r['method'],r['split_id']):r for r in d['results']};e.require(set(entries)==set(predictions),'prediction/summary entries differ')
    grid_checks=0;metric_checks=0;traces=[];grouped=collections.defaultdict(list)
    splits=[r for r in baseline['results'] if r['method']=='rgat' and r['feature_mode']=='strict']
    for split in splits:
        train,val,test=e.partition(rows,split);device_protocol=split['protocol']!='model_holdout'
        # Fixed lexical-first example per observed family. No error or outcome
        # chooses which queries have their anchor evidence exported.
        sample={}
        for i in sorted(test,key=lambda j:rows[j]['record_id']):sample.setdefault(rows[i]['family'],int(i))
        query=np.array(list(sample.values()))
        for method in e.METHODS:
            cache=None if method=='semantic_ridge' else e.neighbors(rows,train,query,method,device_protocol)
            for mode in ['strict','probe']:
                key=(mode,method,split['split_id']);entry=entries[key];stored=predictions[key]
                e.require(set(stored)==set(test.tolist()),'test query support differs from frozen baseline')
                p=np.array([stored[int(i)] for i in test]);actual=e.metrics(rows,test,p)
                for field in ['records','micro_mre_percent','family_macro_mre_percent','model_macro_mre_percent','attention_n']:
                    a,b=actual[field],entry['test_metrics'][field]
                    e.require(math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8),'summary arithmetic mismatch')
                metric_checks+=1
                trials=entry['validation_grid_results']
                if method=='semantic_ridge':
                    e.require(sorted(t['lambda'] for t in trials)==e.RIDGE_LAMBDAS,'ridge grid changed')
                    selected=min(trials,key=lambda x:(x['validation_micro_mre'],x['lambda']))
                else:
                    gammas=e.GAMMAS if mode=='probe' and device_protocol else [0.]
                    expected={(k,t,g) for k in e.KS for t in e.TEMPERATURES for g in gammas}
                    e.require({(q['k'],q['temperature'],q['probe_exponent']) for q in trials}==expected,'retrieval grid changed')
                    selected=min(trials,key=lambda x:(x['validation_micro_mre'],x['k'],x['temperature'],x['probe_exponent']))
                e.require(selected==entry['selected_validation_configuration'],'parameters were not the validation argmin')
                grid_checks+=1;grouped[(mode,method,split['protocol'])].extend((int(i),float(v)) for i,v in zip(test,p))
                if cache is None:continue
                for item in cache:
                    qi=item['query_index'];anchors=item['anchor_indices'][:selected['k']];distance=item['distances'][:selected['k']]
                    e.require(set(anchors).issubset(set(train)),'anchor outside training data')
                    if device_protocol:e.require(all(rows[a]['device_id']!=rows[qi]['device_id'] for a in anchors),'target device label leaked into memory')
                    else:e.require(all(rows[a]['model_id']!=rows[qi]['model_id'] for a in anchors),'query model label leaked into memory')
                    weight=np.exp(-(distance-distance.min())/selected['temperature']);weight/=weight.sum()
                    ratios=np.exp(item['probe_log_ratios'][:selected['k']]);gamma=selected['probe_exponent']
                    estimate=float(np.exp(np.dot(weight,np.log([rows[a]['latency_ms'] for a in anchors])+gamma*np.log(ratios))))
                    e.require(math.isclose(estimate,stored[qi],rel_tol=1e-9,abs_tol=1e-10),'anchor trace does not reproduce frozen prediction')
                    traces.append({'feature_mode':mode,'method':method,'split_id':split['split_id'],'query_record_id':rows[qi]['record_id'],
                        'family':rows[qi]['family'],'tier':item['tier'],'selected_configuration':selected,
                        'prediction_ms':estimate,'query_latency_used_to_compute_prediction':False,
                        'training_anchors':[{'record_id':rows[a]['record_id'],'physical_latency_ms':rows[a]['latency_ms'],
                            'kernel_weight':float(w),'target_source_probe_ratio':float(r)} for a,w,r in zip(anchors,weight,ratios)]})
    for agg in d['aggregate_results']:
        key=(agg['feature_mode'],agg['method'],agg['protocol']);items=grouped[key]
        ids=np.array([i for i,_ in items]);e.require(len(ids)==13905 and len(set(ids))==13905,'aggregate record support changed')
        actual=e.metrics(rows,ids,np.array([p for _,p in items]))
        for field in ['micro_mre_percent','family_macro_mre_percent','model_macro_mre_percent','attention_mre_percent']:
            e.require(math.isclose(actual[field],agg['metrics'][field],rel_tol=1e-9,abs_tol=1e-8),'aggregate arithmetic mismatch')
        metric_checks+=1
    out={'kind':'independent_evidence_memory_support_grid_and_anchor_audit','created_utc':e.utc(),'code_sha256':e.sha(__file__),
        'enhancement_code_sha256':d['code_sha256'],'summary_sha256':e.sha(args.result_dir/'summary.json'),
        'method_freeze_sha256':e.sha(args.result_dir/'method-freeze.json'),'all_checks_passed':True,
        'full_support_entries_checked':66,'metric_objects_checked':metric_checks,'validation_argmin_and_fixed_grids_checked':grid_checks,
        'source_raw_and_metadata_hashes_verified':True,'anchor_trace_examples':len(traces),'traces':traces,
        'trace_selection':'fixed lexical first test query per observed family and fold; no errors used',
        'scope':'No new training, physical timing or parameter selection. All66 validation argmins are verified; sampled explicit training-anchor traces reproduce the retained test predictions.'}
    args.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k] for k in ['all_checks_passed','full_support_entries_checked','metric_objects_checked','validation_argmin_and_fixed_grids_checked','anchor_trace_examples']}))


if __name__=='__main__':main()
