"""Fixed-parameter17-query calibration extension of retained evidence learners.

Original three-probe results and chosen k/temperature/gamma/lambda are immutable.
The new information is14 independently measured standard family workloads.
Failures fall back explicitly to the original frozen three-probe mapping.
"""
import argparse
import collections
import copy
import gzip
import json
import math
import statistics
from pathlib import Path

import numpy as np

import evaluate_evidence_memory as e


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference-summary',required=True,type=Path)
    ap.add_argument('--evidence-summary',required=True,type=Path);ap.add_argument('--profiles-dir',required=True,type=Path)
    ap.add_argument('--family-manifest',required=True,type=Path);ap.add_argument('--family-probe-files',nargs='+',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    e.require(not args.output.exists(),'do not overwrite calibration extension')
    baseline=json.loads(args.reference_summary.read_text());old=json.loads(args.evidence_summary.read_text())
    manifest=json.loads(args.family_manifest.read_text());e.require(manifest['kind']=='frozen_family14_calibration_manifest' and manifest['probe_count']==14,'wrong standard family-probe manifest')
    rows,sources,hardware=e.load_records(baseline,args.profiles_dir)
    e.require(old['reference_summary_sha256']==e.sha(args.reference_summary) and old.get('completed_utc'),'evidence reference incomplete/different')
    e.require(old['code_sha256']==e.sha(e.__file__),'original evidence learner changed')
    by_family={x['specification']['family']:x for x in manifest['probes']}
    observed_families={r['family'] for r in rows}
    e.require(set(by_family)==observed_families and len(observed_families)==14,'auxiliary family set differs from frozen observed14families')
    calibrated={};provenance=[];failures=[];cost=[]
    for path in args.family_probe_files:
        data=json.loads(path.read_text());e.require(data['kind']=='physical_family14_auxiliary_calibration' and data.get('completed_utc'),'incomplete family-probe collection')
        e.require(data['manifest_sha256']==e.sha(args.family_manifest) and data['collector_sha256']==manifest['collector_sha256'],'standard calibration manifest/code mismatch')
        device=data['hardware']['device_id'];e.require(device in hardware and device not in calibrated,'duplicate/unknown calibration runtime stack')
        e.require(data['hardware_source_profile_sha256']==next(x['raw_sha256'] for x in sources if x['device_id']==device),'calibration hardware descriptor source differs from main physical profile')
        e.require(len(data['records'])==14,'missing explicit family-probe outcomes')
        values={};seen=set();incremental_timed=0.
        for record in data['records']:
            family=record['family'];e.require(family in by_family and family not in seen,'unexpected/duplicate family outcome');seen.add(family)
            if record['status']=='measured':
                e.require(record['input_weights_sha256']==by_family[family]['input_weights_sha256'],'cross-stack calibration input/weights differ')
                samples=record['latency_samples_ms'];mean=statistics.mean(samples)
                e.require(len(samples)==20 and all(math.isfinite(x) and x>0 for x in samples),'invalid auxiliary physical timing samples')
                e.require(math.isclose(mean,record['latency_mean_ms'],rel_tol=1e-10,abs_tol=1e-12),'auxiliary mean differs from raw timings')
                values[family]=mean;incremental_timed+=sum(samples)*8
            else:
                failures.append({'device_id':device,'family':family,'status':record['status'],'error':record.get('error')})
                values[family]=hardware[device]['probes'][e.PROBE_INDEX[family]]['latency_mean_ms']
        calibrated[device]=values
        original_timed=sum(sum(p['latency_samples_ms'])*p.get('inner_repeats',8) for p in hardware[device]['probes'])
        cost.append({'device_id':device,'original_query_types':3,'new_query_types':14,'total_query_types':17,
            'successful_new_probes':data['successful_probes'],'original_three_timed_blocks_ms':original_timed,
            'new_fourteen_timed_blocks_ms':incremental_timed,'total_recorded_timed_blocks_ms':original_timed+incremental_timed,
            'new_fourteen_wall_seconds_including_runtime_build_warmup_output_check':data['wall_seconds_including_runtime_validation'],
            'all17_maximum_timed_call_budget':17*20*8,'all17_warmup_call_budget':17*10,'new14_untimed_output_checks':14,
            'scope':'Budget counts standard workload types; each has20 synchronized blocks of8calls. Original3 warmup/setup wall is not retroactively invented.'})
        provenance.append({'path':str(path.resolve()),'sha256':e.sha(path),'device_id':device})
    e.require(set(calibrated)==set(hardware),'family17 comparison requires all5matched physical runtime stacks')
    # Never divide a new-family workload on one stack by an old proxy workload
    # on another. A failure causes a label-free, whole-family fallback on all
    # five stacks, including the ridge information-control descriptor column.
    fallback_families=sorted({r['family'] for r in failures})
    for family in fallback_families:
        for device in calibrated:
            calibrated[device][family]=hardware[device]['probes'][e.PROBE_INDEX[family]]['latency_mean_ms']
    args.output.mkdir(parents=True)
    freeze={'kind':'fixed_parameter_family17_extension_freeze','created_utc':e.utc(),'code_sha256':e.sha(__file__),
        'original_evidence_summary_sha256':e.sha(args.evidence_summary),'family_manifest_sha256':e.sha(args.family_manifest),
        'calibration_sources':provenance,'new_main_query_labels':0,'test_parameter_reselection':False,
        'parameter_policy':'Preserve each original3-probe validation-selected k/temperature/gamma or ridge lambda exactly.',
        'information_control':'Both retrieval baselines use the same family14 ratios; semantic ridge receives original3+14log-probe descriptors and refits coefficients only on the unchanged training fold with its fixed original lambda.',
        'failure_rule':'If any stack fails a family, that entire family uses the same original3 mapping on every stack, preventing mixed-workload ratios.',
        'all_stack_original3_fallback_families':fallback_families}
    (args.output/'extension-freeze.json').write_text(json.dumps(freeze,indent=2))
    # Augment only the ridge information control. Original rows and all original
    #3-probe experiments remain byte-identical and separately archived.
    ridge_rows=copy.deepcopy(rows);family_order=[r['specification']['family'] for r in manifest['probes']]
    for r in ridge_rows:r['probe_features']=r['probe_features']+[math.log(calibrated[r['device_id']][f]) for f in family_order]
    old_entries={(r['feature_mode'],r['method'],r['split_id']):r for r in old['results']}
    result={'kind':'fixed_parameter_independent_family17_calibration_evaluation','created_utc':e.utc(),
        'code_sha256':e.sha(__file__),'extension_freeze_sha256':e.sha(args.output/'extension-freeze.json'),
        'reference_summary_sha256':e.sha(args.reference_summary),'original_evidence_summary_sha256':e.sha(args.evidence_summary),
        'calibration_cost':cost,'calibration_failures':failures,'all_stack_original3_fallback_families':fallback_families,
        'all_stack_fallback_device_family_cells':len(fallback_families)*len(hardware),'results':[],
        'all_original_main_queries_retained':True,'target_model_labels_used_for_calibration':False,'parameter_or_grid_reselection':False}
    aggregate=collections.defaultdict(list);stream=gzip.open(args.output/'predictions.jsonl.gz','wt')
    splits=[r for r in baseline['results'] if r['method']=='rgat' and r['feature_mode']=='strict']
    for split in splits:
        train,val,test=e.partition(rows,split);device_protocol=split['protocol']!='model_holdout'
        for method in e.METHODS:
            old_entry=old_entries[('probe',method,split['split_id'])];params=old_entry['selected_validation_configuration']
            if method=='semantic_ridge':pred=e.ridge_predict(ridge_rows,train,test,'probe',params['lambda'])
            else:
                cache=e.neighbors(rows,train,test,method,device_protocol)
                for item in cache:
                    q=rows[item['query_index']];family=q['family']
                    item['probe_log_ratios']=np.log([calibrated[q['device_id']][family]/calibrated[rows[i]['device_id']][family] for i in item['anchor_indices']])
                pred=e.memory_predict(rows,cache,params['k'],params['temperature'],params['probe_exponent'])
            public={k:v for k,v in split.items() if k in ['split_id','protocol','train_model_ids','validation_model_ids','test_model_ids','heldout_device_id']}
            name=method+'_family17';entry={**public,'feature_mode':'family17','method':name,
                'source_three_probe_validation_selected_configuration':params,'configuration_reselected':False,
                'metrics':e.metrics(rows,test,pred)};result['results'].append(entry)
            e.require(entry['metrics']['evaluation_record_ids_sha256']==old_entry['test_metrics']['evaluation_record_ids_sha256'],'family17 comparison support changed')
            for i,p in zip(test,pred):
                row={k:rows[i][k] for k in ['record_id','model_id','device_id','family']}
                row.update({'method':name,'feature_mode':'family17','split_id':split['split_id'],'protocol':split['protocol'],
                            'true_latency_ms':rows[i]['latency_ms'],'prediction_ms':float(p)})
                stream.write(json.dumps(row,separators=(',',':'))+'\n');aggregate[(name,split['protocol'])].append((int(i),float(p)))
            stream.flush();(args.output/'summary.partial.json').write_text(json.dumps(result,indent=2))
            print(json.dumps({'split':split['split_id'],'method':name,'micro_MRE':entry['metrics']['micro_mre_percent']}),flush=True)
    stream.close();result['aggregate_results']=[]
    for (method,protocol),items in sorted(aggregate.items()):
        indices=np.array([i for i,_ in items]);e.require(len(indices)==13905 and len(set(indices))==13905,'family17 full support changed')
        result['aggregate_results'].append({'feature_mode':'family17','method':method,'protocol':protocol,
                                            'metrics':e.metrics(rows,indices,np.array([p for _,p in items]))})
    result['completed_utc']=e.utc();(args.output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'complete':True,'entries':len(result['results']),'aggregates':len(result['aggregate_results'])}))


if __name__=='__main__':main()
