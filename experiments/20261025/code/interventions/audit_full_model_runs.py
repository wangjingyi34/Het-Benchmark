"""Independent arithmetic, identity and partition audit of actual full-model runs.

This audit verifies saved record consistency, not physical authenticity from a
provenance flag alone. Root must also retain worker execution/transfer receipts.
"""
import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def exact_permutation(values,n=4):
    phi=[0.0]*n
    for order in itertools.permutations(range(n)):
        mask=0
        for i in order:
            nxt=mask|(1<<i);phi[i]+=values[mask]-values[nxt];mask=nxt
    return [v/math.factorial(n) for v in phi]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',type=Path,required=True)
    ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--min-repeats',type=int,default=30);args=ap.parse_args()
    manifest=json.loads((args.assets/'manifest.json').read_text())
    samples={s['sample_id']:s for s in manifest['samples']}
    manifest_sha=sha(args.assets/'manifest.json');groups=defaultdict(list);raws=[]
    counts={'blocks':0,'coalitions':0,'coalition_timed_calls':0,'quality_predictions':0,
            'equivalence_passed_records':0,'non_ok_records':0,'quality_rejections':0}
    for p in sorted(args.raw_dir.rglob('*.json')):
        d=json.loads(p.read_text())
        if d.get('schema_version')!='het.fullmodel.run.v2':continue
        assert d['provenance']=='physical_measurement'
        assert d['dataset_manifest_sha256']==manifest_sha
        run=d['run_index'];phase='train' if run<=3 else 'eval'
        partition='calibration' if run<=3 else 'evaluation'
        assert d['phase']==phase and d['quality_partition']==partition
        assert d['started_utc']<=d['proxy_profile_frozen_at']<=d['completed_utc']
        batch=d['input_shape'][0];assert d['input_shape']==[batch,3,224,224] and batch in [1,4]
        expected_input=manifest['timing_inputs'][partition+f'_b{batch}']['tensor_sha256']
        assert d['input_tensor_sha256']==expected_input
        assert d['model_configuration']['inplace_disabled_modules']
        rows={c['mask']:c for c in d['coalitions']};assert set(rows)==set(range(16))
        assert rows[0]['quality_score']==d['baseline_quality']>0
        expected_ids={i for i,s in samples.items() if s['partition']==partition}
        assert len(expected_ids)==100
        values={};perrun=[]
        for mask,r in rows.items():
            counts['coalitions']+=1
            if r['status']!='ok':counts['non_ok_records']+=1
            if r.get('quality_evaluated') is not True:
                assert r['status']!='ok'
                continue
            assert r['quality_evaluated'] is True
            assert set(r['sample_ids'])==expected_ids and len(r['sample_ids'])==100
            assert len(r['predictions'])==len(r['labels'])==100
            expected_labels=[samples[i]['imagenet_index'] for i in r['sample_ids']]
            assert r['labels']==expected_labels
            assert all(isinstance(v,int) and 0<=v<1000 for v in r['predictions'])
            correctness=[a==b for a,b in zip(r['predictions'],r['labels'])]
            assert correctness==r['correct'] and sum(correctness)==r['correct_count']
            assert abs(sum(correctness)/100-r['quality_score'])<1e-12
            retained=r['quality_score']/rows[0]['quality_score']
            assert abs(retained-r['retention_ratio'])<1e-12
            assert r['quality_threshold_passed']==(retained>=.99)
            if retained<.99:counts['quality_rejections']+=1
            counts['quality_predictions']+=100
            if r['equivalence']['passed']:counts['equivalence_passed_records']+=1
            if r['status']!='ok':continue
            raw=r['latency_samples_ms']
            assert len(raw)>=args.min_repeats and all(math.isfinite(v) and v>0 for v in raw)
            assert abs(statistics.mean(raw)-r['mean_ms'])<1e-9
            assert abs(statistics.median(raw)-r['median_ms'])<1e-9
            assert abs(statistics.stdev(raw)-r['sample_sd_ms'])<1e-9
            counts['coalition_timed_calls']+=len(raw);values[mask]=statistics.mean(raw)
            perrun.append({'mask':mask,'accuracy':r['quality_score'],'retention':retained,
                          'prediction_disagreements_vs_mask0':sum(a!=b for a,b in zip(r['predictions'],rows[0]['predictions']))})
        record={'file':str(p),'sha256':sha(p),'run_id':d['run_id'],'model_id':d['model_id'],
                'device_id':d['device_id'],'phase':phase,'quality_partition':partition,
                'quality':perrun,'source_sha256':d['protocol']['source_sha256'],
                'same_dataset_asset':True,'same_input_asset':True,'raw_statistics_consistent':True}
        if len(values)==16:
            record['independent_24_permutation_physical_shapley_ms']=dict(zip(d['action_ids'],exact_permutation(values)))
            record['all_substitutions_benefit_percent']=100*(values[0]-values[15])/values[0]
        counts['blocks']+=1;raws.append(record);groups[(d['task_id'],d['device_id'])].append(d)
    if not raws:raise ValueError('No formal full-model records found')
    group_rows=[];weight_hashes=defaultdict(set)
    for (task,device),ds in groups.items():
        run_indices=[d['run_index'] for d in ds]
        assert len(run_indices)==len(set(run_indices))
        assert len({d['weights_state_sha256'] for d in ds})==1
        assert len({d['protocol']['source_sha256'] for d in ds})==1
        train={s for d in ds if d['phase']=='train' for s in next(c for c in d['coalitions'] if c['mask']==0).get('sample_ids',[])}
        test={s for d in ds if d['phase']=='eval' for s in next(c for c in d['coalitions'] if c['mask']==0).get('sample_ids',[])}
        assert not train&test
        group_rows.append({'task':task,'device':device,'run_indices':sorted(run_indices),
            'train_runs':sum(d['phase']=='train' for d in ds),'eval_runs':sum(d['phase']=='eval' for d in ds),
            'complete_eight_run_protocol':set(run_indices)==set(range(1,9)),
            'unique_calibration_quality_samples':len(train),'unique_eval_quality_samples':len(test),
            'quality_partitions_disjoint':True})
        weight_hashes[ds[0]['model_id']].update(d['weights_state_sha256'] for d in ds)
    assert all(len(v)==1 for v in weight_hashes.values())
    out={'kind':'independent_full_model_record_consistency_audit','counts':counts,'groups':group_rows,
         'weights_identical_per_model_across_observed_devices':True,
         'unique_sample_definition':'Repeated evaluation of the same 100 IDs is not new independent quality samples.',
         'raw_records':raws,'audit_source_sha256':sha(Path(__file__)),
         'limitations':'Saved-record arithmetic/identity audit; device/run authenticity also requires worker receipts. Recorded logit equivalence cannot be recomputed from hashes alone.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({'output':str(args.output),'counts':counts,'groups':group_rows},indent=2))

if __name__=='__main__':main()
