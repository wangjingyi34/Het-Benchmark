"""Independent summary arithmetic, support, split and provenance audit.

Does not train or profile. Checkpoint bytes are not assumed to be present merely
because the summary records their digests; publication must retain those files.
"""
import argparse
import collections
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODELS = ['resnet18','resnet50','mobilenet_v2','densenet121','efficientnet_b0','squeezenet1_1',
          'text_encoder_ref','text_cnn_ref','audio_conv_ref','audio_dilated_ref','causal_lm_ref','dual_encoder_ref']
METHODS = ['ridge','mlp','gcn','rgat','drop_contains','drop_has_type','drop_dependency','drop_similarity','drop_supports','drop_performance_context']
SEEDS = [11,22,33,44,55]
EXPECTED_EVALUATOR = 'bad26be58e40a0bf6ff939024f15e646a72ec3e1b8bc3fef54dc67745f8cce0f'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(condition, message):
    if not condition:raise ValueError(message)
def close(a, b):return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)
def avg(xs):return statistics.mean(xs)


def bootstrap_model(values):
    values = list(values)
    rng = np.random.default_rng(9153)
    draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def bootstrap_seeds(values):
    rng = np.random.default_rng(61025)
    draws = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def support_metrics(records):
    counts = {k: collections.Counter(r[k] for r in records) for k in ['model_id','family','device_id']}
    return {'records': len(records), 'counts': counts,
            'record_hash': hashlib.sha256('\n'.join(sorted(r['record_id'] for r in records)).encode()).hexdigest()}


def validate_metric(metric, support, totals):
    require(metric['records'] == support['records'], 'metric record count differs from physical support')
    require(metric['evaluation_record_ids_sha256'] == support['record_hash'], 'metric sorted record IDs differ from physical support')
    require(metric['per_family_n'] == dict(support['counts']['family']), 'family support differs')
    require(metric['per_device_n'] == dict(support['counts']['device_id']), 'hardware support differs')
    require(set(metric['per_model_mre_percent']) == set(support['counts']['model_id']), 'model support differs')
    for group, field in [('family','per_family_mre_percent'), ('device_id','per_device_mre_percent'), ('model_id','per_model_mre_percent')]:
        values = metric[field]
        require(set(values) == set(support['counts'][group]), 'metric group keys differ')
        require(all(math.isfinite(v) and v >= 0 for v in values.values()), 'nonfinite/negative error metric')
        weighted = sum(values[k] * n for k, n in support['counts'][group].items()) / support['records']
        require(close(weighted, metric['micro_mre_percent']), 'micro error does not equal group-weighted errors')
    require(close(avg(metric['per_family_mre_percent'].values()), metric['family_macro_mre_percent']), 'family macro arithmetic mismatch')
    require(close(avg(metric['per_model_mre_percent'].values()), metric['model_macro_mre_percent']), 'model macro arithmetic mismatch')
    attention_n = support['counts']['family'].get('Attention', 0)
    require(metric['attention_n'] == attention_n, 'Attention support mismatch')
    require((metric['attention_mre_percent'] is None) == (attention_n == 0), 'missing Attention must be null, not zero')
    if attention_n:require(close(metric['attention_mre_percent'], metric['per_family_mre_percent']['Attention']), 'Attention error mismatch')
    expected_ci = bootstrap_model(metric['per_model_mre_percent'].values())
    require(all(close(x,y) for x,y in zip(expected_ci,metric['model_macro_bootstrap95_percent'])), 'model-cluster bootstrap CI mismatch')
    totals['metric_objects_checked'] += 1
    totals['model_bootstrap_intervals_recomputed'] += 1


def combined_metrics(parts):
    count = sum(s['records'] for _,s in parts)
    result = {'records': count}
    for group, field in [('family','per_family_mre_percent'), ('model_id','per_model_mre_percent'), ('device_id','per_device_mre_percent')]:
        sums = collections.defaultdict(float);ns = collections.Counter()
        for metric, support in parts:
            for key,n in support['counts'][group].items():
                sums[key] += metric[field][key] * n;ns[key] += n
        result[field] = {k:sums[k]/ns[k] for k in sums}
        result[group+'_counts'] = dict(ns)
    result['micro_mre_percent'] = sum(m['micro_mre_percent']*s['records'] for m,s in parts)/count
    result['family_macro_mre_percent'] = avg(result['per_family_mre_percent'].values())
    result['model_macro_mre_percent'] = avg(result['per_model_mre_percent'].values())
    result['attention_mre_percent'] = result['per_family_mre_percent'].get('Attention')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',required=True,type=Path)
    ap.add_argument('--profiles-dir',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    require(not args.output.exists(), 'do not overwrite audit')
    summary_bytes=args.summary.read_bytes();d=json.loads(summary_bytes)
    require(d.get('completed_utc') and d['code_sha256']==EXPECTED_EVALUATOR,'incomplete/wrong evaluator summary')
    require(len(d['results'])==220 and sum(len(r['seed_fits']) for r in d['results'])==1100,'incomplete1100-fit matrix')
    require(d['seeds']==SEEDS and d['epochs_max']==80 and d['patience_validation_only']==15,'training budget changed')
    require(d['label_feature_separation'] is True and d['metadata_enrichment_required'] is True,'label/metadata contract changed')
    records=[];sources=[];hardware=set()
    for manifest in d['source_manifest']:
        path=args.profiles_dir/Path(manifest['path']).name;sidepath=args.profiles_dir/Path(manifest['metadata_sidecar']).name
        require(sha(path)==manifest['sha256'] and sha(sidepath)==manifest['metadata_sidecar_sha256'],'physical/sidecar source hash mismatch')
        raw=json.loads(path.read_text());side=json.loads(sidepath.read_text())
        require(side['raw_bundle_sha256']==manifest['sha256'] and not side['timings_changed_or_added'] and not side['training_scores_seen'],'sidecar not label-free hash-bound metadata')
        device=raw['hardware']['device_id'];hardware.add(device)
        require(device==manifest['hardware'],'manifest runtime stack differs')
        for graph in raw['graphs']:
            nodes={n['node_id']:n for n in graph['nodes']}
            for r in graph['records']:
                if r['status']=='measured':records.append({'record_id':r['record_id'],'model_id':graph['model_id'],'device_id':device,'family':nodes[r['op_instance_id']]['family']})
        sources.append({'raw_file':str(path),'raw_sha256':sha(path),'sidecar_sha256':sha(sidepath),'runtime_stack_id':device})
    require(len(records)==13905 and len(hardware)==5,'physical source size differs')
    all_support=support_metrics(records);entry_keys=set();split_support={};grouped=collections.defaultdict(list)
    totals=collections.Counter();checkpoint_hashes=[];parameter_counts={}
    for r in d['results']:
        key=(r['feature_mode'],r['split_id'],r['method']);require(key not in entry_keys,'duplicate method/split/mode entry');entry_keys.add(key)
        require(r['method'] in METHODS and r['feature_mode'] in ['strict','probe'],'unknown method/features')
        if r['protocol']=='model_holdout':
            fold=int(r['split_id'].replace('model-fold',''));require(0<=fold<6,'unexpected model fold')
            test=MODELS[fold*2:fold*2+2];nxt=(fold+1)%6;val=MODELS[nxt*2:nxt*2+2];train=[m for m in MODELS if m not in test+val]
            tr=[x for x in records if x['model_id'] in train];va=[x for x in records if x['model_id'] in val];te=[x for x in records if x['model_id'] in test]
            require(not(set(train)&set(test) or set(val)&set(test)),'model leakage')
        else:
            require(r['protocol']=='device_holdout_seen_architectures','unknown protocol')
            dev=r['heldout_device_id'];require(r['split_id']=='device-'+dev and dev in hardware,'unknown device holdout')
            test=MODELS;val=MODELS[-2:];train=MODELS[:-2]
            tr=[x for x in records if x['device_id']!=dev and x['model_id'] in train];va=[x for x in records if x['device_id']!=dev and x['model_id'] in val];te=[x for x in records if x['device_id']==dev]
        require(r['train_model_ids']==train and r['validation_model_ids']==val and r['test_model_ids']==test,'fixed model partition changed')
        require(r['train_records']==len(tr) and r['validation_records']==len(va) and r['test_records']==len(te),'partition query count mismatch')
        support=support_metrics(te);split_support[(r['feature_mode'],r['split_id'])]=support
        require([f['seed'] for f in r['seed_fits']]==SEEDS and [s['seed'] for s in r['seed_test_metrics']]==SEEDS,'seed alignment differs')
        for fit in r['seed_fits']:
            require(len(fit['checkpoint_sha256'])==64 and all(c in '0123456789abcdef' for c in fit['checkpoint_sha256']),'invalid checkpoint digest')
            if r['method']=='ridge':
                require(fit['training_epochs']==1 and fit['best_validation_mre'] is None,'fixed closed-form ridge must not report early-stop tuning')
            else:
                require(0<fit['training_epochs']<=80 and math.isfinite(fit['best_validation_mre']),'invalid early-stop evidence')
            checkpoint_hashes.append(fit['checkpoint_sha256'])
            pk=(r['feature_mode'],r['method']);parameter_counts.setdefault(pk,fit['parameter_count']);require(parameter_counts[pk]==fit['parameter_count'],'parameter capacity changed across fits')
        for metric in r['seed_test_metrics']+[r['ensemble_test_metrics']]:validate_metric(metric,support,totals)
        require(close(r['seed_micro_mre_mean'],avg(m['micro_mre_percent'] for m in r['seed_test_metrics'])),'entry micro seed mean mismatch')
        require(close(r['seed_micro_mre_sample_sd'],statistics.stdev(m['micro_mre_percent'] for m in r['seed_test_metrics'])),'entry micro seed SD mismatch')
        require(close(r['seed_family_macro_mre_mean'],avg(m['family_macro_mre_percent'] for m in r['seed_test_metrics'])),'entry family macro seed mean mismatch')
        require(close(r['seed_family_macro_mre_sample_sd'],statistics.stdev(m['family_macro_mre_percent'] for m in r['seed_test_metrics'])),'entry family macro seed SD mismatch')
        grouped[(r['feature_mode'],r['method'],r['protocol'])].append((r,support))
    expected={(mode,split,method) for mode in ['strict','probe'] for split in [f'model-fold{i}' for i in range(6)]+['device-'+x for x in hardware] for method in METHODS}
    require(entry_keys==expected,'missing matrix entry')
    require(len(d['aggregate_results'])==40,'aggregate matrix incomplete')
    aggregates={}
    for a in d['aggregate_results']:
        key=(a['feature_mode'],a['method'],a['protocol']);require(key not in aggregates,'duplicate aggregate');aggregates[key]=a
        parts=grouped[key]
        require([s['seed'] for s in a['seed_metrics']]==SEEDS,'aggregate seed order mismatch')
        for i,metric in enumerate(a['seed_metrics']):
            calc=combined_metrics([(r['seed_test_metrics'][i],s) for r,s in parts])
            validate_metric(metric,all_support,totals)
            for field in ['micro_mre_percent','family_macro_mre_percent','model_macro_mre_percent','attention_mre_percent']:
                require(close(metric[field],calc[field]),'aggregate arithmetic differs: '+str((key,field)))
        calc=combined_metrics([(r['ensemble_test_metrics'],s) for r,s in parts]);validate_metric(a['ensemble_metrics'],all_support,totals)
        for field in ['micro_mre_percent','family_macro_mre_percent','model_macro_mre_percent','attention_mre_percent']:
            require(close(a['ensemble_metrics'][field],calc[field]),'ensemble aggregate arithmetic mismatch')
        for short,long in [('micro','micro_mre_percent'),('family_macro','family_macro_mre_percent'),('model_macro','model_macro_mre_percent')]:
            require(close(a[short+'_mre_seed_mean'],avg(s[long] for s in a['seed_metrics'])),'aggregate seed mean mismatch')
            require(close(a[short+'_mre_seed_sample_sd'],statistics.stdev(s[long] for s in a['seed_metrics'])),'aggregate seed SD mismatch')
    require(len(d['paired_method_comparisons'])==36,'paired comparison matrix incomplete')
    paired_seen=set()
    for comparison in d['paired_method_comparisons']:
        mode,proto,other=comparison['feature_mode'],comparison['protocol'],comparison['comparator'];pk=(mode,proto,other)
        require(pk not in paired_seen,'duplicate paired comparison');paired_seen.add(pk)
        require(comparison['same_record_set_verified'] and comparison['reference']=='rgat','paired support/reference invalid')
        lhs=aggregates[(mode,'rgat',proto)]['seed_metrics'];rhs=aggregates[(mode,other,proto)]['seed_metrics']
        for field,item in comparison['paired_differences'].items():
            values=[x[field]-y[field] for x,y in zip(lhs,rhs)]
            require(item['n_seeds']==5 and all(close(x,y) for x,y in zip(values,item['values_pp'])),'paired per-seed difference mismatch')
            require(close(avg(values),item['mean_pp']) and close(statistics.stdev(values),item['sample_sd_pp']),'paired moment mismatch')
            require(all(close(x,y) for x,y in zip(bootstrap_seeds(values),item['paired_seed_bootstrap95_pp'])),'paired seed CI mismatch')
            totals['paired_seed_bootstrap_intervals_recomputed']+=1
    out={'kind':'independent_formal_five_stack_summary_audit','created_utc':datetime.now(timezone.utc).isoformat(),
         'audit_code_sha256':sha(__file__),'summary_sha256':hashlib.sha256(summary_bytes).hexdigest(),
         'evaluation_code_sha256':d['code_sha256'],'completed_utc':d['completed_utc'],
         'matrix_entries':len(d['results']),'fit_records':len(checkpoint_hashes),'aggregate_entries':len(aggregates),
         'paired_comparisons':len(paired_seen),'source_manifest':sources,'audit_counts':dict(totals),
         'query_support':{'per_protocol_per_seed':all_support['records'],'runtime_stacks':5,'models':12,
                          'family_n':dict(all_support['counts']['family']),'runtime_stack_n':dict(all_support['counts']['device_id'])},
         'parameter_counts':{':'.join(k):v for k,v in parameter_counts.items()},'all_checks_passed':True,
         'scope':['No training or performance timing was performed.',
                  'Raw and sidecar bytes are checked locally;1100 checkpoint digests are structurally validated but checkpoint files still require publication/remote-byte verification.',
                  'All144 paired seed intervals are unadjusted conditional five-seed bootstrap intervals on fixed physical records, not independent hardware-population samples.',
                  'Model-cluster intervals are descriptive with only12 model clusters; hardware holdout includes seen model architectures.',
                  'Probe features use three physical hardware microprofiles and are hardware-calibrated, not zero-measurement transfer.']}
    args.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k] for k in ['matrix_entries','fit_records','aggregate_entries','paired_comparisons','audit_counts','all_checks_passed']}))


if __name__=='__main__':main()
