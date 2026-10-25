"""Audit unchanged-split model-budget curves and exact full-budget reuse."""
import argparse
import collections
import json
import math
from pathlib import Path

from audit_formal_summary import sha,require,support_metrics,validate_metric,combined_metrics,close,EXPECTED_EVALUATOR


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',required=True,type=Path)
    ap.add_argument('--main-summary',required=True,type=Path);ap.add_argument('--profiles-dir',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args();require(not args.output.exists(),'do not overwrite curve audit')
    d=json.loads(args.summary.read_text());main=json.loads(args.main_summary.read_text())
    require(d.get('completed_utc') and d['code_sha256']==main['code_sha256']==EXPECTED_EVALUATOR,'curve source/completion differs')
    require(len(d['results'])==36 and sum(len(r['seed_fits']) for r in d['results'])==180,'incomplete180 fits')
    for k in ['seeds','epochs_max','patience_validation_only','source_manifest','train_runtime']:
        require(d[k]==main[k],'curve/main policy or source differs: '+k)
    reuse=d['learning_curve_full_budget_reference'];require(reuse['sha256']==sha(args.main_summary),'full8 reference hash differs')
    require(len(reuse['reused_results'])==2,'full8 should reuse exactly MLP/RGAT probe results')
    for item in reuse['reused_results']:
        source=next(x for x in main['aggregate_results'] if x['feature_mode']==item['feature_mode'] and x['method']==item['method'] and x['protocol']=='model_holdout')
        for k,v in source.items():
            if k!='protocol':require(item[k]==v,'full8 metrics were not exact reused values')
    records=[];workloads={}
    for item in main['source_manifest']:
        path=args.profiles_dir/Path(item['path']).name;require(sha(path)==item['sha256'],'raw profile changed')
        raw=json.loads(path.read_text());dev=raw['hardware']['device_id']
        for graph in raw['graphs']:
            workloads[graph['model_id']]=graph['workload'];nodes={n['node_id']:n for n in graph['nodes']}
            for r in graph['records']:
                if r['status']=='measured':records.append({'record_id':r['record_id'],'model_id':graph['model_id'],'device_id':dev,'family':nodes[r['op_instance_id']]['family']})
    totals=collections.Counter();groups=collections.defaultdict(list);seen=set()
    for entry in d['results']:
        base=entry['split_id'].split('-budget')[0];budget=entry['training_model_budget']
        control=next(x for x in main['results'] if x['split_id']==base and x['method']==entry['method'] and x['feature_mode']=='probe')
        by=collections.defaultdict(list)
        for model in control['train_model_ids']:by[workloads[model]].append(model)
        order=[]
        while any(by.values()):
            for group in ['CV','NLP','Audio','LLM','Multimodal']:
                if by[group]:order.append(by[group].pop(0))
        require(entry['train_model_ids']==order[:budget] and budget in [2,4,6],'training subset not frozen label-free round robin')
        require(entry['test_model_ids']==control['test_model_ids'] and entry['validation_model_ids']==control['validation_model_ids'],'heldout boundary changed')
        require(entry['train_records']==sum(r['model_id'] in entry['train_model_ids'] for r in records),'training record budget differs')
        key=(entry['split_id'],entry['method']);require(key not in seen,'duplicate budget entry');seen.add(key)
        te=[r for r in records if r['model_id'] in entry['test_model_ids']];support=support_metrics(te)
        require([f['seed'] for f in entry['seed_fits']]==d['seeds'],'seeds differ')
        for metric in entry['seed_test_metrics']+[entry['ensemble_test_metrics']]:validate_metric(metric,support,totals)
        groups[(entry['method'],entry['protocol'])].append((entry,support))
    for agg in d['aggregate_results']:
        parts=groups[(agg['method'],agg['protocol'])];all_support=support_metrics(records)
        for i,metric in enumerate(agg['seed_metrics']):
            validate_metric(metric,all_support,totals);computed=combined_metrics([(e['seed_test_metrics'][i],s) for e,s in parts])
            for field in ['micro_mre_percent','family_macro_mre_percent','model_macro_mre_percent','attention_mre_percent']:
                require(close(metric[field],computed[field]),'curve aggregate arithmetic differs')
        validate_metric(agg['ensemble_metrics'],all_support,totals)
    out={'kind':'independent_model_budget_learning_curve_audit','all_checks_passed':True,'summary_sha256':sha(args.summary),
         'code_sha256':sha(__file__),'fit_records':180,'budget_entries':36,'exact_full8_results_reused':2,
         'audit_counts':dict(totals),'workload_round_robin_train_subset_checked':True,
         'fixed_model_boundaries_and_main_source_hashes_checked':True,
         'scope':'Budget2/4/6 subsets are fixed within training folds and full8 results are reused exactly. Seeds and12model clusters are not independent hardware samples.'}
    args.output.write_text(json.dumps(out,indent=2));print(json.dumps(out))


if __name__=='__main__':main()
