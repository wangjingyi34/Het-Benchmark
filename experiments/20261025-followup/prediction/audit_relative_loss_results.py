"""Independently reconstruct saved selected predictions; report selective costs."""
import collections, csv, gzip, hashlib, json, math
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def weighted_median(ys,ws):
    pairs=sorted(zip(ys,ws),key=lambda x:x[0]);half=sum(ws)/2;total=0
    for y,w in pairs:
        total+=w
        if total>=half:return y
    raise ValueError('empty evidence')
def scalar_predict(anchor,lookup,configuration,relative):
    k=configuration['k'];tmp=configuration['temperature'];gamma=configuration['probe_exponent']
    ys=[lookup[r]['latency']*math.exp(gamma*ratio) for r,ratio in zip(anchor['anchor_record_ids'][:k],anchor['probe_log_ratios'][:k])]
    ds=anchor['distances'][:k];ws=[math.exp(-(d-ds[0])/tmp) for d in ds]
    if relative:p=weighted_median(ys,[w/y for w,y in zip(ws,ys)])
    else:p=math.exp(sum(w*math.log(y) for y,w in zip(ys,ws))/sum(ws))
    return p
def main():
    result=json.loads((OUT/'results.json').read_text());lookup={}
    for s in result['source_manifest']:
        p=Path(s['raw_path']);raw=json.loads(p.read_text());assert sha(p)==s['raw_sha256']
        for g in raw['graphs']:
            for r in g['records']:
                if r['status']=='measured':lookup[r['record_id']]={'latency':sum(r['latency_samples_ms'])/len(r['latency_samples_ms']),
                    'timed_cost_ms':sum(r['latency_samples_ms'])*r['inner_repeats'],'family':next(n['family'] for n in g['nodes'] if n['node_id']==r['op_instance_id'])}
    prediction_rows=[]
    with gzip.open(OUT/'predictions.jsonl.gz','rt') as f:
        for line in f:prediction_rows.append(json.loads(line))
    predlookup={(r['split_id'],r['method'],r['feature_mode'],r['record_id']):r for r in prediction_rows}
    max_delta=0.;checked=0;selected_val=[];max_val_metric_delta=0.;max_test_metric_delta=0.
    for e in result['results']:
        relative=e['method'].startswith('relative_');cfg=e['selected_validation_configuration']
        selected=min(e['validation_grid_results'],key=lambda r:(r['validation_micro_mre'],r['k'],r['temperature'],r['probe_exponent']))
        assert selected==cfg
        train_ids=set(next(f['train_record_ids'] for f in json.loads((OUT/'frozen-folds.json').read_text()) if f['split_id']==e['split_id']))
        val_errors=[];test_errors=[]
        with gzip.open(OUT/e['anchors_path'],'rt') as f:
            for line in f:
                a=json.loads(line);assert set(a['anchor_record_ids']).issubset(train_ids)
                p=scalar_predict(a,lookup,cfg,relative)
                if a['stage']=='test':
                    saved=predlookup[(e['split_id'],e['method'],e['feature_mode'],a['query_id'])]
                    max_delta=max(max_delta,abs(p-saved['prediction_ms'])/max(p,1e-15));checked+=1
                    y=lookup[a['query_id']]['latency'];test_errors.append(100*abs(p-y)/y)
                else:
                    selected_val.append({'split_id':e['split_id'],'method':e['method'],'feature_mode':e['feature_mode'],
                        'record_id':a['query_id'],'true_latency_ms':lookup[a['query_id']]['latency'],'prediction_ms':p})
                    y=lookup[a['query_id']]['latency'];val_errors.append(100*abs(p-y)/y)
        max_val_metric_delta=max(max_val_metric_delta,abs(sum(val_errors)/len(val_errors)-cfg['validation_micro_mre']))
        max_test_metric_delta=max(max_test_metric_delta,abs(sum(test_errors)/len(test_errors)-e['test_metrics']['micro_mre_percent']))
    assert max_delta<1e-12
    assert max_val_metric_delta<1e-9 and max_test_metric_delta<1e-9
    with gzip.open(OUT/'selected-validation-predictions.jsonl.gz','wt') as f:
        for r in selected_val:f.write(json.dumps(r,separators=(',',':'))+'\n')
    selective=[];groups=collections.defaultdict(list)
    for r in prediction_rows:groups[(r['method'],r['feature_mode'],r['protocol'])].append(r)
    for (method,mode,protocol),rs in groups.items():
        retained=[r for r in rs if not r['trigger_measurement_offline']];trigger=[r for r in rs if r['trigger_measurement_offline']]
        attention=[r for r in rs if r['family']=='Attention'];ar=[r for r in attention if not r['trigger_measurement_offline']]
        def mre(items):return float(np.mean([100*abs(r['prediction_ms']-r['true_latency_ms'])/r['true_latency_ms'] for r in items])) if items else None
        selective.append({'method':method,'feature_mode':mode,'protocol':protocol,'records':len(rs),
            'triggered_records':len(trigger),'triggered_percent':100*len(trigger)/len(rs),'all_query_mre_percent':mre(rs),
            'retained_records':len(retained),'retained_mre_percent':mre(retained),'attention_records':len(attention),
            'attention_triggered_records':len(attention)-len(ar),'attention_retained_mre_percent':mre(ar),
            'triggered_recorded_timed_cost_ms':sum(lookup[r['record_id']]['timed_cost_ms'] for r in trigger),
            'all_records_recorded_timed_cost_ms':sum(lookup[r['record_id']]['timed_cost_ms'] for r in rs),
            'cost_scope':'Offline sum of already recorded synchronized timed blocks; excludes setup and warmup. No fresh profiling or deployment receipt.'})
    audit={'kind':'independent_scalar_reconstruction_and_source_anchor_audit','script_sha256':sha(Path(__file__)),
        'results_sha256':sha(OUT/'results.json'),'test_predictions_checked':checked,'max_relative_prediction_delta':max_delta,
        'selected_validation_mre_max_delta':max_val_metric_delta,'test_mre_max_delta':max_test_metric_delta,
        'all_anchor_ids_in_per_fold_train_support':True,'all_grid_selections_validation_minima':True,
        'selected_validation_predictions':len(selected_val),'selected_validation_predictions_sha256':sha(OUT/'selected-validation-predictions.jsonl.gz'),
        'selective_aggregates':selective,'scope':'Offline descriptive evaluation. Retained risk is reported as measured, never assumed to decrease.'}
    (OUT/'result-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    flat=[]
    for a in result['aggregates']:
        m=a['metrics'];flat.append({'method':a['method'],'mode':a['feature_mode'],'protocol':a['protocol'],
            **{k:m[k] for k in ['records','micro_mre_percent','family_macro_mre_percent','model_macro_mre_percent','attention_mre_percent','mae_ms','wape_percent']}})
    with (OUT/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps({k:audit[k] for k in ['test_predictions_checked','max_relative_prediction_delta','selected_validation_predictions']}))

if __name__=='__main__':main()
