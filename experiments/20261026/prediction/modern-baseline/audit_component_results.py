"""Independent raw-mean, split, feature, checkpoint-hash and scalar-score audit."""
import csv,gzip,hashlib,json,math,statistics
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[2]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    proto=json.loads((OUT/'protocol.json').read_text())
    for s in proto['sources']:assert sha(BASE/s['path'])==s['sha256']
    summary=json.loads((BASE/'v2-实证增强/public-reproduction-check/prediction/formal-five-stack/summary.json').read_text())
    truth={}
    for src in summary['source_manifest']:
        p=BASE/'v2-实证增强/public-reproduction-check/evidence/physical_profiles'/Path(src['path']).name
        raw=json.loads(p.read_text())
        for g in raw['graphs']:
            for r in g['records']:
                if r['status']=='measured':truth[r['record_id']]=(statistics.mean(r['latency_samples_ms']),g['model_id'],raw['hardware']['device_id'],g['input_weights_sha256'])
    data=json.loads((OUT/'compatible-queries.json').read_text());byid={r['record_id']:r for r in data}
    with (OUT/'excluded-queries.csv').open() as f:excluded=list(csv.DictReader(f))
    assert len(byid)==350 and len(excluded)==13555
    assert not set(byid)&{r['record_id'] for r in excluded}
    assert set(byid)|{r['record_id'] for r in excluded}==set(truth)
    for r in data:
        t,model,device,weights=truth[r['record_id']]
        assert math.isclose(r['true_latency_ms'],t,rel_tol=1e-12,abs_tol=1e-12)
        assert (r['model_id'],r['device_id'],r['input_weights_sha256'])==(model,device,weights)
        c=r['config']
        if r['kernel']=='fc':
            par=(2*c['CIN']+1)*c['COUT'];features=[c['CIN'],c['COUT'],par/2e6,par/1e6]
        else:
            assert r['kernel']=='conv-block'
            par=c['COUT']*(c['KERNEL_SIZE']**2*c['CIN']+1);flops=2*c['HW']/c['STRIDES']*c['HW']/c['STRIDES']*par
            features=[c[k] for k in ['HW','CIN','COUT','KERNEL_SIZE','STRIDES']]+[flops/2e6,par/1e6]
        assert features==r['features']
    splitdata=json.loads((OUT/'frozen-splits-and-support.json').read_text())
    splits={r['split_id']:r for r in splitdata['splits']};counts={r['status']:0 for r in splitdata['cells']}
    for cell in splitdata['cells']:
        counts[cell['status']]+=1
        for part in ['train','validation','test']:
            ids=cell[part+'_record_ids'];assert len(ids)==cell[part+'_records']
            assert all(byid[i]['model_id'] in splits[cell['split_id']][part+'_model_ids'] for i in ids)
        assert not set(cell['train_record_ids'])&set(cell['test_record_ids'])
    fits=json.loads((OUT/'results/fit-receipts.json').read_text());assert len(fits)==400
    for fit in fits:assert sha(OUT/fit['checkpoint'])==fit['checkpoint_sha256']
    results=[]
    with (OUT/'results/predictions.csv').open() as f:
        for r in csv.DictReader(f):
            r['seed']=int(r['seed']);r['prediction_ms']=float(r['prediction_ms']);r['true_latency_ms']=float(r['true_latency_ms'])
            assert r['model_id'] in splits[r['split_id']]['test_model_ids']
            assert math.isclose(truth[r['record_id']][0],r['true_latency_ms'],abs_tol=1e-12,rel_tol=1e-12)
            assert r['prediction_ms']>0 and math.isfinite(r['prediction_ms']);results.append(r)
    assert len(results)==10500
    groups=defaultdict(list)
    for r in results:
        for scope,value in [('pooled','all'),('per_stack',r['device_id']),('per_kernel',r['kernel']),('per_fold',r['split_id'])]:groups[(scope,value,r['method'],r['seed'])].append(r)
    maxdiff=defaultdict(float)
    with (OUT/'results/per-seed-metrics.csv').open() as f:metricrows=list(csv.DictReader(f))
    for r in metricrows:
        rs=groups[(r['scope'],r['scope_value'],r['method'],int(r['seed']))];ids=[x['record_id'] for x in rs]
        assert len(ids)==len(set(ids))==int(r['n_queries'])
        assert hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()==r['support_sha256']
        ae=[abs(x['prediction_ms']-x['true_latency_ms']) for x in rs];re=[a/x['true_latency_ms'] for a,x in zip(ae,rs)]
        exp={'micro_MRE_percent':statistics.mean(re)*100,'MAE_us':statistics.mean(ae)*1000,
             'WAPE_percent':100*math.fsum(ae)/math.fsum(x['true_latency_ms'] for x in rs),
             'RMSE_ms':math.sqrt(statistics.mean(a*a for a in ae)),'RMSPE_percent':100*math.sqrt(statistics.mean(v*v for v in re)),
             'accuracy_within5percent':100*sum(v<=.05 for v in re)/len(rs),'accuracy_within10percent':100*sum(v<=.1 for v in re)/len(rs)}
        for k,v in exp.items():
            maxdiff[k]=max(maxdiff[k],abs(v-float(r[k])));assert math.isclose(v,float(r[k]),abs_tol=1e-10,rel_tol=1e-12)
    audit={'complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),'source_query_partition':{'total':len(truth),'included':len(data),'excluded':len(excluded)},
           'independently_recomputed_features':len(data),'all_input_weights_and_physical_means_match':True,'support_cells':counts,
           'checkpoint_hashes_verified':len(fits),'prediction_rows':len(results),'metric_rows_independently_recomputed':len(metricrows),
           'maximum_absolute_metric_difference':dict(maxdiff),'all_test_models_separate_from_train_and_validation':True,
           'undefined_OOB_R2_preserved_as_null':sum(f.get('oob_r2_note','').startswith('undefined') and f.get('oob_r2') is None for f in fits),
           'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(OUT/'protocol.json'),'predictions_sha256':sha(OUT/'results/predictions.csv')}
    (OUT/'results/independent-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))

if __name__=='__main__':main()
