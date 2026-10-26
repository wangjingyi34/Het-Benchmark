"""Replay all saved fixed fits; serialize undefined OOB R2 as null.

The original training code and frozen scientific protocol remain unchanged.
This script performs NO training, selection, support change, or profiling.
"""
import sys
sys.dont_write_bytecode=True
import csv, gzip, hashlib, json, math, pickle, statistics, time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import run_kernel_component as k

def main():
    start=time.monotonic();p=json.loads((k.OUT/'protocol.json').read_text())
    for f in p['sources']:assert k.sha(k.BASE/f['path'])==f['sha256'],f['path']
    result=k.OUT/'results';assert not (result/'execution-audit.json').exists()
    inputs=json.loads((k.OUT/'compatible-queries.json').read_text());byid={r['record_id']:r for r in inputs}
    cells=json.loads((k.OUT/'frozen-splits-and-support.json').read_text())['cells']
    torch.set_num_threads(1);predictions=[];fits=[];support=set();undefined=0
    for idx,c in enumerate(cells):
        if c['status']!='execute':continue
        test=[byid[i] for i in c['test_record_ids']];xt=np.array([r['features'] for r in test]);support.update(c['test_record_ids'])
        for seed in k.SEEDS:
            prefix={'split_id':c['split_id'],'device_id':c['device_id'],'kernel':c['kernel'],'seed':seed}
            for method,suffix in [('nn_meter_official_rf_component_adapted','rf.pkl'),('v7_same_information_mlp','mlp.pt')]:
                path=result/'checkpoints'/f'cell{idx}-seed{seed}-{suffix}';assert path.exists();t=time.monotonic()
                if suffix=='rf.pkl':
                    with path.open('rb') as f:model=pickle.load(f)
                    params=model.get_params();assert params['n_estimators']==370 and params['max_depth']==50 and params['max_features']==1.0 and params['random_state']==seed
                    pred=model.predict(xt);raw_oob=float(model.oob_score_);oob=raw_oob if math.isfinite(raw_oob) else None
                    undefined+=oob is None
                    details={'parameters':params,'oob_r2':oob,'oob_r2_note':'undefined because training support contains one query' if oob is None else 'diagnostic only; not used for selection'}
                else:
                    model=torch.load(path,map_location='cpu',weights_only=True);assert model['seed']==seed and model['epochs']==80;pred=k.predict_mlp(model,xt)
                    details={'epochs':80,'hidden':[128,128]}
                assert np.isfinite(pred).all() and (pred>0).all()
                fits.append({**prefix,'method':method,'train_records':c['train_records'],'validation_records_preserved_but_unused':c['validation_records'],
                             'test_records':len(test),'checkpoint':str(path.relative_to(k.OUT)),'checkpoint_sha256':k.sha(path),
                             'replay_seconds':time.monotonic()-t,**details})
                for r,estimate in zip(test,pred):
                    predictions.append({**prefix,'record_id':r['record_id'],'model_id':r['model_id'],'method':method,'true_latency_ms':r['true_latency_ms'],'prediction_ms':float(estimate)})
    assert len(fits)==400 and len(list((result/'checkpoints').glob('*')))==400
    k.dump(result/'fit-receipts.json',fits)
    with gzip.open(k.FORMAL/'predictions.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['protocol']!='model_holdout' or r['feature_mode']!='strict' or r['method'] not in ['ridge','mlp','gcn','rgat'] or r['record_id'] not in support:continue
            q=byid[r['record_id']];assert abs(q['true_latency_ms']-r['true_latency_ms'])<1e-12
            predictions.append({'split_id':r['split_id'],'device_id':r['device_id'],'kernel':q['kernel'],'seed':r['seed'],'record_id':r['record_id'],'model_id':r['model_id'],
                                'method':'V2_archived_'+r['method']+'_richer_training','true_latency_ms':r['true_latency_ms'],'prediction_ms':r['prediction_ms']})
    k.csvout(result/'predictions.csv',predictions)
    grouped=defaultdict(list)
    for r in predictions:
        for scope in ['pooled','per_stack','per_kernel','per_fold']:
            value={'pooled':'all','per_stack':r['device_id'],'per_kernel':r['kernel'],'per_fold':r['split_id']}[scope]
            grouped[(scope,value,r['method'],r['seed'])].append(r)
    metrics=[]
    for (scope,value,method,seed),rs in sorted(grouped.items()):
        ids=[r['record_id'] for r in rs];assert len(ids)==len(set(ids))
        metrics.append({'scope':scope,'scope_value':value,'method':method,'seed':seed,
                        'support_sha256':hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),**k.metric(rs)})
    bysupport=defaultdict(set)
    for r in metrics:bysupport[(r['scope'],r['scope_value'])].add(r['support_sha256'])
    assert all(len(v)==1 for v in bysupport.values());k.csvout(result/'per-seed-metrics.csv',metrics)
    aggregates=[]
    for scope,value,method in sorted(set((r['scope'],r['scope_value'],r['method']) for r in metrics)):
        rs=[r for r in metrics if (r['scope'],r['scope_value'],r['method'])==(scope,value,method)];assert len(rs)==5
        row={'scope':scope,'scope_value':value,'method':method,'n_seeds':5,'n_queries':rs[0]['n_queries']}
        for name in k.metric(predictions[:1]):
            if name=='n_queries':continue
            vals=[r[name] for r in rs];row[name+'_seed_mean']=statistics.mean(vals);row[name+'_seed_sd']=statistics.stdev(vals)
        aggregates.append(row)
    k.csvout(result/'aggregate-metrics.csv',aggregates)
    report={'complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':k.sha(k.OUT/'protocol.json'),
            'completion_code_sha256':k.sha(__file__),'training_code_unchanged_sha256':k.sha(k.OUT/'run_kernel_component.py'),
            'fits':len(fits),'saved_checkpoints_loaded':400,'predictions_including_archived_references':len(predictions),'unique_query_support':len(support),
            'compatible_queries_without_executable_train_support':sorted(set(byid)-support),'undefined_OOB_R2_count':undefined,
            'technical_recovery':'Training all 400 fits finished and saved checkpoints. Original allow_nan=False JSON writer rejected undefined OOB R2 for 25 one-training-query RF fits. This script replays the saved weights and writes null plus reason. No retraining or scientific protocol change.',
            'all_six_methods_same_scoring_support':True,'recovery_and_scoring_wall_seconds':time.monotonic()-start,'new_hardware_measurements':0,
            'pooled_results':[r for r in aggregates if r['scope']=='pooled'],'boundary':p['cannot_claim'],
            'outputs':[{'path':str(f.relative_to(k.OUT)),'sha256':k.sha(f)} for f in sorted(result.glob('*')) if f.is_file()]}
    k.dump(result/'execution-audit.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
