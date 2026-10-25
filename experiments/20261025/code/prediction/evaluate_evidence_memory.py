"""Train-only physical-performance evidence retrieval, separate from RGAT.

Frozen v2 graph/MLP/GCN results are not modified. All retrieval policies are
selected only on existing validation data; device validation masks the query
stack from its anchors to imitate the declared hardware-transfer condition.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluate_profiles import base_vector, NUMERIC_ATTRIBUTES

KS = [1,3,5,10,20]
TEMPERATURES = [.25,1.,4.]
GAMMAS = [0.,.5,1.]
RIDGE_LAMBDAS = [.01,1.,100.]
METHODS = ['family_kernel','semantic_evidence_memory','semantic_ridge']
PROBE_INDEX = {'Conv2D':1,'Conv1D':1,'Composite':1,'Pooling':1,
               'MatMul':0,'Attention':0,'Normalization':2,'Softmax':2,
               'Reduction':2,'Activation':2,'View':2,'Elementwise':2,
               'Embedding':2,'Dropout':2,'Other':2}
DISTANCE_COLUMNS = list(range(20,30)) + list(range(33,58))
EXPECTED_EVAL = 'bad26be58e40a0bf6ff939024f15e646a72ec3e1b8bc3fef54dc67745f8cce0f'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def avg(xs):return statistics.mean(xs)
def utc():return datetime.now(timezone.utc).isoformat()
def require(condition,message):
    if not condition:raise ValueError(message)


def canonical_operator(node):
    if node['module_path']:
        op=node['operator_attributes'].get('module_class')
        require(bool(op),'module class absent from structural sidecar')
    else:
        op=re.sub(r'0x[0-9a-fA-F]+','<address>',node['target'])
        op=re.sub(r'\s+',' ',op).strip()
    return node['fx_op']+'|'+op


def signature(node,op):
    fields=['shape','dtype','stride','numel','element_size']
    obj={'operator':op,'inputs':[{k:x[k] for k in fields} for x in node['input_shapes']],
         'outputs':[{k:x[k] for k in fields} for x in node['output_shapes']],
         'attributes':{k:node['operator_attributes'].get(k,0) for k in NUMERIC_ATTRIBUTES}}
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_records(summary,profiles_dir):
    rows=[];hardware={};manifest=[]
    for source in summary['source_manifest']:
        rawpath=profiles_dir/Path(source['path']).name;sidepath=profiles_dir/Path(source['metadata_sidecar']).name
        require(sha(rawpath)==source['sha256'] and sha(sidepath)==source['metadata_sidecar_sha256'],'source bytes differ from frozen main study')
        raw=json.loads(rawpath.read_text());side=json.loads(sidepath.read_text());h=raw['hardware'];hardware[h['device_id']]=h
        require(side['raw_bundle_sha256']==source['sha256'] and not side['timings_changed_or_added'],'invalid metadata sidecar')
        require(len(h['probes'])==3,'three independent probes required for calibrated arm')
        for graph in raw['graphs']:
            sc=side['graphs'][graph['graph_id']]
            require(sc['input_weights_hash_verified'] and sc['input_weights_sha256']==graph['input_weights_sha256'],'unverified structural enrichment')
            nodes={n['node_id']:{**n,'operator_attributes':sc['nodes'][n['node_id']]} for n in graph['nodes']}
            for record in graph['records']:
                if record['status']!='measured':continue
                samples=record['latency_samples_ms'];y=avg(samples)
                require(len(samples)>=20 and all(math.isfinite(t) and t>0 for t in samples),'invalid physical evidence')
                require(abs(y-record['latency_mean_ms'])<1e-9,'physical label mean mismatch')
                node=nodes[record['op_instance_id']];op=canonical_operator(node)
                rows.append({'record_id':record['record_id'],'node_id':node['node_id'],
                    'model_id':graph['model_id'],'device_id':h['device_id'],'family':node['family'],
                    'operator':op,'exact_signature':signature(node,op),'latency_ms':y,
                    'strict_features':base_vector(node,h,'strict'),'probe_features':base_vector(node,h,'probe'),
                    'probe_latencies_ms':[p['latency_mean_ms'] for p in h['probes']],
                    'input_weights_sha256':graph['input_weights_sha256']})
        manifest.append({'raw_path':str(rawpath.resolve()),'raw_sha256':sha(rawpath),'metadata_sha256':sha(sidepath),'device_id':h['device_id']})
    require(len(rows)==13905,'record support must equal frozen13905 queries')
    return rows,manifest,hardware


def partition(rows,split):
    train_ids=set(split['train_model_ids']);val_ids=set(split['validation_model_ids']);test_ids=set(split['test_model_ids'])
    if split['protocol']=='model_holdout':
        tr=[i for i,r in enumerate(rows) if r['model_id'] in train_ids]
        va=[i for i,r in enumerate(rows) if r['model_id'] in val_ids]
        te=[i for i,r in enumerate(rows) if r['model_id'] in test_ids]
        require(not(train_ids&test_ids or val_ids&test_ids),'model holdout leaks query model')
    else:
        device=split['heldout_device_id']
        tr=[i for i,r in enumerate(rows) if r['device_id']!=device and r['model_id'] in train_ids]
        va=[i for i,r in enumerate(rows) if r['device_id']!=device and r['model_id'] in val_ids]
        te=[i for i,r in enumerate(rows) if r['device_id']==device]
        require(all(rows[i]['device_id']!=device for i in tr+va),'device holdout leaks target stack')
    require(len(tr)==split['train_records'] and len(va)==split['validation_records'] and len(te)==split['test_records'],'partitions differ from frozen main')
    return np.array(tr),np.array(va),np.array(te)


def index_candidates(rows,indices):
    groups={k:collections.defaultdict(list) for k in ['exact_signature','operator','family']}
    for i in indices:
        for key in groups:groups[key][rows[i][key]].append(int(i))
    return {key:{value:np.array(ids,dtype=np.int64) for value,ids in mapping.items()} for key,mapping in groups.items()}


def neighbors(rows,train,queries,method,exclude_query_device):
    # A train-only scaler and only structural operator columns enter distance.
    matrix=np.asarray([r['strict_features'] for r in rows],dtype=np.float64)[:,DISTANCE_COLUMNS]
    mu=matrix[train].mean(0);sd=matrix[train].std(0);sd[sd<1e-8]=1.
    z=(matrix-mu)/sd;groups=index_candidates(rows,train);result=[]
    for qi in queries:
        q=rows[qi];pool=train
        if exclude_query_device:
            pool=pool[np.array([rows[i]['device_id']!=q['device_id'] for i in pool])]
        else:
            same=pool[np.array([rows[i]['device_id']==q['device_id'] for i in pool])]
            if len(same):pool=same
        require(len(pool)>0,'empty hardware-filtered evidence pool')
        permitted=set(pool.tolist());tier='global_train_fallback';candidate=pool
        keys=['exact_signature','operator','family'] if method=='semantic_evidence_memory' else ['family']
        for key in keys:
            ids=groups[key].get(q[key],np.empty(0,dtype=np.int64))
            ids=np.asarray([i for i in ids if int(i) in permitted],dtype=np.int64)
            if len(ids):candidate=ids;tier=key;break
        # Stable record-ID order resolves exact descriptor ties before distance.
        candidate=np.array(sorted(candidate,key=lambda i:rows[i]['record_id']))
        distance=np.mean((z[candidate]-z[qi])**2,axis=1)
        best=np.argsort(distance,kind='stable')[:max(KS)]
        ids=candidate[best];dist=np.maximum(distance[best],0.)
        probe=PROBE_INDEX[q['family']]
        ratios=np.array([q['probe_latencies_ms'][probe]/rows[i]['probe_latencies_ms'][probe] for i in ids])
        result.append({'query_index':int(qi),'anchor_indices':ids,'distances':dist,'probe_log_ratios':np.log(ratios),'tier':tier})
    return result


def memory_predict(rows,cache,k,temperature,gamma):
    values=[]
    for item in cache:
        ids=item['anchor_indices'][:k];distance=item['distances'][:k]
        weight=np.exp(-(distance-distance.min())/temperature);weight/=weight.sum()
        log_y=np.log([rows[i]['latency_ms'] for i in ids])+gamma*item['probe_log_ratios'][:k]
        values.append(float(np.exp(np.dot(weight,log_y))))
    return np.array(values)


def mre(rows,indices,pred):
    y=np.array([rows[i]['latency_ms'] for i in indices]);return float(np.mean(np.abs(pred-y)/np.maximum(y,1e-6))*100)


def ridge_features(rows,train,query,mode):
    classes=sorted({rows[i]['operator'] for i in train});lookup={c:i for i,c in enumerate(classes)}
    def build(indices):
        base=np.array([rows[i][mode+'_features'] for i in indices],dtype=np.float64)
        onehot=np.zeros((len(indices),len(classes)+1))
        for j,i in enumerate(indices):onehot[j,lookup.get(rows[i]['operator'],len(classes))]=1
        return np.concatenate([base,onehot],axis=1)
    tr=build(train);qu=build(query);mu=tr.mean(0);sd=tr.std(0);sd[sd<1e-8]=1.
    return np.column_stack([(tr-mu)/sd,np.ones(len(tr))]),np.column_stack([(qu-mu)/sd,np.ones(len(qu))])


def ridge_predict(rows,train,query,mode,lam):
    x,q=ridge_features(rows,train,query,mode);y=np.log([rows[i]['latency_ms'] for i in train])
    regularizer=np.eye(x.shape[1])*lam;regularizer[-1,-1]=0.
    coef=np.linalg.solve(x.T@x+regularizer,x.T@y)
    return np.exp(np.clip(q@coef,-20,20))


def ridge_validation(rows,train,validation,mode,lam,device_protocol):
    if not device_protocol:return ridge_predict(rows,train,validation,mode,lam)
    result=np.empty(len(validation));devices=sorted({rows[i]['device_id'] for i in validation})
    for device in devices:
        anchor=np.array([i for i in train if rows[i]['device_id']!=device])
        positions=np.array([j for j,i in enumerate(validation) if rows[i]['device_id']==device])
        result[positions]=ridge_predict(rows,anchor,validation[positions],mode,lam)
    return result


def metrics(rows,indices,pred):
    by={k:collections.defaultdict(list) for k in ['family','model_id','device_id']}
    for i,p in zip(indices,pred):
        r=rows[i];e=abs(p-r['latency_ms'])/max(r['latency_ms'],1e-6)*100
        require(math.isfinite(e),'nonfinite test error')
        for key in by:by[key][r[key]].append(e)
    means={k:{name:avg(es) for name,es in value.items()} for k,value in by.items()}
    return {'records':len(indices),'micro_mre_percent':mre(rows,indices,pred),
        'family_macro_mre_percent':avg(means['family'].values()),'model_macro_mre_percent':avg(means['model_id'].values()),
        'attention_mre_percent':means['family'].get('Attention'),'attention_n':len(by['family'].get('Attention',[])),
        'per_family_mre_percent':means['family'],'per_family_n':{k:len(v) for k,v in by['family'].items()},
        'per_model_mre_percent':means['model_id'],'per_device_mre_percent':means['device_id'],
        'per_device_n':{k:len(v) for k,v in by['device_id'].items()},
        'evaluation_record_ids_sha256':hashlib.sha256('\n'.join(sorted(rows[i]['record_id'] for i in indices)).encode()).hexdigest()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference-summary',required=True,type=Path)
    ap.add_argument('--profiles-dir',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--fold-limit',type=int);ap.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    args=ap.parse_args();require(not args.output.exists(),'never overwrite evidence experiment')
    source=json.loads(args.reference_summary.read_text());require(source['code_sha256']==EXPECTED_EVAL and source.get('completed_utc'),'main reference must be frozen complete bad26 result')
    rows,manifest,hardware=load_records(source,args.profiles_dir)
    args.output.mkdir(parents=True)
    plan={'kind':'train_only_physical_evidence_retrieval_method_freeze','frozen_utc':utc(),'code_sha256':sha(__file__),
        'reference_summary_sha256':sha(args.reference_summary),'source_manifest':manifest,
        'methods':args.methods,'k_grid':KS,'kernel_temperature_grid':TEMPERATURES,'probe_exponent_grid':GAMMAS,
        'semantic_ridge_lambda_grid':RIDGE_LAMBDAS,'family_to_probe_index':PROBE_INDEX,
        'semantic_signature':'canonical module class or normalized function/method; exact logical input/output tensor descriptors and25 attributes',
        'candidate_rule':'same device if available for model holdout; exclude query device for device-holdout validation and test; semantic exact->operator->family->global backoff',
        'distance':'train-scaled squared L2 over10 operator shape/size columns and25 attributes; no graph-position or labels',
        'retrieval_target':'geometric kernel mean of actual train measurements with optional independent target/source hardware probe ratio',
        'selection':'one global hyperparameter configuration per method/mode/fold minimizing validation micro MRE; test scores never choose a grid point or family',
        'information_control':'semantic_ridge receives the original strict/probe feature whitelist plus train-vocabulary operation-type onehot. Family kernel controls retrieval without exact operation-type gating.',
        'new_target_physical_measurements':0,'changes_to_bad26':False,
        'fold_limit':args.fold_limit,'research_scope':'Separate evidence-retrieval enhancement; not attributed to RGAT or historical paper numbers.'}
    (args.output/'method-freeze.json').write_text(json.dumps(plan,indent=2))
    splits=[r for r in source['results'] if r['method']=='rgat' and r['feature_mode']=='strict']
    if args.fold_limit:splits=splits[:args.fold_limit]
    result={'kind':'train_only_physical_evidence_retrieval_evaluation','created_utc':utc(),'method_freeze_sha256':sha(args.output/'method-freeze.json'),
        'code_sha256':sha(__file__),'reference_summary_sha256':sha(args.reference_summary),'results':[],
        'source_manifest':manifest,'full_matrix':args.fold_limit is None,'record_support_unchanged':True,
        'timing_or_hardware_accuracy_claim':'New offline evaluation using only previously measured training labels; no new physical profiling.'}
    aggregate=collections.defaultdict(list)
    stream=gzip.open(args.output/'predictions.jsonl.gz','wt')
    for split in splits:
        train,validation,test=partition(rows,split);device_protocol=split['protocol']!='model_holdout'
        # Candidate identities/distances are independent of probe exponent and
        # validation/test target values. Cache once for both information arms.
        cache={}
        for method in args.methods:
            if method!='semantic_ridge':
                cache[(method,'validation')]=neighbors(rows,train,validation,method,device_protocol)
                cache[(method,'test')]=neighbors(rows,train,test,method,device_protocol)
        for mode in ['strict','probe']:
            for method in args.methods:
                trials=[]
                if method=='semantic_ridge':
                    for lam in RIDGE_LAMBDAS:
                        vp=ridge_validation(rows,train,validation,mode,lam,device_protocol)
                        trials.append({'lambda':lam,'validation_micro_mre':mre(rows,validation,vp)})
                    selected=min(trials,key=lambda q:(q['validation_micro_mre'],q['lambda']))
                    pred=ridge_predict(rows,train,test,mode,selected['lambda']);neighbor_summary=None
                else:
                    gammas=GAMMAS if mode=='probe' and device_protocol else [0.]
                    for k in KS:
                        for temp in TEMPERATURES:
                            for gamma in gammas:
                                vp=memory_predict(rows,cache[(method,'validation')],k,temp,gamma)
                                trials.append({'k':k,'temperature':temp,'probe_exponent':gamma,'validation_micro_mre':mre(rows,validation,vp)})
                    selected=min(trials,key=lambda q:(q['validation_micro_mre'],q['k'],q['temperature'],q['probe_exponent']))
                    pred=memory_predict(rows,cache[(method,'test')],selected['k'],selected['temperature'],selected['probe_exponent'])
                    neighbor_summary=dict(collections.Counter(x['tier'] for x in cache[(method,'test')]))
                public={k:v for k,v in split.items() if k in ['split_id','protocol','train_model_ids','validation_model_ids','test_model_ids','heldout_device_id']}
                entry={**public,'feature_mode':mode,'method':method,'selected_validation_configuration':selected,
                    'validation_grid_results':trials,'train_records':len(train),'validation_records':len(validation),
                    'test_metrics':metrics(rows,test,pred),'test_anchor_match_tiers':neighbor_summary,
                    'query_label_used_for_neighbor_selection':False,'target_labels_used_for_hyperparameters':False,
                    'validation_excludes_query_device_anchors':device_protocol,
                    'prediction_seed_scope':'deterministic retrieval/closed-form ridge; no fictitious repeated seed samples'}
                expected=next(r for r in source['results'] if r['split_id']==split['split_id'] and r['method']=='rgat' and r['feature_mode']==mode)
                require(entry['test_metrics']['evaluation_record_ids_sha256']==expected['ensemble_test_metrics']['evaluation_record_ids_sha256'],'test support differs from main regressors')
                result['results'].append(entry)
                for i,p in zip(test,pred):
                    record={k:rows[i][k] for k in ['record_id','model_id','device_id','family']}
                    record.update({'method':method,'feature_mode':mode,'split_id':split['split_id'],'protocol':split['protocol'],
                        'true_latency_ms':rows[i]['latency_ms'],'prediction_ms':float(p)})
                    stream.write(json.dumps(record,separators=(',',':'))+'\n')
                    aggregate[(mode,method,split['protocol'])].append((int(i),float(p)))
                stream.flush();(args.output/'summary.partial.json').write_text(json.dumps(result,indent=2))
                print(json.dumps({'split':split['split_id'],'mode':mode,'method':method,'micro_MRE':entry['test_metrics']['micro_mre_percent'],'selected':selected}),flush=True)
    stream.close();result['aggregate_results']=[]
    for (mode,method,protocol),items in sorted(aggregate.items()):
        ids=np.array([i for i,_ in items]);require(len(set(ids.tolist()))==len(ids),'test queries appear twice in a protocol')
        result['aggregate_results'].append({'feature_mode':mode,'method':method,'protocol':protocol,
            'metrics':metrics(rows,ids,np.array([p for _,p in items]))})
    result['completed_utc']=utc();(args.output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'complete':True,'entries':len(result['results']),'aggregates':len(result['aggregate_results'])}),flush=True)


if __name__=='__main__':main()
