"""Descriptive v4 extension on frozen measured v2 queries; never edits v2.

Our new component predicts the weighted relative-L1 median of train evidence.
It is not an RGAT result, a reconstruction of historical14.3%, or a new timing run.
"""
import argparse, collections, gzip, hashlib, json, math, os, platform, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / 'v2-实证增强'
PUBLIC = V2 / 'public-reproduction-check'
sys.path.insert(0, str(V2/'prediction'))
import evaluate_evidence_memory as em

OUT = Path(__file__).resolve().parent
SOURCE = PUBLIC/'prediction/formal-five-stack/summary.json'
PROFILES = PUBLIC/'evidence/physical_profiles'
BASELINE = PUBLIC/'prediction/evidence-memory-frozen-main/summary.json'
K = [1,3,5,10,20]
TEMP = [.25,1.,4.]
GAMMA = [0.,.5,1.]
METHODS = ['family_kernel','semantic_evidence_memory']
BOOTSTRAP_SEED = 20261025
BOOTSTRAP_N = 10000

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def utc(): return datetime.now(timezone.utc).isoformat()
def idhash(rows,inds): return hashlib.sha256('\n'.join(sorted(rows[int(i)]['record_id'] for i in inds)).encode()).hexdigest()

def matrix(rows, cache):
    n=len(cache); ids=np.zeros((n,max(K)),dtype=int); dist=np.zeros_like(ids,dtype=float)
    ratio=np.zeros_like(dist); valid=np.zeros_like(dist,dtype=bool)
    for j,item in enumerate(cache):
        a=item['anchor_indices'];m=len(a)
        ids[j,:m]=a; dist[j,:m]=item['distances']; ratio[j,:m]=item['probe_log_ratios'];valid[j,:m]=True
        if m < max(K):ids[j,m:]=a[-1]
    y=np.asarray([r['latency_ms'] for r in rows])[ids]
    return ids,dist,ratio,valid,y

def predict(mat,k,temp,gamma,relative):
    ids,dist,ratio,valid,y=mat
    d=dist[:,:k];v=valid[:,:k]
    w=np.exp(-(d-d[:,:1])/temp)*v
    yy=y[:,:k]*np.exp(gamma*ratio[:,:k])
    if relative:
        # argmin_a sum_j kernel_j * |a-y_j|/y_j is a weighted median,
        # with median weights kernel_j/y_j. All y_j are train measurements.
        rw=w/yy;order=np.argsort(yy,axis=1,kind='stable')
        ys=np.take_along_axis(yy,order,axis=1);ws=np.take_along_axis(rw,order,axis=1)
        at=np.argmax(np.cumsum(ws,axis=1)>=.5*ws.sum(axis=1,keepdims=True),axis=1)
        p=ys[np.arange(len(ys)),at]
    else:p=np.exp((w*np.log(yy)).sum(axis=1)/w.sum(axis=1))
    logy=np.log(yy);mu=(w*logy).sum(axis=1)/w.sum(axis=1)
    spread=np.sqrt((w*(logy-mu[:,None])**2).sum(axis=1)/w.sum(axis=1))
    return p,spread

def metrics(rows,inds,pred):
    m=em.metrics(rows,inds,pred);y=np.asarray([rows[int(i)]['latency_ms'] for i in inds])
    m.update({'mae_ms':float(np.abs(pred-y).mean()),'wape_percent':float(100*np.abs(pred-y).sum()/y.sum())})
    return m

def environment():
    import torch
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'numpy':np.__version__,'torch':torch.__version__,'new_physical_measurements':0,
        'network_or_remote_execution':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args()
    source=json.loads(SOURCE.read_text());base=json.loads(BASELINE.read_text())
    rows,manifest,_=em.load_records(source,PROFILES)
    splits=[s for s in source['results'] if s['method']=='rgat' and s['feature_mode']=='strict']
    frozen=[]
    for split in splits:
        tr,va,te=em.partition(rows,split)
        frozen.append({k:split[k] for k in ['split_id','protocol','train_model_ids','validation_model_ids','test_model_ids','heldout_device_id'] if k in split}|{
            'train_records':len(tr),'validation_records':len(va),'test_records':len(te),
            'train_ids_sha256':idhash(rows,tr),'validation_ids_sha256':idhash(rows,va),'test_ids_sha256':idhash(rows,te),
            'train_record_ids':[rows[int(i)]['record_id'] for i in tr],
            'validation_record_ids':[rows[int(i)]['record_id'] for i in va],
            'test_record_ids':[rows[int(i)]['record_id'] for i in te]})
    foldpath=OUT/'frozen-folds.json'
    plan={'kind':'descriptive_relative_loss_evidence_component','frozen_utc':utc(),
        'code_sha256':sha(__file__),'dependency_code_sha256':{'evaluate_evidence_memory.py':sha(V2/'prediction/evaluate_evidence_memory.py'),
        'evaluate_profiles.py':sha(V2/'prediction/evaluate_profiles.py'),'structural_attributes.py':sha(V2/'prediction/structural_attributes.py')},
        'source_summary':str(SOURCE.relative_to(ROOT)),'source_summary_sha256':sha(SOURCE),
        'baseline_summary':str(BASELINE.relative_to(ROOT)),'baseline_summary_sha256':sha(BASELINE),
        'source_manifest':manifest,'records':len(rows),'fold_count':len(frozen),
        'new_component':'kernel-weighted relative-L1 median; weights kernel/actual_train_latency after permitted independent probe calibration',
        'comparators':METHODS,'k_grid':K,'temperature_grid':TEMP,'probe_exponent_grid':GAMMA,
        'validation_selection':'one global configuration per component/candidate-rule/mode/fold, minimum validation micro MRE; deterministic tie k,temp,gamma',
        'support_and_information':'same frozen V2 train/validation/test support, exact/family anchor gate, train-only scaler, strict or 3 independent probe calibration; heldout-stack validation excludes query stack anchors',
        'test_label_policy':'test query latency is accessed only after prediction/configuration selection, for scoring; no method selection, early stop or target score search',
        'selective_policy':'90th percentile of validation anchor log-latency spread, greater-than threshold; report offline retention and recorded cost only, no new measurement/hybrid benefit claim',
        'uncertainty':'10000 paired cluster bootstraps, source model (n12) for model holdout, heldout stack (n5) for stack holdout; descriptive intervals, no multiple-comparison claim',
        'bootstrap_seed':BOOTSTRAP_SEED,'exploratory_status':'Retrospective descriptive extension on an already evaluated dataset. The method was frozen before this run but this is not an independent preregistered confirmatory study.',
        'can_support':'measured-label heldout performance of a new relative-loss retrieval component on these five stacks and twelve architectures',
        'cannot_support':['original RGAT14.3%','original five-platform physical LOPO','arbitrary hardware/model transfer','a causal benefit of measurement fallback','new physical profiling'],
        'folds_sha256':None,'environment':environment()}
    if args.freeze_only:
        if (OUT/'protocol.json').exists():raise ValueError('never overwrite frozen protocol')
        write(foldpath,frozen);plan['folds_sha256']=sha(foldpath);write(OUT/'protocol.json',plan)
        print(json.dumps({'frozen':True,'records':len(rows),'folds':len(frozen),'code':sha(__file__)}));return
    stored=json.loads((OUT/'protocol.json').read_text())
    assert stored['code_sha256']==sha(__file__) and stored['source_summary_sha256']==sha(SOURCE)
    assert stored['folds_sha256']==sha(foldpath) and stored['source_manifest']==manifest
    assert not (OUT/'results.json').exists(),'never overwrite completed run'
    t0=time.time();result={'kind':stored['kind'],'started_utc':utc(),'protocol_sha256':sha(OUT/'protocol.json'),
        'folds_sha256':sha(foldpath),'code_sha256':sha(__file__),'results':[],'aggregates':[],'source_manifest':manifest}
    agg=collections.defaultdict(list);comparison=[]
    with gzip.open(OUT/'predictions.jsonl.gz','wt') as stream:
        for split in splits:
            tr,va,te=em.partition(rows,split);device=split['protocol']!='model_holdout'
            for rule in METHODS:
                vc=em.neighbors(rows,tr,va,rule,device);tc=em.neighbors(rows,tr,te,rule,device)
                vm=matrix(rows,vc);tm=matrix(rows,tc)
                # Anchor identities + distances fully specify the train evidence used.
                anchorfile=OUT/(split['split_id']+'-'+rule+'-anchors.jsonl.gz')
                with gzip.open(anchorfile,'wt') as anchors:
                    for stage,cache in [('validation',vc),('test',tc)]:
                        for c in cache:anchors.write(json.dumps({'stage':stage,'query_id':rows[c['query_index']]['record_id'],
                            'anchor_record_ids':[rows[int(i)]['record_id'] for i in c['anchor_indices']],
                            'distances':c['distances'].tolist(),'probe_log_ratios':c['probe_log_ratios'].tolist(),'tier':c['tier']},separators=(',',':'))+'\n')
                for mode in ['strict','probe']:
                    gm=GAMMA if mode=='probe' and device else [0.]
                    outputs={}
                    for relative in [False,True]:
                        method=('relative_' if relative else 'geometric_')+rule
                        trials=[]
                        for k in K:
                            for temp in TEMP:
                                for gamma in gm:
                                    vp,_=predict(vm,k,temp,gamma,relative)
                                    trials.append({'k':k,'temperature':temp,'probe_exponent':gamma,
                                        'validation_micro_mre':em.mre(rows,va,vp)})
                        chosen=min(trials,key=lambda x:(x['validation_micro_mre'],x['k'],x['temperature'],x['probe_exponent']))
                        vp,vs=predict(vm,chosen['k'],chosen['temperature'],chosen['probe_exponent'],relative)
                        # Config fixed before test prediction/scoring. No test labels enter predict().
                        pred,ts=predict(tm,chosen['k'],chosen['temperature'],chosen['probe_exponent'],relative)
                        threshold=float(np.quantile(vs,.9));keep=ts<=threshold
                        entry={k:split[k] for k in ['split_id','protocol','train_model_ids','validation_model_ids','test_model_ids','heldout_device_id'] if k in split}|{
                            'candidate_rule':rule,'method':method,'feature_mode':mode,'selected_validation_configuration':chosen,
                            'validation_grid_results':trials,'train_records':len(tr),'validation_records':len(va),
                            'train_record_ids_sha256':idhash(rows,tr),'test_record_ids_sha256':idhash(rows,te),
                            'anchors_path':anchorfile.name,'anchors_sha256':sha(anchorfile),
                            'test_metrics':metrics(rows,te,pred),'selective_threshold':threshold,
                            'triggered_count':int((~keep).sum()),'triggered_fraction':float((~keep).mean()),
                            'retained_metrics':metrics(rows,te[keep],pred[keep]) if keep.any() else None,
                            'attention_triggered_count':sum(not keep[j] for j,i in enumerate(te) if rows[int(i)]['family']=='Attention')}
                        reference=next(r for r in base['results'] if r['split_id']==split['split_id'] and r['feature_mode']==mode and r['method']==rule)
                        entry['v2_reference_row']=reference
                        if not relative:
                            delta=abs(reference['test_metrics']['micro_mre_percent']-entry['test_metrics']['micro_mre_percent'])
                            assert delta<1e-9,('baseline reproduction mismatch',delta)
                            entry['baseline_reproduction_max_metric_delta']=delta
                        result['results'].append(entry);outputs[relative]=pred
                        for j,(i,p) in enumerate(zip(te,pred)):
                            row={k:rows[int(i)][k] for k in ['record_id','model_id','device_id','family']}
                            row.update({'method':method,'feature_mode':mode,'split_id':split['split_id'],'protocol':split['protocol'],
                                'true_latency_ms':rows[int(i)]['latency_ms'],'prediction_ms':float(p),'anchor_log_spread':float(ts[j]),
                                'validation_spread_threshold':threshold,'trigger_measurement_offline':bool(not keep[j])})
                            stream.write(json.dumps(row,separators=(',',':'))+'\n')
                            agg[(method,mode,split['protocol'])].append((int(i),float(p)))
                        stream.flush();write(OUT/'results.partial.json',result)
                        print(json.dumps({'fold':split['split_id'],'method':method,'mode':mode,'MRE':entry['test_metrics']['micro_mre_percent']}),flush=True)
                    comparison.append((rule,mode,split['protocol'],te,outputs[False],outputs[True]))
    for (method,mode,protocol),items in agg.items():
        inds=np.array([i for i,p in items]);pred=np.array([p for i,p in items]);assert len(inds)==len(set(inds))
        result['aggregates'].append({'method':method,'feature_mode':mode,'protocol':protocol,'metrics':metrics(rows,inds,pred)})
    result['paired_cluster_bootstrap']=[]
    for rule in METHODS:
        for mode in ['strict','probe']:
            for protocol in sorted({s['protocol'] for s in splits}):
                vals=[x for x in comparison if x[:3]==(rule,mode,protocol)]
                groups=collections.defaultdict(list)
                for _,_,_,inds,a,b in vals:
                    for i,p,q in zip(inds,a,b):
                        r=rows[int(i)];y=r['latency_ms'];cl=r['model_id'] if protocol=='model_holdout' else r['device_id']
                        groups[cl].append(100*(abs(p-y)-abs(q-y))/y)
                ss=np.array([sum(v) for k,v in sorted(groups.items())]);nn=np.array([len(v) for k,v in sorted(groups.items())])
                rng=np.random.default_rng(BOOTSTRAP_SEED);idx=rng.integers(len(ss),size=(BOOTSTRAP_N,len(ss)))
                bs=ss[idx].sum(1)/nn[idx].sum(1)
                result['paired_cluster_bootstrap'].append({'candidate_rule':rule,'feature_mode':mode,'protocol':protocol,
                    'cluster_unit':'model_id' if protocol=='model_holdout' else 'device_id','clusters':len(ss),
                    'baseline_minus_new_micro_MRE_percentage_points':float(ss.sum()/nn.sum()),
                    'descriptive_percentile95_interval':np.quantile(bs,[.025,.975]).tolist(),
                    'seed':BOOTSTRAP_SEED,'bootstrap_resamples':BOOTSTRAP_N})
    result.update({'completed_utc':utc(),'elapsed_seconds':time.time()-t0,'predictions_sha256':sha(OUT/'predictions.jsonl.gz'),
        'all_v2_baselines_reproduced':True,'new_physical_measurements':0})
    write(OUT/'results.json',result);print(json.dumps({'complete':True,'elapsed_seconds':result['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
