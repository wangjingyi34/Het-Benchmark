"""No-training audit independent of the fitting/reporting implementations.

Rebuilds all physical means/features/splits, scalar metrics and raw-tensor
checkpoint forward/adaptation paths without importing either experiment script.
"""
import csv,gzip,hashlib,json,math,statistics,sys
from collections import Counter,defaultdict,OrderedDict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
torch.set_num_threads(1);sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;BASE=OUT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8)
def ranks(v):
    ordered=sorted(range(len(v)),key=lambda i:v[i]);r=[0.]*len(v);j=0
    while j<len(v):
        k=j+1
        while k<len(v) and v[ordered[k]]==v[ordered[j]]:k+=1
        for t in range(j,k):r[ordered[t]]=(j+k-1)/2+1
        j=k
    return r
def scalar_metric(rr):
    y=[r['true_latency_ms'] for r in rr];p=[r['prediction_ms'] for r in rr];ae=[abs(a-b) for a,b in zip(y,p)];re=[a/b for a,b in zip(ae,y)]
    sy=ranks(y);sp=ranks(p);my=statistics.mean(sy);mp=statistics.mean(sp);dy=[a-my for a in sy];dp=[a-mp for a in sp];den=math.sqrt(math.fsum(a*a for a in dy)*math.fsum(a*a for a in dp));rho=math.fsum(a*b for a,b in zip(dy,dp))/den if den else None
    return {'micro_MRE_percent':100*statistics.mean(re),'MAE_us':1000*statistics.mean(ae),'WAPE_percent':100*math.fsum(ae)/math.fsum(y),'RMSE_ms':math.sqrt(statistics.mean(a*a for a in ae)),'RMSPE_percent':100*math.sqrt(statistics.mean(a*a for a in re)),'Spearman':rho,'accuracy_within10percent':100*sum(a<=.1 for a in re)/len(re)}

def network(x,h,p):
    def layer(x,name):return F.linear(x,p['meta_learner.'+name+'.weight'],p['meta_learner.'+name+'.bias'])
    op=F.relu(layer(F.relu(layer(x,'fc1')),'fc2'))
    hw=h.repeat(len(x),1);hw=F.relu(layer(F.relu(layer(hw,'fc_hw1')),'fc_hw2'))
    joint=torch.cat([op,hw],dim=-1)
    return layer(F.relu(layer(F.relu(layer(joint,'fc3')),'fc4')),'fc5')

def independent_predict(ck,method,features,probe_features,probe_labels):
    norm=ck['normalization'];cfg=ck['configuration']
    x=(torch.tensor(features,dtype=torch.float32)-norm['x_mean'])/norm['x_sd']
    xp=(torch.tensor(probe_features,dtype=torch.float32)-norm['x_mean'])/norm['x_sd']
    py=torch.tensor(probe_labels,dtype=torch.float32);h=(py-py.min())/(py.max()-py.min()).clamp_min(1e-12)
    yp=(torch.tensor([math.log(y) for y in probe_labels],dtype=torch.float32).reshape(-1,1)-norm['y_mean'])/norm['y_sd']
    if 'zero_probe' in method:h=ck['source_mean_embedding']
    if method.startswith('help'):
        state=ck['help'];params=OrderedDict((name[6:],v.detach().clone().requires_grad_(True)) for name,v in state.items() if name.startswith('model.'))
        if 'no_z' not in method:
            z=F.linear(F.relu(F.linear(h,state['inference.z_encoder.0.weight'],state['inference.z_encoder.0.bias'])),state['inference.z_encoder.2.weight'],state['inference.z_encoder.2.bias'])
            width=2*cfg['layer_size'];z=z[:(width+1)*3];zw=z[:width*3];zb=z[width*3:]
            for index,layer in enumerate(['fc3','fc4','fc5']):
                wn='meta_learner.'+layer+'.weight';bn='meta_learner.'+layer+'.bias'
                params[wn]=params[wn]*(1+cfg['z_scaling']*zw[index*width:(index+1)*width]);params[bn]=params[bn]+cfg['z_scaling']*zb[index]
        n=0 if 'zero_probe' in method or 'no_inner' in method else cfg['inner_updates']
        for _ in range(n):
            loss=F.mse_loss(network(xp,h,params),yp);grads=torch.autograd.grad(loss,tuple(params.values()))
            params=OrderedDict((name,v-state['alpha.'+name.replace('.','__')]*g) for (name,v),g in zip(params.items(),grads))
    else:
        params=OrderedDict((name,v.detach().clone().requires_grad_(True)) for name,v in ck['pooled'].items())
        for _ in range(cfg['inner_updates'] if 'finetuned' in method else 0):
            loss=F.mse_loss(network(xp,h,params),yp);grads=torch.autograd.grad(loss,tuple(params.values()))
            params=OrderedDict((name,v-cfg['pooled_inner_lr']*g) for (name,v),g in zip(params.items(),grads))
    with torch.no_grad():return torch.exp((network(x,h,params).reshape(-1)*norm['y_sd']+norm['y_mean']).clamp(-20,20)).numpy()

def main():
    proto=json.loads((OUT/'protocol.json').read_text());summary=json.loads((OUT/'results/summary.json').read_text());methods=proto['methods'];seeds=proto['seeds'];truth={}
    for s in proto['sources']:assert sha(BASE/s['path'])==s['sha256'],s['path']
    fc=proto['feature_contract']
    for src in proto['sources']:
        if '/physical_profiles/' not in src['path'] or '.metadata.' in src['path']:continue
        p=BASE/src['path'];raw=json.loads(p.read_text());meta=json.loads(p.with_name(p.stem+'.metadata.json').read_text())
        for g in raw['graphs']:
            nd={n['node_id']:n for n in g['nodes']}
            for rec in g['records']:
                if rec['status']!='measured':continue
                n=nd[rec['op_instance_id']];a=meta['graphs'][g['graph_id']]['nodes'][n['node_id']]
                features=[float(n['family']==f) for f in fc['family_one_hot']]+[math.log1p(float(n['features'].get(k,0))) for k in fc['log1p_numeric']]+[math.log1p(float(a.get(k,0))) for k in fc['log1p_structural']]
                truth[rec['record_id']]={'latency':statistics.mean(rec['latency_samples_ms']),'features':features,'model':g['model_id'],'device':raw['hardware']['device_id'],'input_weights_sha256':g['input_weights_sha256']}
    assert len(truth)==13905
    with gzip.open(OUT/'operator-records.jsonl.gz','rt') as f:data=[json.loads(x) for x in f]
    byid={r['record_id']:r for r in data};assert set(byid)==set(truth)
    for r in data:
        t=truth[r['record_id']];assert r['features']==t['features'] and close(r['true_latency_ms'],t['latency']);assert (r['model_id'],r['device_id'],r['input_weights_sha256'])==(t['model'],t['device'],t['input_weights_sha256'])
    probes=json.loads((OUT/'independent-probes.json').read_text());assert len(probes)==15 and not set(byid)&{p['record_id'] for p in probes}
    for p in probes:assert p['true_latency_ms']==statistics.mean(p['raw_samples_ms'])
    cells=json.loads((OUT/'frozen-splits.json').read_text());assert len(cells)==30
    alltests=[];groups=defaultdict(list);fitcount=0;replayed=0;maxpred=0.;maxnorm=0.;count=0;testcount=0;allpred=set();zero_tests=0
    for c in cells:
        assert len(c['train_stacks'])==3 and len(set(c['train_stacks']+[c['validation_stack'],c['test_stack']]))==5
        assert set(c['train_model_ids']).isdisjoint(c['test_model_ids']) and set(c['validation_model_ids']).isdisjoint(c['test_model_ids']) and set(c['train_model_ids']).isdisjoint(c['validation_model_ids'])
        for phase,ds,ms in [('train',c['train_stacks'],c['train_model_ids']),('validation',[c['validation_stack']],c['validation_model_ids']),('test',[c['test_stack']],c['test_model_ids'])]:
            ids={r['record_id'] for r in data if r['device_id'] in ds and r['model_id'] in ms};assert ids==set(c[phase+'_record_ids']);assert len(ids)==c[phase+'_n']
        alltests+=c['test_record_ids']
        normrows=[byid[i] for i in c['train_record_ids']]+[p for p in probes if p['device_id'] in c['train_stacks']]
        xx=torch.tensor([r['features'] for r in normrows],dtype=torch.float32);yy=torch.tensor([math.log(r['true_latency_ms']) for r in normrows],dtype=torch.float32);sd=xx.std(0,unbiased=False);sd=torch.where(sd<1e-6,torch.ones_like(sd),sd)
        norm={'x_mean':xx.mean(0),'x_sd':sd,'y_mean':yy.mean(),'y_sd':yy.std(unbiased=False).clamp_min(.05)}
        for seed in seeds:
            d=OUT/'runs'/f'{c["cell_id"]}-seed{seed}';receipt=json.loads((d/'receipt.json').read_text());assert receipt['complete'] and receipt['training_episodes']==1000
            for s in receipt['outputs']:assert sha(d/s['path'])==s['sha256']
            ck=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=True);fitcount+=2;assert ck['seed']==seed and ck['cell_id']==c['cell_id'] and ck['protocol_sha256']==sha(OUT/'protocol.json')
            assert ck['configuration']==proto['configuration'] and ck['train_record_ids_sha256']==c['train_sha256']
            for key,value in norm.items():
                delta=float(torch.max(torch.abs(value-ck['normalization'][key])));maxnorm=max(maxnorm,delta);assert delta==0
            with gzip.open(d/'predictions.jsonl.gz','rt') as f:rr=[json.loads(x) for x in f]
            assert len(rr)==receipt['prediction_rows'];local=defaultdict(list)
            for r in rr:
                assert r['seed']==seed and r['cell_id']==c['cell_id'];assert close(r['true_latency_ms'],truth[r['record_id']]['latency']);assert r['prediction_ms']>0 and math.isfinite(r['prediction_ms'])
                assert r['record_id'] in c[r['phase']+'_record_ids'];assert r['target_probe_budget']==(0 if 'zero_probe' in r['method'] else 3)
                key=(r['phase'],r['method'],seed,r['record_id']);assert key not in allpred;allpred.add(key);count+=1;testcount+=r['phase']=='test';local[(r['phase'],r['method'])].append(r)
                for scope,val in [('pooled','all'),('per_stack',r['device_id']),('per_model',r['model_id']),('per_family',r['family']),('per_cell',r['cell_id'])]:groups[(r['phase'],scope,val,r['method'],seed)].append(r)
            for phase,dev in [('validation',c['validation_stack']),('test',c['test_stack'])]:
                pp=sorted([p for p in probes if p['device_id']==dev],key=lambda r:['matmul256','conv2d16','layernorm128'].index(r['probe_name']))
                for method in methods:
                    part=local[(phase,method)];assert {r['record_id'] for r in part}==set(c[phase+'_record_ids'])
                    features=[byid[r['record_id']]['features'] for r in part]
                    values=independent_predict(ck,method,features,[p['features'] for p in pp],[p['true_latency_ms'] for p in pp])
                    old=np.array([r['prediction_ms'] for r in part]);delta=float(np.max(np.abs(values-old)));maxpred=max(maxpred,delta);assert delta==0,(c['cell_id'],seed,method,delta);replayed+=len(part)
                    if 'zero_probe' in method:
                        poisoned=independent_predict(ck,method,features[:2],[p['features'] for p in pp],[1000*p['true_latency_ms'] for p in pp])
                        clean=independent_predict(ck,method,features[:2],[p['features'] for p in pp],[p['true_latency_ms'] for p in pp]);assert np.array_equal(poisoned,clean);zero_tests+=1
    assert fitcount==300 and len(alltests)==len(set(alltests))==13905 and testcount==13905*5*7
    with (OUT/'results/per-seed-metrics.csv').open() as f:mr=list(csv.DictReader(f))
    maxdiff=defaultdict(float)
    for row in mr:
        rr=groups[(row['phase'],row['scope'],row['scope_value'],row['method'],int(row['seed']))];ids=sorted(r['record_id'] for r in rr);assert len(ids)==int(row['n_queries']) and len(ids)==len(set(ids));assert hashlib.sha256('\n'.join(ids).encode()).hexdigest()==row['support_sha256']
        expected=scalar_metric(rr)
        for field,val in expected.items():
            if val is None:assert row[field]=='';continue
            old=float(row[field]);maxdiff[field]=max(maxdiff[field],abs(old-val));assert close(old,val),(field,val,old)
    with (OUT/'results/aggregate-metrics.csv').open() as f:ar=list(csv.DictReader(f))
    for r in ar:
        xs=[x for x in mr if all(x[k]==r[k] for k in ['phase','scope','scope_value','method'])];assert len(xs)==5
        for field in maxdiff:
            vals=[float(x[field]) for x in xs if x[field]!='']
            if vals:
                assert close(statistics.mean(vals),float(r[field+'_seed_mean']));assert close(statistics.stdev(vals),float(r[field+'_seed_sd']))
    report={'complete':True,'utc':datetime.now(timezone.utc).isoformat(),'audit_code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'summary_sha256':sha(OUT/'results/summary.json'),'source_hashes_verified':len(proto['sources']),'physical_operator_means_and_features_recomputed':13905,'independent_reference_records_checked':15,'model_stack_cells_verified':30,'both_model_and_stack_partitions_disjoint':True,'all_300_training_fits_verified':fitcount,'all_150_paired_checkpoints_reloaded':fitcount//2,'raw_tensor_forward_and_inner_gradient_prediction_replays':replayed,'maximum_absolute_replay_difference_ms':maxpred,'maximum_source_only_normalization_difference':maxnorm,'zero_probe_poisoning_checks':zero_tests,'all_prediction_rows':count,'test_prediction_rows':testcount,'per_seed_metric_rows_independently_recomputed':len(mr),'aggregate_rows_recomputed':len(ar),'maximum_scalar_metric_differences':dict(maxdiff),'no_training_or_new_measurements_in_audit':True,'method':'Independent raw-tensor reconstruction of official dense encoder, deterministic z modulation and two learned/fixed inner gradient steps; no import of fitting/report scripts. Scalar metrics use Python statistics/math, including independent average-tie ranking.','bootstrap_scope':'Independent auditor checks base metrics and checkpoints; model-cluster percentile draws are reproducible in separate analysis code, not independently regenerated here.','failure_files':[str(p.relative_to(OUT)) for p in (OUT/'runs').glob('*/failure*.json')]}
    (OUT/'results/independent-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
