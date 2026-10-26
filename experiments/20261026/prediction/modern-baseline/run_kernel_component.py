"""Frozen, semantic-subset nn-Meter RF component comparison; no hardware calls.

This is explicitly NOT a reproduction of nn-Meter fusion/adaptive profiling or
its full-model estimator. Official source bytes and their MIT license are kept.
"""
import sys
sys.dont_write_bytecode=True
import argparse, ast, csv, gzip, hashlib, importlib.util, json, math, pickle, platform, random, statistics, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import sklearn

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[2]
SOURCE=BASE/'v2-实证增强/public-reproduction-check'
FORMAL=SOURCE/'prediction/formal-five-stack'
OFFICIAL=OUT/'official-source'
SEEDS=[11,22,33,44,55]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def dump(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def csvout(p,rs):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

LIB=load_module('official_nnmeter_rf',OFFICIAL/'predictor_lib.py')
EVAL=load_module('official_nnmeter_formulas',OFFICIAL/'evaluation.py')
UTIL=load_module('official_nnmeter_padding',OFFICIAL/'kernel_utils.py')
TREE=ast.parse((OFFICIAL/'extract_feature.py').read_text())
FEATURE_FIELDS=next(ast.literal_eval(n.value) for n in TREE.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='feature_for_kernel' for t in n.targets))

def encode(kernel,config):
    feature=[float(config[k]) for k in FEATURE_FIELDS[kernel]]
    if kernel=='conv-block':flops,params=EVAL.get_conv_flop_params(config['HW'],config['CIN'],config['COUT'],config['KERNEL_SIZE'],config['STRIDES'])
    elif kernel=='dwconv-block':flops,params=EVAL.get_dwconv_flop_params(config['HW'],config['CIN'],config['KERNEL_SIZE'],config['STRIDES'])
    elif kernel=='fc':flops,params=EVAL.get_fc_flop_params(config['CIN'],config['COUT'])
    else:raise ValueError(kernel)
    return feature+[flops/2e6,params/1e6]

def compatible(node,attrs):
    module=attrs.get('module_class','')
    inputs=node.get('input_shapes',[]);outputs=node.get('output_shapes',[])
    if module not in ('torch.nn.modules.conv.Conv2d','torch.nn.modules.linear.Linear'):return None,'outside_Conv2d_Linear_modules'
    if len(inputs)!=1 or len(outputs)!=1:return None,'not_single_tensor_input_output'
    x=inputs[0]['shape'];y=outputs[0]['shape']
    if inputs[0]['dtype']!='torch.float32' or outputs[0]['dtype']!='torch.float32':return None,'dtype_not_float32'
    if attrs.get('bias_present')!=1:return None,'bias_absent_official_kernel_has_bias'
    if module.endswith('Linear'):
        if len(x)!=2 or x[0]!=1:return None,'FC_not_single_batch_vector'
        cin=int(attrs['in_features']);cout=int(attrs['out_features'])
        if x!=[1,cin] or y!=[1,cout]:return None,'FC_shape_mismatch'
        return ('fc',{'CIN':cin,'COUT':cout}),'included'
    if len(x)!=4 or len(y)!=4 or x[0]!=1 or y[0]!=1:return None,'Conv_not_batch1_NCHW'
    if x[2]!=x[3] or y[2]!=y[3] or attrs.get('input_channels_last',0)!=0:return None,'Conv_square_contiguous_NCHW_required'
    if attrs['kernel_h']!=attrs['kernel_w'] or attrs['stride_h']!=attrs['stride_w']:return None,'Conv_nonsquare_kernel_or_stride'
    if attrs['dilation_h']!=1 or attrs['dilation_w']!=1:return None,'Conv_dilation_not_one'
    cin=int(attrs['in_channels']);cout=int(attrs['out_channels']);groups=int(attrs['groups'])
    if groups==1:kernel='conv-block'
    elif groups==cin and cout==cin:kernel='dwconv-block'
    else:return None,'Conv_grouping_not_builtin_kernel'
    k=int(attrs['kernel_h']);s=int(attrs['stride_h']);hw=x[2];padding=UTIL.get_padding(k,s,hw)
    if attrs['padding_h']!=padding or attrs['padding_w']!=padding:return None,'Conv_padding_differs_from_official_generator'
    if y[1]!=cout or y[2]!=(hw-1)//s+1:return None,'Conv_output_shape_differs_from_official_generator'
    return (kernel,{'HW':hw,'CIN':cin,'COUT':cout,'KERNEL_SIZE':k,'STRIDES':s}),'included'

def build_inputs():
    summary=json.loads((FORMAL/'summary.json').read_text());rows=[];exclusions=[];sources=[]
    for src in summary['source_manifest']:
        rp=SOURCE/'evidence/physical_profiles'/Path(src['path']).name
        mp=SOURCE/'evidence/physical_profiles'/Path(src['metadata_sidecar']).name
        assert sha(rp)==src['sha256'] and sha(mp)==src['metadata_sidecar_sha256']
        sources.extend([rp,mp]);raw=json.loads(rp.read_text());meta=json.loads(mp.read_text())
        for g in raw['graphs']:
            records={r['op_instance_id']:r for r in g['records'] if r['status']=='measured'}
            for n in g['nodes']:
                if n['node_id'] not in records:continue
                r=records[n['node_id']];a=meta['graphs'][g['graph_id']]['nodes'][n['node_id']]
                comp,reason=compatible(n,a)
                if comp is None:
                    exclusions.append({'record_id':r['record_id'],'model_id':g['model_id'],'device_id':raw['hardware']['device_id'],'family':n['family'],'reason':reason});continue
                kernel,cfg=comp
                rows.append({'record_id':r['record_id'],'model_id':g['model_id'],'shape_id':g['shape_id'],
                             'device_id':raw['hardware']['device_id'],'kernel':kernel,'config':cfg,'features':encode(kernel,cfg),
                             'true_latency_ms':statistics.mean(r['latency_samples_ms']),
                             'input_weights_sha256':g['input_weights_sha256']})
    splits=[]
    for r in summary['results']:
        if r['method']=='mlp' and r['feature_mode']=='strict' and r['protocol']=='model_holdout':
            splits.append({k:r[k] for k in ['split_id','train_model_ids','validation_model_ids','test_model_ids']})
    return rows,exclusions,splits,sources

def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('protocol exists')
    rows,exclusions,splits,sources=build_inputs()
    dump(OUT/'compatible-queries.json',rows);csvout(OUT/'excluded-queries.csv',exclusions)
    coverage=[]
    for split in splits:
        assert not (set(split['train_model_ids'])&set(split['test_model_ids']))
        assert not (set(split['validation_model_ids'])&set(split['test_model_ids']))
        for device in sorted(set(r['device_id'] for r in rows)):
            for kernel in ['conv-block','dwconv-block','fc']:
                cell={'split_id':split['split_id'],'device_id':device,'kernel':kernel}
                for part in ['train','validation','test']:
                    selected=sorted(r['record_id'] for r in rows if r['device_id']==device and r['kernel']==kernel and r['model_id'] in split[part+'_model_ids'])
                    cell[part+'_records']=len(selected);cell[part+'_record_ids']=selected
                cell['status']='execute' if cell['test_records'] and cell['train_records'] else 'no_test_support' if not cell['test_records'] else 'no_training_support'
                coverage.append(cell)
    dump(OUT/'frozen-splits-and-support.json',{'splits':splits,'cells':coverage})
    inputs=sources+[FORMAL/'summary.json',FORMAL/'predictions.jsonl.gz',Path(__file__),OUT/'compatible-queries.json',OUT/'excluded-queries.csv',OUT/'frozen-splits-and-support.json']+sorted(OFFICIAL.glob('*'))
    inputs=[p for p in inputs if p.is_file()]
    p={'kind':'prospectively_frozen_retrospective_nn_meter_kernel_component_adaptation','created_utc':datetime.now(timezone.utc).isoformat(),
       'official_repository':'microsoft/nn-Meter','official_commit':'cd8dab49b735d58d03746141f73ef5934559ae68','official_license':'MIT, retained verbatim',
       'publications':['https://github.com/microsoft/nn-Meter','https://www.microsoft.com/en-us/research/publication/nn-meter-towards-accurate-latency-prediction-of-dnn-inference-on-diverse-edge-devices/'],
       'baseline_name':'nn_meter_official_rf_component_adapted','not_full_method_reproduction':True,
       'feature_contract':{'conv-block':FEATURE_FIELDS['conv-block']+['official_flops/2e6','official_params/1e6'],
                           'dwconv-block':FEATURE_FIELDS['dwconv-block']+['official_flops/2e6','official_params/1e6'],
                           'fc':FEATURE_FIELDS['fc']+['official_flops/2e6','official_params/1e6']},
       'structural_filter':'Only measured FP32 bias=True Conv2d/Linear records matching official single-input batch1 generators. Conv requires square NCHW, dilation1, official padding/output, groups1 or depthwise multiplier1; FC requires shape [1,CIN] -> [1,COUT]. No filter uses latency, error, or prior method outcomes.',
       'split':'Original six V2 source-model folds; per-stack/per-kernel fits, train/validation/test source model membership unchanged. Validation records are preserved but not used by either fixed estimator. No test tuning.',
       'rf':'Call verbatim official init_predictor(kernel, het_pytorch_eager). It takes official generic fallback: 370 trees, max_depth50, min_leaf1, min_split2, oob=True. Set random_state to each frozen seed, n_jobs1. Translate legacy sklearn max_features=auto to mathematically equivalent 1.0; no other parameter change.',
       'matched_control':'New MLP on exactly the same per-stack/per-kernel features and training labels; two hidden128 GELU layers; train-only x mean/population SD; log-y mean/population SD floor.05; AdamW lr.003 weight_decay.0001; gradient clip5; exactly80 epochs; fixed final epoch, no validation/test selection. Seed11/22/33/44/55. The MLP is a new same-information control, not the archived V2 MLP.',
       'calibration_probe_budget':0,'new_hardware_measurements':0,'training_device':'local CPU only','seeds':SEEDS,
       'included_queries':len(rows),'excluded_queries':len(exclusions),'exclusion_reason_counts':dict(Counter(r['reason'] for r in exclusions)),
       'included_per_stack':dict(Counter(r['device_id'] for r in rows)),'included_per_kernel':dict(Counter(r['kernel'] for r in rows)),
       'included_source_models':sorted(set(r['model_id'] for r in rows)),
       'executable_cells':sum(c['status']=='execute' for c in coverage),'total_cells_including_empty':len(coverage),
       'reference_controls':'Archived strict V2 ridge/MLP/GCN/RGAT scores are recomputed only on the exact executable test-query support, with the same source-model folds and five seeds. Those models were trained on more records/all stacks/other features; the reference comparison is not an equal-information causal test.',
       'metrics':['micro_MRE_percent','MAE_us','WAPE_percent','RMSE_ms','RMSPE_percent','accuracy_within5percent','accuracy_within10percent'],
       'report_all':'Retain every fold, seed, kernel, stack, exclusion and worse result; no errors dropped.',
       'cannot_claim':['Complete nn-Meter pipeline reproduction: no target-backend fusion discovery or adaptive kernel sampling is run.','Full-model latency accuracy: only isolated operator timings are scored.','Unknown-device transfer: every RF receives same-stack training labels.','Original four-test identities, original RGAT14.3, original five-hardware LOPO, HELP or BRP-NAS reproduction.'],
       'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'platform':platform.platform()},
       'sources':[{'path':str(f.relative_to(BASE)),'sha256':sha(f),'bytes':f.stat().st_size} for f in inputs]}
    dump(OUT/'protocol.json',p)
    print(json.dumps({k:p[k] for k in ['included_queries','excluded_queries','included_per_stack','included_per_kernel','included_source_models','executable_cells','total_cells_including_empty']}),flush=True)

def network(dim):
    return torch.nn.Sequential(torch.nn.Linear(dim,128),torch.nn.GELU(),torch.nn.Linear(128,128),torch.nn.GELU(),torch.nn.Linear(128,1))

def fit_mlp(x,y,seed):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    tx=torch.tensor(x,dtype=torch.float32);ty=torch.tensor(y,dtype=torch.float32)
    mu=tx.mean(0);sd=tx.std(0,unbiased=False);sd=torch.where(sd<1e-6,torch.ones_like(sd),sd)
    logy=ty.log();ym=logy.mean();ys=logy.std(unbiased=False).clamp_min(.05);z=(logy-ym)/ys
    net=network(tx.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=1e-4)
    tx=(tx-mu)/sd
    for epoch in range(80):
        opt.zero_grad();loss=torch.nn.functional.mse_loss(net(tx).squeeze(-1),z);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
    return {'state':net.state_dict(),'mean':mu,'sd':sd,'ymean':ym,'ysd':ys,'input_dim':tx.shape[1],'epochs':80,'seed':seed}

def predict_mlp(fit,x):
    net=network(fit['input_dim']);net.load_state_dict(fit['state']);net.eval()
    with torch.no_grad():
        tx=(torch.tensor(x,dtype=torch.float32)-fit['mean'])/fit['sd'];return torch.exp((net(tx).squeeze(-1)*fit['ysd']+fit['ymean']).clamp(-20,20)).numpy()

def metric(rs):
    y=np.array([r['true_latency_ms'] for r in rs]);p=np.array([r['prediction_ms'] for r in rs]);ae=np.abs(p-y);re=ae/y
    return {'n_queries':len(rs),'micro_MRE_percent':float(re.mean()*100),'MAE_us':float(ae.mean()*1000),'WAPE_percent':float(ae.sum()/y.sum()*100),
            'RMSE_ms':float(np.sqrt(np.mean((p-y)**2))),'RMSPE_percent':float(np.sqrt(np.mean(re**2))*100),
            'accuracy_within5percent':float(np.mean(re<=.05)*100),'accuracy_within10percent':float(np.mean(re<=.1)*100)}

def run():
    p=json.loads((OUT/'protocol.json').read_text())
    for f in p['sources']:assert sha(BASE/f['path'])==f['sha256'],f['path']
    result=OUT/'results';result.mkdir(exist_ok=False);(result/'checkpoints').mkdir()
    rows=json.loads((OUT/'compatible-queries.json').read_text());byid={r['record_id']:r for r in rows}
    cells=json.loads((OUT/'frozen-splits-and-support.json').read_text())['cells'];torch.set_num_threads(1)
    predictions=[];fits=[];support=set();start=time.monotonic()
    for cell_idx,c in enumerate(cells):
        if c['status']!='execute':continue
        train=[byid[i] for i in c['train_record_ids']];test=[byid[i] for i in c['test_record_ids']]
        support.update(c['test_record_ids']);x=np.array([r['features'] for r in train]);y=np.array([r['true_latency_ms'] for r in train]);xt=np.array([r['features'] for r in test])
        for seed in SEEDS:
            prefix={'split_id':c['split_id'],'device_id':c['device_id'],'kernel':c['kernel'],'seed':seed}
            for method in ['nn_meter_official_rf_component_adapted','v7_same_information_mlp']:
                t=time.monotonic()
                if method.startswith('nn_meter'):
                    model=LIB.init_predictor(c['kernel'],'het_pytorch_eager');model.set_params(max_features=1.0,random_state=seed,n_jobs=1);model.fit(x,y);pred=model.predict(xt)
                    path=result/'checkpoints'/f'cell{cell_idx}-seed{seed}-rf.pkl'
                    with path.open('wb') as f:pickle.dump(model,f,protocol=5)
                    with path.open('rb') as f:reloaded=pickle.load(f)
                    assert np.array_equal(reloaded.predict(xt),pred)
                    details={'parameters':model.get_params(),'oob_score':float(model.oob_score_)}
                else:
                    model=fit_mlp(x,y,seed);pred=predict_mlp(model,xt);path=result/'checkpoints'/f'cell{cell_idx}-seed{seed}-mlp.pt';torch.save(model,path)
                    reloaded=torch.load(path,map_location='cpu',weights_only=True);assert np.array_equal(predict_mlp(reloaded,xt),pred);details={'epochs':80,'hidden':[128,128]}
                assert np.all(np.isfinite(pred)) and np.all(pred>0)
                fits.append({**prefix,'method':method,'train_records':len(train),'validation_records_preserved_but_unused':c['validation_records'],'test_records':len(test),
                             'checkpoint':str(path.relative_to(OUT)),'checkpoint_sha256':sha(path),'fit_and_checkpoint_replay_seconds':time.monotonic()-t,**details})
                for r,estimate in zip(test,pred):
                    predictions.append({**prefix,'record_id':r['record_id'],'model_id':r['model_id'],'method':method,
                                        'true_latency_ms':r['true_latency_ms'],'prediction_ms':float(estimate)})
        print(json.dumps({'complete_cells':cell_idx+1,'total_fits':len(fits),'elapsed_seconds':time.monotonic()-start}),flush=True)
    dump(result/'fit-receipts.json',fits)
    # Historical models are clearly labelled as richer-training references.
    with gzip.open(FORMAL/'predictions.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['protocol']!='model_holdout' or r['feature_mode']!='strict' or r['method'] not in ['ridge','mlp','gcn','rgat'] or r['record_id'] not in support:continue
            q=byid[r['record_id']];assert abs(q['true_latency_ms']-r['true_latency_ms'])<1e-12
            predictions.append({'split_id':r['split_id'],'device_id':r['device_id'],'kernel':q['kernel'],'seed':r['seed'],
                                'record_id':r['record_id'],'model_id':r['model_id'],'method':'V2_archived_'+r['method']+'_richer_training',
                                'true_latency_ms':r['true_latency_ms'],'prediction_ms':r['prediction_ms']})
    csvout(result/'predictions.csv',predictions)
    grouped=defaultdict(list)
    for r in predictions:
        for scope in ['pooled','per_stack','per_kernel','per_fold']:
            value={'pooled':'all','per_stack':r['device_id'],'per_kernel':r['kernel'],'per_fold':r['split_id']}[scope]
            grouped[(scope,value,r['method'],r['seed'])].append(r)
    metrics=[]
    for (scope,value,method,seed),rs in sorted(grouped.items()):
        ids=[r['record_id'] for r in rs];assert len(ids)==len(set(ids))
        metrics.append({'scope':scope,'scope_value':value,'method':method,'seed':seed,
                        'support_sha256':hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),**metric(rs)})
    bysupport=defaultdict(set)
    for r in metrics:bysupport[(r['scope'],r['scope_value'])].add(r['support_sha256'])
    assert all(len(v)==1 for v in bysupport.values())
    csvout(result/'per-seed-metrics.csv',metrics)
    aggregates=[]
    for scope,value,method in sorted(set((r['scope'],r['scope_value'],r['method']) for r in metrics)):
        rs=[r for r in metrics if (r['scope'],r['scope_value'],r['method'])==(scope,value,method)];assert len(rs)==5
        row={'scope':scope,'scope_value':value,'method':method,'n_seeds':5,'n_queries':rs[0]['n_queries']}
        for k in metric(predictions[:1]):
            if k=='n_queries':continue
            v=[r[k] for r in rs];row[k+'_seed_mean']=statistics.mean(v);row[k+'_seed_sd']=statistics.stdev(v)
        aggregates.append(row)
    csvout(result/'aggregate-metrics.csv',aggregates)
    report={'complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.json'),
            'fits':len(fits),'prediction_rows_including_archived_references':len(predictions),'unique_query_support':len(support),
            'compatible_queries_without_executable_train_support':sorted(set(byid)-support),'all_checkpoint_prediction_replays_exact':True,
            'all_six_methods_same_scoring_support':True,'wall_seconds':time.monotonic()-start,'new_hardware_measurements':0,
            'pooled_results':[r for r in aggregates if r['scope']=='pooled'],
            'boundary':p['cannot_claim'],'outputs':[{'path':str(f.relative_to(OUT)),'sha256':sha(f)} for f in sorted(result.glob('*')) if f.is_file()]}
    dump(result/'execution-audit.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args();freeze() if a.freeze else run()
