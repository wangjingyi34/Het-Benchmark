"""Pinned official HELP mechanisms adapted to contextual operator latency.

No hardware measurement occurs here. --selfcheck uses only synthetic data.
--freeze locks data/splits/code before --run. --run resumes only complete,
hash-matching cells, and never selects checkpoints from target outcomes.
Official source remains verbatim under official-help, with its MIT license.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, copy, csv, gzip, hashlib, importlib.util, json, math, platform
import random, statistics, sys, time, traceback
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]
SOURCE=BASE/'v2-实证增强/public-reproduction-check'
OFFICIAL=OUT/'official-help'
sys.dont_write_bytecode=True
sys.path.insert(0,str(OFFICIAL))
spec=importlib.util.spec_from_file_location('official_help_net',OFFICIAL/'net.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
torch.set_num_threads(1)
SEEDS=[11,22,33,44,55]
FAMILIES=['Conv2D','Conv1D','MatMul','Attention','Normalization','Softmax','Embedding','Pooling','Dropout','Activation','View','Reduction','Elementwise','Composite','Other','Context']
NUMERIC=['input_numel','output_numel','parameter_numel','input_bytes','output_bytes','rank','dim0','dim1','dim2','dim3']
ATTRS=['in_channels','out_channels','groups','kernel_h','kernel_w','stride_h','stride_w','padding_h','padding_w','dilation_h','dilation_w','in_features','out_features','num_embeddings','embedding_dim','normalized_dim','eps','attention_heads','attention_head_dim','is_causal','dropout_p','bias_present','training_flag','input_channels_last','composite_leaf']
PROBE_NAMES=['matmul256','conv2d16','layernorm128']
METHODS=['help_zero_probe_source_embedding','help_three_probe_full','help_three_probe_no_inner_update','help_three_probe_no_z_modulation','pooled_zero_probe','pooled_three_probe_conditioned','pooled_three_probe_finetuned']
CFG={'layer_size':32,'episodes':1000,'source_tasks_per_episode':3,'query_batch_size':64,'inner_updates':2,'initial_meta_sgd_lr':.001,'pooled_inner_lr':.001,'outer_lr':.0001,'z_scaling':.01,'kl_scaling':.1,'deterministic_z':True,'second_order':True,'gradient_clip_norm':10.0,'training_device':'cpu','torch_threads':1,'checkpoint_selection':'fixed final episode; validation descriptive only'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def writejson(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def now():return datetime.now(timezone.utc).isoformat()

def encode(node,attrs):
    v=[float(node['family']==f) for f in FAMILIES]
    v += [math.log1p(float(node['features'].get(k,0))) for k in NUMERIC]
    v += [math.log1p(float(attrs.get(k,0))) for k in ATTRS]
    assert all(math.isfinite(x) for x in v)
    return v

def probe_features():
    # Exact functions/inputs from frozen collect_profiles.hardware, no new timing.
    definitions=[('MatMul',[[256,256],[256,256]],[256,256],{}),
        ('Conv2D',[[1,16,32,32],[16,16,3,3]],[1,16,32,32],{'in_channels':16,'out_channels':16,'groups':1,'kernel_h':3,'kernel_w':3,'stride_h':1,'stride_w':1,'padding_h':1,'padding_w':1,'dilation_h':1,'dilation_w':1}),
        ('Normalization',[[128,128]],[128,128],{'normalized_dim':128,'eps':1e-5})]
    rows=[]
    for name,(fam,inputs,out,attrs) in zip(PROBE_NAMES,definitions):
        dims=inputs[0][:4]+[0]*(4-len(inputs[0]))
        n={'family':fam,'features':{'input_numel':sum(math.prod(s) for s in inputs),'output_numel':math.prod(out),'parameter_numel':0,'input_bytes':4*sum(math.prod(s) for s in inputs),'output_bytes':4*math.prod(out),'rank':len(inputs[0]),**dict(zip(['dim0','dim1','dim2','dim3'],dims))}}
        rows.append({'name':name,'family':fam,'input_shapes':inputs,'output_shape':out,'parameters':0,'attributes':attrs,'features':encode(n,attrs),'semantics':'functional operation; weight tensors are explicit inputs, not module parameters; FP32; no bias/affine parameters'})
    return rows

def collect():
    summary=json.loads((SOURCE/'prediction/formal-five-stack/summary.json').read_text())
    rows=[];probes=[];sources=[]
    pfeatures=probe_features()
    for src in summary['source_manifest']:
        rp=SOURCE/'evidence/physical_profiles'/Path(src['path']).name
        mp=SOURCE/'evidence/physical_profiles'/Path(src['metadata_sidecar']).name
        assert sha(rp)==src['sha256'] and sha(mp)==src['metadata_sidecar_sha256']
        sources += [rp,mp]
        raw=json.loads(rp.read_text());meta=json.loads(mp.read_text());device=raw['hardware']['device_id']
        assert [p['name'] for p in raw['hardware']['probes']]==PROBE_NAMES
        for p,feat in zip(raw['hardware']['probes'],pfeatures):
            assert len(p['latency_samples_ms'])>=20 and math.isclose(statistics.mean(p['latency_samples_ms']),p['latency_mean_ms'],rel_tol=1e-12)
            probes.append({'record_id':device+'/independent-hardware-probe/'+p['name'],'device_id':device,'probe_name':p['name'],'features':feat['features'],'true_latency_ms':statistics.mean(p['latency_samples_ms']),'raw_samples_ms':p['latency_samples_ms'],'inputs_seed':p['inputs_seed'],'source':str(rp.relative_to(BASE))})
        for g in raw['graphs']:
            nodes={n['node_id']:n for n in g['nodes']}
            for r in g['records']:
                if r['status']!='measured':continue
                n=nodes[r['op_instance_id']];a=meta['graphs'][g['graph_id']]['nodes'][n['node_id']]
                rows.append({'record_id':r['record_id'],'device_id':device,'model_id':g['model_id'],'shape_id':g['shape_id'],'node_id':n['node_id'],'family':n['family'],'features':encode(n,a),'true_latency_ms':statistics.mean(r['latency_samples_ms']),'input_weights_sha256':g['input_weights_sha256'],'n_timing_samples':len(r['latency_samples_ms']),'source':str(rp.relative_to(BASE))})
    assert len(rows)==13905 and len(probes)==15
    splits=[]
    for r in summary['results']:
        if r['method']=='mlp' and r['feature_mode']=='strict' and r['protocol']=='model_holdout':
            splits.append({k:r[k] for k in ['split_id','train_model_ids','validation_model_ids','test_model_ids']})
    assert len(splits)==6
    return rows,probes,splits,sources

def freeze():
    assert not (OUT/'protocol.json').exists(),'protocol already frozen'
    rows,probes,model_splits,sources=collect();devices=sorted({r['device_id'] for r in rows});cells=[]
    for di,target in enumerate(devices):
        valid=devices[(di+1)%len(devices)];train=[d for d in devices if d not in [target,valid]]
        for s in model_splits:
            c={'cell_id':f'stack{di}-{s["split_id"]}','test_stack':target,'validation_stack':valid,'train_stacks':train,**s}
            for phase,ds,ms in [('train',train,s['train_model_ids']),('validation',[valid],s['validation_model_ids']),('test',[target],s['test_model_ids'])]:
                ids=sorted(r['record_id'] for r in rows if r['device_id'] in ds and r['model_id'] in ms)
                c[phase+'_record_ids']=ids;c[phase+'_n']=len(ids);c[phase+'_sha256']=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
            assert not set(c['train_model_ids'])&set(c['test_model_ids'])
            assert not set(c['validation_model_ids'])&set(c['test_model_ids'])
            assert not set(c['train_stacks'])&{target,valid}
            assert not set(c['train_record_ids'])&set(c['test_record_ids'])
            cells.append(c)
    testids=[i for c in cells for i in c['test_record_ids']]
    assert len(testids)==len(set(testids))==13905
    assert not set(testids)&{p['record_id'] for p in probes}
    with gzip.open(OUT/'operator-records.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    writejson(OUT/'independent-probes.json',probes);writejson(OUT/'probe-semantics.json',probe_features());writejson(OUT/'frozen-splits.json',cells)
    sources += [SOURCE/'prediction/formal-five-stack/summary.json',BASE/'v2-实证增强/prediction/collect_profiles.py',Path(__file__)]
    sources += list(OFFICIAL.glob('*'))+[OUT/'operator-records.jsonl.gz',OUT/'independent-probes.json',OUT/'probe-semantics.json',OUT/'frozen-splits.json',OUT/'engineering-selfcheck.json']
    sources=[p for p in sources if p.is_file()]
    protocol={'kind':'prospective_code_and_split_freeze_for_retrospective_HELP_operator_domain_experiment','created_utc':now(),'official_repository':'https://github.com/HayeonLee/HELP','official_commit':'45f70eb92d0d735970a33b608d1544ad3c52f285','paper':'https://proceedings.neurips.cc/paper/2021/file/e3251075554389fe91d17a794861d47b-Paper.pdf','seeds':SEEDS,'configuration':CFG,'n_operator_records':13905,'n_unique_operator_configurations':2781,'n_source_models':12,'n_stacks':5,'crossed_model_stack_cells':30,'n_HELP_fits':150,'n_pooled_control_fits':150,'n_independent_probe_records':15,'methods':METHODS,'feature_contract':{'family_one_hot':FAMILIES,'log1p_numeric':NUMERIC,'log1p_structural':ATTRS,'dimension':len(FAMILIES)+len(NUMERIC)+len(ATTRS),'excluded':['latency labels','device IDs','model IDs','graph positions','target normalization bounds']},'source_split':'Each cell has 3 training stacks x 8 source models, a separate validation stack x 2 validation models, and a target stack x 2 test models. All shapes of each source model stay together. Target stack and both test models are excluded from training and validation. Thirty crossed cells cover every existing physical query once per seed/method.','reference_budget':'Exactly three independent matmul256/conv2d16/layernorm128 references per stack, reused as both hardware embedding and adaptation support. These are 15 existing measurement records with 20 block samples each, disjoint from 13905 evaluation queries. References are fixed across all six model folds; no additional target labels.','zero_probe_contract':'No target hardware latency is read by the zero-probe prediction path. It uses the arithmetic mean of the three source-stack minmax probe embeddings; no target adaptation. This is a blinded transfer control, not standard reference-conditioned HELP.','normalization':'Operator features and log latency are standardized using only current training stack/model rows plus current source-stack reference rows. No validation or target latency extrema are used. Three-probe hardware embedding minmax-normalizes only its three reference means, as official Eq3.','HELP_mechanism':'Verbatim official Net and InferenceNetwork classes. Learned hardware embedding, z modulation of fc3/fc4/fc5 initial weights/biases as official get_params_z, learned per-parameter Meta-SGD steps, two inner support-gradient updates, exact second-order meta-gradients on disjoint source-query losses. Deterministic posterior mean z is an official supported setting.','controls':'Paired source data, feature normalization, reference information, architecture initialization and episode query samples. Pooled MLP uses supervised query/support losses and the same 1000 outer steps; it has no z or learned inner rates. 0-probe, 3-probe conditioned and 3-probe fixed-step finetuning results all reported. HELP no-inner/no-z are test-time mechanism diagnostics, not independently retrained ablations.','adaptations_from_official':['Prediction unit is contextual isolated operator, not whole NAS architecture; official dense encoder accepts the new 51-D structural feature vector.','Reference dimension and support reduced from 10/20 to exactly 3 measured operator references.','Hidden layer 32 instead of default 100; 1000 fixed outer episodes instead of default 2000, source meta-batch 3 and query batch64.','No outlier trimming. All successful measured records retained, including large relative errors.','Global source-only log-target standardization replaces per-device normalization that would otherwise require unseen target labels.','CPU execution and deterministic z, gradient norm clip10 and fixed final checkpoint; no target-based selection.'],'statistical_units':'Five seeds measure optimization variability. Query/model/stack metrics retain their units; no hardware-population guarantee since some stacks share physical hosts.','metrics':['micro_MRE_percent','MAE_us','WAPE_percent','RMSE_ms','RMSPE_percent','Spearman','accuracy_within10percent'],'reporting':'Every stack/model cell, all five seeds and all seven inference conditions retained. Failure recorded without dropping or replacing seeds. No result-based tuning or extra training.','cannot_claim':['Original five historical platforms LOPO reproduced','Exact published HELP NAS benchmark/table or end-to-end NAS reproduced','Universal cross-hardware or unseen-model guarantee','Independent new timing repeats or physical measurements'],'new_hardware_measurements':0,'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()},'sources':[{'path':str(p.relative_to(BASE)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sources]}
    writejson(OUT/'protocol.json',protocol);print(json.dumps({'frozen':str(OUT/'protocol.json'),'sha256':sha(OUT/'protocol.json'),'cells':30,'fits':300}),flush=True)

class WrappedNet(nn.Module):
    def __init__(self,dim):
        super().__init__();self.meta_learner=official.Net(dim,True,3,CFG['layer_size'])
    def forward(self,x,h,params=None):return self.meta_learner(x,h,params)

class Adaptation(nn.Module):
    def __init__(self,dim):
        super().__init__();self.model=WrappedNet(dim);self.inference=official.InferenceNetwork(True,3,CFG['layer_size'],True)
        self.alpha=nn.ParameterDict({name.replace('.','__'):nn.Parameter(torch.full_like(p,CFG['initial_meta_sgd_lr'])) for name,p in self.model.named_parameters()})
    def parameters_for(self,h,modulate=True):
        params=OrderedDict((n,p.clone()) for n,p in self.model.named_parameters())
        if not modulate:return params,torch.tensor(0.)
        z,kl=self.inference((None,None,h));size=2*CFG['layer_size']
        for name,p in list(params.items()):
            for idx,layer in enumerate(['fc3','fc4','fc5']):
                if layer+'.weight' in name:params[name]=p*(1+CFG['z_scaling']*z['w'][idx*size:(idx+1)*size])
                elif layer+'.bias' in name:params[name]=p+CFG['z_scaling']*z['b'][idx]
        return params,kl
    def adapted(self,h,xs,ys,steps,create_graph,modulate=True):
        params,kl=self.parameters_for(h,modulate)
        for _ in range(steps):
            loss=F.mse_loss(self.model(xs,h,params),ys)
            grads=torch.autograd.grad(loss,tuple(params.values()),create_graph=create_graph)
            params=OrderedDict((n,p-self.alpha[n.replace('.','__')]*g) for (n,p),g in zip(params.items(),grads))
        return params,kl

def embedding(y):
    y=torch.tensor(y,dtype=torch.float32)
    return (y-y.min())/(y.max()-y.min()).clamp_min(1e-12)

def make_pair(dim,seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    a=Adaptation(dim);p=copy.deepcopy(a.model)
    return a,p

def train_episode(a,p,ao,po,tasks,gen):
    meta=0.;pool=0.
    for task in tasks:
        x,y,xs,ys,h=task
        idx=torch.randperm(len(x),generator=gen)[:CFG['query_batch_size']];xq=x[idx];yq=y[idx]
        params,kl=a.adapted(h,xs,ys,CFG['inner_updates'],True)
        meta=meta+F.mse_loss(a.model(xq,h,params),yq)/len(tasks)+CFG['kl_scaling']*kl/len(tasks)
        pool=pool+F.mse_loss(p(torch.cat([xq,xs]),h),torch.cat([yq,ys]))/len(tasks)
    ao.zero_grad();meta.backward();nn.utils.clip_grad_norm_(a.parameters(),CFG['gradient_clip_norm']);ao.step()
    po.zero_grad();pool.backward();nn.utils.clip_grad_norm_(p.parameters(),CFG['gradient_clip_norm']);po.step()
    if not math.isfinite(float(meta.detach())) or not math.isfinite(float(pool.detach())):raise FloatingPointError('nonfinite loss')
    return float(meta.detach()),float(pool.detach())

def selfcheck():
    assert not (OUT/'protocol.json').exists()
    a,p=make_pair(51,20261027);gen=torch.Generator().manual_seed(20261027)
    tasks=[(torch.randn(80,51),torch.randn(80,1),torch.randn(3,51),torch.randn(3,1),torch.rand(3)) for _ in range(3)]
    ao=torch.optim.Adam(a.parameters(),lr=CFG['outer_lr']);po=torch.optim.Adam(p.parameters(),lr=CFG['outer_lr'])
    before=copy.deepcopy(a.state_dict());start=time.monotonic()
    for _ in range(10):loss=train_episode(a,p,ao,po,tasks,gen)
    elapsed=time.monotonic()-start
    assert all(torch.isfinite(t).all() for t in a.state_dict().values())
    assert any(not torch.equal(v,a.state_dict()[n]) for n,v in before.items() if n.startswith('alpha'))
    assert any(not torch.equal(v,a.state_dict()[n]) for n,v in before.items() if n.startswith('inference'))
    params,_=a.adapted(tasks[0][4],tasks[0][2],tasks[0][3],2,False)
    assert torch.isfinite(a.model(tasks[0][0],tasks[0][4],params)).all()
    result={'pass':True,'utc':now(),'synthetic_only':True,'real_target_metrics_seen':False,'episodes':10,'seconds':elapsed,'seconds_per_paired_training_episode':elapsed/10,'projected_150_pairs_training_seconds':elapsed/10*1000*150,'second_order_gradient_executes':True,'meta_sgd_parameters_updated':True,'z_modulator_parameters_updated':True,'finite_inner_adaptation':True,'final_synthetic_losses':loss,'code_sha256':sha(__file__)}
    writejson(OUT/'engineering-selfcheck.json',result);print(json.dumps(result),flush=True)

def load_data():
    with gzip.open(OUT/'operator-records.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
    probes=json.loads((OUT/'independent-probes.json').read_text());cells=json.loads((OUT/'frozen-splits.json').read_text())
    return rows,probes,cells

def tensors_for_cell(c,rows,probes):
    byid={r['record_id']:r for r in rows};train=[byid[i] for i in c['train_record_ids']]
    trainprobe=[r for r in probes if r['device_id'] in c['train_stacks']]
    normrows=train+trainprobe
    rawx=torch.tensor([r['features'] for r in normrows],dtype=torch.float32);rawy=torch.tensor([math.log(r['true_latency_ms']) for r in normrows],dtype=torch.float32)
    mu=rawx.mean(0);sd=rawx.std(0,unbiased=False);sd=torch.where(sd<1.0e-6,torch.ones_like(sd),sd);ym=rawy.mean();ys=rawy.std(unbiased=False).clamp_min(.05)
    def to_tensors(rr):
        x=(torch.tensor([r['features'] for r in rr],dtype=torch.float32)-mu)/sd
        y=(torch.tensor([math.log(r['true_latency_ms']) for r in rr],dtype=torch.float32).reshape(-1,1)-ym)/ys
        return x,y
    tasks=[];ph={}
    for device in sorted({r['device_id'] for r in probes}):
        pp=sorted([r for r in probes if r['device_id']==device],key=lambda p:PROBE_NAMES.index(p['probe_name']))
        xp,yp=to_tensors(pp);h=embedding([r['true_latency_ms'] for r in pp]);ph[device]=(xp,yp,h)
        if device in c['train_stacks']:
            tr=[r for r in train if r['device_id']==device];tx,ty=to_tensors(tr);tasks.append((tx,ty,xp,yp,h))
    source_h=torch.stack([ph[d][2] for d in c['train_stacks']]).mean(0)
    return tasks,ph,source_h,{'x_mean':mu,'x_sd':sd,'y_mean':ym,'y_sd':ys},byid

def predict_conditions(a,p,raw_features,probe_tuple,source_h,norm):
    x=(torch.tensor(raw_features,dtype=torch.float32)-norm['x_mean'])/norm['x_sd'];xp,yp,h=probe_tuple
    estimates={};clipped={}
    for method in METHODS:
        if method.startswith('help'):
            hh=source_h if 'zero_probe' in method else h
            steps=0 if ('zero_probe' in method or 'no_inner' in method) else CFG['inner_updates']
            params,_=a.adapted(hh,xp,yp,steps,False,modulate='no_z' not in method)
            with torch.no_grad():z=a.model(x,hh,params)
        else:
            hh=source_h if 'zero_probe' in method else h
            params=OrderedDict((n,w.clone()) for n,w in p.named_parameters())
            if 'finetuned' in method:
                for _ in range(CFG['inner_updates']):
                    loss=F.mse_loss(p(xp,hh,params),yp);grads=torch.autograd.grad(loss,tuple(params.values()))
                    params=OrderedDict((n,w-CFG['pooled_inner_lr']*g) for (n,w),g in zip(params.items(),grads))
            with torch.no_grad():z=p(x,hh,params)
        log_ms=z.reshape(-1)*norm['y_sd']+norm['y_mean'];clipped[method]=int(((log_ms<-20)|(log_ms>20)).sum())
        estimates[method]=torch.exp(log_ms.clamp(-20,20)).cpu().numpy()
        assert np.isfinite(estimates[method]).all() and (estimates[method]>0).all()
    return estimates,clipped

def run(shard_index=0,shard_count=1):
    proto=json.loads((OUT/'protocol.json').read_text())
    for s in proto['sources']:assert sha(BASE/s['path'])==s['sha256'],s['path']
    assert proto['configuration']==CFG
    assert 0<=shard_index<shard_count<=4
    rows,probes,cells=load_data();cells=[c for i,c in enumerate(cells) if i%shard_count==shard_index];runs=OUT/'runs';runs.mkdir(exist_ok=True);start=time.monotonic();finished=0
    for c in cells:
        tasks,ph,source_h,norm,byid=tensors_for_cell(c,rows,probes)
        for seed in SEEDS:
            cellpath=runs/f'{c["cell_id"]}-seed{seed}';cellpath.mkdir(exist_ok=True);receipt=cellpath/'receipt.json'
            if receipt.exists():
                prev=json.loads(receipt.read_text());assert prev['protocol_sha256']==sha(OUT/'protocol.json') and prev['complete']
                assert all(sha(cellpath/s['path'])==s['sha256'] for s in prev['outputs']);finished+=1;continue
            if list(cellpath.glob('failure*.json')):raise RuntimeError('Earlier failed cell requires recorded investigation, not silent retry')
            try:
                t=time.monotonic();a,p=make_pair(len(rows[0]['features']),seed)
                ao=torch.optim.Adam(a.parameters(),lr=CFG['outer_lr']);po=torch.optim.Adam(p.parameters(),lr=CFG['outer_lr'])
                gen=torch.Generator().manual_seed(seed);losslog=[]
                for episode in range(CFG['episodes']):
                    l=train_episode(a,p,ao,po,tasks,gen)
                    if (episode+1)%100==0:losslog.append({'episode':episode+1,'help_meta_loss':l[0],'pooled_loss':l[1]})
                checkpoint={'help':a.state_dict(),'pooled':p.state_dict(),'normalization':norm,'source_mean_embedding':source_h,'seed':seed,'cell_id':c['cell_id'],'configuration':CFG,'protocol_sha256':sha(OUT/'protocol.json'),'train_record_ids_sha256':c['train_sha256']}
                torch.save(checkpoint,cellpath/'checkpoint.pt');writejson(cellpath/'training-loss.json',losslog)
                predictions=[];clipping={}
                for phase,device in [('validation',c['validation_stack']),('test',c['test_stack'])]:
                    rr=[byid[i] for i in c[phase+'_record_ids']]
                    est,clip=predict_conditions(a,p,[r['features'] for r in rr],ph[device],source_h,norm);clipping[phase]=clip
                    for method in METHODS:
                        for r,estimate in zip(rr,est[method]):
                            predictions.append({'cell_id':c['cell_id'],'seed':seed,'phase':phase,'method':method,'record_id':r['record_id'],'device_id':r['device_id'],'model_id':r['model_id'],'shape_id':r['shape_id'],'family':r['family'],'true_latency_ms':r['true_latency_ms'],'prediction_ms':float(estimate),'target_probe_budget':0 if 'zero_probe' in method else 3})
                # Fresh reconstruction/reload before any receipt is marked complete.
                loaded=torch.load(cellpath/'checkpoint.pt',map_location='cpu',weights_only=True);aa,pp=make_pair(len(rows[0]['features']),seed);aa.load_state_dict(loaded['help']);pp.load_state_dict(loaded['pooled'])
                testrows=[byid[i] for i in c['test_record_ids']];replay,_=predict_conditions(aa,pp,[r['features'] for r in testrows],ph[c['test_stack']],loaded['source_mean_embedding'],loaded['normalization'])
                original={m:np.array([r['prediction_ms'] for r in predictions if r['phase']=='test' and r['method']==m]) for m in METHODS}
                assert all(np.array_equal(replay[m],original[m]) for m in METHODS)
                # Zero-probe output is invariant to target probe poisoning.
                xp,yp,h=ph[c['test_stack']];poisoned=(xp,yp+10,torch.flip(h,[0]))
                poison,_=predict_conditions(aa,pp,[r['features'] for r in testrows[:2]],poisoned,loaded['source_mean_embedding'],loaded['normalization'])
                base2,_=predict_conditions(aa,pp,[r['features'] for r in testrows[:2]],ph[c['test_stack']],loaded['source_mean_embedding'],loaded['normalization'])
                assert all(np.array_equal(poison[m],base2[m]) for m in METHODS if 'zero_probe' in m)
                with gzip.open(cellpath/'predictions.jsonl.gz','wt') as f:
                    for row in predictions:f.write(json.dumps(row,separators=(',',':'))+'\n')
                result={'complete':True,'created_utc':now(),'cell_id':c['cell_id'],'seed':seed,'protocol_sha256':sha(OUT/'protocol.json'),'training_episodes':CFG['episodes'],'training_stack_count':3,'train_model_ids':c['train_model_ids'],'validation_model_ids':c['validation_model_ids'],'test_model_ids':c['test_model_ids'],'train_records':c['train_n'],'validation_records':c['validation_n'],'test_records':c['test_n'],'training_reference_records':9,'target_reference_records':3,'reference_test_overlap':0,'prediction_rows':len(predictions),'log_output_clipping_counts':clipping,'checkpoint_replay_exact':True,'zero_probe_target_reference_poisoning_invariant':True,'seconds':time.monotonic()-t,'outputs':[{'path':f.name,'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(cellpath.glob('*')) if f.is_file()]}
                writejson(receipt,result);finished+=1
                print(json.dumps({'shard':shard_index,'complete_fit_pairs':finished,'total_pairs':len(cells)*5,'cell':c['cell_id'],'seed':seed,'pair_seconds':result['seconds'],'elapsed_seconds':time.monotonic()-start}),flush=True)
            except Exception:
                writejson(cellpath/'failure.json',{'utc':now(),'error':traceback.format_exc(),'protocol_sha256':sha(OUT/'protocol.json'),'cell_id':c['cell_id'],'seed':seed});raise
    writejson(OUT/f'training-complete-shard{shard_index}.json',{'complete':True,'utc':now(),'shard_index':shard_index,'shard_count':shard_count,'fit_pairs':finished,'n_estimator_fits':2*finished,'protocol_sha256':sha(OUT/'protocol.json'),'elapsed_this_invocation_seconds':time.monotonic()-start,'new_physical_measurements':0})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selfcheck',action='store_true');parser.add_argument('--freeze',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=1);a=parser.parse_args()
    if a.selfcheck:selfcheck()
    elif a.freeze:freeze()
    elif a.run:run(a.shard_index,a.shard_count)
    else:parser.error('choose --selfcheck, --freeze or --run')
