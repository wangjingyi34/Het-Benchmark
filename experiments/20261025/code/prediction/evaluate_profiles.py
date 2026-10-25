"""Inductive real-label latency prediction and typed graph ablations.

No historical table, simulated label or target latency enters the features.
The six relation families are explicitly defined here; performance_context is
unlabelled query context, not a disguised target-value edge.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
import random
import statistics
import sys
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from structural_attributes import NUMERIC_ATTRIBUTES

FAMILIES=['Conv2D','Conv1D','MatMul','Attention','Normalization','Softmax',
          'Embedding','Pooling','Dropout','Activation','View','Reduction',
          'Elementwise','Composite','Other','Context']
RELATIONS=['contains','has_type','dependency','similarity','supports','performance_context']
NUMERIC=['input_numel','output_numel','parameter_numel','input_bytes','output_bytes',
         'rank','dim0','dim1','dim2','dim3','position_fraction','fanin','fanout']
STRICT=['backend_cpu','backend_cuda','backend_mps','backend_npu','host_arm64',
        'host_logical_cpus','threads','device_memory_bytes','compute_major','compute_minor']
MODEL_ORDER=['resnet18','resnet50','mobilenet_v2','densenet121','efficientnet_b0','squeezenet1_1',
             'text_encoder_ref','text_cnn_ref','audio_conv_ref','audio_dilated_ref','causal_lm_ref','dual_encoder_ref']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mean(x):return sum(x)/len(x)

def hardware_vector(h,mode):
    base=[float(h['descriptor_strict'][k]) for k in STRICT]
    # Backend/architecture flags stay binary; positive scalar sizes are log1p.
    base=[v if i<5 else math.log1p(v) for i,v in enumerate(base)]
    if mode=='probe':
        assert len(h['probes'])==3
        for p in h['probes']:
            assert len(p['latency_samples_ms'])>=20
            v=p['latency_mean_ms'];assert math.isfinite(v) and v>0
            base.append(math.log(v))
    return base

def base_vector(node,h,mode,node_type='operator'):
    # Explicit whitelist; unknown keys (including latency) cannot silently enter.
    family=node.get('family','Context');fam=[float(f==family) for f in FAMILIES]
    types=[float(node_type==t) for t in ['operator','model','family','hardware']]
    v=node.get('features',{})
    nums=[]
    for k in NUMERIC:
        x=float(v.get(k,0.0));assert math.isfinite(x) and x>=0
        nums.append(x if k=='position_fraction' else math.log1p(x))
    attrs=node.get('operator_attributes',{})
    for k in NUMERIC_ATTRIBUTES:
        value=float(attrs.get(k,0.0));assert math.isfinite(value) and value>=0
        nums.append(value if k in ['is_causal','bias_present','training_flag','input_channels_last','composite_leaf'] else math.log1p(value))
    return fam+types+nums+hardware_vector(h,mode)

def build_graph(g,h,mode,minimum_repeats,unlabelled=False):
    raw_nodes=g['nodes'];ids={n['node_id']:i for i,n in enumerate(raw_nodes)}
    if len(ids)!=len(raw_nodes):raise ValueError('duplicate FX nodes')
    records={r['op_instance_id']:r for r in g['records']}
    x=[base_vector(n,h,mode) for n in raw_nodes]
    q=[];y=[];meta=[];errors=[]
    for n in raw_nodes:
        r=records.get(n['node_id'])
        if unlabelled:
            if n['family']=='Context':continue
            q.append(ids[n['node_id']]);y.append(float('nan'))
            meta.append({'record_id':None,'node_id':n['node_id'],'model_id':g['model_id'],
                         'shape_id':g['shape_id'],'family':n['family'],'module_path':n['module_path'],
                         'device_id':h['device_id'],'backend':h['backend'],
                         'true_latency_ms':None,'timed_measurement_cost_ms':None,
                         'input_weights_sha256':g.get('input_weights_sha256'),
                         'target_latency_available':False})
            continue
        if not r or r['status']!='measured':continue
        samples=r['latency_samples_ms']
        if len(samples)<minimum_repeats:raise ValueError('too few physical samples: '+r['record_id'])
        if not all(isinstance(t,(int,float)) and not isinstance(t,bool) and math.isfinite(t) and t>0 for t in samples):raise ValueError('invalid physical samples')
        recomputed=mean(samples)
        if abs(recomputed-r['latency_mean_ms'])>1e-9:raise ValueError('raw mean mismatch')
        q.append(ids[n['node_id']]);y.append(recomputed)
        meta.append({'record_id':r['record_id'],'node_id':n['node_id'],'model_id':g['model_id'],
                     'shape_id':g['shape_id'],'family':n['family'],'module_path':n['module_path'],
                     'device_id':h['device_id'],'backend':h['backend'],
                     'true_latency_ms':recomputed,
                     'timed_measurement_cost_ms':sum(samples)*r['inner_repeats'],
                     'measurement_cost_scope':'recorded synchronized timed blocks only; excludes warmup/setup',
                     'input_weights_sha256':g['input_weights_sha256']})
    if not q:return None
    src=[];dst=[];rel=[]
    def edge(a,b,r,bidirectional=False):
        src.append(a);dst.append(b);rel.append(r)
        if bidirectional:src.append(b);dst.append(a);rel.append(r)
    # Virtual nodes carry only the same whitelisted structural/hardware inputs.
    model_idx=len(x);x.append(base_vector({},h,mode,'model'))
    hardware_idx=len(x);x.append(base_vector({},h,mode,'hardware'))
    family_idx={}
    for f in sorted({n['family'] for n in raw_nodes if n['family']!='Context'}):
        family_idx[f]=len(x);x.append(base_vector({'family':f},h,mode,'family'))
    byfamily=collections.defaultdict(list)
    for i,n in enumerate(raw_nodes):
        edge(model_idx,i,0,True)
        if n['family'] in family_idx:
            edge(family_idx[n['family']],i,1,True);byfamily[n['family']].append(i)
            edge(i,hardware_idx,5,True)
    for e in g['edges']:
        if e['relation']!='dependency':raise ValueError('unexpected raw relation')
        edge(ids[e['source']],ids[e['target']],2,True)
    for f,indices in byfamily.items():
        # Two closest label-free input/output-size neighbours within this graph.
        for a in indices:
            va=raw_nodes[a]['features']
            ranked=sorted((b for b in indices if b!=a),key=lambda b:(
                abs(math.log1p(va['input_numel'])-math.log1p(raw_nodes[b]['features']['input_numel']))+
                abs(math.log1p(va['output_numel'])-math.log1p(raw_nodes[b]['features']['output_numel'])),b))
            for b in ranked[:2]:edge(a,b,3)
        # Adapter candidate family, not proof of target latency or task legality.
        edge(family_idx[f],hardware_idx,4,True)
    return {'graph_id':g['graph_id'],'model_id':g['model_id'],'device_id':h['device_id'],
            'workload':g['workload'],'shape_id':g['shape_id'],'architecture_source':g['architecture_source'],
            'x':torch.tensor(x,dtype=torch.float32),'edge_index':torch.tensor([src,dst],dtype=torch.long),
            'edge_type':torch.tensor(rel,dtype=torch.long),'query':torch.tensor(q,dtype=torch.long),
            'y':torch.tensor(y,dtype=torch.float32),'meta':meta}

def load_data(paths,mode,minimum_repeats,sidecars=None,allow_unenriched=False):
    graphs=[];manifest=[];seen=set();hardware={}
    metadata={}
    for side in sidecars or []:
        sc=json.loads(Path(side).read_text())
        if sc['kind']!='label_free_structural_metadata_sidecar' or sc['timings_changed_or_added']:
            raise ValueError('invalid structural metadata sidecar')
        metadata[sc['raw_bundle_sha256']]=(sc,Path(side))
    for path in paths:
        d=json.loads(Path(path).read_text())
        if d['schema_version']!='het-physical-profile-v2.1':raise ValueError('not physical v2.1')
        if not d['provenance'].get('completed_utc'):raise ValueError('incomplete profile bundle '+str(path))
        raw_sha=digest(path);side=metadata.get(raw_sha)
        if not side and not allow_unenriched:raise ValueError('formal prediction requires hash-verified structural sidecar: '+str(path))
        h=d['hardware'];hardware[h['device_id']]=h
        for g in d['graphs']:
            k=(h['device_id'],g['graph_id'])
            if k in seen:raise ValueError('duplicate device/graph '+str(k))
            seen.add(k)
            if side:
                sc=side[0]['graphs'][g['graph_id']]
                if not sc['input_weights_hash_verified'] or sc['input_weights_sha256']!=g['input_weights_sha256']:
                    raise ValueError('unverified structural enrichment')
                g={**g,'nodes':[{**n,'operator_attributes':sc['nodes'][n['node_id']]} for n in g['nodes']]}
            built=build_graph(g,h,mode,minimum_repeats)
            if built:graphs.append(built)
        manifest.append({'path':str(Path(path).resolve()),'sha256':raw_sha,
                         'code_sha256':d['provenance']['code_sha256'],'graphs':len(d['graphs']),
                         'failures':d.get('failures',[]),'hardware':h['device_id'],
                         'record_status_counts':dict(collections.Counter(r['status'] for g in d['graphs'] for r in g['records'])),
                         'metadata_sidecar':str(side[1].resolve()) if side else None,
                         'metadata_sidecar_sha256':digest(side[1]) if side else None})
    return graphs,manifest,hardware

def collate(graphs,device,scaler=None):
    if not graphs:raise ValueError('empty partition')
    xs=[];es=[];rs=[];qs=[];ys=[];meta=[];offset=0
    for g in graphs:
        xs.append(g['x']);es.append(g['edge_index']+offset);rs.append(g['edge_type'])
        qs.append(g['query']+offset);ys.append(g['y']);meta.extend(g['meta']);offset+=len(g['x'])
    x=torch.cat(xs).to(device)
    if scaler:x=(x-scaler[0])/scaler[1]
    edge=torch.cat(es,dim=1).to(device)
    # Vanilla GCN sees the same structure with relation names removed. Collapse
    # parallel typed edges and symmetrize once, outside all training epochs.
    loops=torch.arange(len(x),device=device);loops=torch.stack([loops,loops])
    untyped=torch.unique(torch.cat([edge,edge.flip(0),loops],dim=1),dim=1)
    degree=torch.bincount(untyped[1],minlength=len(x)).to(x.dtype)
    normalization=(degree[untyped[0]]*degree[untyped[1]]).rsqrt()
    return {'x':x,'edge_index':edge,'edge_type':torch.cat(rs).to(device),
            'gcn_edge_index':untyped,'gcn_normalization':normalization,
            'query':torch.cat(qs).to(device),'y':torch.cat(ys).to(device),'meta':meta}

class MLP(nn.Module):
    def __init__(self,dim,hidden=128):
        super().__init__();self.body=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.head=nn.Linear(hidden,1)
    def forward(self,b,embedding=False):
        z=self.body(b['x']);return (self.head(z).squeeze(-1)[b['query']],z[b['query']]) if embedding else self.head(z).squeeze(-1)[b['query']]

class RelationalAttention(nn.Module):
    def __init__(self,dim,heads=4):
        super().__init__();self.heads=heads;self.depth=dim//heads
        self.w=nn.Parameter(torch.empty(6,dim,dim));nn.init.xavier_uniform_(self.w)
        self.a_src=nn.Parameter(torch.empty(6,heads,self.depth));self.a_dst=nn.Parameter(torch.empty(6,heads,self.depth))
        nn.init.xavier_uniform_(self.a_src);nn.init.xavier_uniform_(self.a_dst)
        self.self_linear=nn.Linear(dim,dim);self.norm=nn.LayerNorm(dim)
    def forward(self,x,edge_index,edge_type,drop=None):
        if drop is not None:
            keep=edge_type!=drop;edge_index=edge_index[:,keep];edge_type=edge_type[keep]
        s,t=edge_index
        # Relation-specific transformation and attention, normalized across all
        # incoming relation messages for each destination and attention head.
        # Project each node once per relation rather than expanding a D-by-D
        # weight matrix for every edge (which needlessly consumes GPU memory).
        projected=torch.einsum('ni,rij->nrj',x,self.w)
        hs=projected[s,edge_type].reshape(-1,self.heads,self.depth)
        ht=projected[t,edge_type].reshape(-1,self.heads,self.depth)
        logits=F.leaky_relu((hs*self.a_src[edge_type]).sum(-1)+(ht*self.a_dst[edge_type]).sum(-1),negative_slope=.2)
        index=t[:,None].expand(-1,self.heads)
        maxima=torch.full((len(x),self.heads),-torch.inf,device=x.device,dtype=x.dtype)
        maxima.scatter_reduce_(0,index,logits,reduce='amax',include_self=True)
        numerator=torch.exp(logits-maxima[t]);denominator=torch.zeros_like(maxima)
        denominator.scatter_add_(0,index,numerator)
        alpha=numerator/denominator[t].clamp_min(1e-12)
        message=(hs*alpha[:,:,None]).reshape(-1,self.heads*self.depth)
        out=torch.zeros_like(x);out.index_add_(0,t,message)
        return self.norm(self.self_linear(x)+F.gelu(out))

class RGAT(nn.Module):
    def __init__(self,dim,hidden=32,drop=None):
        super().__init__();self.input=nn.Linear(dim,hidden);self.layer1=RelationalAttention(hidden)
        self.layer2=RelationalAttention(hidden);self.head=nn.Linear(hidden,1);self.drop=drop
    def forward(self,b,embedding=False):
        z=F.gelu(self.input(b['x']));z=self.layer1(z,b['edge_index'],b['edge_type'],self.drop)
        z=self.layer2(z,b['edge_index'],b['edge_type'],self.drop)
        zq=z[b['query']];pred=self.head(zq).squeeze(-1)
        return (pred,zq) if embedding else pred

class GCN(nn.Module):
    """Untyped symmetric-normalized graph-convolution latency baseline.

    Two propagation layers with width80 plus input encoder and pointwise output.
    Parameter count is reported; it is close to (slightly above) RGAT width32.
    This is not a reproduction of the complete BRP-NAS search procedure.
    """
    def __init__(self,dim,hidden=80):
        super().__init__();self.input=nn.Linear(dim,hidden)
        self.layers=nn.ModuleList([nn.Linear(hidden,hidden),nn.Linear(hidden,hidden)])
        self.norms=nn.ModuleList([nn.LayerNorm(hidden),nn.LayerNorm(hidden)])
        self.head=nn.Linear(hidden,1)
    def forward(self,b,embedding=False):
        z=F.gelu(self.input(b['x']));s,t=b['gcn_edge_index'];weight=b['gcn_normalization']
        for layer,norm in zip(self.layers,self.norms):
            value=layer(z);aggregated=torch.zeros_like(value)
            aggregated.index_add_(0,t,value[s]*weight[:,None])
            z=norm(F.gelu(aggregated))
        zq=z[b['query']];out=self.head(zq).squeeze(-1)
        return (out,zq) if embedding else out

def make_splits(graphs,protocol,folds):
    ids=[m for m in MODEL_ORDER if m in {g['model_id'] for g in graphs}]
    if len(ids)<6 or len(ids)%2:raise ValueError('need an even number of at least six physical model IDs')
    pairs=[ids[i:i+2] for i in range(0,len(ids),2)]
    result=[]
    if protocol in ['model','both']:
        for f in range(min(folds,len(pairs))):
            test=pairs[f];val=pairs[(f+1)%len(pairs)];train=[m for m in ids if m not in test+val]
            result.append({'split_id':f'model-fold{f}','protocol':'model_holdout',
                'train_model_ids':train,'validation_model_ids':val,'test_model_ids':test,
                'train':[g for g in graphs if g['model_id'] in train],
                'validation':[g for g in graphs if g['model_id'] in val],
                'test':[g for g in graphs if g['model_id'] in test]})
    if protocol in ['device','both']:
        devices=sorted({g['device_id'] for g in graphs})
        if len(devices)<2:raise ValueError('device holdout requires >=2 physically measured devices')
        # This protocol holds hardware out, not all test model architectures.
        val=pairs[-1];train=[m for m in ids if m not in val]
        for dev in devices:
            result.append({'split_id':'device-'+dev,'protocol':'device_holdout_seen_architectures',
                'heldout_device_id':dev,'train_model_ids':train,'validation_model_ids':val,'test_model_ids':ids,
                'train':[g for g in graphs if g['device_id']!=dev and g['model_id'] in train],
                'validation':[g for g in graphs if g['device_id']!=dev and g['model_id'] in val],
                'test':[g for g in graphs if g['device_id']==dev]})
    for s in result:
        assert {g['model_id'] for g in s['train']}==set(s['train_model_ids'])
        if s['protocol']=='model_holdout':
            assert not(set(s['train_model_ids'])&set(s['test_model_ids']))
            assert not(set(s['validation_model_ids'])&set(s['test_model_ids']))
        else:assert all(g['device_id']!=s['heldout_device_id'] for g in s['train']+s['validation'])
    return result

def mre(y,p):return (torch.abs(p-y)/y.clamp_min(1e-6)).mean().item()*100

def fit_model(method,train,val,seed,args):
    if not all(torch.isfinite(b['y']).all() and (b['y']>0).all() for b in [train,val]):
        raise ValueError('Training requires positive physical labels; unlabelled inference graphs are forbidden')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    x=train['x'];mu=x.mean(0);sd=x.std(0,unbiased=False)
    # An unseen hardware flag must not become 1e6 merely because its training
    # column was constant. Constant columns use scale1, fitted on train only.
    sd=torch.where(sd<1e-6,torch.ones_like(sd),sd)
    scaler=(mu,sd)
    tr={**train,'x':(x-mu)/sd};va={**val,'x':(val['x']-mu)/sd}
    log_y=torch.log(tr['y']);ym=log_y.mean();ys=log_y.std(unbiased=False).clamp_min(.05)
    z_y=(log_y-ym)/ys
    if method=='ridge':
        xx=tr['x'][tr['query']];xx=torch.cat([xx,torch.ones(len(xx),1,device=x.device)],1)
        eye=torch.eye(xx.shape[1],device=x.device);eye[-1,-1]=0
        # Fixed L2=1.0; never select a ridge coefficient using the test split.
        weight=torch.linalg.solve(xx.T@xx+eye,xx.T@z_y)
        return {'kind':'ridge','weight':weight,'scaler':scaler,'target_scale':(ym,ys),
                'training_epochs':1,'best_validation_mre':None,'seed':seed,'parameter_count':weight.numel(),
                'seed_scope':'deterministic closed-form baseline; repeated seeds do not create independent training runs'}
    drop=RELATIONS.index(method[5:]) if method.startswith('drop_') else None
    net=MLP(x.shape[1]) if method=='mlp' else GCN(x.shape[1]) if method=='gcn' else RGAT(x.shape[1],drop=drop)
    net.to(x.device);optim=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=1e-4)
    best=None;best_score=float('inf');bad=0
    for epoch in range(args.epochs):
        net.train();optim.zero_grad();pred=net(tr);loss=F.mse_loss(pred,z_y);loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),5.);optim.step()
        net.eval()
        with torch.no_grad():vp=torch.exp((net(va)*ys+ym).clamp(-20,20));score=mre(va['y'],vp)
        if score<best_score-1e-5:
            best_score=score;best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};bad=0
        else:bad+=1
        if bad>=args.patience:break
    net.load_state_dict(best);net.eval()
    return {'kind':method,'net':net,'state':best,'scaler':scaler,'target_scale':(ym,ys),
            'training_epochs':epoch+1,'best_validation_mre':best_score,'seed':seed,
            'parameter_count':sum(p.numel() for p in net.parameters())}

def predict(fit,b,embeddings=False):
    mu,sd=fit['scaler'];ym,ys=fit['target_scale'];bb={**b,'x':(b['x']-mu)/sd}
    with torch.no_grad():
        if fit['kind']=='ridge':
            x=bb['x'][bb['query']];x=torch.cat([x,torch.ones(len(x),1,device=x.device)],1)
            pred=x@fit['weight'];emb=None
        elif embeddings:pred,emb=fit['net'](bb,embedding=True)
        else:pred=fit['net'](bb);emb=None
        p=torch.exp((pred*ys+ym).clamp(-20,20)).cpu().numpy()
        return (p,emb.cpu().numpy() if emb is not None else None) if embeddings else p

def metrics(rows):
    if not rows:return {}
    errs=[abs(r['prediction_ms']-r['true_latency_ms'])/max(r['true_latency_ms'],1e-6)*100 for r in rows]
    bymodel=collections.defaultdict(list);byfamily=collections.defaultdict(list);bydevice=collections.defaultdict(list)
    for r,e in zip(rows,errs):
        bymodel[r['model_id']].append(e);byfamily[r['family']].append(e);bydevice[r['device_id']].append(e)
    modelmeans={k:mean(v) for k,v in bymodel.items()};values=list(modelmeans.values())
    # The bootstrap unit is model, not operators or repeated timing calls.
    rng=np.random.default_rng(9153)
    boot=[mean(rng.choice(values,size=len(values),replace=True).tolist()) for _ in range(2000)]
    return {'records':len(rows),'micro_mre_percent':mean(errs),'model_macro_mre_percent':mean(values),
        'family_macro_mre_percent':mean([mean(v) for v in byfamily.values()]),
        'attention_mre_percent':mean(byfamily['Attention']) if byfamily.get('Attention') else None,
        'attention_n':len(byfamily.get('Attention',[])),
        'model_macro_bootstrap95_percent':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
        'bootstrap_scope':'descriptive percentile bootstrap over held-out model clusters; small model count limits population inference',
        'per_model_mre_percent':modelmeans,'per_family_mre_percent':{k:mean(v) for k,v in byfamily.items()},
        'per_family_n':{k:len(v) for k,v in byfamily.items()},'per_device_mre_percent':{k:mean(v) for k,v in bydevice.items()},
        'per_device_n':{k:len(v) for k,v in bydevice.items()},
        'evaluation_record_ids_sha256':hashlib.sha256('\n'.join(sorted(r['record_id'] for r in rows)).encode()).hexdigest()}

def accumulate(store,key,rows):
    a=store.setdefault(key,{'n':0,'sum':0.,'model':{},'family':{},'device':{},'record_ids':[]})
    for r in rows:
        e=abs(r['prediction_ms']-r['true_latency_ms'])/max(r['true_latency_ms'],1e-6)*100
        a['n']+=1;a['sum']+=e
        a['record_ids'].append(r['record_id'])
        for group,column in [('model','model_id'),('family','family'),('device','device_id')]:
            q=a[group].setdefault(r[column],[0.,0]);q[0]+=e;q[1]+=1

def accumulated_metrics(a):
    mm={k:v[0]/v[1] for k,v in a['model'].items()};values=list(mm.values());rng=np.random.default_rng(9153)
    boot=[mean(rng.choice(values,size=len(values),replace=True).tolist()) for _ in range(2000)]
    return {'records':a['n'],'micro_mre_percent':a['sum']/a['n'],'model_macro_mre_percent':mean(values),
        'family_macro_mre_percent':mean([v[0]/v[1] for v in a['family'].values()]),
        'attention_mre_percent':a['family']['Attention'][0]/a['family']['Attention'][1] if 'Attention' in a['family'] else None,
        'attention_n':a['family'].get('Attention',[0.,0])[1],
        'model_macro_bootstrap95_percent':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
        'bootstrap_scope':'descriptive model-cluster percentile bootstrap; small model count limits population inference',
        'per_model_mre_percent':mm,'per_family_mre_percent':{k:v[0]/v[1] for k,v in a['family'].items()},
        'per_family_n':{k:v[1] for k,v in a['family'].items()},'per_device_mre_percent':{k:v[0]/v[1] for k,v in a['device'].items()},
        'per_device_n':{k:v[1] for k,v in a['device'].items()},
        'evaluation_record_ids_sha256':hashlib.sha256('\n'.join(sorted(a['record_ids'])).encode()).hexdigest()}

def paired_seed_interval(values):
    if not values:return None
    rng=np.random.default_rng(61025)
    bootstrap=[float(np.mean(rng.choice(values,len(values),replace=True))) for _ in range(4000)]
    return {'n_seeds':len(values),'values_pp':values,'mean_pp':mean(values),
            'sample_sd_pp':statistics.stdev(values) if len(values)>1 else None,
            'paired_seed_bootstrap95_pp':[float(np.quantile(bootstrap,.025)),float(np.quantile(bootstrap,.975))] if len(values)>1 else None,
            'scope':'conditional paired seed uncertainty on the same physical test records; five seeds are a small sample'}

def fallback_policy(val_rows,test_rows,budget):
    # Uncertainty is ensemble relative prediction dispersion, not test error.
    choices=sorted({r['uncertainty'] for r in val_rows})
    thresholds=[float('inf')]+[float(np.quantile(choices,q)) for q in np.linspace(0,1,21)]
    feasible=[]
    for threshold in thresholds:
        trigger=[r['uncertainty']>threshold for r in val_rows]
        fraction=mean(trigger)
        if fraction<=budget+1e-12:
            error=mean([0. if t else abs(r['prediction_ms']-r['true_latency_ms'])/max(r['true_latency_ms'],1e-6)*100 for r,t in zip(val_rows,trigger)])
            feasible.append((error,fraction,threshold))
    chosen=min(feasible,key=lambda x:(x[0],x[1],-x[2]))
    threshold=chosen[2];trigger=[r['uncertainty']>threshold for r in test_rows]
    unmeasured=[r for r,t in zip(test_rows,trigger) if not t]
    hybrid=[{**r,'prediction_ms':r['true_latency_ms'] if t else r['prediction_ms']} for r,t in zip(test_rows,trigger)]
    return {'threshold':threshold if math.isfinite(threshold) else None,'threshold_infinite':not math.isfinite(threshold),
        'threshold_selected_only_on_validation':True,'validation_budget_fraction':budget,
        'validation_measured_fraction':chosen[1],'test_measured_fraction':mean(trigger),
        'test_measured_count':sum(trigger),'test_total':len(test_rows),
        'pure_prediction':metrics(test_rows),'unmeasured_selective_prediction':metrics(unmeasured),
        'hybrid_replay':metrics(hybrid),
        'recorded_timed_measurement_cost_ms':sum(r['timed_measurement_cost_ms'] for r,t in zip(test_rows,trigger) if t),
        'cost_scope':'Offline policy replay on physically measured held-out targets. Timed-block cost excludes warmup/setup; not an online trigger deployment.',
        'trigger_record_ids':[r['record_id'] for r,t in zip(test_rows,trigger) if t]}

def save_fit(fit,path,split,mode,method,manifest):
    data={'kind':fit['kind'],'seed':fit['seed'],'feature_mode':mode,'method':method,
          'training_model_ids':split['train_model_ids'],'validation_model_ids':split['validation_model_ids'],
          'test_model_ids':split['test_model_ids'],'split_id':split['split_id'],
          'source_manifest':manifest,'scaler':[t.detach().cpu() for t in fit['scaler']],
          'target_scale':[t.detach().cpu() for t in fit['target_scale']],
          'state_dict':fit.get('state'),'ridge_weight':fit.get('weight',None)}
    if isinstance(data['ridge_weight'],torch.Tensor):data['ridge_weight']=data['ridge_weight'].cpu()
    torch.save(data,path);return digest(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',nargs='+',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--train-device',default='cpu')
    ap.add_argument('--protocol',choices=['model','device','both'],default='both')
    ap.add_argument('--feature-modes',nargs='+',choices=['strict','probe'],default=['strict','probe'])
    ap.add_argument('--metadata-sidecars',nargs='*',type=Path,default=[])
    ap.add_argument('--allow-unenriched-smoke',action='store_true')
    ap.add_argument('--methods',nargs='+',default=['ridge','mlp','gcn','rgat']+['drop_'+r for r in RELATIONS])
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55]);ap.add_argument('--folds',type=int,default=6)
    ap.add_argument('--epochs',type=int,default=80);ap.add_argument('--patience',type=int,default=15)
    ap.add_argument('--min-repeats',type=int,default=20);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--fallback-budget',type=float,default=.2)
    ap.add_argument('--learning-budgets',nargs='*',type=int,default=[])
    ap.add_argument('--full-budget-reference',type=Path)
    args=ap.parse_args();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Never overwrite an earlier predictor run; choose a new output directory')
    if args.train_device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA training requested but unavailable')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    args.output.mkdir(parents=True,exist_ok=True)
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'code_sha256':digest(__file__),
        'scope':'New physical v2 prediction, not rerun of original v21 accuracy. Model- and device-held-out protocols are separate.',
        'relation_semantics':{'contains':'model/FX node membership, bidirectional messages',
            'has_type':'node/family taxonomy','dependency':'real directed FX dependencies, bidirectional messages',
            'similarity':'two label-free size-nearest same-family nodes within model graph',
            'supports':'adapter candidate-family/device context, not target latency or task feasibility',
            'performance_context':'unlabelled operator query/device relation; no performance value'},
        'label_feature_separation':True,'min_repeats':args.min_repeats,'seeds':args.seeds,
        'train_runtime':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
            'device':args.train_device,'threads':args.threads,
            'cuda_version':torch.version.cuda,'tf32':False,
            'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled()},
        'command_configuration':{k:([str(x) if isinstance(x,Path) else x for x in v] if isinstance(v,list)
                                    else str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},
        'numeric_feature_whitelist':NUMERIC,'structural_attribute_whitelist':NUMERIC_ATTRIBUTES,
        'metadata_enrichment_required':not args.allow_unenriched_smoke,
        'epochs_max':args.epochs,'patience_validation_only':args.patience,'results':[],'embedding_exports':[]}
    # Stream millions of per-record/seed predictions rather than retaining a
    # multi-gigabyte Python list and serializing it in a second memory copy.
    seed_acc={};ensemble_acc={};prediction_count=0;embedding_rows=[]
    prediction_file=gzip.open(args.output/'predictions.jsonl.gz','wt',encoding='utf-8')
    ensemble_file=gzip.open(args.output/'ensemble-predictions.jsonl.gz','wt',encoding='utf-8')
    for mode in args.feature_modes:
        graphs,manifest,hws=load_data(args.inputs,mode,args.min_repeats,args.metadata_sidecars,args.allow_unenriched_smoke)
        splits=make_splits(graphs,args.protocol,args.folds)
        if args.learning_budgets:
            if args.protocol!='model':raise ValueError('learning curves use model holdout only')
            smaller=[]
            workload_by_model={g['model_id']:g['workload'] for g in graphs}
            for split in splits:
                by_workload=collections.defaultdict(list)
                for model in split['train_model_ids']:by_workload[workload_by_model[model]].append(model)
                order=[]
                # Fixed label-free round robin, never ranking models by errors.
                while any(by_workload.values()):
                    for group in ['CV','NLP','Audio','LLM','Multimodal']:
                        if by_workload[group]:order.append(by_workload[group].pop(0))
                for budget in args.learning_budgets:
                    if budget<=0 or budget>=len(order):raise ValueError('learning budgets must be positive and below the main full training-model count')
                    selected=order[:budget]
                    smaller.append({**split,'split_id':split['split_id']+f'-budget{budget}',
                        'protocol':f'model_holdout_budget{budget}','base_protocol':'model_holdout',
                        'training_model_budget':budget,'train_model_ids':selected,
                        'train':[g for g in split['train'] if g['model_id'] in selected],
                        'budget_selection_rule':'fixed workload round robin within train IDs; no labels, validation errors or test errors used'})
            splits=smaller
        summary.setdefault('source_manifest',manifest)
        summary.setdefault('hardware',hws)
        summary.setdefault('record_counts',{'graphs':len(graphs),'labels':sum(len(g['y']) for g in graphs),
            'models':len({g['model_id'] for g in graphs}),'physical_runtime_stacks':len(hws),
            'scope':'Runtime stacks are not independent hardware architectures: M5 CPU/MPS share a machine; cloud CPU and MIG share a host.'})
        for split in splits:
            tr=collate(split['train'],args.train_device);va=collate(split['validation'],args.train_device)
            te=collate(split['test'],args.train_device)
            public_split={k:v for k,v in split.items() if k not in ['train','validation','test']}
            for method in args.methods:
                started=time.monotonic();seed_predictions=[];seed_val=[];fits=[];seed_scores=[]
                for seed in args.seeds:
                    fit=fit_model(method,tr,va,seed,args);vp=predict(fit,va);tp=predict(fit,te)
                    path=args.output/f"{mode}-{split['split_id']}-{method}-seed{seed}.pt"
                    ch=save_fit(fit,path,split,mode,method,manifest)
                    fits.append({'seed':seed,'checkpoint':str(path.resolve()),'checkpoint_sha256':ch,
                        'training_epochs':fit['training_epochs'],'best_validation_mre':fit['best_validation_mre'],
                        'parameter_count':fit.get('parameter_count')})
                    seed_predictions.append(tp);seed_val.append(vp)
                    rows=[{**m,'prediction_ms':float(p),'seed':seed,'feature_mode':mode,'method':method,
                           'split_id':split['split_id'],'protocol':split['protocol']} for m,p in zip(te['meta'],tp)]
                    seed_scores.append({'seed':seed,**metrics(rows)})
                    for row in rows:prediction_file.write(json.dumps(row,separators=(',',':'))+'\n')
                    prediction_file.flush();prediction_count+=len(rows)
                    accumulate(seed_acc,(mode,method,split['protocol'],seed),rows)
                    # The fold excluding ResNet is the one exported for planner
                    # state. Only features/structure are read at inference.
                    if method=='rgat' and seed==args.seeds[0] and split['protocol']=='model_holdout':
                        _,emb=predict(fit,te,embeddings=True)
                        for m,z in zip(te['meta'],emb):
                            if m['module_path']:
                                embedding_rows.append({k:m[k] for k in ['model_id','module_path','device_id','shape_id']}|
                                    {'embedding':z.astype(float).tolist(),'checkpoint_sha256':ch,
                                     'training_model_ids':split['train_model_ids'],'used_inference_latency_labels':False,
                                     'feature_mode':mode,'split_id':split['split_id']})
                        # Make the frozen inductive embeddings available to the
                        # planner as soon as each held-out fold is completed.
                        (args.output/'graph-embeddings.partial.json').write_text(json.dumps(embedding_rows,indent=2))
                    del fit
                def ensemble_rows(meta,preds):
                    a=np.stack(preds);mu=a.mean(0);sd=a.std(0,ddof=1) if len(preds)>1 else np.zeros_like(mu)
                    return [{**m,'prediction_ms':float(v),'uncertainty':float(s/max(v,1e-6))} for m,v,s in zip(meta,mu,sd)]
                erows=ensemble_rows(te['meta'],seed_predictions);vrows=ensemble_rows(va['meta'],seed_val)
                for row in erows:
                    ensemble_file.write(json.dumps({**row,'feature_mode':mode,'method':method,
                        'split_id':split['split_id'],'protocol':split['protocol']},separators=(',',':'))+'\n')
                ensemble_file.flush();accumulate(ensemble_acc,(mode,method,split['protocol']),erows)
                entry={'feature_mode':mode,'method':method,**public_split,'train_records':len(tr['y']),
                    'validation_records':len(va['y']),'test_records':len(te['y']),'seed_fits':fits,
                    'seed_test_metrics':seed_scores,
                    'seed_micro_mre_mean':mean([r['micro_mre_percent'] for r in seed_scores]),
                    'seed_micro_mre_sample_sd':statistics.stdev([r['micro_mre_percent'] for r in seed_scores]) if len(seed_scores)>1 else None,
                    'seed_family_macro_mre_mean':mean([r['family_macro_mre_percent'] for r in seed_scores]),
                    'seed_family_macro_mre_sample_sd':statistics.stdev([r['family_macro_mre_percent'] for r in seed_scores]) if len(seed_scores)>1 else None,
                    'ensemble_test_metrics':metrics(erows),'fallback':fallback_policy(vrows,erows,args.fallback_budget),
                    'evaluation_record_ids_sha256':hashlib.sha256('\n'.join(r['record_id'] for r in te['meta']).encode()).hexdigest(),
                    'elapsed_seconds':time.monotonic()-started}
                summary['results'].append(entry)
                (args.output/'summary.partial.json').write_text(json.dumps(summary,indent=2))
                print(json.dumps({'mode':mode,'split':split['split_id'],'method':method,
                    'MRE':entry['ensemble_test_metrics']['micro_mre_percent'],'seconds':entry['elapsed_seconds']}),flush=True)
            del tr,va,te
    prediction_file.close();ensemble_file.close()
    (args.output/'graph-embeddings.json').write_text(json.dumps(embedding_rows,indent=2))
    summary['embedding_exports']={'file':str((args.output/'graph-embeddings.json').resolve()),'rows':len(embedding_rows),
        'sha256':digest(args.output/'graph-embeddings.json'),'dimension':32,'target_latency_used_at_inference':False}
    summary['completed_utc']=datetime.now(timezone.utc).isoformat()
    summary['aggregate_results']=[]
    for mode,method,protocol in sorted(ensemble_acc):
        ss=[]
        for seed in args.seeds:
            ss.append({'seed':seed,**accumulated_metrics(seed_acc[(mode,method,protocol,seed)])})
        summary['aggregate_results'].append({'feature_mode':mode,'method':method,'protocol':protocol,
            'seed_metrics':ss,'micro_mre_seed_mean':mean([r['micro_mre_percent'] for r in ss]),
            'micro_mre_seed_sample_sd':statistics.stdev([r['micro_mre_percent'] for r in ss]) if len(ss)>1 else None,
            'family_macro_mre_seed_mean':mean([r['family_macro_mre_percent'] for r in ss]),
            'family_macro_mre_seed_sample_sd':statistics.stdev([r['family_macro_mre_percent'] for r in ss]) if len(ss)>1 else None,
            'model_macro_mre_seed_mean':mean([r['model_macro_mre_percent'] for r in ss]),
            'model_macro_mre_seed_sample_sd':statistics.stdev([r['model_macro_mre_percent'] for r in ss]) if len(ss)>1 else None,
            'ensemble_metrics':accumulated_metrics(ensemble_acc[(mode,method,protocol)])})
    summary['paired_method_comparisons']=[]
    bykey={(r['feature_mode'],r['method'],r['protocol']):r for r in summary['aggregate_results']}
    for (mode,method,protocol),baseline in sorted(bykey.items()):
        reference=bykey.get((mode,'rgat',protocol))
        if method=='rgat' or reference is None:continue
        paired={}
        for field in ['micro_mre_percent','family_macro_mre_percent','model_macro_mre_percent','attention_mre_percent']:
            deltas=[]
            for left,right in zip(reference['seed_metrics'],baseline['seed_metrics']):
                if left['evaluation_record_ids_sha256']!=right['evaluation_record_ids_sha256']:
                    raise ValueError('unfair method comparison: different physical test records')
                if left[field] is not None and right[field] is not None:deltas.append(left[field]-right[field])
            paired[field]=paired_seed_interval(deltas)
        summary['paired_method_comparisons'].append({'feature_mode':mode,'protocol':protocol,
            'reference':'rgat','comparator':method,'sign':'RGAT minus comparator; negative MRE difference favors RGAT',
            'paired_differences':paired,'same_record_set_verified':True})
    if args.full_budget_reference:
        reference=json.loads(args.full_budget_reference.read_text())
        expected={(m['sha256'],m['metadata_sidecar_sha256']) for m in summary['source_manifest']}
        actual={(m['sha256'],m['metadata_sidecar_sha256']) for m in reference['source_manifest']}
        if actual!=expected or reference['seeds']!=args.seeds or reference['epochs_max']!=args.epochs or reference['patience_validation_only']!=args.patience or reference['code_sha256']!=summary['code_sha256']:
            raise ValueError('full-budget reference differs in physical data, seeds or training policy')
        if reference['train_runtime']['device']!=summary['train_runtime']['device'] or reference['train_runtime']['torch']!=summary['train_runtime']['torch']:
            raise ValueError('learning-curve training runtime must match the full-budget reference')
        reused=[]
        for row in reference['aggregate_results']:
            if row['protocol']=='model_holdout' and row['method'] in args.methods and row['feature_mode'] in args.feature_modes:
                reused.append({**row,'protocol':'model_holdout_budget8_reused','training_model_budget':8,
                    'reused_from_summary_sha256':digest(args.full_budget_reference)})
        summary['learning_curve_full_budget_reference']={'path':str(args.full_budget_reference.resolve()),
            'sha256':digest(args.full_budget_reference),'reused_results':reused,
            'reason':'full8-model fits already executed in primary matrix; no duplicate test selection'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'complete':True,'results':len(summary['results']),'predictions':prediction_count,'embedding_rows':len(embedding_rows)}),flush=True)

if __name__=='__main__':main()
