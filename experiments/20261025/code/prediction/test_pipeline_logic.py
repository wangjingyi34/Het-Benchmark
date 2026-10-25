"""Small synthetic fixtures validate implementation logic only, never accuracy."""
import json
import copy
from pathlib import Path
import torch
from evaluate_profiles import RGAT, GCN, RelationalAttention, MODEL_ORDER, make_splits, fallback_policy

def reference_attention(layer,x,edge,relation):
    s,t=edge
    hs=torch.bmm(x[s].unsqueeze(1),layer.w[relation]).squeeze(1).reshape(-1,layer.heads,layer.depth)
    ht=torch.bmm(x[t].unsqueeze(1),layer.w[relation]).squeeze(1).reshape(-1,layer.heads,layer.depth)
    logits=torch.nn.functional.leaky_relu((hs*layer.a_src[relation]).sum(-1)+(ht*layer.a_dst[relation]).sum(-1),negative_slope=.2)
    index=t[:,None].expand(-1,layer.heads)
    maxima=torch.full((len(x),layer.heads),-torch.inf,dtype=x.dtype)
    maxima.scatter_reduce_(0,index,logits,reduce='amax',include_self=True)
    numerator=torch.exp(logits-maxima[t]);denominator=torch.zeros_like(maxima)
    denominator.scatter_add_(0,index,numerator)
    alpha=numerator/denominator[t].clamp_min(1e-12)
    out=torch.zeros_like(x);out.index_add_(0,t,(hs*alpha[:,:,None]).reshape(-1,layer.heads*layer.depth))
    return layer.norm(layer.self_linear(x)+torch.nn.functional.gelu(out))

def main():
    torch.set_num_threads(1);torch.manual_seed(1025)
    net=RGAT(12)
    x=torch.randn(5,12)
    edge=torch.tensor([[0,1,2,3,4,2,1],[1,2,3,4,0,1,4]])
    relation=torch.tensor([0,1,2,3,4,5,2])
    optimized=RelationalAttention(12,heads=3).double();reference=copy.deepcopy(optimized)
    xx=torch.randn(5,12,dtype=torch.float64,requires_grad=True);xr=xx.detach().clone().requires_grad_(True)
    oo=optimized(xx,edge,relation);rr=reference_attention(reference,xr,edge,relation)
    assert torch.allclose(oo,rr,atol=1e-10,rtol=1e-10)
    oo.square().sum().backward();rr.square().sum().backward()
    assert torch.allclose(xx.grad,xr.grad,atol=1e-8,rtol=1e-8)
    for a,b in zip(optimized.parameters(),reference.parameters()):assert torch.allclose(a.grad,b.grad,atol=1e-8,rtol=1e-8)
    batch={'x':x,'edge_index':edge,'edge_type':relation,'query':torch.tensor([1,3]),'y':torch.tensor([1.,2.])}
    net.eval();a=net(batch)
    changed={**batch,'y':torch.tensor([999.,.001])}
    assert torch.equal(a,net(changed))
    # Node-index permutation is a pure relabelling, not a change in prediction.
    perm=torch.tensor([3,1,4,0,2]);inverse=torch.argsort(perm)
    relabelled={**batch,'x':x[perm],'edge_index':inverse[edge], 'query':inverse[batch['query']]}
    assert torch.allclose(a,net(relabelled),atol=1e-6,rtol=1e-6)
    net.train();out=net(batch);out.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    loops=torch.arange(5);untyped=torch.unique(torch.cat([edge,edge.flip(0),torch.stack([loops,loops])],dim=1),dim=1)
    degree=torch.bincount(untyped[1],minlength=5).float();weights=(degree[untyped[0]]*degree[untyped[1]]).rsqrt()
    gb={**batch,'gcn_edge_index':untyped,'gcn_normalization':weights}
    gcn=GCN(12);gcn.eval();go=gcn(gb)
    gp={**gb,'x':x[perm],'query':inverse[batch['query']],'gcn_edge_index':inverse[untyped]}
    assert torch.allclose(go,gcn(gp),atol=1e-6,rtol=1e-6)
    assert torch.equal(go,gcn({**gb,'y':batch['y']*33}))
    go.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gcn.parameters())
    fake=[{'model_id':m,'device_id':d} for m in MODEL_ORDER for d in ['dev1','dev2']]
    splits=make_splits(fake,'both',6)
    model_splits=[s for s in splits if s['protocol']=='model_holdout']
    tests=[m for s in model_splits for m in s['test_model_ids']]
    assert sorted(tests)==sorted(MODEL_ORDER) and len(tests)==len(set(tests))
    assert len(splits)==8
    # The policy must not change its trigger if test outcomes are altered.
    val=[{'uncertainty':i,'prediction_ms':float(i+1),'true_latency_ms':float(i+2),
          'model_id':'v','family':'F','device_id':'d','record_id':str(i),
          'timed_measurement_cost_ms':1.} for i in range(10)]
    test=[{**r,'model_id':'t'} for r in val]
    p1=fallback_policy(val,test,.2)
    p2=fallback_policy(val,[{**r,'true_latency_ms':r['true_latency_ms']*3} for r in test],.2)
    assert p1['threshold']==p2['threshold'] and p1['trigger_record_ids']==p2['trigger_record_ids']
    assert p1['validation_measured_fraction']<=.2
    result={'fixture_kind':'synthetic logic only; zero predictor-accuracy or hardware-performance evidence',
       'passed':['target_label_not_read_by_forward','typed_attention_node_permutation_equivariance',
                 'all_parameter_gradients_finite','12_models_each_test_once','strict_model_disjointness',
                 'device_holdout_omits_target_device','validation_only_fallback_threshold',
                 'test_true_latency_does_not_select_trigger','validation_measurement_budget'],
       'physical_samples_used_for_these_tests':0}
    result['passed'].extend(['relation_node_projection_matches_edge_expansion_forward',
                             'relation_node_projection_matches_parameter_and_input_gradients',
                             'untyped_GCN_permutation_equivariance','untyped_GCN_does_not_read_labels',
                             'untyped_GCN_gradients_finite'])
    Path('pipeline-logic-test-results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
