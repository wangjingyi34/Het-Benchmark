"""Post-result audit only: independent cluster arithmetic and source embeddings.

No fitting or scientific protocol changes. Base metrics were independently
checked against every raw prediction by audit_help_independent.py.
"""
import csv, hashlib, json, math, statistics
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    proto=json.loads((OUT/'protocol.json').read_text())
    cells=json.loads((OUT/'frozen-splits.json').read_text())
    probes=json.loads((OUT/'independent-probes.json').read_text())
    max_embedding_difference=0.;embeddings_checked=0
    names=['matmul256','conv2d16','layernorm128']
    for cell in cells:
        embedded=[]
        for device in cell['train_stacks']:
            selected=sorted([p for p in probes if p['device_id']==device],key=lambda p:names.index(p['probe_name']))
            assert len(selected)==3
            y=torch.tensor([p['true_latency_ms'] for p in selected],dtype=torch.float32)
            embedded.append((y-y.min())/(y.max()-y.min()).clamp_min(1e-12))
        expected=torch.stack(embedded).mean(0)
        assert cell['test_stack'] not in cell['train_stacks'] and cell['validation_stack'] not in cell['train_stacks']
        for seed in proto['seeds']:
            checkpoint=torch.load(OUT/'runs'/f'{cell["cell_id"]}-seed{seed}'/'checkpoint.pt',weights_only=True,map_location='cpu')
            delta=float((expected-checkpoint['source_mean_embedding']).abs().max())
            max_embedding_difference=max(max_embedding_difference,delta)
            assert delta==0
            embeddings_checked+=1
    with (OUT/'results/per-seed-metrics.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['phase']=='test' and r['scope']=='per_model']
    models=sorted({r['scope_value'] for r in rows});assert len(models)==12
    counts={model:int(next(r['n_queries'] for r in rows if r['scope_value']==model)) for model in models}
    observed=json.loads((OUT/'results/paired-model-cluster-comparisons.json').read_text())
    weights=np.random.default_rng(20261038).multinomial(12,np.full(12,1/12),size=10000).tolist()
    max_scalar_delta=0.;max_interval_delta=0.;recomputed=[]
    def percentile(values,q):
        values=sorted(values);index=(len(values)-1)*q;lo=math.floor(index);hi=math.ceil(index)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    for contrast in observed:
        differences=[]
        for model in models:
            values={method:statistics.mean(float(r['micro_MRE_percent']) for r in rows if r['method']==method and r['scope_value']==model) for method in [contrast['first'],contrast['second']]}
            differences.append(values[contrast['first']]-values[contrast['second']])
        n=[counts[model] for model in models]
        micro=math.fsum(d*c for d,c in zip(differences,n))/sum(n)
        macro=statistics.mean(differences)
        bm=[math.fsum(w*d*c for w,d,c in zip(weight,differences,n))/sum(w*c for w,c in zip(weight,n)) for weight in weights]
        ba=[math.fsum(w*d for w,d in zip(weight,differences))/12 for weight in weights]
        ci_micro=[percentile(bm,q) for q in [.025,.975]];ci_macro=[percentile(ba,q) for q in [.025,.975]]
        for actual,ref in [(micro,contrast['micro_MRE_difference_pp']),(macro,contrast['model_macro_MRE_difference_pp'])]:
            max_scalar_delta=max(max_scalar_delta,abs(actual-ref));assert math.isclose(actual,ref,abs_tol=1e-8)
        for actual,ref in zip(ci_micro+ci_macro,contrast['conditional_model_cluster95_micro_pp']+contrast['conditional_model_cluster95_macro_pp']):
            max_interval_delta=max(max_interval_delta,abs(actual-ref));assert math.isclose(actual,ref,abs_tol=1e-8)
        assert sum(d<0 for d in differences)==contrast['models_improved']
        assert sum(d>0 for d in differences)==contrast['models_worse']
        recomputed.append({'first':contrast['first'],'second':contrast['second'],'micro_difference_pp':micro,'macro_difference_pp':macro,'micro95':ci_micro,'macro95':ci_macro})
    out={'complete':True,'utc':datetime.now(timezone.utc).isoformat(),'code_sha256':sha(__file__),'scope':'Post-result numerical audit; no retraining, new measurements, selection, or scientific protocol change. Uses independent scalar reconstruction from per-model metrics already verified from raw predictions.','source_mean_embeddings_reconstructed_from_training_probes':embeddings_checked,'maximum_source_embedding_difference':max_embedding_difference,'cluster_contrasts_independently_recomputed':len(recomputed),'draws_per_contrast':10000,'maximum_scalar_difference_pp':max_scalar_delta,'maximum_interval_endpoint_difference_pp':max_interval_delta,'recomputed':recomputed,'inputs':[{'path':str(p.relative_to(OUT)),'sha256':sha(p)} for p in [OUT/'protocol.json',OUT/'results/per-seed-metrics.csv',OUT/'results/paired-model-cluster-comparisons.json',OUT/'results/independent-audit.json']]}
    (OUT/'results/independent-cluster-source-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
