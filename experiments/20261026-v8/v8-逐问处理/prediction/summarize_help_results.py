"""Complete reporting for the frozen HELP operator-domain experiment.
Analysis specified before reading any scientific model outputs. No selection.
"""
import csv,gzip,hashlib,json,math,statistics
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

OUT=Path(__file__).resolve().parent
FIELDS=['micro_MRE_percent','MAE_us','WAPE_percent','RMSE_ms','RMSPE_percent','Spearman','accuracy_within10percent']
CONTRASTS=[('help_three_probe_full','help_zero_probe_source_embedding'),('help_three_probe_full','pooled_three_probe_finetuned'),('help_three_probe_full','pooled_three_probe_conditioned'),('help_three_probe_full','help_three_probe_no_inner_update')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(rs):
    y=np.array([r['true_latency_ms'] for r in rs]);p=np.array([r['prediction_ms'] for r in rs]);ae=np.abs(p-y);re=ae/y
    corr=float(spearmanr(y,p).statistic) if len(set(p))>1 and len(set(y))>1 else None
    return {'n_queries':len(rs),'micro_MRE_percent':float(100*re.mean()),'MAE_us':float(1000*ae.mean()),'WAPE_percent':float(100*ae.sum()/y.sum()),'RMSE_ms':float(np.sqrt(np.mean((p-y)**2))),'RMSPE_percent':float(100*np.sqrt(np.mean(re**2))),'Spearman':corr,'accuracy_within10percent':float(100*np.mean(re<=.1))}

def main():
    proto=json.loads((OUT/'protocol.json').read_text());cells=json.loads((OUT/'frozen-splits.json').read_text());res=OUT/'results';res.mkdir(exist_ok=True)
    for i in range(4):assert json.loads((OUT/f'training-complete-shard{i}.json').read_text())['complete']
    groups=defaultdict(list);fits=[];cluster=defaultdict(lambda:defaultdict(lambda:[0.,0.,0.,0.,0.]))
    total=0;testtotal=0;clipcount=defaultdict(int)
    for c in cells:
        for seed in proto['seeds']:
            d=OUT/'runs'/f'{c["cell_id"]}-seed{seed}';receipt=json.loads((d/'receipt.json').read_text());assert receipt['complete'] and receipt['seed']==seed
            assert all(sha(d/s['path'])==s['sha256'] for s in receipt['outputs'])
            fits.append({'cell_id':c['cell_id'],'seed':seed,'test_stack':c['test_stack'],'validation_stack':c['validation_stack'],'train_n':c['train_n'],'validation_n':c['validation_n'],'test_n':c['test_n'],'checkpoint_sha256':sha(d/'checkpoint.pt'),'training_episodes':receipt['training_episodes'],'seconds':receipt['seconds']})
            for phase,ms in receipt['log_output_clipping_counts'].items():
                for method,n in ms.items():clipcount[(phase,method)]+=n
            with gzip.open(d/'predictions.jsonl.gz','rt') as f:
                for line in f:
                    r=json.loads(line);total+=1
                    for scope,val in [('pooled','all'),('per_stack',r['device_id']),('per_model',r['model_id']),('per_family',r['family']),('per_cell',r['cell_id'])]:groups[(r['phase'],scope,val,r['method'],seed)].append(r)
                    if r['phase']=='test':
                        testtotal+=1;a=abs(r['prediction_ms']-r['true_latency_ms']);s=cluster[(r['method'],seed)][r['model_id']]
                        s[0]+=1;s[1]+=100*a/r['true_latency_ms'];s[2]+=1000*a;s[3]+=r['true_latency_ms'];s[4]+=a*a
    assert len(fits)==150 and testtotal==13905*5*7
    rows=[]
    for (phase,scope,val,method,seed),rr in sorted(groups.items()):
        ids=[r['record_id'] for r in rr];assert len(set(ids))==len(ids)
        rows.append({'phase':phase,'scope':scope,'scope_value':val,'method':method,'seed':seed,'support_sha256':hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),**metrics(rr)})
    agg=[]
    regroup=defaultdict(list)
    for r in rows:regroup[(r['phase'],r['scope'],r['scope_value'],r['method'])].append(r)
    for (phase,scope,val,method),rr in sorted(regroup.items()):
        assert len(rr)==5 and len({r['support_sha256'] for r in rr})==1
        row={'phase':phase,'scope':scope,'scope_value':val,'method':method,'n_seeds':5,'n_queries':rr[0]['n_queries']}
        for k in FIELDS:
            vals=[r[k] for r in rr if r[k] is not None];mean=statistics.mean(vals) if vals else None;sd=statistics.stdev(vals) if len(vals)>1 else None
            row.update({k+'_seed_mean':mean,k+'_seed_sd':sd,k+'_conditional_seed_t95_low':None if sd is None else mean-2.7764451051977987*sd/math.sqrt(len(vals)),k+'_conditional_seed_t95_high':None if sd is None else mean+2.7764451051977987*sd/math.sqrt(len(vals)),k+'_defined_seeds':len(vals)})
        agg.append(row)
    comparisons=[];models=sorted({m for d in cluster.values() for m in d});assert len(models)==12
    weights=np.random.default_rng(20261026+12).multinomial(12,np.full(12,1/12),size=10000)
    for first,second in CONTRASTS:
        left=np.array([np.mean([cluster[(first,s)][m] for s in proto['seeds']],axis=0) for m in models]);right=np.array([np.mean([cluster[(second,s)][m] for s in proto['seeds']],axis=0) for m in models]);assert np.array_equal(left[:,0],right[:,0])
        n=left[:,0];delta=left[:,1]-right[:,1];micro=delta.sum()/n.sum();mac=(delta/n).mean()
        boot_micro=(weights@delta)/(weights@n);boot_macro=(weights@(delta/n))/12
        comparisons.append({'first':first,'second':second,'sign':'first minus second; negative favors first','micro_MRE_difference_pp':float(micro),'model_macro_MRE_difference_pp':float(mac),'conditional_model_cluster95_micro_pp':[float(x) for x in np.quantile(boot_micro,[.025,.975])],'conditional_model_cluster95_macro_pp':[float(x) for x in np.quantile(boot_macro,[.025,.975])],'models_improved':int((delta<0).sum()),'models_worse':int((delta>0).sum()),'models_tied':int((delta==0).sum()),'cluster_unit':'source model, all 5 target stacks and seed-average errors retained together','n_clusters':12,'draws':10000,'population_or_multiple_testing_claim':False})
    csvout(res/'per-seed-metrics.csv',rows);csvout(res/'aggregate-metrics.csv',agg);csvout(res/'fit-ledger.csv',fits);dump(res/'paired-model-cluster-comparisons.json',comparisons)
    summary={'complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.json'),'analysis_code_sha256':sha(Path(__file__)),'fits_HELP':150,'fits_pooled':150,'prediction_rows_validation_and_test':total,'test_prediction_rows':testtotal,'unique_test_queries_per_seed_method':13905,'training_seeds':proto['seeds'],'pooled_test_results':[r for r in agg if r['phase']=='test' and r['scope']=='pooled'],'per_stack_test_results':[r for r in agg if r['phase']=='test' and r['scope']=='per_stack'],'paired_comparisons':comparisons,'log_prediction_cap_counts':[{'phase':ph,'method':m,'count':n} for (ph,m),n in sorted(clipcount.items())],'interval_limits':'Seed intervals describe optimizer randomness conditional on fixed labels and split; paired source-model cluster intervals describe this 12-model cohort. Five stacks are not five independent physical architectures. No significance or broad population guarantee.','source_and_target_scope':'Both target runtime stack and target source models are held out; fixed independent 3-query probes are disjoint from the scoring cohort. Original historical 5-platform LOPO remains unverified.','new_measurements':0,'failures_found':[str(p.relative_to(OUT)) for p in (OUT/'runs').glob('*/failure*.json')],'outputs':[{'path':str(p.relative_to(OUT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(res.glob('*')) if p.is_file()]}
    dump(res/'summary.json',summary);print(json.dumps({'complete':True,'fits':300,'test_prediction_rows':testtotal,'pooled_results':summary['pooled_test_results'],'comparisons':comparisons},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
