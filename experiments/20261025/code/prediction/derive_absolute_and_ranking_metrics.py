"""Read-only MAE/WAPE/ranking/regret on the unchanged physical query support.

No parameter selection, tiny-op removal, profiling, or training is performed.
Operator sums are isolated-call diagnostics, never full-model application time.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(c,m):
    if not c:raise ValueError(m)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference-summary',required=True,type=Path)
    ap.add_argument('--profiles-dir',required=True,type=Path);ap.add_argument('--prediction-files',nargs='+',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    require(not args.output.exists(),'do not overwrite derived analysis')
    reference=json.loads(args.reference_summary.read_text());records=[]
    for source in reference['source_manifest']:
        p=args.profiles_dir/Path(source['path']).name;require(sha(p)==source['sha256'],'physical reference hash differs')
        raw=json.loads(p.read_text());device=raw['hardware']['device_id']
        for graph in raw['graphs']:
            nodes={n['node_id']:n for n in graph['nodes']}
            for r in graph['records']:
                if r['status']!='measured':continue
                y=statistics.mean(r['latency_samples_ms'])
                records.append({'record_id':r['record_id'],'node_id':r['op_instance_id'],'model_id':graph['model_id'],
                                'device_id':device,'family':nodes[r['op_instance_id']]['family'],
                                'graph_id':graph['graph_id'],'true_ms':y})
    require(len(records)==13905,'frozen main query support changed')
    ids={r['record_id']:i for i,r in enumerate(records)};require(len(ids)==len(records),'duplicate physical query')
    true=np.array([r['true_ms'] for r in records]);support_sha=hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
    groups={};file_hashes=[];row_count=0
    for path in args.prediction_files:
        file_hashes.append({'path':str(path.resolve()),'sha256':sha(path)})
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt') as stream:
            for line in stream:
                row=json.loads(line);key=(row['feature_mode'],row['method'],row['protocol'],str(row.get('seed','seedless')))
                i=ids.get(row['record_id']);require(i is not None,'prediction not in frozen record support')
                require(row['model_id']==records[i]['model_id'] and row['device_id']==records[i]['device_id'] and row['family']==records[i]['family'],'prediction semantic identity differs')
                require(math.isclose(row['true_latency_ms'],true[i],rel_tol=1e-9,abs_tol=1e-10),'prediction physical truth differs')
                p=float(row['prediction_ms']);require(math.isfinite(p) and p>0,'invalid prediction')
                if key not in groups:groups[key]=np.full(len(records),np.nan)
                require(np.isnan(groups[key][i]),'duplicate method/protocol/seed query')
                groups[key][i]=p;row_count+=1
    # Pair weights are fixed from independent physical labels. Every graph's
    # operator pair remains included, including small View/identity operations.
    graph_groups=collections.defaultdict(list);family_groups=collections.defaultdict(list);device_groups=collections.defaultdict(list)
    device_names=sorted({r['device_id'] for r in records});node_names=sorted({r['node_id'] for r in records})
    lookup={(r['node_id'],r['device_id']):i for i,r in enumerate(records)}
    device_matrix=np.array([[lookup[(node,dev)] for dev in device_names] for node in node_names])
    tmat=true[device_matrix];oracle=tmat.min(axis=1)
    left=[];right=[];weight=[];sign=[];top_oracle=0.;top_configs=[]
    for i,r in enumerate(records):
        graph_groups[(r['device_id'],r['graph_id'])].append(i);family_groups[r['family']].append(i);device_groups[r['device_id']].append(i)
    for graph,indices in graph_groups.items():
        indices=np.array(indices);li,ri=np.triu_indices(len(indices),1)
        a=indices[li];b=indices[ri];delta=true[a]-true[b]
        left.extend(a);right.extend(b);weight.extend(np.abs(delta));sign.extend(np.sign(delta))
        k=max(1,math.ceil(.1*len(indices)));top_oracle+=float(np.sort(true[indices])[-k:].sum());top_configs.append((indices,k))
    left=np.array(left);right=np.array(right);weight=np.array(weight);sign=np.array(sign)
    positive_weight=weight>0;left=left[positive_weight];right=right[positive_weight];sign=sign[positive_weight];weight=weight[positive_weight]
    outputs=[]
    for key,pred in sorted(groups.items()):
        require(np.isfinite(pred).all(),'incomplete full-support prediction group')
        absolute=np.abs(pred-true);order=np.sign(pred[left]-pred[right]);correct=(order==sign).astype(float)+.5*(order==0)
        capture=sum(float(true[ix[np.argpartition(pred[ix],len(ix)-k)[-k:]]].sum()) for ix,k in top_configs)
        chosen=np.argmin(pred[device_matrix],axis=1);actual=tmat[np.arange(len(node_names)),chosen];regret=actual-oracle
        row={'feature_mode':key[0],'method':key[1],'protocol':key[2],'seed':key[3],
             'records':len(records),'evaluation_record_ids_sha256':support_sha,
             'micro_MRE_percent':float(np.mean(absolute/np.maximum(true,1e-6))*100),
             'MAE_ms':float(absolute.mean()),'MAE_microseconds':float(absolute.mean()*1000),
             'WAPE_percent':float(absolute.sum()/true.sum()*100),'sum_true_isolated_latency_ms':float(true.sum()),
             'operator_latency_gap_weighted_pair_order_accuracy_percent':float(np.dot(weight,correct)/weight.sum()*100),
             'operator_top10percent_oracle_latency_capture_percent':float(capture/top_oracle*100),
             'cross_stack_operator_fastest_device_hit_percent':float(np.mean(actual==oracle)*100),
             'cross_stack_operator_device_choice_mean_regret_percent':float(np.mean(regret/np.maximum(oracle,1e-6))*100),
             'cross_stack_operator_device_choice_oracle_latency_weighted_regret_percent':float(regret.sum()/oracle.sum()*100),
             'cross_stack_operator_device_choice_mean_absolute_regret_ms':float(regret.mean()),
             'cross_stack_choice_groups':len(node_names),'per_family':{},'per_runtime_stack':{},
             'cross_stack_scope':'Same-architecture/input/weights query across5stacks. Device-holdout uses stitched crossvalidated predictors, not a single deployed model or a complete migration plan.'}
        for label,indexmap in [('per_family',family_groups),('per_runtime_stack',device_groups)]:
            for name,indices in indexmap.items():
                index=np.array(indices);ae=absolute[index];ts=true[index]
                row[label][name]={'n':len(index),'MAE_ms':float(ae.mean()),'MAE_microseconds':float(ae.mean()*1000),
                                  'WAPE_percent':float(ae.sum()/ts.sum()*100),'micro_MRE_percent':float(np.mean(ae/np.maximum(ts,1e-6))*100)}
        outputs.append(row)
    out={'kind':'read_only_absolute_error_and_operator_ranking_diagnostics','created_utc':datetime.now(timezone.utc).isoformat(),
         'code_sha256':sha(__file__),'reference_summary_sha256':sha(args.reference_summary),'prediction_sources':file_hashes,
         'prediction_rows_read':row_count,'groups':outputs,'no_parameter_or_threshold_reselection':True,'no_query_or_family_removed':True,
         'scope':['MAE and WAPE supplement, and never replace, the original instance micro-MRE.',
                  'All groups use the same13905 operator records; independent physical means supply ground truth.',
                  'Weighted pair ordering pools within-graph operator pairs weighted by absolute physical latency gap; predicted ties receive half credit.',
                  'Top10% capture compares the true isolated latency of predicted top-ceil(10%×n) operators with the oracle top set.',
                  'Device-choice regret compares the selected stack with the physically fastest stack for the identical operator query; communication, resource capacity, quality and total model execution are outside this diagnostic.',
                  'Operator sums and probe block counts are not application end-to-end latency or independent architecture samples.']}
    args.output.mkdir(parents=True);(args.output/'metrics.json').write_text(json.dumps(out,indent=2))
    flat=[{k:v for k,v in row.items() if not isinstance(v,dict)} for row in outputs]
    with (args.output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps({'complete':True,'groups':len(outputs),'prediction_rows':row_count,'support_each':len(records),'output':str(args.output)}))


if __name__=='__main__':main()
