"""Account for every captured FX node in the independently measured cohort.

The structural 34-model catalogue and original 5000-edge claim are not inputs.
"""
import argparse,csv,hashlib,json,statistics,math,collections
from pathlib import Path
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profiles',type=Path,required=True);ap.add_argument('--output',type=Path,default=P/'measured-ledger');a=ap.parse_args();a.output.mkdir(exist_ok=True)
 rows=[];summaries=[];sources=[];identities={};configurations={};allids=set()
 for name in ['m5cpu','m5mps','xeon_cpu','a100mig','npub3']:
  f=a.profiles/(name+'.json');d=json.loads(f.read_text());did=d['hardware']['device_id'];counts=collections.Counter();status=collections.Counter();graphids=[]
  sources.append({'file':f.name,'sha256':sha(f),'graphs':len(d['graphs']),'collection_failures':d['failures']})
  assert len(d['graphs'])==36 and not d['failures']
  for g in d['graphs']:
   graphids.append(g['graph_id']);portable_shapes=[{k:v for k,v in x.items() if k!='backend_storage_format'} for x in g['input_shapes']];identity=(g['input_weights_sha256'],json.dumps(portable_shapes,sort_keys=True),g['dtype'])
   if g['graph_id'] in identities:assert identities[g['graph_id']]==identity
   identities[g['graph_id']]=identity;configurations[g['graph_id']]=g['model_id']
   recs={r['op_instance_id']:r for r in g['records']};assert len(recs)==len(g['records'])
   for n in g['nodes']:
    counts['all_fx_nodes']+=1;key=did+'/'+n['node_id'];assert key not in allids;allids.add(key)
    r=recs.get(n['node_id']);exclusion=''
    if r:
     assert r['status']=='measured';t=r['latency_samples_ms'];assert len(t)==20 and all(math.isfinite(x) and x>0 for x in t)
     assert math.isclose(statistics.mean(t),r['latency_mean_ms'],rel_tol=1e-12)
     s='measured';counts['measured']+=1;counts['timing_blocks']+=len(t)
    else:
     assert n['family']=='Context'
     if n['fx_op'] in ['placeholder','output']:exclusion='graph_input_or_output'
     else:
      assert n['fx_op']=='call_method' and n['target']=='size';exclusion='integer_shape_query'
     s='excluded_context_or_non_tensor_execution';counts[s]+=1
    status[s]+=1
    rows.append({'device_id':did,'graph_id':g['graph_id'],'model_id':g['model_id'],'shape_id':g['shape_id'],'node_id':n['node_id'],'fx_op':n['fx_op'],'family':n['family'],'status':s,'exclusion_reason':exclusion,'run_id':r.get('run_id','') if r else '', 'input_weights_sha256':g['input_weights_sha256'],'profile_file':f.name,'profile_sha256':sources[-1]['sha256'],'raw_blocks':len(r['latency_samples_ms']) if r else 0,'latency_mean_ms':r['latency_mean_ms'] if r else ''})
  assert len(set(graphids))==36
  summaries.append({'device_id':did,**dict(counts),'failed_record':0,'unsupported_record':0,'eligible_missing_record':0,'derived_record':0,'boundary':'Only the executed 12-architecture/3-configuration cohort; no coverage assertion for unexecuted models or original historical hardware.'})
 with (a.output/'node-status-ledger.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (a.output/'platform-status-counts.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
 out={'status':'passed','cohort_models':len(set(configurations.values())),'graph_configurations':len(identities),'platform_graphs':180,'all_fx_node_platform_rows':len(rows),'measured_query_platform_records':sum(r['measured'] for r in summaries),'excluded_context_or_non_tensor_execution_nodes':sum(r['excluded_context_or_non_tensor_execution'] for r in summaries),'measured_timing_blocks':sum(r['timing_blocks'] for r in summaries),'all_matching_graph_input_hashes_agree':True,'shape_comparison_scope':'Portable shape/dtype/stride/element metadata agree; Ascend-only backend_storage_format is retained in raw profiles but excluded from this comparison. This does not assert identical physical storage layout.','sources':sources,'platforms':summaries,'original_5000_claim_authenticated':False,'scope':'Recount and identity verification of existing raw physical evidence; no new measurement or inferred labels.'}
 (a.output/'ledger-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['sources','platforms']}))
if __name__=='__main__':main()
