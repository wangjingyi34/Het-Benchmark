"""Independent scalar audit and process-level descriptive summaries."""
import csv,hashlib,json,math,statistics,copy
from pathlib import Path
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(d,p,protocol_hash):
 assert d['status']=='completed' and d['protocol_sha256']==protocol_hash
 assert len(d['rows'])==8 and len({r['id'] for r in d['rows']})==8
 for row in d['rows']:
  i=next(j for j,c in enumerate(p['configurations']) if c['id']==row['id'])
  assert row['identity']==p['identities'][i]
  assert row['status']=='measured' and row['finite']
  assert len(row['timings'])==100
  for t in row['timings']:
   assert t['end_ns']>t['start_ns'] and math.isfinite(t['latency_ms'])
   assert math.isclose(t['latency_ms'],(t['end_ns']-t['start_ns'])/1e6,rel_tol=1e-12)
  assert all(x['end_ns']<=y['start_ns'] for x,y in zip(row['timings'],row['timings'][1:]))
  vals=[t['latency_ms'] for t in row['timings']]
  assert math.isclose(row['mean_ms'],statistics.mean(vals),rel_tol=1e-12)
  assert math.isclose(row['sd_ms'],statistics.stdev(vals),rel_tol=1e-12)
 return True
def main():
 p=json.loads((P/'configuration-protocol.json').read_text());h=sha(P/'configuration-protocol.json');assert sha(P/'configuration_remeasurement.py')==p['script_sha256']
 rows=[];files=[];raw=[]
 for backend in ['a100mig','m5mps']:
  fs=sorted((P/backend).glob('process-*.json'));assert len(fs)==5
  data=[json.loads(f.read_text()) for f in fs];assert sorted(d['process'] for d in data)==list(range(1,6));raw+=data
  for f,d in zip(fs,data):
   validate(d,p,h);files.append({'path':str(f.relative_to(P)),'sha256':sha(f)})
   assert p['created_utc']<d['created_utc']<d['completed_utc']
   if backend=='a100mig':assert 'A100' in d['hardware']['device_name'] and 19e9<d['hardware']['memory_bytes']<23e9
   else:assert d['hardware']['chip']=='Apple M5'
  for i,c in enumerate(p['configurations']):
   means=[next(r for r in d['rows'] if r['id']==c['id'])['mean_ms'] for d in data];mu=statistics.mean(means);sd=statistics.stdev(means);half=2.7764451051977987*sd/math.sqrt(5)
   rows.append({'backend':backend,'workload':c['id'],'shape':str(c['input']),'operation':c['operation'],'layout':c['layout'],'dtype':'float16','processes':5,'timed_calls':500,'mean_ms':mu,'process_sd_ms':sd,'descriptive_t95_low_ms':mu-half,'descriptive_t95_high_ms':mu+half,'input_sha256':p['identities'][i]['input_sha256'],'weights_sha256':p['identities'][i]['weights_sha256']})
 # Two corruption controls ensure identity and individual observations are checked.
 controls=[]
 for kind in ['input_hash','raw_timing']:
  x=copy.deepcopy(raw[0])
  if kind=='input_hash':x['rows'][0]['identity']['input_sha256']='0'*64
  else:x['rows'][0]['timings'][0]['latency_ms']+=1
  try:validate(x,p,h)
  except AssertionError:controls.append({'corruption':kind,'rejected':True})
  else:raise AssertionError('corruption was not detected')
 with (P/'configuration-measurements.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 out={'status':'passed','protocol_sha256':h,'audit_script_sha256':sha(__file__),'source_runs':files,'fresh_processes':10,'workload_process_rows':80,'individual_timing_calls':8000,'warmup_calls':1600,'all_input_weight_hashes_match':True,'failures':0,'negative_controls':controls,'rows':rows,'scope':'Fresh FP16 eval/inference measurements of exactly stated configurations; no original Table4 full-device identity or historical4.35/4.441 authentication. Intervals describe five process means, not a hardware-population effect. Untrained weights have no accuracy claim.'}
 (P/'configuration-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({'status':'passed','rows':80,'calls':8000,'controls':controls,'latency_rows':[{k:r[k] for k in ['backend','workload','mean_ms']} for r in rows]}))
if __name__=='__main__':main()
