"""Independent stdlib audit of frozen inputs, all raw timing blocks and endpoints."""
import hashlib,json,math,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
checks=[];count=0
for backend in ['a100mig','m5mps']:
 d=read(P/(backend+'-decisions.json'));protocol=read(P/'protocol.json');assert d['protocol_sha256']==sha(P/'protocol.json')
 expected=sorted(d['decisions'],key=lambda x:(-x['uncertainty'],x['case']))[:3];assert [x['case'] for x in expected]==d['anchors']
 c=read(P/backend/'correction.json');a=read(P/backend/'anchors.json');assert c['anchor_measurements_sha256']==sha(P/backend/'anchors.json')
 assert protocol['created_utc']<d['created_utc']<a['created_utc']<c['created_utc']
 truth={r['case']:r['latency_mean_ms'] for r in a['rows']}
 for r in c['predictions']:
  i=r['case'];dist=[sum((math.log2(x)-math.log2(y))**2 for x,y in zip(protocol['configurations'][i],protocol['configurations'][j])) for j in d['anchors']]
  w=[math.exp(-.5*(x-min(dist))) for x in dist];w=[x/sum(w) for x in w]
  close(r['calibrated_ms'],r['pure_ms']*math.exp(sum(x*math.log(truth[j]/d['decisions'][j]['prediction_ms']) for x,j in zip(w,d['anchors']))))
 s=read(P/backend/'summary.json');reference={}
 for n in range(1,6):
  m=read(P/backend/f'eval-{n}.json');assert m['correction_sha256']==sha(P/backend/'correction.json') and c['created_utc']<m['created_utc'];assert not m['errors']
  for r in m['rows']:
   assert len(r['latency_samples_ms'])==10 and all(math.isfinite(t) and t>0 for t in r['latency_samples_ms']);close(statistics.mean(r['latency_samples_ms']),r['latency_mean_ms']);count+=1
   reference.setdefault(r['case'],[]).append(r['latency_mean_ms'])
 assert len(reference)==21 and not set(reference)&set(d['anchors'])
 for k in ['pure','calibrated']:
  errors=[abs(r[k+'_ms']-statistics.mean(reference[r['case']])) for r in c['predictions']];actual=[statistics.mean(reference[r['case']]) for r in c['predictions']]
  close(s['metrics'][k]['mre_pct'],100*statistics.mean(e/t for e,t in zip(errors,actual)));close(s['metrics'][k]['wape_pct'],100*sum(errors)/sum(actual))
 checks.append({'backend':backend,'experiment':'three_query','metrics':s['metrics'],'passed':True})
 f=P/('support-'+backend);r=read(f/'reference.json');sp=read(P/'support-protocol.json');assert r['protocol_sha256']==sha(P/'support-protocol.json') and sp['created_utc']<r['created_utc'];assert len(r['rows'])==24
 reference={}
 for n in range(1,6):
  m=read(f/f'repeat-{n}.json');assert m['reference_sha256']==sha(f/'reference.json') and r['created_utc']<m['created_utc'] and not m['failures']
  for x in m['rows']:
   assert len(x['latency_samples_ms'])==10 and all(math.isfinite(t) and t>0 for t in x['latency_samples_ms']);close(statistics.mean(x['latency_samples_ms']),x['latency_mean_ms']);count+=1
   reference.setdefault(x['case'],[]).append(x['latency_mean_ms'])
 errors=[abs(x['latency_mean_ms']-statistics.mean(reference[x['case']])) for x in r['rows']];truth=[statistics.mean(reference[x['case']]) for x in r['rows']];s=read(f/'summary.json')
 close(s['mre_pct'],100*statistics.mean(e/t for e,t in zip(errors,truth)));close(s['mae_ms'],statistics.mean(errors));close(s['wape_pct'],100*sum(errors)/sum(truth))
 checks.append({'backend':backend,'experiment':'support_gate','mre_pct':s['mre_pct'],'profile_seconds':s['profiling_wall_seconds'],'passed':True})
for family in ['','support-']:
 name='anchors.json' if not family else 'reference.json';aa=read(P/(family+'a100mig')/name);bb=read(P/(family+'m5mps')/name)
 ha={x['case']:x['input_weights_sha256'] for x in aa['rows']};hb={x['case']:x['input_weights_sha256'] for x in bb['rows']}
 assert all(ha[i]==hb[i] for i in ha.keys()&hb.keys())
out={'status':'passed','independent_implementation':'Python stdlib math/statistics, no original summarizer import','evaluation_query_process_rows_verified':count,'checks':checks,
     'scope':'Audits prospective file ordering, exact frozen decisions, independent correction equations, all timing means and reported endpoints. Does not authenticate the machine outside recorded runtime evidence.'}
(P/'independent-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
