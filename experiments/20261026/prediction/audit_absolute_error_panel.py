"""Independent scalar recomputation of every detailed point-estimate row."""
import csv
import gzip
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]
KEYS=('protocol','feature_mode','method','split_id','seed')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    result=OUT/'absolute-error-results'
    with (result/'per-fold-seed-model.csv').open() as f:
        panel={tuple(r[k] for k in KEYS)+ (r['model_id'],):r for r in csv.DictReader(f)}
    seen=set(); max_delta=defaultdict(float); count=0
    source=BASE/'v2-实证增强/public-reproduction-check/prediction/formal-five-stack/predictions.jsonl.gz'
    with gzip.open(source,'rt') as f:
        stream=(json.loads(s) for s in f)
        for key,rows in itertools.groupby(stream,key=lambda r:tuple(str(r[k]) for k in KEYS)):
            groups=defaultdict(list)
            for r in rows:groups[r['model_id']].append(r);count+=1
            for model,rs in groups.items():
                k=key+(model,);assert k not in seen;seen.add(k);p=panel[k]
                ae=[abs(r['prediction_ms']-r['true_latency_ms'])*1000 for r in rs]
                are=[100*abs(r['prediction_ms']-r['true_latency_ms'])/r['true_latency_ms'] for r in rs]
                expected={'mae_us':statistics.mean(ae),'ae_sample_variance_us2':statistics.variance(ae),
                          'ae_sample_sd_us':statistics.stdev(ae),'micro_mre_percent':statistics.mean(are),
                          'are_sample_variance_percent2':statistics.variance(are),
                          'wape_percent':math.fsum(ae)/(10*math.fsum(r['true_latency_ms'] for r in rs))}
                assert int(p['n_queries'])==len(rs)
                assert int(p['n_clusters'])==len(set(r['shape_id'] for r in rs))==3
                for field,value in expected.items():
                    delta=abs(value-float(p[field]));max_delta[field]=max(max_delta[field],delta)
                    assert math.isclose(value,float(p[field]),rel_tol=1e-12,abs_tol=1e-10),(k,field,value,p[field])
                for field in ['mae','ae_variance','mre']:
                    unit={'mae':'us','ae_variance':'us2','mre':'percent'}[field]
                    assert float(p[f'{field}_cluster_ci95_low_{unit}'])<=float(p[f'{field}_cluster_ci95_high_{unit}'])
    assert seen==set(panel) and len(seen)==7200 and count==2781000
    report={'complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),
            'checked_raw_prediction_rows':count,'independently_recomputed_panel_rows':len(seen),
            'scalar_statistics_implementation':'Python statistics.mean/variance/stdev and math.fsum; main panel used NumPy',
            'maximum_absolute_difference':dict(max_delta),
            'source_sha256':sha(source),'panel_sha256':sha(result/'per-fold-seed-model.csv'),
            'code_sha256':sha(Path(__file__)),
            'bootstrap_audit_scope':'Cluster cardinality and interval ordering checked; percentile draws are specified reproducibly in frozen analysis code, not independently regenerated here.'}
    (result/'independent-scalar-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
