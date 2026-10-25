"""Produce explicit printed-cell aggregates and provenance annotations.

This is table reconciliation, never reconstruction of missing per-query labels.
"""
from pathlib import Path
import csv,hashlib,json,statistics
ROOT=Path(__file__).resolve().parent;PUBLIC=ROOT.parent/'public-repo';OUT=ROOT/'reconciled-tables';OUT.mkdir(exist_ok=True)
source=PUBLIC/'results/paper_tables/prediction_mre.csv'
rows=list(csv.DictReader(source.open()));families=[r for r in rows if r['operator_family']!='Overall'];common=families[:3]
out=[]
for name,scope in [('shared_three_families',common),('all_eight_printed_families',families)]:
 for method in ['deepbench','roofline','train','validation','test']:
  values=[float(r[method]) for r in scope if r[method]]
  if len(values)!=len(scope):continue
  out.append({'family_set':name,'method_or_split':method,'families':len(values),'weighting':'equal family weights',
      'printed_cell_mean_mre_percent':statistics.mean(values),'source':'reported family cells, not recovered per-query predictions'})
with (OUT/'table3-explicit-aggregates.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
source_flags=json.loads((PUBLIC/'results/cross_platform_results.json').read_text());flags=[]
for kind in ['operators','models']:
 for name,entry in source_flags[kind].items():
  for hw,r in entry['platforms'].items():
   flags.append({'kind':kind,'item':name,'platform':hw,'reported_latency_ms':r['latency_ms'],
      'source_measured_flag':r['measured'],'source_status':'source_marks_measured' if r['measured'] else 'specification_calibrated',
      'calibration_method':r.get('estimation_method','hardware_spec_ratio' if not r['measured'] else ''),
      'paper_row_run_identity_verified':False})
with (OUT/'legacy-source-row-provenance.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(flags[0]));w.writeheader();w.writerows(flags)
printed=[]
for table in ['model_latency_ms','operator_relative_throughput']:
 for row in csv.DictReader((PUBLIC/'results/paper_tables'/f'{table}.csv').open()):
  name=row.get('model') or row['operator_shape']
  for hw,value in list(row.items())[1:]:
   printed.append({'table':table,'item':name,'platform':hw,'printed_value':value,
    'provenance_annotation':'A100 reference in legacy generation scheme; exact printed run not recovered' if hw=='a100' else 'specification-calibrated generation scheme; not claimed as physical run',
    'status_scope':'generation-class annotation, not identification of the original 5,000 labelled graph edges'})
with (OUT/'legacy-printed-table-provenance.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(printed[0]));w.writeheader();w.writerows(printed)
receipt={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'historical_five_family_values':[12.4,11.5,36.9,6.6,4.0],
 'historical_five_family_equal_mean':14.28,'common3_test_mean':statistics.mean(float(r['test']) for r in common),
 'common3_roofline_mean':statistics.mean(float(r['roofline']) for r in common),'eight_family_test_macro':statistics.mean(float(r['test']) for r in families),
 'source_json_rows_flagged':len(flags),'printed_table_cells_annotated':len(printed),
 'old_5000_per_edge_sources_reconstructed':False,'new_empirical_measurements':0,
 'interpretation':'Makes equal-family and provenance reporting explicit. It does not certify old per-query overall MRE or authenticate exact printed-run identity.'}
(OUT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
