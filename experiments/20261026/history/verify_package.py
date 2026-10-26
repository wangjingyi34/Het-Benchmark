"""Verify the public package bytes and optionally recompute all numerical outputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def scientific_equal(expected, actual, at='$'):
 """Exact structure/order/scalars; tolerate only finite float rounding noise.

 Python 3.12 changed built-in sum's floating-point summation algorithm. The
 archived decimal values stay unchanged; recomputation may differ by a few
 ulps across Python versions. Source and artifact hashes remain byte-exact.
 """
 if isinstance(expected,float) or isinstance(actual,float):
  assert isinstance(expected,(int,float)) and not isinstance(expected,bool),at
  assert isinstance(actual,(int,float)) and not isinstance(actual,bool),at
  assert math.isfinite(expected) and math.isfinite(actual),at
  assert math.isclose(expected,actual,rel_tol=1e-12,abs_tol=1e-12),(at,expected,actual)
 elif isinstance(expected,dict):
  assert isinstance(actual,dict) and expected.keys()==actual.keys(),at
  for key in expected:scientific_equal(expected[key],actual[key],at+'/'+key)
 elif isinstance(expected,list):
  assert isinstance(actual,list) and len(expected)==len(actual),at
  for i,(x,y) in enumerate(zip(expected,actual)):scientific_equal(x,y,at+'/'+str(i))
 else:
  assert type(expected) is type(actual) and expected==actual,(at,expected,actual)

def comparison_controls():
 # Reproduce observed 3.9/3.12 ulp differences, and reject meaningful changes.
 scientific_equal({'mean':14.279999999999998},{'mean':14.280000000000001})
 for left,right in [({'mean':14.28},{'mean':14.29}),([1,2],[2,1]),({'ok':False},{'ok':0}),({'x':1},{'y':1}),({'x':float('nan')},{'x':float('nan')})]:
  try:scientific_equal(left,right)
  except AssertionError:pass
  else:raise AssertionError('Scientific comparison negative control failed')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--recompute',action='store_true');args=parser.parse_args()
 manifest=json.loads((ROOT/'package-manifest.json').read_text())
 for entry in manifest['files']:
  path=ROOT/entry['path'];assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],entry['path']
 checks={'manifest_files':len(manifest['files']),'manifest_pass':True}
 if args.recompute:
  comparison_controls()
  with tempfile.TemporaryDirectory(prefix='historical-arithmetic-') as temp:
   dest=Path(temp)/'numeric'
   subprocess.run([sys.executable,str(ROOT/'recompute_history_findings.py'),'--output',str(dest)],check=True)
   expected=list((ROOT/'results').iterdir());assert len(expected)==len(list(dest.iterdir()))
   for f in expected:
    if f.suffix=='.json':scientific_equal(json.loads(f.read_text()),json.loads((dest/f.name).read_text()),f.name)
    else:assert f.read_bytes()==(dest/f.name).read_bytes(),f.name
   subprocess.run([sys.executable,str(ROOT/'test_original_lopo_validator.py')],check=True)
   report=Path(temp)/'gate.json'
   run=subprocess.run([sys.executable,str(ROOT/'validate_original_lopo.py'),str(ROOT/'current-original-data-candidate.json'),'--report',str(report)])
   assert run.returncode==2
   gate=json.loads(report.read_text());assert gate['pass'] is False and gate['record_count']==0 and gate['unrecorded_opportunities']==31220
   assert gate==json.loads((ROOT/'current-original-platform-data-gate.json').read_text())
   checks.update({'numerical_outputs_recomputed':len(expected),'synthetic_validator_controls_pass':True,'truthful_empty_label_gate_failed_as_expected':True,'float_rounding_tolerance':{'relative':1e-12,'absolute':1e-12},'comparison_negative_controls_pass':True})
 print(json.dumps(checks))

if __name__=='__main__':main()
