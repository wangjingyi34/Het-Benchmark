"""Run preregistered train-size curves after the primary fit matrix finishes."""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

root=Path('/work')
primary=root/'prediction/formal-five-stack/summary.json'
deadline=time.monotonic()+1800
while not primary.exists():
    if time.monotonic()>deadline:raise TimeoutError('Primary result not complete')
    time.sleep(5)
d=json.loads(primary.read_text())
assert d.get('completed_utc') and len(d['results'])==220
assert d['code_sha256']=='bad26be58e40a0bf6ff939024f15e646a72ec3e1b8bc3fef54dc67745f8cce0f'
names=['m5cpu','m5mps','a100mig','xeon_cpu','npub3']
command=[sys.executable,str(root/'prediction/evaluate_profiles.py'),'--inputs']
command += [str(root/'evidence/physical_profiles'/(n+'.json')) for n in names]
command += ['--metadata-sidecars']+[str(root/'evidence/physical_profiles'/(n+'.metadata.json')) for n in names]
command += ['--output',str(root/'prediction/formal-learning-curves'),'--protocol','model','--folds','6',
            '--methods','mlp','rgat','--seeds','11','22','33','44','55','--epochs','80','--patience','15',
            '--feature-modes','probe','--train-device','cuda','--threads','4',
            '--learning-budgets','2','4','6','--full-budget-reference',str(primary)]
receipt={'started_utc':datetime.now(timezone.utc).isoformat(),'command':command,'scope':'frozen workload-round-robin train-only subsets; full8 reused'}
path=root/'prediction/learning-curve-run-receipt.json'
if path.exists():raise FileExistsError('Inspect existing curve job before retrying')
path.write_text(json.dumps(receipt,indent=2))
result=subprocess.run(command)
receipt.update({'exit_code':result.returncode,'finished_utc':datetime.now(timezone.utc).isoformat()})
path.write_text(json.dumps(receipt,indent=2))
raise SystemExit(result.returncode)
