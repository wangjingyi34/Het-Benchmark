"""Restore a fresh planning-analysis workspace using SHA256-matched public inputs."""
import argparse,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prior-release',type=Path,required=True);ap.add_argument('--restored-root',type=Path,required=True);ap.add_argument('--workspace',type=Path,required=True);a=ap.parse_args()
 out=a.workspace.resolve();assert not out.exists(),'Use a new empty destination'
 sources=[a.restored_root.resolve(),a.prior_release.resolve(),a.prior_release.resolve().parent/'20261025-followup',HERE]
 by_name={}
 for root in sources:
  for p in root.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:by_name.setdefault(p.name,[]).append(p)
 inputs=json.loads((HERE/'analysis-freeze.json').read_text())['inputs']+json.loads((HERE/'input-freeze-supplement.json').read_text())['inputs'];matched=[];missing=[]
 for r in inputs:
  rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts
  found=next((p for p in by_name.get(rel.name,[]) if sha(p)==r['sha256']),None)
  if found is None:missing.append(r)
  else:matched.append((r,found))
 if missing:raise RuntimeError('Missing hash-matched public inputs: '+json.dumps(missing))
 out.mkdir();dest=out/'v7-剩余条件实证闭环/planning';shutil.copytree(HERE,dest,ignore=shutil.ignore_patterns('__pycache__'))
 for r,p in matched:
  q=out/r['path'];q.parent.mkdir(parents=True,exist_ok=True)
  if not q.exists():q.symlink_to(p)
 # Import-only historical modules were not all numerical inputs in the freeze.
 for p in (a.prior_release/'code/methods').glob('*.py'):
  q=out/'v2-实证增强/methods'/p.name;q.parent.mkdir(parents=True,exist_ok=True)
  if not q.exists():q.symlink_to(p.resolve())
 receipt={'source_hash_checks':len(matched),'workspace':str(out),'planning_directory':str(dest),'original_sources_mutated':False}
 (out/'planning-replay-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
