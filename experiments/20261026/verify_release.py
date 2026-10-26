"""Byte-exact verification of all scientific files in the 20261026 release."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
manifest=json.loads((P/'RELEASE-SHA256.json').read_text())
for r in manifest['files']:
 p=P/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
print(json.dumps({'verified_files':len(manifest['files']),'bytes':sum(r['bytes'] for r in manifest['files']),'status':'passed'}))
