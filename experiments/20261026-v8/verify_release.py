from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
m=json.loads((P/'RELEASE-SHA256.json').read_text())
for r in m['files']:
 f=P/r['path'];assert f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],r['path']
print('PASS',len(m['files']),'files')
