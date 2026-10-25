"""Verify the public release and restore its immutable research artifacts."""
import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def safe(root,name):
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Archive path escapes destination')
    return p
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--extract',action='store_true');ap.add_argument('--destination',type=Path,default=ROOT/'run-data');a=ap.parse_args()
    manifest=json.loads((ROOT/'RELEASE-SHA256.json').read_text())
    for row in manifest['files']:
        p=ROOT/row['path'];assert p.is_file() and p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],row['path']
    print('Verified',len(manifest['files']),'release files')
    if not a.extract:return
    dest=a.destination.resolve();dest.mkdir(parents=True,exist_ok=True)
    parts=json.loads((ROOT/'archives/prediction-parts.json').read_text())
    joined=dest/parts['output']
    if not joined.exists():
        tmp=joined.with_suffix('.partial')
        with tmp.open('wb') as f:
            for row in parts['parts']:
                p=ROOT/row['path'];assert digest(p)==row['sha256']
                with p.open('rb') as src:shutil.copyfileobj(src,f)
        assert digest(tmp)==parts['sha256'];tmp.replace(joined)
    else:assert digest(joined)==parts['sha256']
    for p in sorted((ROOT/'archives').glob('*.zip'))+[joined]:
        target=dest/'prediction' if p==joined else dest
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():safe(target,name)
            z.extractall(target)
    for p in sorted((ROOT/'archives').glob('*.tar.gz')):
        target=dest/'archived-method-runs'/p.name.removesuffix('.tar.gz');target.mkdir(parents=True,exist_ok=True)
        with tarfile.open(p) as t:
            for m in t.getmembers():
                safe(target,m.name)
                if not (m.isfile() or m.isdir()):raise ValueError('Unexpected archive link')
            t.extractall(target,filter='data')
    # Private original absolute paths remain in immutable records. Explicit
    # content-hash-resolved path maps make the same records usable elsewhere.
    rawdir=dest/'evidence/fullmodel_raw';mapping={}
    for p in (dest/'evidence/final-tasks').glob('*.json'):
        task=json.loads(p.read_text())
        if 'source_artifacts' not in task:continue
        for item in task['source_artifacts']:
            actual=rawdir/Path(item['path']).name
            assert digest(actual)==item['sha256']
            mapping[item['path']]=str(actual)
    (dest/'explicit-raw-path-map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    print('Restored artifacts to',dest)
if __name__=='__main__':main()
