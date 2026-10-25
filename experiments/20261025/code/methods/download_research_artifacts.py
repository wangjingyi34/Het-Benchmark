"""Hash-verified bounded chunks avoid truncating large kubectl stdout streams."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from cloud_early_bridge import remote,require,infra

def main():
    p=argparse.ArgumentParser();p.add_argument('--remote-root',required=True);p.add_argument('--members',nargs='+',required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--snapshot',required=True);a=p.parse_args()
    if not a.remote_root.startswith('/work/') or any(v in ['.','..'] for v in Path(a.remote_root).parts):raise ValueError('isolated /work subtree required')
    if '/' in a.snapshot or '..' in a.snapshot:raise ValueError('plain snapshot name required')
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in a.members):raise ValueError('relative artifact members required')
    a.destination.mkdir(parents=True,exist_ok=True)
    request={'root':a.remote_root,'members':a.members,'snapshot':a.snapshot}
    code="import pathlib,json,tarfile,hashlib,os; os.nice(19); d="+repr(request)+"; root=pathlib.Path(d['root']); out=root/(d['snapshot']+'.tar.gz'); assert not out.exists(); assert all((root/n).exists() for n in d['members']); tf=tarfile.open(out,'w:gz'); [tf.add(root/n,arcname=n) for n in d['members']]; tf.close(); print(json.dumps({'remote_path':str(out),'size':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))"
    meta=json.loads(require(remote(['exec','-n',infra.NS,infra.POD,'--','python','-c',code],timeout=90)))
    archive=a.destination/(a.snapshot+'.tar.gz')
    if archive.exists():raise FileExistsError('preserve previous archive')
    chunk=512*1024;digest=hashlib.sha256()
    with archive.open('xb') as f:
        for offset in range(0,meta['size'],chunk):
            n=min(chunk,meta['size']-offset)
            code="import sys; f=open("+repr(meta['remote_path'])+",'rb'); f.seek("+str(offset)+"); sys.stdout.buffer.write(f.read("+str(n)+"))"
            data=require(remote(['exec','-n',infra.NS,infra.POD,'--','python','-c',code],timeout=40))
            if len(data)!=n:raise ValueError('truncated chunk: '+str(offset))
            f.write(data);digest.update(data)
    if digest.hexdigest()!=meta['sha256']:raise ValueError('downloaded archive SHA256 differs')
    with tarfile.open(archive,'r:gz') as tf:tf.extractall(a.destination,filter='data')
    receipt={**meta,'local_path':str(archive.resolve()),'verified_sha256':digest.hexdigest(),'chunk_bytes':chunk,'members':a.members}
    (a.destination/(a.snapshot+'-transfer-receipt.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
