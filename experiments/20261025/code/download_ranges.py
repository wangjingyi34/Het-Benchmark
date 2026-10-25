"""Fetch a public immutable HTTP object with checked range boundaries and ETag."""
import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--url',required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--chunk',type=int,default=1048576)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError(args.output)
    def request(start,end):
        sep='&' if '?' in args.url else '?'
        req=urllib.request.Request(args.url+sep+f'het_range={start}-{end}',headers={'Range':f'bytes={start}-{end}'})
        return urllib.request.urlopen(req,timeout=45)
    with request(0,0) as r:
        first=r.read();cr=r.headers.get('Content-Range','');etag=r.headers.get('ETag')
    match=re.fullmatch(r'bytes 0-0/(\d+)',cr)
    if not match or len(first)!=1:raise RuntimeError('Server does not honor exact byte range: '+cr)
    total=int(match.group(1));parts=args.output.with_name(args.output.name+'.chunks');parts.mkdir(exist_ok=True)
    count=(total+args.chunk-1)//args.chunk
    print(json.dumps({'total':total,'etag':etag,'parts':count,'workers':args.workers}),flush=True)
    def get(i):
        start=i*args.chunk;end=min(total-1,start+args.chunk-1);out=parts/f'{i:06d}.part'
        if out.exists() and out.stat().st_size==end-start+1:return i
        last=None
        for attempt in range(3):
            try:
                with request(start,end) as r:
                    got=r.headers.get('Content-Range','');tag=r.headers.get('ETag');data=r.read()
                    if got!=f'bytes {start}-{end}/{total}' or len(data)!=end-start+1 or tag!=etag:
                        raise RuntimeError(f'Range/length/ETag mismatch: {got} {tag}')
                temp=out.with_suffix('.tmp');temp.write_bytes(data);temp.replace(out);return i
            except Exception as e:last=e;time.sleep(attempt+1)
        raise RuntimeError(f'chunk {i}: {last}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for k,i in enumerate(pool.map(get,range(count)),1):
            if k%8==0 or k==count:print(json.dumps({'completed_chunks':k,'total_chunks':count}),flush=True)
    h=hashlib.sha256();tmp=args.output.with_suffix(args.output.suffix+'.assembling')
    with tmp.open('wb') as f:
        for i in range(count):
            data=(parts/f'{i:06d}.part').read_bytes();f.write(data);h.update(data)
    assert tmp.stat().st_size==total;tmp.replace(args.output)
    info={'url':args.url,'bytes':total,'etag':etag,'sha256':h.hexdigest(),'range_length_etag_checked':True,
          'limitations':'Hash seals received bytes; no independent upstream SHA256 was supplied.'}
    args.output.with_suffix(args.output.suffix+'.download.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info),flush=True)

if __name__=='__main__':main()
