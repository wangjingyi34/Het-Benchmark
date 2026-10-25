"""Prepare a locked, labeled ImageNet-val subset from the official Imagenette tar.

Only original ILSVRC2012_val_* identities are eligible. No model inference or
quality outcome is observed while selecting samples. No pickle is loaded here.
"""
import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

SYNSETS = ['n01440764','n02102040','n02979186','n03000684','n03028079',
           'n03394916','n03417042','n03425413','n03445777','n03888257']
CLASS_INDEX_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
DATASET_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'

def sha(b): return hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--class-index',type=Path)
    ap.add_argument('--per-class',type=int,default=20)
    args=ap.parse_args()
    if args.per_class<4 or args.per_class%2: raise ValueError('even per-class >=4 required')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Do not overwrite prepared assets')
    if args.class_index:
        mapping_bytes=args.class_index.read_bytes()
    else:
        with urllib.request.urlopen(CLASS_INDEX_URL,timeout=30) as r:mapping_bytes=r.read()
    mapping=json.loads(mapping_bytes)
    assert len(mapping)==1000 and set(map(int,mapping))==set(range(1000))
    exact={v[0]:{'imagenet_index':int(k),'imagenet_label':v[1]} for k,v in mapping.items()}
    assert all(w in exact for w in SYNSETS)
    (out/'imagenet_class_index.json').write_bytes(mapping_bytes)
    # Streaming read avoids extracting unwanted training images or unsafe paths.
    candidates={w:{} for w in SYNSETS}
    archive_hash=hashlib.sha256()
    with args.archive.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):archive_hash.update(b)
    with tarfile.open(args.archive,'r:gz') as tf:
        for m in tf:
            p=PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or m.issym() or m.islnk():
                raise ValueError('Unsafe archive member: '+m.name)
            if not m.isfile() or not p.name.startswith('ILSVRC2012_val_'):continue
            if len(p.parts)<3 or p.parent.name not in SYNSETS:continue
            w=p.parent.name
            data=tf.extractfile(m).read()
            if p.name in candidates[w]:
                if candidates[w][p.name]['sha256']!=sha(data):raise ValueError('Conflicting original identity')
                continue
            candidates[w][p.name]={'original_archive_path':m.name,'filename':p.name,
                'sha256':sha(data),'data':data}
    counts={w:len(x) for w,x in candidates.items()}
    if min(counts.values())<args.per_class:
        (out/'insufficient-original-val-identities.json').write_text(json.dumps(counts,indent=2))
        raise ValueError('Insufficient official-val identities; do not replace with train data: '+str(counts))
    # Common fixed preprocessing is the torchvision IMAGENET1K_V1 recipe.
    from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize,InterpolationMode
    transform=Compose([Resize(256,interpolation=InterpolationMode.BILINEAR),CenterCrop(224),
                       ToTensor(),Normalize([.485,.456,.406],[.229,.224,.225])])
    samples=[];arrays=[]
    for w in SYNSETS:
        chosen=sorted(candidates[w].values(),key=lambda r:sha(r['filename'].encode()))[:args.per_class]
        for j,r in enumerate(chosen):
            im=Image.open(io.BytesIO(r['data'])).convert('RGB')
            x=transform(im).numpy().astype(np.float32,copy=False)
            record={k:v for k,v in r.items() if k!='data'}
            record.update({'sample_id':w+'/'+r['filename'],'synset':w,**exact[w],
                'partition':'calibration' if j<args.per_class//2 else 'evaluation',
                'selection_hash':sha(r['filename'].encode()),'tensor_sha256':sha(x.tobytes())})
            samples.append(record);arrays.append(x)
    images=np.stack(arrays);labels=np.array([s['imagenet_index'] for s in samples],dtype=np.int64)
    # Save one portable NumPy asset; all devices consume these exact tensor bytes.
    np.savez_compressed(out/'quality_inputs.npz',images=images,labels=labels)
    timing={}
    for partition in ['calibration','evaluation']:
        indices=[next(i for i,s in enumerate(samples) if s['synset']==w and s['partition']==partition)
                 for w in SYNSETS[:4]]
        for batch in [1,4]:
            x=images[indices[:batch]].copy()
            name=f'{partition}_b{batch}.npy';np.save(out/name,x,allow_pickle=False)
            timing[partition+f'_b{batch}']={'file':name,'shape':list(x.shape),
                'tensor_sha256':sha(x.tobytes()),'file_sha256':sha((out/name).read_bytes()),
                'sample_ids':[samples[i]['sample_id'] for i in indices[:batch]]}
    d={'schema_version':'het.imagenet-val-subset.v2','created_utc':datetime.now(timezone.utc).isoformat(),
       'dataset_id':'imagenette2-160_original_ILSVRC2012_val_sha20perclass',
       'dataset_source_url':DATASET_URL,'archive_sha256':archive_hash.hexdigest(),
       'mapping_source_url':CLASS_INDEX_URL,'mapping_sha256':sha(mapping_bytes),
       'selection':'only original ILSVRC2012_val_ filenames, deduplicate by original identity; per synset filename SHA256 ascending before observing inference',
       'original_val_candidates_per_class':counts,'samples_per_class':args.per_class,
       'preprocessing':{'resize_short_side':256,'center_crop':224,'interpolation':'PIL bilinear',
          'rgb':True,'scale':'0..1','mean':[.485,.456,.406],'std':[.229,.224,.225]},
       'quality_metric':'unrestricted ImageNet-1000 top-1 accuracy, not ten-class-restricted logits',
       'quality_partition':'first half of selected identities per class calibration; second half evaluation',
       'samples':samples,'timing_inputs':timing,'quality_asset_sha256':sha((out/'quality_inputs.npz').read_bytes()),
       'image_tensor_bytes_sha256':sha(images.tobytes()),'labels_bytes_sha256':sha(labels.tobytes()),
       'sample_count':len(samples),'limitations':['A fixed 200-image ten-class audit, not full ImageNet generalization.',
       'Original validation identity is established from upstream archive filenames, not an independent ImageNet license database.']}
    (out/'manifest.json').write_text(json.dumps(d,indent=2))
    print(json.dumps({'output':str(out),'sample_count':len(samples),'candidate_counts':counts,
                      'mapping':{w:exact[w] for w in SYNSETS},'images_shape':list(images.shape)},indent=2))

if __name__=='__main__':main()
