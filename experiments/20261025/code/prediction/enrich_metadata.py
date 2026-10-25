"""Create a label-free sidecar; never rewrite a physical measurement bundle."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import torch
from collect_profiles import build_model,tensor_hash
from structural_attributes import attributes_for_node,NUMERIC_ATTRIBUTES

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    if args.output.exists():raise ValueError('never overwrite a sealed sidecar')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    bundle=json.loads(args.input.read_text())
    out={'kind':'label_free_structural_metadata_sidecar','created_utc':datetime.now(timezone.utc).isoformat(),
         'raw_bundle_sha256':sha(args.input),'raw_bundle_path':str(args.input.resolve()),
         'sidecar_code_sha256':sha(__file__),'attributes_code_sha256':sha(Path(__file__).with_name('structural_attributes.py')),
         'collector_code_sha256':bundle['provenance']['code_sha256'],'training_scores_seen':False,
         'timings_changed_or_added':False,'numerical_feature_whitelist':NUMERIC_ATTRIBUTES,'graphs':{}}
    with torch.inference_mode():
        for graph in bundle['graphs']:
            model,inputs,_=build_model(graph['model_id'],graph['shape_id'])
            reconstructed=tensor_hash(model,inputs)
            if reconstructed!=graph['input_weights_sha256']:raise ValueError('architecture/input hash mismatch: '+graph['graph_id'])
            nodes={n['node_id']:attributes_for_node(model,n,graph['model_id']) for n in graph['nodes']}
            out['graphs'][graph['graph_id']]={'input_weights_hash_verified':True,
                'input_weights_sha256':reconstructed,'nodes':nodes}
            del model,inputs
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({'output':str(args.output),'graphs':len(out['graphs']),
        'nodes':sum(len(g['nodes']) for g in out['graphs'].values()),'raw_timings_modified':False,
        'all_input_weight_hashes_match':True}))

if __name__=='__main__':main()
