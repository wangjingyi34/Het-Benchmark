"""Attach actual frozen heldout embeddings with exact model/shape/action matching.

Device-ID aliases require an explicit caller mapping plus retained runtime
evidence. No fuzzy name matching silently binds a device or a model.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from measured_planner import MeasuredTable,CalibrationEnv


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def attach(task,embedding_payload,embedding_device_id=None,mapping_evidence=None):
    out=copy.deepcopy(task);device=embedding_device_id or task['device_id']
    if device!=task['device_id'] and mapping_evidence is None:
        raise ValueError('device alias requires explicit runtime evidence path')
    if embedding_payload.get('kind')!='frozen_inductive_action_embeddings':raise ValueError('wrong embedding artifact')
    if not task.get('model_id') or not task.get('input_shape'):raise ValueError('task model and full input shape required')
    candidates=[r for r in embedding_payload['records'] if r['model_id']==task['model_id']
                and r['input_shape']==task['input_shape'] and r['device_id']==device]
    for metadata in out['action_metadata']:
        paths=metadata.get('module_paths')
        if not paths:raise ValueError('action module paths missing')
        matched=[r for r in candidates if sorted(r['module_paths'])==sorted(paths)]
        if len(matched)!=1:raise ValueError(f"action {metadata['id']} has {len(matched)} exact embedding matches")
        r=matched[0];embedding=r['graph_embedding']
        if len(embedding)!=32 or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in embedding) or not any(abs(v)>1e-12 for v in embedding):
            raise ValueError('invalid/nonfinite/placeholder embedding')
        metadata['graph_embedding']=embedding
        metadata['embedding_provenance']={**r['embedding_provenance'],'source_embedding_device_id':device,
                                           'target_task_device_id':task['device_id']}
        if mapping_evidence is not None:
            metadata['embedding_provenance']['explicit_device_alias_evidence']={
                'path':str(Path(mapping_evidence).resolve()),'sha256':sha(mapping_evidence)}
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True,type=Path)
    ap.add_argument('--embeddings',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--embedding-device-id');ap.add_argument('--device-mapping-evidence',type=Path)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('preserve previous task freeze')
    task=json.loads(args.task.read_text());emb=json.loads(args.embeddings.read_text())
    out=attach(task,emb,args.embedding_device_id,args.device_mapping_evidence)
    out['embedding_artifact']={'path':str(args.embeddings.resolve()),'sha256':sha(args.embeddings),
                               'source_task_path':str(args.task.resolve()),'source_task_sha256':sha(args.task)}
    table=MeasuredTable(out)
    CalibrationEnv(table,min(2,table.n),with_embedding=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({'output':str(args.output),'matched_actions':table.n,'dimension':32,
                      'eval_labels_read_for_matching':False,'explicit_device_alias':args.embedding_device_id is not None}))


if __name__=='__main__':main()
