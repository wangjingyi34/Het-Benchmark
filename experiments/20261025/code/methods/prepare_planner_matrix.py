"""Bind explicit runtime identity and actual model-heldout action embeddings."""
import hashlib
import json
from pathlib import Path
import torch
from attach_embeddings import attach
from measured_planner import MeasuredTable,CalibrationEnv

ROOT=Path(__file__).resolve().parent;V2=ROOT.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(1)
    directory=ROOT/'planner-matrix-inputs';outdir=directory/'tasks';outdir.mkdir(exist_ok=True)
    ck=directory/'formal-five-stack/probe-model-fold0-rgat-seed11.pt'
    groups=[{'backend':'npu','task_device':'npu_Ascend910B3','embedding_device':'Ascend-910B3-64GB','profile':'npub3',
             'task_paths':sorted((ROOT/'bridge-final-inputs/npu-tasks').glob('resnet*.json'))},
            {'backend':'cuda','task_device':'cuda_NVIDIA_A100-SXM4-80GB_MIG_2g.20gb','embedding_device':'A100-SXM4-MIG-2g20GB','profile':'a100mig',
             'task_paths':sorted((V2/'evidence/early-tasks').glob('*_2c4332169b.json'))}]
    index=[]
    for group in groups:
        profilepath=V2/'evidence/physical_profiles'/(group['profile']+'.json');profile=json.loads(profilepath.read_text())
        ep=directory/('action-embeddings-'+group['profile']+'.json');emb=json.loads(ep.read_text());h=profile['hardware']
        assert h['backend']==group['backend'] and h['device_id']==group['embedding_device']
        assert len(emb['records'])==16 and {r['device_id'] for r in emb['records']}=={group['embedding_device']}
        for row in emb['records']:
            p=row['embedding_provenance']
            assert p['hardware_source_sha256']==sha(profilepath) and p['checkpoint_sha256']==sha(ck)
            assert p['backend']==h['backend'] and p['hardware_name']==h['hardware_name'] and p['hardware_threads']==h['threads']
            assert p['used_inference_latency_labels'] is False and p['target_latency_used_at_inference'] is False
        evidence={'kind':'explicit_verified_device_identity_mapping','fullmodel_device_id':group['task_device'],
            'embedding_device_id':group['embedding_device'],'hardware_profile':{'path':str(profilepath),'sha256':sha(profilepath),'hardware':h,
                'runtime':profile['provenance']},'embedding_artifact':{'path':str(ep),'sha256':sha(ep)},
            'checkpoint':{'path':str(ck),'sha256':sha(ck)},'fullmodel_raw_artifacts':[],
            'scope':'Exact chip/backend/thread/runtime/profile binding, not fuzzy device-name inference; embedding is target-model heldout, hardware can be represented in predictor training.'}
        for path in group['task_paths']:
            task=json.loads(path.read_text());assert task['device_id']==group['task_device']
            for art in task['source_artifacts']:
                rawpath=Path(art['path']);assert sha(rawpath)==art['sha256'];r=json.loads(rawpath.read_text());runtime=r['runtime']
                assert r['device_id']==group['task_device'] and runtime['hardware']==h['hardware_name'] and runtime['threads']==h['threads']
                for key in ['torch','python','platform']:assert runtime[key]==profile['provenance'][key]
                if group['backend']=='cuda':assert runtime['memory_bytes']==h['visible_device_memory_bytes']
                evidence['fullmodel_raw_artifacts'].append({'path':str(rawpath),'sha256':sha(rawpath),'run_id':r['run_id'],'runtime':runtime})
        if group['backend']=='npu':
            receipts=[]
            for path in sorted((V2/'evidence/npu-main-backup/launch-receipts').glob('*.json')):
                r=json.loads(path.read_text())
                if Path(r.get('script','')).name=='full_model_coalitions.py':
                    assert r['device']==h['hardware_name'] and r['torch']==profile['provenance']['torch'] and r['visible_device_count']==1
                    assert r['conv_allow_hf32'] is False and r['matmul_allow_hf32'] is False
                    receipts.append({'path':str(path),'sha256':sha(path),'output_arg':r['args'][-1],
                                     'script_sha256':r['script_sha256'],'visible_device_count':r['visible_device_count']})
            assert len(receipts)>=32;evidence['npu_exact_single_visible_device_launch_receipts']=receipts
        mapfile=directory/('device-mapping-'+group['profile']+'.json')
        if mapfile.exists():raise FileExistsError('preserve mapping freeze')
        mapfile.write_text(json.dumps(evidence,indent=2))
        for path in group['task_paths']:
            task=json.loads(path.read_text());out=attach(task,emb,group['embedding_device'],mapfile)
            out['embedding_artifact']={'path':str(ep),'sha256':sha(ep),'source_task_path':str(path),'source_task_sha256':sha(path)}
            table=MeasuredTable(out);CalibrationEnv(table,2,with_embedding=True)
            destination=outdir/(group['backend']+'-'+task['task_id']+'.json')
            if destination.exists():raise FileExistsError('preserve task freeze')
            destination.write_text(json.dumps(out,indent=2))
            index.append({'file':str(destination),'sha256':sha(destination),'task_id':task['task_id'],'device_id':task['device_id'],
                          'mapping_evidence':str(mapfile),'mapping_sha256':sha(mapfile),'original_task_sha256':sha(path)})
    (directory/'index.json').write_text(json.dumps(index,indent=2));print(json.dumps({'tasks':len(index),'actions_per_task':4,'embedding_dimension':32,'checkpoint_sha256':sha(ck)}))

if __name__=='__main__':main()
