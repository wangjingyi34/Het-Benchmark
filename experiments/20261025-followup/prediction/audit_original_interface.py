"""Read-only feasibility audit of archived original checkpoint, not evaluation."""
import hashlib, json, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SUB=ROOT/'submitted-v21/HET_AAAI27_Final_All_Files_v21/05_Code_Data/HET_AAAI27_Code_Data_v21'
sys.path.insert(0,str(ROOT/'v2-实证增强/prediction'))
import evaluate_evidence_memory as em

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=SUB/'models/rgat_final.pt';s=SUB/'src/train_rgat.py'
    cp=torch.load(p,map_location='cpu',weights_only=False)
    lines=s.read_text().splitlines();excerpts=[]
    for i,line in enumerate(lines,1):
        if any(t in line for t in ['node_dim = 64','node_features = torch.randn','targets = torch.randn','synthetic for training']):
            excerpts.append({'file':str(s.relative_to(ROOT)),'line':i,'text':line.strip()})
    public=ROOT/'v2-实证增强/public-reproduction-check'
    summary=public/'prediction/formal-five-stack/summary.json'
    rows,manifest,_=em.load_records(json.loads(summary.read_text()),public/'evidence/physical_profiles')
    audit={'kind':'original_checkpoint_fresh_physical_query_interface_feasibility',
        'audit_script_sha256':sha(Path(__file__)),'checkpoint_path':str(p.relative_to(ROOT)),
        'checkpoint_sha256':sha(p),'checkpoint_keys':list(cp),'checkpoint_config':cp['config'],
        'checkpoint_first_layer_shape':list(cp['model_state_dict']['node_embedding.weight'].shape),
        'archived_source_sha256':sha(s),'archived_source_excerpts':excerpts,
        'fresh_queries':{'count':len(rows),'strict_feature_dimension':len(rows[0]['strict_features']),
            'three_probe_feature_dimension':len(rows[0]['probe_features']),'source_manifest':manifest},
        'interface_compatible':False,'latency_experiment_executed_with_original_checkpoint':False,
        'missing_dependencies':['saved semantically defined original64-dimensional feature transform or original node features and mapping',
            'normalization/target transform and physical units','training label record IDs and measured/calibrated lineage',
            'source-model split and exact graph edge masking linked to this checkpoint'],
        'finding':'Our archived checkpoint cannot be evaluated meaningfully on the new structural-query schema without the original input/target contract. Equal-width padding or random features would not repair semantic compatibility.',
        'evidence_limit':'The archived training example explicitly generates random features and targets, but the checkpoint has no code/label provenance binding. This audit does not establish that every historical paper result was generated with that example.',
        'historical_14_3_certified':False,'new_retrieval_result_is_original_rgat':False}
    (OUT/'original-checkpoint-interface-audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'checkpoint':audit['checkpoint_sha256'],'config':audit['checkpoint_config'],'new_feature_dims':[len(rows[0]['strict_features']),len(rows[0]['probe_features'])],'compatible':False}))

if __name__=='__main__':main()
