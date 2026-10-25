"""Independent arithmetic plus saved-policy reload audit of actual CPU fits."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import torch

from measured_planner import MeasuredTable,CalibrationEnv,ActorCritic,select_plan
from summarize_bridge import vector

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def avg(xs):return statistics.mean(xs)

def independent_phi(actions,times):
    # Closed finite n=4 subset weights, implemented independently of COPA.
    if len(actions)!=4:raise ValueError('this audit is the registered four-player study')
    weights={0:1/4,1:1/12,2:1/12,3:1/4}
    return {a:sum(weights[m.bit_count()]*(times[m]-times[m|(1<<i)]) for m in range(16) if not m&(1<<i)) for i,a in enumerate(actions)}

def raw_gate_evaluation(task,mask):
    rows=[]
    for block in task['eval']:
        records={r['mask']:r for r in block['coalitions']};base=records[0];r=records[mask]
        b=avg(base['latency_samples_ms']);t=avg(r['latency_samples_ms']);q=r['quality_score']/base['quality_score']
        raw=100*(b-t)/b;accept=r['status']=='ok' and q>=.99 and t<=b
        rows.append({'run_id':block['run_id'],'raw_benefit_percent':raw,'accepted_benefit_percent':raw if accept else 0.,'accepted':accept})
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backup-root',type=Path,required=True);ap.add_argument('--input-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError('preserve earlier audit')
    torch.set_num_threads(1);root=a.backup_root;index=json.loads((a.input_dir/'index.json').read_text());out={'kind':'actual_fullmodel_PPO_matrix_independent_audit','contexts':[],
        'policy_fits':0,'checkpoint_sha_checks':0,'checkpoint_reload_plan_checks':0,'independent_physical_phi_checks':0,'independent_surrogate_phi_checks':0,
        'scope':'Physical input tables, real CPU policy training and independent arithmetic/reload checks; no new hardware measurements.'}
    for entry in index:
        taskpath=Path(entry['file']);task=json.loads(taskpath.read_text());name=taskpath.stem;path=root/'results'/(name+'.json');d=json.loads(path.read_text())
        assert d['input_sha256']==entry['sha256']==sha(taskpath) and d['status']=='completed'
        assert len(d['ppo'])==20 and len(d['training_runs'])==3 and len(d['evaluation_runs'])==5
        assert set(d['training_runs']).isdisjoint(d['evaluation_runs']) and d['quality_sample_disjointness_checked'] is True
        table=MeasuredTable(task);configuration={};fixed={}
        for key,r in d['fixed_strategies'].items():
            if r['status']!='evaluated':continue
            expected=raw_gate_evaluation(task,r['mask'])
            for x,y in zip(expected,r['evaluation']):
                assert x['run_id']==y['run_id'] and x['accepted']==y['accepted'] and abs(x['raw_benefit_percent']-y['raw_benefit_percent'])<1e-9
            fixed[key]={'mask':r['mask'],'raw_gain_percent':vector([x['raw_benefit_percent'] for x in expected]),'accepted_gain_percent':vector([x['accepted_benefit_percent'] for x in expected])}
        for r in d['ppo']:
            assert r['eval_access_during_training'] is False and r['unit_fixture_data'] is False and 4096<=r['steps']<=4097 and r['plan']['mask'].bit_count()<=2
            assert set(r['training_run_ids'])==set(d['training_runs']) and len(r['history'])>=16
            ck=root/Path(r['checkpoint']).relative_to('/work/planner-matrix-v2')
            assert sha(ck)==r['checkpoint_sha256'];out['checkpoint_sha_checks']+=1
            env=CalibrationEnv(table,2,r['hardware_mask'],r['with_embedding'],r['drop_embedding']);env.reset(table.splits['train'][0])
            model=ActorCritic(len(env.state()),5);model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True));model.eval()
            replay=select_plan(model,table,2,r['hardware_mask'],r['with_embedding'],r['drop_embedding'])
            assert replay['mask']==r['plan']['mask'];out['checkpoint_reload_plan_checks']+=1
            expected=raw_gate_evaluation(task,r['plan']['mask'])
            for x,y in zip(expected,r['evaluation']):assert x['accepted']==y['accepted'] and abs(x['raw_benefit_percent']-y['raw_benefit_percent'])<1e-9
            configuration.setdefault(r['configuration'],[]).append((r,expected))
        assert set(configuration)=={'ppo_hardware_mask','ppo_without_hardware_mask','ppo_with_heldout_embedding','ppo_embedding_zeroed_same_capacity'}
        ppo={}
        for name,rows in configuration.items():
            assert {r['seed'] for r,_ in rows}=={11,22,33,44,55}
            raw=[avg(e[j]['raw_benefit_percent'] for _,e in rows) for j in range(5)]
            accepted=[avg(e[j]['accepted_benefit_percent'] for _,e in rows) for j in range(5)]
            ppo[name]={'seed_masks':{str(r['seed']):r['plan']['mask'] for r,_ in rows},'raw_gain_averaged_within_each_physical_block_percent':vector(raw),
                'accepted_gain_averaged_within_each_physical_block_percent':vector(accepted),'policy_seeds':5,'physical_blocks':5}
        runtime={}
        physical_by_run={}
        for block in task['eval']:
            times={r['mask']:avg(r['latency_samples_ms']) for r in block['coalitions']};physical_by_run[block['run_id']]=independent_phi(task['action_ids'],times)
        assert d['runtime_ablation']['fit']['fit_coalition_count']==11 and d['runtime_ablation']['fit']['evaluation_access_during_fit'] is False
        assert {int(m) for m in d['runtime_ablation']['fit']['fit_masks']}=={m for m in range(16) if m.bit_count()<=2}
        for name,r in d['runtime_ablation']['variants'].items():
            assert r['clamped_predictions']==0
            predictions={int(m):v for m,v in r['predictions_ms'].items()}
            if not r['nonpositive_or_nonfinite_masks']:
                exact=independent_phi(task['action_ids'],predictions)
                assert all(abs(exact[k]-r['exact_surrogate_shapley']['values'][k])<1e-9 for k in exact);out['independent_surrogate_phi_checks']+=1
            for e in r['eval']:
                expected=physical_by_run[e['run_id']]
                assert all(abs(expected[k]-e['physical_exact_shapley']['values'][k])<1e-9 for k in expected);out['independent_physical_phi_checks']+=1
            runtime[name]={'selected_mask':r.get('selected_mask'),'nonpositive_masks':r['nonpositive_or_nonfinite_masks'],'clamped':r['clamped_predictions'],
                'heldout_train_highorder_mae_ms':r['heldout_train_coalition_mae_ms'],
                'eval_all_coalition_mae_ms':avg(e['all_coalition_mae_ms'] for e in r['eval']),
                'eval_all_coalition_mre_percent':100*avg(e['all_coalition_mre'] for e in r['eval']),
                'physical_attribution_signed_mae_ms':avg(e['physical_attribution_error']['signed_mae_ms'] for e in r['eval']) if r.get('exact_surrogate_shapley') else None,
                'same_surrogate_sampler_signed_mae_ms':avg(s['sampling_error']['signed_mae_ms'] for s in r['sampler']) if r['sampler'] else None}
        out['contexts'].append({'task_id':task['task_id'],'device_id':task['device_id'],'fixed_strategies':fixed,'ppo':ppo,'runtime_ablation':runtime,
            'input_path':str(taskpath),'input_sha256':sha(taskpath),'result_path':str(path),'result_sha256':sha(path)})
        out['policy_fits']+=len(d['ppo'])
    assert out['policy_fits']==160 and out['checkpoint_sha_checks']==160 and out['checkpoint_reload_plan_checks']==160
    out['passed']=True;out['analyzer_sha256']=sha(__file__);a.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='contexts'}))

if __name__=='__main__':main()
