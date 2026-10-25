"""Pure stdlib identity/math/quality bridge tests, no heavy local model fitting."""
import copy
import hashlib
import json
import unittest
from pathlib import Path

from migration_bridge import (build_operator_migration_game,canonical_graph,
    grouped_action_priority,matched_control_plans,paired_migration_quality,run_bridge)
from copa_v2 import estimate_shapley,exact_shapley


def raw(device,source=True):
    costs={'x':0.,'a':4. if source else 2.,'b':3. if source else 4.,'c':1. if source else 1.5,'output':0.}
    graph={'nodes':[{'id':'x','operation':'placeholder','target':'x','shape':[1,3],'dtype':'torch.float32'},
                    {'id':'a','operation':'call_module','target':'a_mod','shape':[1,3],'dtype':'torch.float32'},
                    {'id':'b','operation':'call_module','target':'b_mod','shape':[1,3],'dtype':'torch.float32'},
                    {'id':'c','operation':'call_function','target':'<built-in function add>','shape':[1,3],'dtype':'torch.float32'},
                    {'id':'output','operation':'output','target':'output','shape':[1,3],'dtype':'torch.float32'}],
           'edges':[{'source':a,'target':b,'relation':'actual_fx_dependency'} for a,b in [('x','a'),('x','b'),('a','c'),('b','c'),('c','output')]]}
    profiles=[]
    for n in graph['nodes']:
        cost=costs[n['id']]
        profiles.append({'node_id':n['id'],'status':'ok' if cost else 'structural_zero_cost',
                         'mean_ms':cost,**({'latency_samples_ms':[cost,cost]} if cost else {})})
    return {'schema_version':'het.fullmodel.run.v2','provenance':'UNIT_FIXTURE','phase':'train',
        'quality_partition':'calibration','run_id':device+'-run1','run_index':1,'device_id':device,
        'task_id':'UNIT_TASK','model_id':'unit-model','input_shape':[1,3],
        'weights_state_sha256':'a'*64,'input_tensor_sha256':'b'*64,'dataset_manifest_sha256':'c'*64,
        'dataset_id':'UNIT_DATA','model_configuration':{'inplace_disabled_modules':['relu']},
        'protocol':{'dtype':'float32','tf32':False,'source_sha256':'d'*64,'mps_fallback_env':None},
        'runtime':{'hardware':device},'graph':graph,'node_profiles':profiles,
        'coalitions':[{'mask':0,'status':'ok','latency_samples_ms':[10.,10.] if source else [14.,14.]}]}


def task(device):
    return {'task_id':'UNIT_TASK','device_id':device,'model_id':'unit-model','input_shape':[1,3],
        'quality_definition':{'metric':'UNIT_TOP1','dataset_id':'UNIT_DATA','mode':'retention_ratio','threshold':.99},
        'action_ids':['opt-a','opt-b'],'train':[{'run_id':device+'-run1'}],
        'action_metadata':[{'id':'opt-a','module_paths':['a_mod'],'hardware_supported':True,
                           'operator_latency_ms':2.,'candidate_latency_ms':1.5,'local_benefit_ms':.5,
                           'profile_run_ids':[device+'-run1']},
                          {'id':'opt-b','module_paths':['b_mod'],'hardware_supported':True,
                           'operator_latency_ms':4.,'candidate_latency_ms':3.,'local_benefit_ms':1.,
                           'profile_run_ids':[device+'-run1']}]}


def quality_record(mask,correct,latency_ms,ids=None):
    ids=ids or [f'UNIT_IMAGE_{i}' for i in range(10)]
    return {'mask':mask,'status':'ok','latency_samples_ms':[latency_ms]*2,'quality_evaluated':True,
            'sample_ids':ids,'predictions':[0]*correct+[1]*(10-correct),'labels':[0]*10,
            'quality_score':correct/10}


class BridgeTests(unittest.TestCase):
    def test_antithetic_quadratic_exactness_is_algebraic_not_physical_fidelity(self):
        players=['a','b','c','d'];main=[-2.,1.,3.,-.5];pairs={(0,1):4.,(0,3):-1.,(1,2):2.,(2,3):-.25}
        def value(mask):
            return 12.+sum(v for i,v in enumerate(main) if mask&(1<<i))+sum(v for (i,j),v in pairs.items() if mask&(1<<i) and mask&(1<<j))
        expected={p:main[i]+.5*sum(v for ij,v in pairs.items() if i in ij) for i,p in enumerate(players)}
        for seed in [11,22,33,44,55]:
            result=estimate_shapley(players,value,2,seed,True)
            self.assertEqual(result['values'],expected)
            self.assertTrue(all(v is None for v in result['standard_errors'].values()))

    def test_real_profile_factory_sign_and_unallocated_runtime(self):
        game,c=build_operator_migration_game([raw('SRC')],[raw('TGT',False)],True)
        self.assertEqual(c['players'],['a','b','c'])
        exact=exact_shapley(c['players'],game)['values']
        self.assertEqual(exact,{'a':-.5,'b':.5,'c':.5})
        self.assertEqual(c['plain_target_minus_source_ms'],{'a':-2.,'b':1.,'c':.5})
        self.assertEqual(c['physical_migration_gap_ms'],4.)
        self.assertEqual(c['operator_surrogate_gap_ms'],.5)
        self.assertEqual(c['unattributed_physical_gap_residual_ms'],3.5)
        self.assertFalse(c['endpoint_anchor_applied'])

    def test_only_train_matching_identity_allowed(self):
        for field,value in [('weights_state_sha256','f'*64),('input_tensor_sha256','f'*64),
                            ('dataset_manifest_sha256','f'*64),('input_shape',[2,3]),
                            ('model_configuration',{'different':True})]:
            t=raw('TGT',False);t[field]=value
            with self.assertRaises(ValueError):build_operator_migration_game([raw('SRC')],[t],True)
        t=raw('TGT',False);t['phase']='eval';t['quality_partition']='evaluation';t['run_index']=4
        with self.assertRaises(ValueError):build_operator_migration_game([raw('SRC')],[t],True)
        with self.assertRaises(ValueError):build_operator_migration_game([raw('SRC')],[raw('SRC',False)],True)

    def test_no_fake_missing_profile_or_executed_zero(self):
        t=raw('TGT',False);t['node_profiles'][1]['status']='failed'
        with self.assertRaises(ValueError):build_operator_migration_game([raw('SRC')],[t],True)
        t=raw('TGT',False);t['node_profiles'][1]={'node_id':'a','status':'structural_zero_cost','mean_ms':0.}
        with self.assertRaises(ValueError):build_operator_migration_game([raw('SRC')],[t],True)
        t=raw('TGT',False);t['node_profiles'].pop()
        with self.assertRaises(ValueError):build_operator_migration_game([raw('SRC')],[t],True)

    def test_process_repr_addresses_normalized_only_for_functions(self):
        a=raw('SRC')['graph'];b=copy.deepcopy(a)
        a['nodes'][3]['target']='<built-in method flatten of type object at 0xabc123>'
        b['nodes'][3]['target']='<built-in method flatten of type object at 0xdef456>'
        self.assertEqual(canonical_graph(a),canonical_graph(b))
        b['nodes'][3]['target']='<built-in method add of type object at 0xdef456>'
        self.assertNotEqual(canonical_graph(a),canonical_graph(b))

    def test_grouping_covariance_headroom_and_frozen_rank(self):
        game,c=build_operator_migration_game([raw('SRC')],[raw('TGT',False)],True)
        estimate=estimate_shapley(c['players'],game,20,42)
        p=grouped_action_priority(task('TGT'),c,estimate,1)
        self.assertEqual(p['selected_action_ids'],['opt-b']);self.assertEqual(p['selected_mask'],2)
        self.assertEqual(p['actions'][1]['grouped_migration_phi_ms'],.5)
        self.assertAlmostEqual(p['actions'][1]['positive_headroom_ratio'],.25)
        self.assertFalse(p['headroom_fallback_used'])
        changed=task('TGT');changed['action_metadata'][1]['profile_run_ids']=['eval4']
        with self.assertRaises(ValueError):grouped_action_priority(changed,c,estimate,1)
        changed=task('TGT');changed['action_metadata'][1]['module_paths']=['unknown']
        with self.assertRaises(ValueError):grouped_action_priority(changed,c,estimate,1)

    def test_no_positive_migration_burden_explicit_headroom_fallback(self):
        game,c=build_operator_migration_game([raw('SRC')],[raw('TGT',False)],True)
        estimate={'values':{'a':-1.,'b':-2.,'c':0.},'unit_values':[[-1.,-2.,0.]]*10}
        p=grouped_action_priority(task('TGT'),c,estimate,1)
        self.assertTrue(p['headroom_fallback_used']);self.assertEqual(p['selected_action_ids'],['opt-b'])

    def test_same_budget_controls_and_invalid_budget(self):
        game,c=build_operator_migration_game([raw('SRC')],[raw('TGT',False)],True)
        estimate=estimate_shapley(c['players'],game,20,42)
        p=grouped_action_priority(task('TGT'),c,estimate,1)
        plans=matched_control_plans(task('TGT'),p,1)
        self.assertEqual(set(plans),{'cross_plain_sum','target_headroom','target_operator_latency'})
        for plan in plans.values():
            self.assertEqual(plan['selected_action_ids'],['opt-b'])
            self.assertEqual(plan['selected_mask'],2)
            self.assertEqual(plan['budget_at_most'],1)
        for budget in [-1,0,True,3]:
            with self.assertRaises(ValueError):grouped_action_priority(task('TGT'),c,estimate,budget)

    def test_same_surrogate_reference_is_named_not_physical_truth(self):
        r=run_bridge(task('SRC'),task('TGT'),[raw('SRC')],[raw('TGT',False)],1,20,42,allow_unit_fixtures=True)
        self.assertEqual(r['reference']['kind'],'exact same diagnostic surrogate, not physical attribution truth')
        self.assertAlmostEqual(r['operator_estimate']['efficiency_residual'],0.)
        self.assertAlmostEqual(r['sample_sum_vs_physical_gap_residual_ms'],-3.5)

    def test_source_retention_catches_target_local_only_false_acceptance(self):
        source={'run_id':'src-eval4','records':{0:quality_record(0,9,10.)}}
        target={'run_id':'tgt-eval4','records':{0:quality_record(0,8,14.),1:quality_record(1,8,12.)}}
        r=paired_migration_quality(source,target,1,{'requested_mask':1,'accepted':True,'reason':'ok'})
        self.assertEqual(r['candidate_retention_to_target'],1.)
        self.assertAlmostEqual(r['candidate_retention_to_source'],8/9)
        self.assertFalse(r['migration_accepted']);self.assertEqual(r['fallback'],'original_source_baseline')

    def test_physical_gap_trace_after_frozen_plan(self):
        source={'run_id':'src-eval4','records':{0:quality_record(0,8,10.)}}
        target={'run_id':'tgt-eval4','records':{0:quality_record(0,8,14.),1:quality_record(1,8,12.)}}
        r=paired_migration_quality(source,target,1,{'requested_mask':1,'accepted':True,'reason':'ok'})
        self.assertTrue(r['migration_accepted']);self.assertEqual(r['physical_gap_before_ms'],4.)
        self.assertEqual(r['physical_gap_after_requested_plan_ms'],2.)
        self.assertAlmostEqual(r['raw_target_optimization_benefit_percent'],100*2/14)

    def test_raw_accuracy_and_sample_label_mismatch_rejected(self):
        source={'run_id':'src-eval4','records':{0:quality_record(0,8,10.)}}
        target={'run_id':'tgt-eval4','records':{0:quality_record(0,8,14.),1:quality_record(1,8,12.)}}
        target['records'][1]['quality_score']=.9
        with self.assertRaises(ValueError):paired_migration_quality(source,target,1,{'requested_mask':1,'accepted':True})
        target['records'][1]=quality_record(1,8,12.,[f'OTHER_{i}' for i in range(10)])
        with self.assertRaises(ValueError):paired_migration_quality(source,target,1,{'requested_mask':1,'accepted':True})


class Recorder(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.records=[]
    def addSuccess(self,test):super().addSuccess(test);self.records.append({'id':test.id(),'passed':True})
    def addFailure(self,test,err):super().addFailure(test,err);self.records.append({'id':test.id(),'passed':False})
    def addError(self,test,err):super().addError(test,err);self.records.append({'id':test.id(),'passed':False})


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2,resultclass=Recorder).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    root=Path(__file__).parent
    out={'scope':'Pure stdlib lightweight UNIT_FIXTURE identity/math/paired-quality tests only',
         'new_physical_runs':0,'heavy_training_runs':0,'passed':result.wasSuccessful(),'cases':result.records,
         'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}}
    (root/'migration-bridge-unit-results.json').write_text(json.dumps(out,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
