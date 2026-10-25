"""Executed mathematical and logic tests, explicitly not physical experiments."""
import copy
import hashlib
import json
import math
import statistics
import unittest
import tempfile
from pathlib import Path

import torch

from copa_v2 import ActionDAGGame, exact_shapley, estimate_shapley, improvement_game
from measured_planner import (MeasuredTable, CalibrationEnv, train_ppo, select_plan,
                              evaluate_plan, attribution_rankings, feasible_oracle,
                              uniform_random_expectation,calibration_search,ActorCritic)
from runtime_calibration import truncated_quadratic_residual,build_runtime_variants
from attach_embeddings import attach


def fixture():
    def block(run,split,latencies):
        return {'run_id':run,'provenance':'UNIT_FIXTURE','quality_partition':split,
                'quality_sample_ids':[split+'-sample'], 'baseline_quality':0.8,
                'coalitions':[{'mask':m,'status':'ok','latency_samples_ms':[v,v],
                               'quality_score':0.8,'quality_evaluated':True,
                               'artifact_id':f'UNIT_FIXTURE:{run}:{m}'} for m,v in enumerate(latencies)]}
    return {'schema_version':'het.measured.coalitions.v2','task_id':'UNIT_FIXTURE','model_id':'unit-model',
            'device_id':'UNIT_FIXTURE_NO_PHYSICAL_DEVICE','action_ids':['a','b'],
            'quality_definition':{'metric':'UNIT_FIXTURE_QUALITY','dataset_id':'UNIT_FIXTURE',
                                  'threshold':0.99,'mode':'retention_ratio'},
            'action_metadata':[{'id':'a','hardware_supported':True,'operator_latency_ms':8.,
                                'local_benefit_ms':1.,'profile_run_ids':['tr1','tr2']},
                               {'id':'b','hardware_supported':True,'operator_latency_ms':4.,
                                'local_benefit_ms':2.,'profile_run_ids':['tr1','tr2']}],
            'train':[block('tr1','calibration',[10.,9.,8.,6.]),block('tr2','calibration',[10.1,9.1,8.1,6.1])],
            'eval':[block('ev1','evaluation',[10.,9.,8.,6.]) ]}


def four_action_fixture():
    f=fixture();actions=['a','b','c','d'];f['action_ids']=actions
    f['action_metadata']=[{'id':a,'hardware_supported':True,'operator_latency_ms':1.,
                           'local_benefit_ms':.1,'profile_run_ids':['tr1','tr2']} for a in actions]
    for split in ['train','eval']:
        for block in f[split]:
            block['coalitions']=[{'mask':m,'status':'ok','latency_samples_ms':[10-m.bit_count()*.1],
                                  'quality_score':.8,'quality_evaluated':True,
                                  'artifact_id':f"UNIT_FIXTURE:{block['run_id']}:{m}"} for m in range(16)]
    f['copa_game']={'base_costs_ms':{a:1. for a in actions},'edges':[(actions[i],actions[i+1]) for i in range(3)],
                    'action_cost_updates_ms':{a:{a:.9} for a in actions},'runtime_overhead_ms':0.,
                    'profile_run_ids':['tr1','tr2']}
    return f


class MathTests(unittest.TestCase):
    def test_branch_path_switch_and_sum_ablation(self):
        args=(['a','b'],{'a':4.,'b':3.},[],{'a':{'a':1.},'b':{'b':2.}})
        path=ActionDAGGame(*args);total=ActionDAGGame(*args,aggregation='sum')
        self.assertEqual(exact_shapley(['a','b'],improvement_game(path))['values'],{'a':1.5,'b':.5})
        self.assertEqual(exact_shapley(['a','b'],improvement_game(total))['values'],{'a':3.,'b':1.})

    def test_serial_degeneracy_and_constant_cancellation(self):
        args=(['a','b'],{'a':4.,'b':3.},[('a','b')],{'a':{'a':1.},'b':{'b':2.}})
        p=exact_shapley(['a','b'],improvement_game(ActionDAGGame(*args)))['values']
        c=exact_shapley(['a','b'],improvement_game(ActionDAGGame(*args,runtime_overhead=10)))['values']
        self.assertEqual(p,{'a':3.,'b':1.});self.assertEqual(p,c)

    def test_coalition_runtime_changes_allocation(self):
        args=(['a','b'],{'a':4.,'b':3.},[],{'a':{'a':1.},'b':{'b':2.}})
        g=ActionDAGGame(*args,runtime_overhead=lambda m:1. if m==1 else 0.)
        self.assertEqual(exact_shapley(['a','b'],improvement_game(g))['values'],{'a':1.,'b':1.})

    def test_pair_units_se_and_budget(self):
        vals=[0.,1.,2.,5.,4.,7.,8.,20.]
        r=estimate_shapley(['a','b','c'],lambda m:vals[m],100,11,True)
        self.assertEqual(r['independent_units'],50);self.assertEqual(len(r['permutations']),100)
        expected=statistics.stdev(u[0] for u in r['unit_values'])/math.sqrt(50)
        self.assertAlmostEqual(r['standard_errors']['a'],expected)
        self.assertAlmostEqual(r['efficiency_residual'],0.,places=12)
        one=estimate_shapley(['a','b','c'],lambda m:vals[m],2,11,True)
        self.assertIsNone(one['standard_errors']['a'])
        with self.assertRaises(ValueError):estimate_shapley(['a','b'],lambda m:m,3)

    def test_negative_contribution_retained(self):
        r=exact_shapley(['a','b'],lambda m:[0,2,-1,1][m])
        self.assertEqual(r['values'],{'a':2.,'b':-1.})

    def test_endpoint_closure_cannot_identify_coalitions(self):
        a=exact_shapley(['a','b'],lambda m:[0,10,0,10][m])
        b=exact_shapley(['a','b'],lambda m:[0,0,10,10][m])
        self.assertEqual(a['efficiency_residual'],b['efficiency_residual'])
        self.assertNotEqual(a['values'],b['values'])

    def test_invalid_graph_overlap_nonfinite_rejected(self):
        with self.assertRaises(ValueError):ActionDAGGame(['a'],{'x':1.,'y':1.},[('x','y'),('y','x')],{'a':{'x':0.}})
        with self.assertRaises(ValueError):ActionDAGGame(['a','b'],{'x':1.},[],{'a':{'x':0.},'b':{'x':2.}})
        with self.assertRaises(ValueError):exact_shapley(['a'],lambda m:float('nan'))


class TableAndGateTests(unittest.TestCase):
    def test_synthetic_data_rejected_without_explicit_test_mode(self):
        with self.assertRaises(ValueError):MeasuredTable(fixture())

    def test_duplicate_run_and_quality_sample_leak_rejected(self):
        f=fixture();f['eval'][0]['run_id']='tr1'
        with self.assertRaises(ValueError):MeasuredTable(f,True)
        f=fixture();f['eval'][0]['quality_sample_ids']=['calibration-sample']
        with self.assertRaises(ValueError):MeasuredTable(f,True)

    def test_metadata_eval_leak_rejected(self):
        f=fixture();f['action_metadata'][0]['profile_run_ids']=['ev1']
        with self.assertRaises(ValueError):MeasuredTable(f,True)

    def test_mask_and_budget_state(self):
        t=MeasuredTable(fixture(),True);env=CalibrationEnv(t,1)
        env.reset(t.splits['train'][0]);start=env.state()
        _,reward,done,_=env.step(1)
        self.assertTrue(done);self.assertAlmostEqual(reward,2/10.05)
        self.assertEqual(int(env.state()[1]),1);self.assertEqual(float(env.state()[2]),0.)
        self.assertFalse(env.valid_mask()[0]);self.assertTrue(env.valid_mask()[-1])
        with self.assertRaises(ValueError):env.step(0)
        with self.assertRaises(ValueError):env.reset(t.splits['eval'][0])

    def test_hardware_mask_and_final_gate_separate(self):
        f=fixture();f['action_metadata'][1]['hardware_supported']=False
        t=MeasuredTable(f,True);env=CalibrationEnv(t,2,True);env.reset(t.splits['train'][0])
        self.assertFalse(env.valid_mask()[1])
        unmasked=CalibrationEnv(t,2,False);unmasked.reset(t.splits['train'][0])
        self.assertTrue(unmasked.valid_mask()[1]);_,r,done,info=unmasked.step(1)
        self.assertTrue(done and info['invalid']);self.assertEqual(r,-2.)
        outcome=t.final_gate(t.splits['eval'][0],2)
        self.assertFalse(outcome['accepted']);self.assertEqual(outcome['accepted_mask'],0)

    def test_quality_modes_and_block_specific_baseline(self):
        f=fixture();f['eval'][0]['baseline_quality']=.5
        for rec in f['eval'][0]['coalitions']:rec['quality_score']=.5
        t=MeasuredTable(f,True);self.assertTrue(t.final_gate(t.splits['eval'][0],1)['accepted'])
        f['quality_definition']['mode']='absolute'
        t=MeasuredTable(f,True);self.assertFalse(t.final_gate(t.splits['eval'][0],1)['accepted'])

    def test_calibration_point9_eval_point8_paired_retention_accepts(self):
        f=fixture()
        for b in f['train']:
            b['baseline_quality']=.9
            for rec in b['coalitions']:rec['quality_score']=.9
        f['eval'][0]['baseline_quality']=.8
        for rec in f['eval'][0]['coalitions']:rec['quality_score']=.8
        t=MeasuredTable(f,True)
        self.assertEqual(t.calibrated_observation(0)[1],1.)
        accepted=t.final_gate(t.splits['eval'][0],1)
        self.assertTrue(accepted['accepted']);self.assertEqual(accepted['normalized_quality'],1.)
        self.assertEqual(accepted['baseline_quality'],.8)

    def test_quality_missing_nan_and_slowdown_fail_closed(self):
        f=fixture();f['eval'][0]['coalitions'][1]['quality_evaluated']=False
        t=MeasuredTable(f,True);self.assertFalse(t.final_gate(t.splits['eval'][0],1)['accepted'])
        f=fixture();f['eval'][0]['coalitions'][1]['quality_score']=float('nan')
        t=MeasuredTable(f,True);self.assertFalse(t.final_gate(t.splits['eval'][0],1)['accepted'])
        f=fixture();f['eval'][0]['coalitions'][1]['latency_samples_ms']=[11.,11.]
        t=MeasuredTable(f,True);o=t.final_gate(t.splits['eval'][0],1)
        self.assertFalse(o['accepted']);self.assertAlmostEqual(o['raw_benefit_percent'],-10.)
        self.assertEqual(o['accepted_benefit_percent'],0.)

    def test_placeholder_embedding_rejected(self):
        f=fixture()
        with self.assertRaises(ValueError):CalibrationEnv(MeasuredTable(f,True),1,with_embedding=True)

    def test_embedding_control_capacity_and_model_holdout(self):
        f=fixture()
        for m in f['action_metadata']:
            m['graph_embedding']=[.1,.2,.3]
            m['embedding_provenance']={'checkpoint_sha256':'UNIT_FIXTURE','training_model_ids':['other-model'],
                                       'used_inference_latency_labels':False,'split_id':'UNIT_SPLIT',
                                       'test_model_ids':['unit-model'],'validation_model_ids':[]}
        t=MeasuredTable(f,True)
        yes=CalibrationEnv(t,1,with_embedding=True);zero=CalibrationEnv(t,1,with_embedding=True,drop_embedding=True)
        yes.reset(t.splits['train'][0]);zero.reset(t.splits['train'][0])
        self.assertEqual(len(yes.state()),len(zero.state()))
        self.assertTrue(torch.equal(yes.state()[:-6],zero.state()[:-6]))
        self.assertEqual(float(zero.state()[-6:].abs().sum()),0.)
        self.assertGreater(float(yes.state()[-6:].abs().sum()),0.)
        f['action_metadata'][0]['embedding_provenance']['training_model_ids']=['unit-model']
        with self.assertRaises(ValueError):CalibrationEnv(MeasuredTable(f,True),1,with_embedding=True)
        for m in f['action_metadata']:m['graph_embedding']=[0.,0.]
        with self.assertRaises(ValueError):CalibrationEnv(MeasuredTable(f,True),1,with_embedding=True)

    def test_oracle_and_uniform_expectation(self):
        t=MeasuredTable(fixture(),True)
        self.assertEqual(feasible_oracle(t,1)[0]['requested_mask'],2)
        self.assertAlmostEqual(uniform_random_expectation(t,1)[0]['raw_benefit_percent'],15.)
        self.assertAlmostEqual(uniform_random_expectation(t,1,True)[0]['raw_benefit_percent'],10.)
        self.assertEqual(uniform_random_expectation(t,1,True)[0]['subset_count'],3)
        self.assertEqual(attribution_rankings(t,1)['measured_calibration_shapley']['mask'],2)
        self.assertEqual(calibration_search(t,1,True),2)
        self.assertEqual(calibration_search(t,1,False),2)

    def test_missing_training_quality_causes_rejection_not_crash(self):
        f=fixture();f['train'][0]['coalitions'][1]['quality_evaluated']=False
        t=MeasuredTable(f,True);env=CalibrationEnv(t,1);env.reset(t.splits['train'][0])
        _,reward,done,info=env.step(0)
        self.assertTrue(done and info['invalid']);self.assertEqual(reward,-2.)

    def test_no_feasible_absolute_quality_oracle(self):
        f=fixture();f['quality_definition']['mode']='absolute'
        t=MeasuredTable(f,True)
        self.assertEqual(feasible_oracle(t,1)[0]['reason'],'no_feasible_plan')


class PPOTests(unittest.TestCase):
    def test_ppo_learns_true_table_and_ignores_eval_labels(self):
        t=MeasuredTable(fixture(),True)
        model,log=train_ppo(t,budget=1,seed=7,total_steps=2048,batch_steps=128,epochs=4)
        choice=select_plan(model,t,1)
        self.assertEqual(choice['mask'],2)
        self.assertFalse(log['eval_access_during_training'])
        self.assertTrue(all(math.isfinite(r['loss']) for r in log['history']))
        changed=fixture();changed['eval'][0]['coalitions'][1]['latency_samples_ms']=[.01,.01]
        changed['eval'][0]['coalitions'][2]['latency_samples_ms']=[100.,100.]
        t2=MeasuredTable(changed,True)
        other,_=train_ppo(t2,budget=1,seed=7,total_steps=2048,batch_steps=128,epochs=4)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.parameters(),other.parameters())))
        self.assertEqual(select_plan(other,t2,1)['mask'],2)
        self.assertFalse(evaluate_plan(t2,2)[0]['accepted'])
        self.assertEqual(evaluate_plan(t,2)[0]['accepted_mask'],2)


class RuntimeCalibrationTests(unittest.TestCase):
    def test_quadratic_residual_recovers_low_order_without_high_order_access(self):
        actions=['a','b','c','d']
        game=ActionDAGGame(actions,{a:1. for a in actions},[(actions[i],actions[i+1]) for i in range(3)],
                           {a:{a:.9} for a in actions})
        def truth(m):return game.raw_runtime(m)+.5+.1*m.bit_count()+.03*math.comb(m.bit_count(),2)
        observed={m:truth(m) for m in range(16) if m.bit_count()<=2}
        overhead,fit=truncated_quadratic_residual(game,observed,4)
        self.assertEqual(len(fit['fit_masks']),11);self.assertEqual(len(fit['withheld_masks']),5)
        for m in range(16):self.assertAlmostEqual(game.raw_runtime(m)+overhead(m),truth(m),places=12)

    def test_fitted_models_invariant_to_high_order_and_eval_labels(self):
        f=four_action_fixture();a,trace=build_runtime_variants(MeasuredTable(f,True))
        changed=copy.deepcopy(f)
        for split in ['train','eval']:
            for block in changed[split]:
                for rec in block['coalitions']:
                    if split=='eval' or rec['mask'].bit_count()>2:rec['latency_samples_ms']=[100+rec['mask']]
        b,trace2=build_runtime_variants(MeasuredTable(changed,True))
        self.assertEqual(trace,trace2)
        for name in a:
            self.assertEqual([a[name].raw_runtime(m) for m in range(16)],
                             [b[name].raw_runtime(m) for m in range(16)])

    def test_nonpositive_extrapolation_retained_not_clamped(self):
        actions=['a','b','c','d'];game=ActionDAGGame(actions,{a:1. for a in actions},[],{a:{a:1.} for a in actions},aggregation='sum')
        observed={m:1. if m.bit_count()!=1 else 2. for m in range(16) if m.bit_count()<=2}
        overhead,_=truncated_quadratic_residual(game,observed,4)
        fitted=ActionDAGGame(actions,{a:1. for a in actions},[],{a:{a:1.} for a in actions},overhead,aggregation='sum')
        self.assertAlmostEqual(fitted.raw_runtime(15),-7.)
        with self.assertRaises(ValueError):fitted(15)

    def test_runtime_profile_eval_leak_rejected(self):
        f=four_action_fixture();f['copa_game']['profile_run_ids']=['ev1']
        with self.assertRaises(ValueError):MeasuredTable(f,True)


class EmbeddingAttachmentTests(unittest.TestCase):
    def setup_payload(self):
        f=fixture();f['input_shape']=[1,3,224,224]
        rows=[]
        for i,m in enumerate(f['action_metadata']):
            m['module_paths']=[f'conv{i}',f'bn{i}']
            rows.append({'model_id':'unit-model','device_id':f['device_id'],
                         'input_shape':f['input_shape'],'module_paths':m['module_paths'],
                         'graph_embedding':[.1]*32,'embedding_provenance':{
                             'checkpoint_sha256':'UNIT_FIXTURE','training_model_ids':['other-model'],
                             'used_inference_latency_labels':False,'split_id':'UNIT_SPLIT',
                             'test_model_ids':['unit-model'],'validation_model_ids':[]}})
        return f,{'kind':'frozen_inductive_action_embeddings','records':rows}

    def test_exact_match_and_duplicate_rejection(self):
        f,e=self.setup_payload();out=attach(f,e)
        env=CalibrationEnv(MeasuredTable(out,True),1,with_embedding=True)
        self.assertEqual(len(env.embedding),64)
        e['records'].append(copy.deepcopy(e['records'][0]))
        with self.assertRaises(ValueError):attach(f,e)

    def test_device_alias_requires_explicit_evidence(self):
        f,e=self.setup_payload()
        for r in e['records']:r['device_id']='explicit-other-ID'
        with self.assertRaises(ValueError):attach(f,e,'explicit-other-ID')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'UNIT_FIXTURE-runtime.json';p.write_text('{"kind":"UNIT_FIXTURE"}')
            out=attach(f,e,'explicit-other-ID',p)
            self.assertIn('explicit_device_alias_evidence',out['action_metadata'][0]['embedding_provenance'])


class RecordingResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.records=[]
    def addSuccess(self,test):super().addSuccess(test);self.records.append({'id':test.id(),'passed':True})
    def addFailure(self,test,err):super().addFailure(test,err);self.records.append({'id':test.id(),'passed':False,'type':'failure'})
    def addError(self,test,err):super().addError(test,err);self.records.append({'id':test.id(),'passed':False,'type':'error'})


if __name__=='__main__':
    torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2,resultclass=RecordingResult).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    root=Path(__file__).parent
    d={'scope':'New v2 mathematical/logic/PPO implementation unit tests only; all data are UNIT_FIXTURE',
       'physical_measurements':0,'task_accuracy_evaluations':0,'tests':result.records,
       'passed':result.wasSuccessful(),'torch':torch.__version__,
       'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}}
    (root/'unit-test-results.json').write_text(json.dumps(d,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
