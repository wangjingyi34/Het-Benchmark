"""Arithmetic/unit fixtures only; not hardware or empirical performance evidence."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from summarize_bridge import analyze,vector
from test_migration_bridge import quality_record,raw,task
from migration_bridge import run_bridge,paired_migration_quality

class SummaryTests(unittest.TestCase):
    def fixture(self,directory):
        source=task('SRC');target=task('TGT');source['eval']=[];target['eval']=[]
        trace=run_bridge(source,target,[raw('SRC')],[raw('TGT',False)],1,20,42,allow_unit_fixtures=True)
        trace['physical_eval_by_plan']={k:[] for k in ['cross_dag',*trace['matched_control_plans']]}
        for idx in range(4,9):
            sb={'run_id':f'src-r{idx}','coalitions':[quality_record(0,8,10.)]}
            tb={'run_id':f'tgt-r{idx}','coalitions':[quality_record(0,8,14.),quality_record(1,8,15.),quality_record(2,8,12.),quality_record(3,8,10.)]}
            source['eval'].append(sb);target['eval'].append(tb)
            for name in trace['physical_eval_by_plan']:
                row=paired_migration_quality({'run_id':sb['run_id'],'records':{r['mask']:r for r in sb['coalitions']}},
                    {'run_id':tb['run_id'],'records':{r['mask']:r for r in tb['coalitions']}},2,
                    {'requested_mask':2,'accepted':True,'reason':'ok'})
                trace['physical_eval_by_plan'][name].append(row)
        trace['physical_eval']=trace['physical_eval_by_plan']['cross_dag']
        s=Path(directory)/'source.json';t=Path(directory)/'target.json'
        s.write_text(json.dumps(source));t.write_text(json.dumps(target))
        trace['source_task_sha256']=hashlib.sha256(s.read_bytes()).hexdigest();trace['target_task_sha256']=hashlib.sha256(t.read_bytes()).hexdigest()
        return trace,s,t

    def test_vector_uses_process_count_and_descriptive_interval(self):
        v=vector([1,2,3,4,5]);self.assertEqual(v['n_process_blocks'],5);self.assertEqual(v['mean'],3.)
        self.assertLess(v['descriptive_t95'][0],3.);self.assertIsNone(vector([1,2])['descriptive_t95'])

    def test_budgeted_oracle_and_random_raw_negative_retained(self):
        with tempfile.TemporaryDirectory() as d:
            trace,s,t=self.fixture(d);r=analyze(trace,s,t)
            self.assertTrue(all(o['best_feasible_mask']==2 for o in r['heldout_feasible_oracle']))
            self.assertEqual(r['plan_results']['cross_dag']['target_oracle_regret_ms']['mean'],0.)
            random=r['random_controls']['random_uniform_at_most_k'][0]
            self.assertEqual(random['subset_count'],3)
            self.assertLess(random['raw_gain_percent'],random['accepted_gain_percent'])

    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            trace,s,t=self.fixture(d);t.write_text(t.read_text()+' ')
            with self.assertRaises(ValueError):analyze(trace,s,t)

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SummaryTests))
    Path(__file__).with_name('bridge-summary-unit-results.json').write_text(json.dumps({'scope':'UNIT_FIXTURE arithmetic only','physical_samples':0,'cases':r.testsRun,'passed':r.wasSuccessful(),'source_sha256':hashlib.sha256(Path(__file__).with_name('summarize_bridge.py').read_bytes()).hexdigest()},indent=2))
    raise SystemExit(0 if r.wasSuccessful() else 1)
