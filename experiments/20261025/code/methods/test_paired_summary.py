"""Pure arithmetic fixtures; never collected or labeled as physical runs."""
import copy
import hashlib
import json
import unittest
from pathlib import Path
from summarize_paired_methods import paired_row
from test_migration_bridge import quality_record

def fixture():
    q=quality_record(0,8,10.);pairs=[]
    for i in range(20):
        a=10. if i<10 else 20.;b=10.
        pairs.append({'pair_index':i,'status':'ok','order':'AB' if i<10 else 'BA',
            'A':{'latency_ms':a},'B':{'latency_ms':b},'baseline_ms':a,'candidate_ms':b,
            'paired_delta_ms':a-b,'paired_gain_percent':100*(a-b)/a,'output_equivalent':True})
    return {'mask':6,'status':'ok','pairs':pairs,'candidate_valid':True,'final_candidate_feasible':True,
            'baseline_quality':q,'candidate_quality':copy.deepcopy(q),'summary':{'ratio_of_means_gain_percent':100/3}}

class Tests(unittest.TestCase):
    def test_ratio_of_means_distinct_from_mean_percentages(self):
        r=paired_row(fixture());self.assertAlmostEqual(r['raw_gain_percent'],100/3)
        self.assertAlmostEqual(r['mean_pair_gain_auxiliary_percent'],25.)

    def test_failed_timed_equivalence_rejects_final_feasibility(self):
        f=fixture();f['pairs'][0]['output_equivalent']=False;f['final_candidate_feasible']=False
        self.assertFalse(paired_row(f)['candidate_feasible'])
        f['final_candidate_feasible']=True
        with self.assertRaises(ValueError):paired_row(f)

    def test_negative_control_not_clipped(self):
        f=fixture();f['mask']=0
        for p in f['pairs']:
            p['A']['latency_ms']=10.;p['B']['latency_ms']=11.;p['baseline_ms']=10.;p['candidate_ms']=11.
            p['paired_delta_ms']=-1.;p['paired_gain_percent']=-10.
        f['summary']['ratio_of_means_gain_percent']=-10.
        self.assertEqual(paired_row(f)['raw_gain_percent'],-10.)

    def test_duplicate_pair_or_wrong_order_rejected(self):
        f=fixture();f['pairs'][1]['pair_index']=0
        with self.assertRaises(ValueError):paired_row(f)
        f=fixture();f['pairs'][0]['order']='BA'
        with self.assertRaises(ValueError):paired_row(f)

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    Path(__file__).with_name('paired-summary-unit-results.json').write_text(json.dumps({'kind':'UNIT_FIXTURE arithmetic only','physical_samples':0,'passed':r.wasSuccessful(),'cases':r.testsRun,'analyzer_sha256':hashlib.sha256(Path(__file__).with_name('summarize_paired_methods.py').read_bytes()).hexdigest()},indent=2))
    raise SystemExit(0 if r.wasSuccessful() else 1)
