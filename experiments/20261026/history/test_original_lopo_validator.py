#!/usr/bin/env python3
"""Synthetic software test fixtures only. These are never benchmark observations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import validate_original_lopo as v

HERE=Path(__file__).resolve().parent

class ValidatorControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='selftest-',dir=HERE)
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'SYNTHETIC_SOFTWARE_TEST_ONLY.json'
        config={'input_shapes':[[1,3,2,2]],'output_shapes':[[1,4]],'dtype':'float16','layout':'NCHW','operator_parameters':{'kernel':[1,1]}}
        self.r={'record_id':'synthetic-selftest-only','status':'measured','platform_id':'A100_80GB','source_model_id':'train_model','operator_instance_id':'train_op','run_id':'synthetic-test-not-physical','captured_at_utc':'2026-10-26T00:00:00Z','samples_ms':[1.,2.,3.,4.,5.],'statistic':'mean','latency_ms':3.,'configuration':config,'configuration_sha256':v.canonical_sha(config),'model_weights_sha256':'a'*64,'input_sha256':'b'*64,'software_environment_sha256':'c'*64}
        self.hw={'platform_id':'A100_80GB','device_name':v.PLATFORMS['A100_80GB'],'partition':'full_device','device_variant':'software-test-fixture','driver_version':'test','runtime_version':'test','framework_version':'test','host_inventory_sha256':'d'*64}
        self.reset_evidence()

    def reset_evidence(self):
        raw={k:value for k,value in self.r.items() if k not in ['raw_evidence','hardware_evidence']}
        self.path.write_text(json.dumps({'purpose':'SYNTHETIC SOFTWARE VALIDATOR TEST ONLY; NOT EXPERIMENTAL EVIDENCE','records':[raw],'hardware':self.hw}))
        digest=v.sha256(self.path)
        self.r['raw_evidence']={'path':str(self.path),'sha256':digest,'pointer':'/records/0'}
        self.r['hardware_evidence']={'path':str(self.path),'sha256':digest,'pointer':'/hardware'}

    def run_record(self,r=None):
        a=v.Audit();v.measured_record(r or self.r,a);return a

    def test_valid_integrity_control(self):
        self.assertFalse(self.run_record().counts)

    def test_means_without_repeats_rejected(self):
        del self.r['samples_ms']
        self.assertIn('RAW_REPEATS_REQUIRED',self.run_record().counts)

    def test_modified_source_file_rejected(self):
        self.path.write_text('{}')
        self.assertIn('SOURCE_INTEGRITY',self.run_record().counts)

    def test_raw_status_and_values_must_match(self):
        d=json.loads(self.path.read_text());d['records'][0]['status']='estimated';self.path.write_text(json.dumps(d));self.r['raw_evidence']['sha256']=v.sha256(self.path);self.r['hardware_evidence']['sha256']=v.sha256(self.path)
        self.assertIn('RAW_RECORD_MISMATCH',self.run_record().counts)

    def test_mig_cannot_replace_original_full_device(self):
        self.hw['partition']='MIG-2g.20gb';self.reset_evidence()
        self.assertIn('PARTITION_SUBSTITUTION',self.run_record().counts)

    def test_hardware_identity_mismatch(self):
        self.hw['device_name']='Apple M5 MPS';self.reset_evidence()
        self.assertIn('WRONG_HARDWARE',self.run_record().counts)

    def test_target_labels_and_test_models_cannot_train(self):
        records={'leak':{'record_id':'leak','status':'measured','platform_id':'A100_80GB','source_model_id':'test_model'}}
        f={'held_out_platform':'A100_80GB','train_record_ids':['leak'],'validation_record_ids':[],'test_record_ids':[],'feature_names':['latency_ms'],'normalization_fit_record_ids':['leak'],'feature_selection_fit_record_ids':['leak'],'training_performance_edge_record_ids':['leak'],'hyperparameter_selection_record_ids':[],'training_similarity_edges':[{'source_model':'train_model','target_model':'test_model'}]}
        a=v.Audit();v.validate_fold(f,records,{'train_model':'train','test_model':'test'},a)
        for code in ['SOURCE_MODEL_LEAKAGE','HELDOUT_PLATFORM_LABEL_LEAKAGE','UNAPPROVED_FEATURE','SIMILARITY_EDGE_LEAKAGE']:self.assertIn(code,a.counts)

    def test_estimated_labels_cannot_score(self):
        records={'estimate':{'record_id':'estimate','status':'estimated','platform_id':'A100_80GB','source_model_id':'test_model'}}
        f={'held_out_platform':'A100_80GB','train_record_ids':[],'validation_record_ids':[],'test_record_ids':['estimate'],'feature_names':['input_shapes']}
        a=v.Audit();v.validate_fold(f,records,{'test_model':'test'},a)
        self.assertIn('NONMEASURED_LABEL_USED',a.counts)

    def test_five_new_runtime_stacks_rejected(self):
        b=json.loads((HERE/'current-original-data-candidate.json').read_text());b['platform_ids']=['m5cpu','m5mps','a100mig','xeon_cpu','npub3']
        self.assertIn('PLATFORM_SUBSTITUTION',v.validate(b)['error_counts'])

    def test_current_history_gate_fails_honestly(self):
        report=v.validate(json.loads((HERE/'current-original-data-candidate.json').read_text()))
        self.assertFalse(report['pass'])
        self.assertEqual(report['unrecorded_opportunities'],31220)
        self.assertIn('CANONICAL5000_MISSING',report['error_counts'])
        self.assertIn('NO_PHYSICAL_LABELS',report['error_counts'])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ValidatorControls)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    (HERE/'validator-selftest-results.json').write_text(json.dumps({'purpose':'Software validation only; all positive fixture latency values are synthetic and prohibited from paper results','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'pass':result.wasSuccessful(),'validator_sha256':v.sha256(HERE/'validate_original_lopo.py'),'tests_sha256':v.sha256(__file__)},indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
