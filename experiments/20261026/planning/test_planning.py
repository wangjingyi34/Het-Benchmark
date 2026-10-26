"""Fault injection and invariance tests on in-memory copies; no new experiment."""
from copy import deepcopy
import json
from pathlib import Path
import statistics as st
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import recompute_planning as r


class AuditGateTests(unittest.TestCase):
    def setUp(self):
        self.raw = r.read(r.RAW / 'npu-resnet18-b1-r4.json')
        self.candidate = next(x for x in self.raw['coalitions'] if x['mask'] == 6)

    def test_real_positive_block_accepted(self):
        outcome = r.strict_gate(self.raw, 6, [True] * 4)
        self.assertTrue(outcome['accepted'])
        self.assertEqual(outcome['candidate_top1'], .62)
        self.assertAlmostEqual(outcome['raw_gain_percent'], 1.169611130254616)

    def test_raw_score_spoof_rejected(self):
        i = next(i for i, (a, b) in enumerate(zip(self.candidate['labels'], self.candidate['predictions'])) if a == b)
        self.candidate['predictions'][i] = (self.candidate['labels'][i] + 1) % 1000
        with self.assertRaisesRegex(ValueError, 'accuracy disagrees'):
            r.strict_gate(self.raw, 6, [True] * 4)

    def test_duplicate_sample_identity_rejected(self):
        self.candidate['sample_ids'][1] = self.candidate['sample_ids'][0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            r.strict_gate(self.raw, 6, [True] * 4)

    def test_quality_failure_rolls_back(self):
        self.candidate['predictions'] = [(label + 1) % 1000 for label in self.candidate['labels']]
        self.candidate.update(quality_score=0.0, correct_count=0, correct=[False] * 100)
        outcome = r.strict_gate(self.raw, 6, [True] * 4)
        self.assertEqual(outcome['reason'], 'quality_below_threshold')
        self.assertEqual(outcome['accepted_mask'], 0)
        self.assertEqual(outcome['accepted_gain_percent'], 0)

    def test_latency_failure_rolls_back(self):
        self.candidate['latency_samples_ms'] = [2 * x for x in self.candidate['latency_samples_ms']]
        self.candidate['mean_ms'] = st.mean(self.candidate['latency_samples_ms'])
        self.candidate['sample_sd_ms'] = st.stdev(self.candidate['latency_samples_ms'])
        outcome = r.strict_gate(self.raw, 6, [True] * 4)
        self.assertEqual(outcome['reason'], 'latency_regression')
        self.assertLess(outcome['raw_gain_percent'], 0)
        self.assertEqual(outcome['accepted_gain_percent'], 0)

    def test_hardware_failure_rejected(self):
        outcome = r.strict_gate(self.raw, 6, [True, False, True, True])
        self.assertEqual(outcome['reason'], 'hardware_unsupported')
        self.assertFalse(outcome['accepted'])

    def test_failed_status_rejected(self):
        self.candidate['status'] = 'failed'
        self.assertEqual(r.strict_gate(self.raw, 6, [True] * 4)['reason'], 'failed')

    def test_calibration_search_ignores_heldout_outcomes(self):
        tp = r.METHODS / 'planner-matrix-inputs/tasks/npu-resnet18_b1_imagenet_val10class_v2.json'
        task, blocks, _, _ = r.validate_task(tp)
        original = r.independent_search(task, blocks)
        changed = deepcopy(blocks)
        changed['eval'] = []
        self.assertEqual(original, r.independent_search(task, changed))

    def test_physical_process_interval_does_not_count_policy_seeds(self):
        s = r.summary([1, 2, 3, 4, 5])
        self.assertEqual(s['physical_processes'], 5)
        self.assertAlmostEqual(s['descriptive_t95_df4'][1], 3 + r.T95 * st.stdev([1, 2, 3, 4, 5]) / 5 ** .5)


class ArchivedGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(r.ROOT / '.experiment-deps'))
        sys.path.insert(0, str(r.METHODS))
        from measured_planner import MeasuredTable
        from migration_bridge import paired_migration_quality
        cls.Table = MeasuredTable
        cls.migration_gate = staticmethod(paired_migration_quality)

    def setUp(self):
        self.task = r.read(r.METHODS / 'planner-matrix-inputs/tasks/npu-resnet18_b1_imagenet_val10class_v2.json')

    def test_archived_quality_hard_gate_rejects_low_score(self):
        table = self.Table(self.task)
        block = deepcopy(table.splits['eval'][0])
        block['records'][6]['quality_score'] = .60
        outcome = table.final_gate(block, 6)
        self.assertFalse(outcome['accepted'])
        self.assertEqual(outcome['reason'], 'quality_below_threshold')
        self.assertTrue(outcome['rollback_to_baseline'])

    def test_archived_gate_requires_trusted_or_prevalidated_score(self):
        # Explicit limitation: original final_gate trusts supplied quality_score.
        table = self.Table(self.task)
        block = deepcopy(table.splits['eval'][0])
        block['records'][6]['predictions'] = [(x + 1) % 1000 for x in block['records'][6]['labels']]
        self.assertTrue(table.final_gate(block, 6)['accepted'])
        with self.assertRaisesRegex(ValueError, 'accuracy disagrees'):
            r.quality(block['records'][6])

    def test_archived_cross_device_quality_gate_rejects(self):
        stask = r.read(r.ROOT / 'v2-实证增强/evidence/early-tasks/resnet18_b1_imagenet_val10class_v2_2c4332169b.json')
        source = self.Table(stask)
        target = self.Table(self.task)
        sb = deepcopy(source.splits['eval'][0])
        baseline = sb['records'][0]
        baseline.update(predictions=list(baseline['labels']), quality_score=1.0, correct_count=100, correct=[True] * 100)
        tb = target.splits['eval'][0]
        local_gate = target.final_gate(tb, 6)
        self.assertTrue(local_gate['accepted'])
        result = self.migration_gate(sb, tb, 6, local_gate, .99)
        self.assertFalse(result['migration_accepted'])
        self.assertEqual(result['migration_rejection_reason'], 'cross_device_quality_below_source_threshold')
        self.assertEqual(result['fallback'], 'original_source_baseline')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    r.dump('quality-gate-tests.json', {'status': 'passed' if result.wasSuccessful() else 'failed',
        'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'test_source_sha256': r.sha(__file__), 'audit_validator_sha256': r.sha(r.HERE / 'recompute_planning.py'),
        'scope': 'Fault injection on in-memory copies and direct original gate calls; no hardware-quality failure trial.',
        'original_gate_limitation': 'Original final_gate trusts quality_score; raw-ledger spoof is accepted there but rejected by this isolated strict validator. Original runtime files were not modified.',
        'implementation_chain': ['raw heldout labeled predictions -> audited top1', 'local target retention >=.99',
                                 'hardware support and valid status', 'nonregression or baseline rollback',
                                 'cross-device candidate/source retention >=.99, else source fallback'],
        'deployed_runtime_hardening_claim': False})
    sys.exit(0 if result.wasSuccessful() else 1)
