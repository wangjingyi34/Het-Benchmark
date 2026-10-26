"""Software-only adversarial tests on detached copies of an actual ledger."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('HET_EVIDENCE_BASE', str(HERE.parent.parent))).resolve()
sys.path.insert(0, str(HERE))
import strict_quality_gate as gate

OBSERVED = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class StrictGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'v2-实证增强/evidence/fullmodel_raw/npu-resnet18-b1-r4.json'
        frozen = json.loads((HERE / 'analysis-freeze.json').read_text())
        cls.expected_hash = next(x['sha256'] for x in frozen['inputs'] if x['path'] == str(cls.path.relative_to(ROOT)))
        cls.verified_run = gate.load_verified_run(cls.path, cls.expected_hash)
        cls.binding = gate.make_binding(cls.verified_run, 6)
        cls.base = gate.packet_from_run(cls.verified_run, 0)
        cls.candidate = gate.packet_from_run(cls.verified_run, 6)

    def rejection(self, name, baseline, candidate, binding=None):
        result = gate.strict_final_gate(baseline, candidate, binding or self.binding)
        self.assertFalse(result['accepted'], name)
        self.assertEqual(result['accepted_mask'], 0, name)
        self.assertEqual(result['accepted_benefit_percent'], 0, name)
        self.assertEqual(result['fallback'], 'baseline', name)
        self.assertTrue(result['rollback_to_baseline'], name)
        OBSERVED.append({'case': name, 'accepted': result['accepted'], 'reason': result['reason'],
                         'fallback': result['fallback'], 'requires_verified_baseline_reload': result['requires_verified_baseline_reload']})
        return result

    def test_original_actual_ledger_passes(self):
        result = gate.strict_final_gate(self.base, self.candidate, self.binding)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['baseline_correct'], 62)
        self.assertEqual(result['candidate_correct'], 62)
        self.assertEqual(result['quality_total'], 100)
        self.assertEqual(result['normalized_quality'], 1)
        self.assertAlmostEqual(result['raw_benefit_percent'], 1.169611130254616)

    def test_quality_score_tampering(self):
        c = deepcopy(self.candidate)
        c['record']['quality_score'] = .99
        result = self.rejection('quality_score_tampered', self.base, c)
        self.assertEqual(result['reason'], 'quality_score_disagrees_with_raw_ledger')

    def test_prediction_tampering(self):
        c = deepcopy(self.candidate)
        i = next(i for i, (p, y) in enumerate(zip(c['record']['predictions'], c['record']['labels'])) if p == y)
        c['record']['predictions'][i] = (c['record']['labels'][i] + 1) % 1000
        self.rejection('prediction_tampered', self.base, c)

    def test_consistent_prediction_and_score_forgery(self):
        c = deepcopy(self.candidate)
        row = c['record']
        row.update(predictions=list(row['labels']), correct=[True] * 100, correct_count=100, quality_score=1.0)
        result = self.rejection('prediction_and_score_consistent_but_not_frozen', self.base, c)
        self.assertEqual(result['reason'], 'record_changed_after_collector_freeze')

    def test_label_tampering(self):
        c = deepcopy(self.candidate)
        c['record']['labels'][0] = (c['record']['labels'][0] + 1) % 1000
        result = self.rejection('label_tampered', self.base, c)
        self.assertEqual(result['reason'], 'sample_label_binding_mismatch')

    def test_wrong_sample(self):
        c = deepcopy(self.candidate)
        c['record']['sample_ids'][0] = 'different-image.jpeg'
        self.rejection('wrong_sample', self.base, c)

    def test_duplicate_id(self):
        c = deepcopy(self.candidate)
        c['record']['sample_ids'][1] = c['record']['sample_ids'][0]
        result = self.rejection('duplicate_sample_id', self.base, c)
        self.assertEqual(result['reason'], 'sample_ids_duplicate')

    def test_order_preserving_pairs_but_permuting_batch_rejected(self):
        c = deepcopy(self.candidate)
        for key in ['sample_ids', 'labels', 'predictions', 'correct']:
            c['record'][key] = list(reversed(c['record'][key]))
        result = self.rejection('consistent_joint_sample_permutation', self.base, c)
        self.assertEqual(result['reason'], 'sample_identity_or_order_mismatch')

    def test_nan_and_infinity_fields(self):
        for key in ['quality_score', 'correct_count', 'sample_count', 'retention_ratio', 'mean_ms']:
            for val in [float('nan'), float('inf')]:
                with self.subTest(field=key, value=str(val)):
                    c = deepcopy(self.candidate)
                    c['record'][key] = val
                    self.rejection(key + '_' + str(val), self.base, c)
        for key in ['labels', 'predictions', 'latency_samples_ms']:
            with self.subTest(vector=key):
                c = deepcopy(self.candidate)
                c['record'][key][0] = float('nan')
                self.rejection(key + '_nan', self.base, c)

    def test_missing_fields(self):
        for key in ['sample_ids', 'labels', 'predictions', 'correct', 'correct_count', 'sample_count',
                    'quality_score', 'quality_evaluated', 'artifact_id', 'equivalence', 'status', 'retention_ratio']:
            with self.subTest(field=key):
                c = deepcopy(self.candidate)
                del c['record'][key]
                self.rejection('missing_' + key, self.base, c)

    def test_input_model_and_run_bindings(self):
        replacements = {'task_id': 'other_task', 'device_id': 'other_device', 'model_id': 'resnet50',
                        'run_id': 'other_run', 'run_index': 5, 'dataset_id': 'other_dataset',
                        'weights_state_sha256': '0' * 64, 'input_tensor_sha256': '0' * 64,
                        'dataset_manifest_sha256': '0' * 64, 'input_shape': [4, 3, 224, 224],
                        'phase': 'train', 'quality_partition': 'calibration',
                        'model_configuration': {'different_runtime_configuration': True}}
        for key, value in replacements.items():
            with self.subTest(identity=key):
                c = deepcopy(self.candidate)
                c['identity'][key] = value
                self.rejection('identity_' + key, self.base, c)

    def test_missing_identity_field(self):
        c = deepcopy(self.candidate)
        del c['identity']['input_tensor_sha256']
        self.rejection('missing_input_identity', self.base, c)

    def test_artifact_and_action_binding(self):
        for key, value in [('artifact_id', 'different_artifact'), ('mask', 5), ('action_ids', ['stem'])]:
            with self.subTest(field=key):
                c = deepcopy(self.candidate)
                c['record'][key] = value
                self.rejection('candidate_binding_' + key, self.base, c)

    def test_source_hash_binding(self):
        c = deepcopy(self.candidate)
        c['source_sha256'] = '0' * 64
        self.rejection('packet_source_hash', self.base, c)
        result = gate.gate_verified_raw(self.path, '0' * 64, 6)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['fallback'], 'baseline')
        self.assertTrue(result['requires_verified_baseline_reload'])
        OBSERVED.append({'case': 'file_source_hash_mismatch', **result})

    def test_invalid_baseline_requires_reload(self):
        b = deepcopy(self.base)
        b['record']['quality_score'] = .99
        result = self.rejection('baseline_corrupted', b, self.candidate)
        self.assertTrue(result['requires_verified_baseline_reload'])
        self.assertIsNone(result['accepted_latency_ms'])

    def test_measurement_failure(self):
        for key, value in [('status', 'failed'), ('equivalence', {'passed': False})]:
            c = deepcopy(self.candidate)
            c['record'][key] = value
            self.rejection('measurement_' + key, self.base, c)

    def test_unsupported_action_and_budget(self):
        binding = replace(self.binding, hardware_supported=(True, False, True, True))
        result = self.rejection('unsupported_action', self.base, self.candidate, binding)
        self.assertEqual(result['reason'], 'hardware_unsupported')
        binding = gate.make_binding(self.verified_run, 7, max_actions=2)
        result = self.rejection('budget_exceeded', self.base, gate.packet_from_run(self.verified_run, 7), binding)
        self.assertEqual(result['reason'], 'action_budget_exceeded')

    def test_quality_failure_with_consistent_ledger(self):
        # In-memory fault injection: emulate a collector-frozen bad candidate.
        # This is deliberately not labelled as a new physical measurement.
        c = deepcopy(self.candidate)
        row = c['record']
        row.update(predictions=[(y + 1) % 1000 for y in row['labels']], correct=[False] * 100,
                   correct_count=0, quality_score=0.0, retention_ratio=0.0, quality_threshold_passed=False)
        digest = hashlib.sha256(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        binding = replace(self.binding, candidate_record_sha256=digest)
        result = self.rejection('injected_consistent_quality_failure', self.base, c, binding)
        self.assertEqual(result['reason'], 'quality_below_threshold')
        self.assertEqual(result['candidate_correct'], 0)

    def test_actual_latency_regression(self):
        path = ROOT / 'v2-实证增强/evidence/fullmodel_raw/npu-resnet18-b1-r5.json'
        frozen = json.loads((HERE / 'analysis-freeze.json').read_text())
        h = next(x['sha256'] for x in frozen['inputs'] if x['path'] == str(path.relative_to(ROOT)))
        result = gate.gate_verified_raw(path, h, 6)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'latency_regression')
        self.assertEqual(result['fallback'], 'baseline')
        self.assertLess(result['raw_benefit_percent'], 0)
        OBSERVED.append({'case': 'actual_run5_latency_regression', **result})

    def test_malformed_packet_and_binding(self):
        for packet in [None, {}, {'identity': {}}]:
            self.rejection('malformed_packet_' + str(packet), self.base, packet)
        result = gate.strict_final_gate(self.base, self.candidate, None)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['fallback'], 'baseline')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'negative_cases_executed': len(OBSERVED), 'negative_case_outcomes': OBSERVED,
              'module_sha256': sha(HERE / 'strict_quality_gate.py'), 'test_sha256': sha(__file__),
              'scope': 'Detached-copy software corruption tests plus replay of an existing adverse physical block; no new hardware.',
              'all_rejections_conservatively_fallback_to_baseline': all(not x['accepted'] and x['fallback'] == 'baseline' for x in OBSERVED),
              'public_runtime_modified': False, 'deployed': False}
    (HERE / 'strict-quality-gate-tests.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    sys.exit(0 if result.wasSuccessful() else 1)
