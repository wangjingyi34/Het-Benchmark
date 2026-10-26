"""QG-P1-001/QG-P2-002 regression tests; no original data are modified."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json
import os
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('HET_EVIDENCE_BASE', str(HERE.parent.parent))).resolve()
sys.path[:0] = [str(HERE), str(ROOT / '.experiment-deps'), str(ROOT / 'v2-实证增强/methods')]
from measured_planner import MeasuredTable
from upgrade_final_gate_example import upgraded_final_gate
import strict_quality_gate as gate

OUTCOMES = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class QualityContractAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((ROOT / 'v2-实证增强/methods/planner-matrix-inputs/tasks/npu-resnet18_b1_imagenet_val10class_v2.json').read_text())
        cls.raw_path = ROOT / 'v2-实证增强/evidence/fullmodel_raw/npu-resnet18-b1-r4.json'
        frozen = json.loads((HERE / 'analysis-freeze.json').read_text())
        cls.raw_hash = next(r['sha256'] for r in frozen['inputs'] if r['path'] == str(cls.raw_path.relative_to(ROOT)))
        cls.raw = json.loads(cls.raw_path.read_text())

    def setUp(self):
        self.table = MeasuredTable(deepcopy(self.payload))
        self.block = self.table.splits['eval'][0]

    def check_reject(self, name, table, block):
        result = upgraded_final_gate(table, block, 6, raw_path=self.raw_path, expected_source_sha256=self.raw_hash)
        self.assertFalse(result['accepted'], name)
        self.assertEqual(result['accepted_mask'], 0, name)
        self.assertEqual(result['fallback'], 'baseline', name)
        self.assertTrue(result['requires_verified_baseline_reload'], name)
        self.assertIsNone(result['accepted_latency_ms'], name)
        self.assertFalse(result['rollback_executed'], name)
        self.assertFalse(result['deployment_or_rollback_execution_proved'], name)
        self.assertFalse(result['quality_contract_evaluated'], name)
        self.assertEqual(result['quality_mode'], 'not_evaluated', name)
        OUTCOMES.append({'case': name, 'reason': result['reason'], 'accepted': result['accepted'],
                         'fallback': result['fallback'], 'requires_verified_baseline_reload': True,
                         'rollback_executed': False})
        return result

    def test_valid_declared_retention_contract(self):
        result = upgraded_final_gate(self.table, self.block, 6, raw_path=self.raw_path, expected_source_sha256=self.raw_hash)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['quality_mode'], 'retention_ratio')
        self.assertEqual(result['quality_metric'], 'ImageNet1000_top1_accuracy')
        self.assertEqual(result['quality_dataset_id'], self.raw['dataset_id'])
        self.assertEqual(result['threshold'], self.raw['protocol']['quality_threshold'])

    def test_exact_review_absolute_counterexample(self):
        changed = deepcopy(self.payload)
        changed['quality_definition']['mode'] = 'absolute'
        table = MeasuredTable(changed)
        block = table.splits['eval'][0]
        old = table.final_gate(block, 6)
        self.assertFalse(old['accepted'])
        self.assertEqual(old['normalized_quality'], .62)
        result = self.check_reject('review_absolute_mode_accuracy062_threshold099', table, block)
        self.assertEqual(result['reason'], 'table_quality_mode_unsupported')

    def test_unknown_or_inconsistent_modes(self):
        for name, mode, declared in [('unknown_mode', 'unknown', 'unknown'),
                                     ('mode_attribute_changed', 'absolute', 'retention_ratio'),
                                     ('mode_declaration_changed', 'retention_ratio', 'absolute')]:
            with self.subTest(case=name):
                t = deepcopy(self.table)
                t.mode = mode
                t.payload['quality_definition']['mode'] = declared
                self.check_reject(name, t, self.block)

    def test_exact_review_dataset_and_metric_counterexamples(self):
        for field, value in [('dataset_id', 'other-dataset'), ('metric', 'different_metric')]:
            with self.subTest(field=field):
                changed = deepcopy(self.payload)
                changed['quality_definition'][field] = value
                table = MeasuredTable(changed)
                self.check_reject('review_' + field + '_mismatch', table, table.splits['eval'][0])

    def test_thresholds_bound_to_declaration_and_frozen_raw_protocol(self):
        for name, attr, declared in [('attribute_threshold_changed', .98, .99),
                                    ('declared_threshold_changed', .99, .98),
                                    ('both_thresholds_differ_from_raw', .98, .98),
                                    ('threshold_nearby_but_not_exact', .99000000000001, .99000000000001)]:
            with self.subTest(case=name):
                t = deepcopy(self.table)
                t.threshold = attr
                t.payload['quality_definition']['threshold'] = declared
                self.check_reject(name, t, self.block)
        for value in [None, '0.99', True, float('nan'), float('inf'), -1, 0, 2, 10 ** 1000]:
            with self.subTest(value=str(value)[:20]):
                t = deepcopy(self.table)
                t.threshold = value
                t.payload['quality_definition']['threshold'] = value
                self.check_reject('invalid_threshold_' + str(value)[:20], t, self.block)

    def test_exact_review_none_and_missing_attribute_counterexamples(self):
        self.check_reject('review_table_none', None, self.block)
        self.check_reject('review_block_none', self.table, None)
        for field in ['threshold', 'mode', 'payload', 'supported']:
            t = deepcopy(self.table)
            delattr(t, field)
            self.check_reject('missing_table_attribute_' + field, t, self.block)

    def test_empty_mappings_and_malformed_object_boundaries(self):
        for obj in [{}, [], SimpleNamespace()]:
            self.check_reject('malformed_table_' + type(obj).__name__, obj, self.block)
        for block in [{}, [], '', {'records': {}}]:
            self.check_reject('malformed_block_' + type(block).__name__ + str(len(block)), self.table, block)
        for payload in [None, {}, []]:
            t = deepcopy(self.table)
            t.payload = payload
            self.check_reject('malformed_payload_' + str(payload), t, self.block)
        t = deepcopy(self.table)
        t.supported = None
        self.check_reject('support_none', t, self.block)

    def test_missing_quality_definition_and_block_fields(self):
        for definition in [None, {}, [], {'mode': 'retention_ratio'}]:
            t = deepcopy(self.table)
            t.payload['quality_definition'] = definition
            self.check_reject('incomplete_quality_definition_' + str(definition), t, self.block)
        for key in ['mode', 'threshold', 'metric', 'dataset_id']:
            t = deepcopy(self.table)
            del t.payload['quality_definition'][key]
            self.check_reject('missing_quality_definition_' + key, t, self.block)
        for key in ['run_id', 'quality_partition', 'provenance', 'quality_sample_ids', 'records']:
            b = deepcopy(self.block)
            del b[key]
            self.check_reject('missing_block_' + key, self.table, b)
        for key in ['quality_sample_ids', 'records']:
            b = deepcopy(self.block)
            b[key] = None
            self.check_reject('invalid_block_' + key, self.table, b)

    def test_explicit_metric_mapping_not_textual_guess(self):
        t = deepcopy(self.table)
        t.payload['quality_definition']['metric'] = self.raw['quality_metric']
        result = self.check_reject('raw_description_is_not_declared_table_metric_name', t, self.block)
        self.assertEqual(result['reason'], 'table_quality_metric_unsupported')

    def test_core_rejects_unsupported_pinned_raw_contracts(self):
        for name, mutate in [
            ('raw_absolute_mode', lambda d: d['protocol'].update(quality_mode='absolute')),
            ('raw_unknown_mode', lambda d: d['protocol'].update(quality_mode='unknown')),
            ('raw_unknown_metric', lambda d: d.update(quality_metric='other_metric')),
            ('raw_threshold_differs_from_requested', lambda d: d['protocol'].update(quality_threshold=.98)),
            ('raw_missing_mode', lambda d: d['protocol'].pop('quality_mode')),
            ('raw_missing_threshold', lambda d: d['protocol'].pop('quality_threshold')),
        ]:
            with self.subTest(case=name):
                # Mock file bytes are software fixtures, not new physical results.
                d = deepcopy(self.raw)
                mutate(d)
                data = json.dumps(d, ensure_ascii=False).encode()
                digest = hashlib.sha256(data).hexdigest()
                with patch.object(Path, 'read_bytes', return_value=data):
                    result = gate.gate_verified_raw(self.raw_path, digest, 6)
                self.assertFalse(result['accepted'])
                self.assertTrue(result['requires_verified_baseline_reload'])
                self.assertEqual(result['fallback'], 'baseline')
                self.assertFalse(result['rollback_executed'])
                OUTCOMES.append({'case': name, 'reason': result['reason'], 'accepted': False,
                                 'fallback': 'baseline', 'requires_verified_baseline_reload': True,
                                 'rollback_executed': False})


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    review = HERE / 'prior-versions/strict-gate-v1-pre-quality-review/quality-independent-review.json'
    report = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'negative_cases_executed': len(OUTCOMES), 'negative_case_outcomes': OUTCOMES,
              'addresses_findings': ['QG-P1-001', 'QG-P2-002'],
              'independent_review_sha256': sha(review) if review.exists() else None, 'private_review_not_distributed': not review.exists(), 'module_sha256': sha(HERE / 'strict_quality_gate.py'),
              'adapter_sha256': sha(HERE / 'upgrade_final_gate_example.py'), 'test_sha256': sha(__file__),
              'all_rejections_require_verified_baseline_reload': all(r['requires_verified_baseline_reload'] for r in OUTCOMES),
              'no_rollback_execution_claimed': all(not r['rollback_executed'] for r in OUTCOMES),
              'scope': 'Software regression tests for the independent review findings; existing raw data and old reviewed version are preserved.',
              'deployed': False, 'independent_rereview_status': 'pending'}
    (HERE / 'quality-contract-regression-tests.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    sys.exit(0 if result.wasSuccessful() else 1)
