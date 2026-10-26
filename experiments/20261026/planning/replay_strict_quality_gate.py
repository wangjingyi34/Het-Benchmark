"""Replay reusable strict gate and its upgrade adapter on existing raw ledgers."""
from copy import deepcopy
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('HET_EVIDENCE_BASE', str(HERE.parent.parent))).resolve()
METHODS = ROOT / 'v2-实证增强/methods'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / '.experiment-deps'))
sys.path.insert(0, str(METHODS))
import strict_quality_gate as gate
from upgrade_final_gate_example import upgraded_final_gate
from measured_planner import MeasuredTable


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-9), (a, b)


def main():
    freeze = {r['path']: r['sha256'] for r in read(HERE / 'analysis-freeze.json')['inputs']}
    rows, raw_manifest, cache = [], {}, {}
    binding_example = None
    for tp in sorted((METHODS / 'planner-matrix-inputs/tasks').glob('*.json')):
        assert sha(tp) == freeze[str(tp.relative_to(ROOT))]
        task = read(tp)
        table = MeasuredTable(task)
        by_id = {b['run_id']: b for b in task['eval']}
        for block in table.splits['eval']:
            source = by_id[block['run_id']]
            path = ROOT / 'v2-实证增强/evidence/fullmodel_raw' / Path(source['artifact_path']).name
            expected = freeze[str(path.relative_to(ROOT))]
            assert expected == source['artifact_sha256']
            run = gate.load_verified_run(path, expected)
            raw_manifest[path.name] = {'path': str(path.relative_to(ROOT)), 'sha256': expected, 'run_id': run.raw['run_id']}
            for mask in [m for m in range(16) if bin(m).count('1') <= 2]:
                binding = gate.make_binding(run, mask)
                strict = gate.strict_final_gate(gate.packet_from_run(run, 0), gate.packet_from_run(run, mask), binding)
                upgraded = upgraded_final_gate(table, block, mask, raw_path=path, expected_source_sha256=expected)
                original = table.final_gate(block, mask)
                assert strict == upgraded
                for key in ['accepted', 'reason', 'requested_mask', 'accepted_mask', 'run_id', 'rollback_to_baseline']:
                    assert strict[key] == original[key], (path.name, mask, key)
                for key in ['raw_latency_ms', 'raw_benefit_percent', 'normalized_quality', 'accepted_latency_ms', 'accepted_benefit_percent', 'baseline_quality']:
                    close(strict[key], original[key])
                assert strict['quality_total'] == 100
                rows.append({'context': tp.stem, 'raw_path': str(path.relative_to(ROOT)),
                             'raw_sha256': expected, 'raw_run_index': run.raw['run_index'], **strict})
                cache[(tp.stem, block['run_id'], mask)] = strict
                if tp.stem == 'npu-resnet18_b1_imagenet_val10class_v2' and run.raw['run_index'] == 4 and mask == 6:
                    binding_example = {'raw_path': str(path.relative_to(ROOT)), 'pinned_source_sha256': expected,
                                       'binding': binding.to_dict(), 'strict_decision': strict}
    assert len(rows) == 440 and len(raw_manifest) == 40
    scores = list(csv.DictReader((HERE / 'planner-seedwise-process.csv').open()))
    assert len(scores) == 800
    for score in scores:
        decision = cache[(score['context'], score['run_id'], int(score['mask']))]
        close(float(score['raw_gain_percent']), decision['raw_benefit_percent'])
        close(float(score['accepted_gain_percent']), decision['accepted_benefit_percent'])
        assert score['accepted'] == str(decision['accepted'])
    # Integration faults exercise the actual upgrade call, not a parallel stub.
    tp = METHODS / 'planner-matrix-inputs/tasks/npu-resnet18_b1_imagenet_val10class_v2.json'
    table = MeasuredTable(read(tp))
    block = table.splits['eval'][0]
    path = ROOT / binding_example['raw_path']
    expected = binding_example['pinned_source_sha256']
    corrupt = deepcopy(block)
    corrupt['records'][6]['predictions'] = [(y + 1) % 1000 for y in corrupt['records'][6]['labels']]
    assert table.final_gate(corrupt, 6)['accepted']  # Reproduce the original trust boundary.
    integration = []
    tests = [('mutated_table_prediction_ledger', table, corrupt, expected)]
    wrong_run = deepcopy(block)
    wrong_run['run_id'] = 'different_run'
    tests.append(('wrong_block_run_id', table, wrong_run, expected))
    wrong_order = deepcopy(block)
    wrong_order['quality_sample_ids'] = list(reversed(wrong_order['quality_sample_ids']))
    tests.append(('wrong_block_sample_order', table, wrong_order, expected))
    wrong_model_table = deepcopy(table)
    wrong_model_table.payload['model_id'] = 'resnet50'
    tests.append(('wrong_table_model_id', wrong_model_table, block, expected))
    tests.append(('wrong_pinned_hash', table, block, '0' * 64))
    for name, t, b, digest in tests:
        result = upgraded_final_gate(t, b, 6, raw_path=path, expected_source_sha256=digest)
        assert not result['accepted'] and result['accepted_mask'] == 0
        assert result['fallback'] == 'baseline' and result['rollback_to_baseline']
        integration.append({'case': name, 'reason': result['reason'], 'fallback': result['fallback'], 'accepted': result['accepted']})
    for source in raw_manifest.values():
        assert sha(ROOT / source['path']) == source['sha256']
    report = {'status': 'passed', 'existing_eval_processes': 40,
              'all_at_most_two_candidate_decisions_replayed': 440,
              'upgrade_adapter_decisions_matched': 440,
              'all_160_fit_seed_process_scores_matched': 800,
              'raw_quality_ledgers_recomputed': 880,
              'unique_eval_images_per_context': 100,
              'accepted_decisions': sum(r['accepted'] for r in rows),
              'rejected_decisions': sum(not r['accepted'] for r in rows),
              'raw_sources_unchanged_after_replay': True,
              'raw_source_manifest': list(raw_manifest.values()),
              'integration_fault_tests_passed': len(integration), 'integration_fault_results': integration,
              'module_sha256': sha(HERE / 'strict_quality_gate.py'),
              'adapter_sha256': sha(HERE / 'upgrade_final_gate_example.py'), 'replay_source_sha256': sha(__file__),
              'scope': 'Software upgrade replay on the same existing physical tables; no new hardware, retraining, deployment, or historical rewrite.',
              'deployment_status': 'not_deployed', 'original_runtime_changed': False,
              'baseline_fallback_is_a_decision_not_proof_of_execution': True}
    (HERE / 'strict-quality-gate-replay.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (HERE / 'strict-quality-gate-binding-example.json').write_text(json.dumps(binding_example, ensure_ascii=False, indent=2) + '\n')
    with (HERE / 'strict-quality-gate-decisions.csv').open('w', newline='') as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in report.items() if k not in ['raw_source_manifest', 'integration_fault_results']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
