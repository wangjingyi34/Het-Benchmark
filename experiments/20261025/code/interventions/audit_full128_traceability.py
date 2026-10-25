"""Reproduce frozen-source, real-DAG and train-only assembly traceability checks.

Run audit_full_model_runs.py separately for per-image and timing arithmetic.
Only JSON/hash processing is performed; no physical measurements are launched.
"""
import argparse
import collections
import hashlib
import json
import re
import statistics
from pathlib import Path

FROZEN = '04c84a5283a3c5c1c286cf3b7dc6300c60d90fbad5275437a1ec2f15907cacdb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', type=Path, required=True)
    ap.add_argument('--task-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert sha(Path(__file__).with_name('full_model_coalitions.py')) == FROZEN
    raw, topology, dag_counts = {}, {}, {}
    devices = collections.Counter()
    weights = collections.defaultdict(set)
    profile_status = collections.Counter()
    for p in args.raw_dir.glob('*.json'):
        d = json.loads(p.read_text())
        if d.get('schema_version') != 'het.fullmodel.run.v2':
            continue
        assert d['protocol']['source_sha256'] == FROZEN
        assert d['protocol']['warmup'] == 10 and d['protocol']['repeats'] == 30
        assert d['protocol']['quality_threshold'] == .99
        assert d['protocol']['quality_mode'] == 'retention_ratio'
        if d['device_id'].startswith('mps_'):
            assert d['protocol']['mps_fallback_env'] != '1'
        graph = d['graph']
        nodes = {n['id']: n for n in graph['nodes']}
        assert len(nodes) == len(graph['nodes'])
        edges = [(e['source'], e['target']) for e in graph['edges']]
        assert len(edges) == len(set(edges))
        position = {n['id']: i for i, n in enumerate(graph['nodes'])}
        assert all(a in nodes and b in nodes and position[a] < position[b] for a, b in edges)
        indegree = collections.Counter(b for a, b in edges)
        residuals = [n for n in graph['nodes'] if n['operation'] == 'call_function'
                     and n['target'] == '<built-in function add>']
        expected = {'resnet18': (71, 78, 8), 'resnet50': (177, 192, 16)}[d['model_id']]
        assert (len(nodes), len(edges), len(residuals)) == expected
        assert all(indegree[n['id']] == 2 for n in residuals)
        assert sum(indegree[n['id']] > 1 for n in graph['nodes']) == len(residuals)
        normalized = {
            'nodes': [(n['id'], n['operation'], re.sub(r'0x[0-9a-fA-F]+', '0xADDR', n['target']),
                       n['shape'], n['dtype']) for n in graph['nodes']], 'edges': edges,
        }
        key = (d['model_id'], tuple(d['input_shape']))
        signature = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
        if key in topology:
            assert topology[key] == signature
        topology[key] = signature
        dag_counts[d['model_id']] = {'nodes': expected[0], 'edges': expected[1],
                                     'residual_add_nodes': expected[2], 'all_residuals_have_two_inputs': True}
        if d['phase'] == 'train':
            assert {r['node_id'] for r in d['node_profiles']} == set(nodes)
            for row in d['node_profiles']:
                assert row['status'] in ['ok', 'structural_zero_cost']
                profile_status[row['status']] += 1
                if row['status'] == 'ok':
                    assert len(row['latency_samples_ms']) == 30
                    assert all(v > 0 for v in row['latency_samples_ms'])
        else:
            assert d['node_profiles'] == []
        for a in d['action_profiles']:
            assert a['numerical_equivalence'] and a['hardware_supported']
            assert a['input_shape'][0] == d['input_shape'][0]
            assert len(a['baseline']['latency_samples_ms']) == len(a['candidate']['latency_samples_ms']) == 30
            assert abs(a['local_benefit_ms'] - a['baseline']['mean_ms'] + a['candidate']['mean_ms']) < 1e-12
        assert d['run_id'] not in raw
        raw[d['run_id']] = (p, d)
        devices[d['device_id']] += 1
        weights[d['model_id']].add(d['weights_state_sha256'])
    assert len(raw) == 128 and len(devices) == 4 and set(devices.values()) == {32}
    assert all(len(v) == 1 for v in weights.values())
    index_path = args.task_dir / 'index.json'
    index = json.loads(index_path.read_text())
    assert len(index) == 16
    seen, trace = set(), []
    for entry in index:
        # Basename resolution is explicit and portable; file hash must still agree.
        p = args.task_dir / Path(entry['file']).name
        assert sha(p) == entry['sha256']
        task = json.loads(p.read_text())
        assert len(task['train']) == entry['train_runs'] == 3
        assert len(task['eval']) == entry['eval_runs'] == 5
        assert len(task['source_artifacts']) == 8
        tr = {b['run_id'] for b in task['train']}
        ev = {b['run_id'] for b in task['eval']}
        assert not tr & ev
        references = {Path(a['path']).name: a['sha256'] for a in task['source_artifacts']}
        for split in ['train', 'eval']:
            for block in task[split]:
                original_path, d = raw[block['run_id']]
                assert block['run_id'] not in seen
                seen.add(block['run_id'])
                assert block['coalitions'] == d['coalitions']
                assert block['baseline_quality'] == d['baseline_quality']
                assert block['artifact_sha256'] == references[original_path.name] == sha(original_path)
                assert block['input_tensor_sha256'] == d['input_tensor_sha256']
                assert block['weights_state_sha256'] == d['weights_state_sha256']
                assert d['phase'] == split
                assert block['quality_sample_ids'] == next(r for r in d['coalitions'] if r['mask'] == 0)['sample_ids']
        for md in task['action_metadata']:
            assert set(md['profile_run_ids']) == tr
            profiles = [next(a for a in raw[r][1]['action_profiles'] if a['id'] == md['id']) for r in tr]
            assert abs(md['operator_latency_ms'] - statistics.mean(a['baseline']['mean_ms'] for a in profiles)) < 1e-12
            assert abs(md['candidate_latency_ms'] - statistics.mean(a['candidate']['mean_ms'] for a in profiles)) < 1e-12
            assert abs(md['local_benefit_ms'] - statistics.mean(a['local_benefit_ms'] for a in profiles)) < 1e-12
            assert md['source_target_delta_ms'] is None
        assert set(task['copa_game']['profile_run_ids']) == tr
        assert task['copa_game']['runtime_overhead_ms'] == 0
        trace.append({'task': task['task_id'], 'device': task['device_id'],
                      'artifact_sha256': sha(p), 'train': sorted(tr), 'eval': sorted(ev),
                      'exact_original_blocks_and_train_only_metadata': True})
    assert seen == set(raw)
    out = {'kind': 'reproducible_full128_DAG_and_assembly_audit', 'all_checks_passed': True,
           'blocks': 128, 'contexts': 16, 'blocks_per_backend': dict(devices), 'source_sha256': FROZEN,
           'DAGs': dag_counts, 'same_topology_per_model_shape_across_devices': True,
           'calibration_node_profile_statuses': dict(profile_status),
           'identical_weights_across_backends': {m: sorted(s) for m, s in weights.items()},
           'all_128_blocks_covered_exactly_once': True, 'index_sha256': sha(index_path),
           'tasks': trace, 'audit_source_sha256': sha(Path(__file__)),
           'limitations': 'Recorded identity and dependency checks; hashes do not reconstruct logits. Physical worker receipts remain separate. Dependency DAGs do not assert concurrent execution or performance gain.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != 'tasks'}, indent=2))


if __name__ == '__main__':
    main()
