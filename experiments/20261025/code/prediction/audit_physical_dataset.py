"""Independent raw/sidecar integrity audit; no profiling or training."""
import argparse
import collections
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profiles', nargs='+', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('do not overwrite audit output')
    datasets = []
    reference = None
    ids = set()
    for p in args.profiles:
        d = json.loads(p.read_text())
        if d['schema_version'] != 'het-physical-profile-v2.1' or not d['provenance'].get('completed_utc'):
            raise ValueError('profile schema/status invalid')
        sid = d['hardware']['device_id']
        if sid in ids:
            raise ValueError('duplicate runtime stack')
        ids.add(sid)
        sidepath = p.with_suffix('.metadata.json')
        side = json.loads(sidepath.read_text())
        if side['raw_bundle_sha256'] != sha(p) or side['timings_changed_or_added'] or side['training_scores_seen']:
            raise ValueError('sidecar provenance invalid')
        graph_identity = {}
        family = collections.Counter()
        workload = collections.Counter()
        samples = 0
        max_mean_disagreement = 0.
        operator_ids = set()
        status = collections.Counter()
        for graph in d['graphs']:
            nodes = {n['node_id']: n for n in graph['nodes']}
            if len(nodes) != len(graph['nodes']):
                raise ValueError('duplicate node')
            if not all(e['source'] in nodes and e['target'] in nodes and e['relation'] == 'dependency' for e in graph['edges']):
                raise ValueError('invalid real dependency edge')
            sc = side['graphs'][graph['graph_id']]
            if not sc['input_weights_hash_verified'] or sc['input_weights_sha256'] != graph['input_weights_sha256'] or set(sc['nodes']) != set(nodes):
                raise ValueError('unverified or incomplete structural enrichment')
            # Backend storage-format codes are intentionally backend-specific.
            # Compare portable logical shape/layout fields and retain backend
            # formats in the raw profile rather than demanding identical codes.
            portable_inputs = [{k: x[k] for k in ['shape', 'dtype', 'stride', 'numel', 'element_size']} for x in graph['input_shapes']]
            graph_identity[graph['graph_id']] = (graph['input_weights_sha256'], graph['dtype'], portable_inputs, sorted(nodes))
            workload[graph['workload']] += 1
            for r in graph['records']:
                status[r['status']] += 1
                if r['status'] != 'measured':
                    continue
                if r['op_instance_id'] in operator_ids:
                    raise ValueError('duplicate operator query')
                operator_ids.add(r['op_instance_id'])
                if r['record_id'] != sid + '/' + r['op_instance_id'] or r['op_instance_id'] not in nodes:
                    raise ValueError('record identity invalid')
                ts = r['latency_samples_ms']
                if len(ts) != 20 or r['warmup'] != 10 or r['inner_repeats'] != 8:
                    raise ValueError('timing budget differs from frozen protocol')
                if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) and t > 0 for t in ts):
                    raise ValueError('nonpositive/nonfinite physical observation')
                delta = abs(statistics.mean(ts) - r['latency_mean_ms'])
                max_mean_disagreement = max(max_mean_disagreement, delta)
                if delta > 1e-9:
                    raise ValueError('reported label differs from raw mean')
                samples += len(ts)
                family[nodes[r['op_instance_id']]['family']] += 1
        identity = {'graphs': graph_identity, 'operator_ids': sorted(operator_ids)}
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError('cross-runtime input/weights/shapes/FX/query support differs')
        datasets.append({'path': str(p.resolve()), 'raw_sha256': sha(p),
                         'metadata_sha256': sha(sidepath), 'runtime_stack_id': sid,
                         'backend': d['hardware']['backend'], 'hardware_name': d['hardware']['hardware_name'],
                         'torch': d['provenance']['torch'], 'python': d['provenance']['python'],
                         'collector_code_sha256': d['provenance']['code_sha256'],
                         'adapter_provenance': d['hardware'].get('adapter_provenance'),
                         'graphs': len(d['graphs']), 'measured_operator_queries': len(operator_ids),
                         'raw_timing_blocks': samples, 'warmup_calls_per_query': 10,
                         'calls_per_timed_block': 8, 'status_counts': dict(status),
                         'graph_failure_count': len(d['failures']),
                         'per_family_n': dict(family), 'per_workload_graphs': dict(workload),
                         'raw_mean_max_absolute_disagreement_ms': max_mean_disagreement,
                         'all_input_weight_shape_fx_and_query_ids_match_reference': True})
    out = {'kind': 'independent_physical_profile_integrity_audit',
           'created_utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': sha(__file__),
           'runtime_stacks': len(datasets), 'graphs': sum(d['graphs'] for d in datasets),
           'measured_operator_queries': sum(d['measured_operator_queries'] for d in datasets),
           'raw_timing_blocks': sum(d['raw_timing_blocks'] for d in datasets),
           'datasets': datasets, 'all_checks_passed': True,
           'scope': ['This audit recomputes label means and verifies identity/provenance; it does not certify task accuracy or historical v21 figures.',
                     'Five runtime stacks are not five independent hardware architectures. M5 CPU/MPS share a host, and cloud CPU/A100 MIG share a host.',
                     'NPU uses torch/torch_npu2.6.0 and other stacks use torch2.9.1; results compare measured runtime stacks, not isolated chip effects.',
                     '20 timing blocks of eight calls are repeated timing observations, not20 independent model samples.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ('runtime_stacks', 'graphs', 'measured_operator_queries', 'raw_timing_blocks', 'all_checks_passed')}))


if __name__ == '__main__':
    main()
