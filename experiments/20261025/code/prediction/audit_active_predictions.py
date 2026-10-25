"""Recompute frozen decisions from local checkpoints and unlabelled FX inputs.

No fitting or physical timing. Explicit checkpoint relocation leaves the exact
deployed policy unchanged. Physical records are removed before graph building.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from active_fallback import load_fit
from evaluate_profiles import build_graph, collate, predict


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True, type=Path)
    ap.add_argument('--decisions', required=True, type=Path)
    ap.add_argument('--profile', required=True, type=Path)
    ap.add_argument('--metadata', required=True, type=Path)
    ap.add_argument('--checkpoint-dir', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():raise ValueError('do not overwrite prediction audit')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    policy = json.loads(args.policy.read_text());frozen = json.loads(args.decisions.read_text())
    raw = json.loads(args.profile.read_text());side = json.loads(args.metadata.read_text())
    if frozen['policy_sha256'] != sha(args.policy):raise ValueError('deployment policy differs')
    if side['raw_bundle_sha256'] != sha(args.profile):raise ValueError('metadata does not bind the exact physical profile')
    if frozen['device_id'] != raw['hardware']['device_id']:raise ValueError('runtime stack differs')
    expected = {r['record_id']: r for r in frozen['decisions']};got = {}
    fits = None
    for graph in raw['graphs']:
        if graph['model_id'] not in policy['test_model_ids'] or graph['shape_id'] not in policy['shape_ids']:continue
        sc = side['graphs'][graph['graph_id']]
        if not sc['input_weights_hash_verified'] or sc['input_weights_sha256'] != graph['input_weights_sha256']:
            raise ValueError('structural enrichment hash differs')
        # Operator target records are deliberately absent during inference.
        unlabelled = {**graph, 'records': [], 'nodes': [{**n, 'operator_attributes': sc['nodes'][n['node_id']]} for n in graph['nodes']]}
        batch = collate([build_graph(unlabelled, raw['hardware'], policy['feature_mode'], 20, unlabelled=True)], 'cpu')
        if not torch.isnan(batch['y']).all():raise ValueError('latency target entered inference')
        if fits is None:
            fits = [load_fit(args.checkpoint_dir / Path(c['path']).name, c['sha256'], batch['x'].shape[1], policy) for c in policy['checkpoints']]
        values = np.stack([predict(f, batch) for f in fits])
        mu = values.mean(0);sd = values.std(0, ddof=1)
        for meta, p, s in zip(batch['meta'], mu, sd):
            key = raw['hardware']['device_id'] + '/' + meta['node_id']
            u = float(s / max(float(p), 1e-6));threshold = math.inf if policy['threshold_infinite'] else policy['threshold']
            got[key] = {'prediction_ms': float(p), 'uncertainty': u, 'trigger_measurement': u > threshold}
    if set(got) != set(expected):raise ValueError('query support differs')
    pdelta = [abs(got[k]['prediction_ms'] - expected[k]['prediction_ms']) for k in got]
    udelta = [abs(got[k]['uncertainty'] - expected[k]['uncertainty']) for k in got]
    mismatches = [k for k in got if got[k]['trigger_measurement'] != expected[k]['trigger_measurement']]
    close = all(math.isclose(got[k]['prediction_ms'], expected[k]['prediction_ms'], rel_tol=2e-5, abs_tol=1e-7)
                and math.isclose(got[k]['uncertainty'], expected[k]['uncertainty'], rel_tol=2e-5, abs_tol=1e-6) for k in got)
    out = {'kind': 'checkpoint_recomputed_unlabelled_active_decision_audit', 'created_utc': datetime.now(timezone.utc).isoformat(),
           'code_sha256': sha(__file__), 'policy_sha256': sha(args.policy), 'decisions_sha256': sha(args.decisions),
           'physical_profile_sha256': sha(args.profile), 'metadata_sha256': sha(args.metadata),
           'checkpoint_directory_explicitly_relocated': str(args.checkpoint_dir.resolve()),
           'query_count': len(got), 'checkpoint_count': len(fits), 'prediction_max_absolute_difference_ms': max(pdelta),
           'uncertainty_max_absolute_difference': max(udelta), 'all_predictions_numerically_close': close,
           'trigger_mismatch_record_ids': mismatches, 'all_triggers_match': not mismatches,
           'deployed_policy_bytes_changed': False, 'target_latency_used_for_inference': False,
           'new_physical_timings': 0, 'model_fitting': False,
           'torch': torch.__version__, 'scope': 'Read-only recomputation using stored structural metadata and exact local checkpoint hashes; cross-runtime FP32 tolerance2e-5 relative is declared.'}
    args.output.write_text(json.dumps(out, indent=2));print(json.dumps(out))
    if not close or mismatches:raise SystemExit(1)


if __name__ == '__main__':main()
