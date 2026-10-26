"""Read-only retrospective uncertainty panel for the complete frozen V2 experiment.

Run --freeze first, then run without arguments. All writes remain beside this file.
No training, profiling, threshold selection, or source-data changes occur here.
"""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
SOURCE = BASE / 'v2-实证增强/public-reproduction-check'
FORMAL = SOURCE / 'prediction/formal-five-stack'
KEYS = ('protocol', 'feature_mode', 'method', 'split_id', 'seed')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csvout(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def freeze():
    dest = OUT / 'absolute-error-protocol.json'
    if dest.exists():
        raise FileExistsError('Frozen protocol already exists; never silently overwrite it')
    summary = json.loads((FORMAL / 'summary.json').read_text())
    files = [FORMAL / 'summary.json', FORMAL / 'predictions.jsonl.gz',
             BASE / 'v2-实证增强/prediction/evaluate_profiles.py', Path(__file__)]
    for src in summary['source_manifest']:
        p = SOURCE / 'evidence/physical_profiles' / Path(src['path']).name
        m = SOURCE / 'evidence/physical_profiles' / Path(src['metadata_sidecar']).name
        assert sha(p) == src['sha256'] and sha(m) == src['metadata_sidecar_sha256']
        files.extend([p, m])
    checkpoints = []
    for r in summary['results']:
        for s in r['seed_fits']:
            p = FORMAL / Path(s['checkpoint']).name
            actual = sha(p)
            assert actual == s['checkpoint_sha256'], str(p)
            checkpoints.append({**{k: r[k] for k in KEYS[:-1]}, 'seed': s['seed'],
                                'path': str(p.relative_to(BASE)), 'sha256': actual,
                                'bytes': p.stat().st_size})
    csvout(OUT / 'checkpoint-source-hashes.csv', checkpoints)
    files.append(OUT / 'checkpoint-source-hashes.csv')
    protocol = {
        'schema': 'het-v7-absolute-error-panel/v1',
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen before the new statistics calculation; retrospective analysis of an already evaluated cohort',
        'scope': 'all 10 V2 methods, 2 feature modes, 6 model folds and 5 stack folds, 5 training seeds; no old four-test-model reconstruction',
        'sources': [{'path': str(p.relative_to(BASE)), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in files],
        'seeds': summary['seeds'], 'expected_fits': 1100, 'expected_unique_queries': 13905,
        'units': {'absolute_error': 'microseconds', 'absolute_error_variance': 'microseconds squared',
                  'relative_error': 'percent', 'relative_error_variance': 'percentage-points squared'},
        'point_estimands': {
            'mae_us': 'mean(abs(prediction_ms - true_latency_ms))*1000; all queries retained',
            'ae_sample_variance_us2': 'sample variance (ddof=1) of per-query absolute error; dispersion, not variance of MAE estimator',
            'micro_mre_percent': 'mean(abs(prediction_ms - true_latency_ms)/true_latency_ms)*100',
            'wape_percent': '100*sum(abs(prediction_ms - true_latency_ms))/sum(true_latency_ms)',
            'quantiles': 'empirical query absolute-error 2.5/50/97.5 percentiles; not confidence limits'},
        'detailed_intervals': 'For every fold/seed/source-model, cluster bootstrap paired shape IDs across stacks. There are only 3 configurations/model. For fold/seed overall, use source-model clusters. 10000 percentile draws, seed 20261026+n_clusters; same resampling weights across methods. These are conditional descriptive intervals, not new-model or hardware-population guarantees.',
        'aggregate_intervals': 'First average sufficient statistics over the five seeds. In model_holdout resample 12 source-model clusters, keeping all configurations/stacks and seed-average errors together. In device_holdout_seen_architectures resample 5 held-out-stack clusters, keeping all models/configurations together. 10000 percentile draws. No claim of independent hardware architectures (shared host/machine stacks).',
        'seed_summary': 'Five seed metrics on identical labels/splits: arithmetic mean, sample SD and sample variance, plus conditional Student-t interval using df4 critical value 2.7764451051977987. Seeds describe fitting randomness, not five hardware repeats. Ridge duplicate fits remain, with zero seed variance.',
        'variance_bootstrap': 'Absolute-error sample variance is recomputed inside every cluster resample from count, sum(error), sum(error squared), using ddof=1.',
        'bootstrap_replicates': 10000, 'bootstrap_seed': 20261026,
        'selection_policy': 'No tuning, model selection, exclusion of tiny operations, or result-dependent stopping.',
        'limitations': ['No repair of original RGAT 14.3% provenance or historical four-test identities.',
                       'No timing-replicate or system-noise uncertainty propagation; labels are frozen means.',
                       'All intervals are descriptive and unadjusted across this complete panel.',
                       'V4 relative-loss retrieval is a separate method and is not relabelled RGAT.'],
        'new_physical_measurements': 0, 'remote_execution': False}
    dump(dest, protocol)
    print(json.dumps({'frozen': str(dest), 'checkpoint_hashes_verified': len(checkpoints), 'protocol_sha256': sha(dest)}))


COUNTS = {}


def bootstrap(stats):
    """Columns n,sumAE_us,sumAE_us2,sumARE_pct,sumTrue_ms."""
    s = np.asarray(stats, dtype=np.float64)
    n = len(s)
    if n not in COUNTS:
        rng = np.random.default_rng(20261026 + n)
        COUNTS[n] = rng.multinomial(n, np.full(n, 1 / n), size=10000)
    sampled = COUNTS[n] @ s
    count = sampled[:, 0]
    mae = sampled[:, 1] / count
    var = np.maximum(0, (sampled[:, 2] - sampled[:, 1] ** 2 / count) / (count - 1))
    mre = sampled[:, 3] / count
    lo, hi = np.quantile(np.stack([mae, var, mre], axis=1), [0.025, 0.975], axis=0)
    return {'n_clusters': n, 'mae_cluster_ci95_low_us': float(lo[0]), 'mae_cluster_ci95_high_us': float(hi[0]),
            'ae_variance_cluster_ci95_low_us2': float(lo[1]), 'ae_variance_cluster_ci95_high_us2': float(hi[1]),
            'mre_cluster_ci95_low_percent': float(lo[2]), 'mre_cluster_ci95_high_percent': float(hi[2])}


def sufficient(y, p):
    ae = np.abs(p - y) * 1000
    re = ae / (y * 10)
    return np.array([len(y), ae.sum(), (ae ** 2).sum(), re.sum(), y.sum()])


def metrics(y, p):
    ae = np.abs(p - y) * 1000
    re = ae / (y * 10)
    q = np.quantile(ae, [0.025, 0.5, 0.975])
    return {'n_queries': len(y), 'mae_us': float(ae.mean()), 'ae_sample_sd_us': float(ae.std(ddof=1)),
            'ae_sample_variance_us2': float(ae.var(ddof=1)), 'micro_mre_percent': float(re.mean()),
            'are_sample_variance_percent2': float(re.var(ddof=1)),
            'wape_percent': float(ae.sum() / (10 * y.sum())),
            'ae_q025_us': float(q[0]), 'ae_median_us': float(q[1]), 'ae_q975_us': float(q[2])}


def main():
    protocol = json.loads((OUT / 'absolute-error-protocol.json').read_text())
    for src in protocol['sources']:
        assert sha(BASE / src['path']) == src['sha256'], src['path']
    results = OUT / 'absolute-error-results'
    results.mkdir(exist_ok=False)
    summary = json.loads((FORMAL / 'summary.json').read_text())
    fits = {(r['protocol'], r['feature_mode'], r['method'], r['split_id'], s['seed']): (r, s)
            for r in summary['results'] for s in r['seed_test_metrics']}
    truths = {}
    for src in summary['source_manifest']:
        p = SOURCE / 'evidence/physical_profiles' / Path(src['path']).name
        raw = json.loads(p.read_text())
        for g in raw['graphs']:
            for r in g['records']:
                if r['status'] == 'measured':
                    assert r['record_id'] not in truths
                    truths[r['record_id']] = (statistics.mean(r['latency_samples_ms']), g['model_id'],
                                               g['shape_id'], raw['hardware']['device_id'], g['input_weights_sha256'])
    assert len(truths) == 13905
    detailed = []; fold_rows = []; suff_rows = []; seen_fits = set()
    aggregate = defaultdict(dict); total_rows = 0; max_true_diff = 0.; max_mre_diff = 0.
    with gzip.open(FORMAL / 'predictions.jsonl.gz', 'rt') as f:
        stream = (json.loads(line) for line in f)
        for key, iterator in itertools.groupby(stream, key=lambda r: tuple(r[k] for k in KEYS)):
            assert key not in seen_fits, 'Noncontiguous or duplicate fit'
            seen_fits.add(key)
            rows = list(iterator)
            rsum, ssum = fits[key]
            ids = [r['record_id'] for r in rows]
            assert len(set(ids)) == len(ids) == rsum['test_records']
            assert hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest() == ssum['evaluation_record_ids_sha256']
            for r in rows:
                truth, model, shape, device, weights = truths[r['record_id']]
                diff = abs(r['true_latency_ms'] - truth)
                max_true_diff = max(max_true_diff, diff)
                assert diff < 1e-12
                assert (r['model_id'], r['shape_id'], r['device_id'], r['input_weights_sha256']) == (model, shape, device, weights)
                assert math.isfinite(r['prediction_ms']) and r['prediction_ms'] > 0
            y = np.array([r['true_latency_ms'] for r in rows]); p = np.array([r['prediction_ms'] for r in rows])
            models = np.array([r['model_id'] for r in rows]); shapes = np.array([r['shape_id'] for r in rows])
            devices = np.array([r['device_id'] for r in rows]); prefix = dict(zip(KEYS, key))
            met = metrics(y, p)
            max_mre_diff = max(max_mre_diff, abs(met['micro_mre_percent'] - ssum['micro_mre_percent']))
            assert abs(met['micro_mre_percent'] - ssum['micro_mre_percent']) < 1e-8
            model_stats = []
            for model in sorted(set(models)):
                mask = models == model
                cluster_stats = []
                for shape in sorted(set(shapes[mask])):
                    sel = mask & (shapes == shape)
                    stats = sufficient(y[sel], p[sel]); cluster_stats.append(stats)
                    suff_rows.append({**prefix, 'model_id': model, 'shape_id': int(shape),
                                      **dict(zip(('n_queries','sum_ae_us','sum_ae_squared_us2','sum_are_percent','sum_true_ms'), map(float,stats)))})
                model_stats.append(sufficient(y[mask], p[mask]))
                detailed.append({**prefix, 'model_id': model, **metrics(y[mask], p[mask]),
                                 'cluster_unit': 'shape_id (same configuration across stacks kept together)',
                                 **bootstrap(cluster_stats)})
            fold_rows.append({**prefix, **met, 'cluster_unit': 'source_model', **bootstrap(model_stats)})
            ak = key[:3]
            cluster_ids = models if key[0] == 'model_holdout' else devices
            for cid in sorted(set(cluster_ids)):
                sel = cluster_ids == cid
                ck = (str(cid), key[-1])
                assert ck not in aggregate[ak]
                aggregate[ak][ck] = sufficient(y[sel], p[sel])
            total_rows += len(rows)
            if len(seen_fits) % 100 == 0:
                print(json.dumps({'fits_processed': len(seen_fits), 'prediction_rows': total_rows}), flush=True)
    assert len(seen_fits) == 1100 and set(fits) == seen_fits
    csvout(results / 'per-fold-seed-model.csv', detailed)
    csvout(results / 'per-fold-seed.csv', fold_rows)
    csvout(results / 'per-fold-seed-model-shape-sufficient.csv', suff_rows)
    agg_rows = []; seed_rows = []
    for key, clusters in sorted(aggregate.items()):
        prefix = dict(zip(KEYS[:3], key)); cids = sorted(set(c for c,s in clusters))
        seed_metrics = []
        for seed in protocol['seeds']:
            st = np.sum([clusters[(c, seed)] for c in cids], axis=0)
            assert st[0] == 13905
            met = {'mae_us': st[1]/st[0], 'micro_mre_percent': st[3]/st[0],
                   'wape_percent': st[1]/(10*st[4]),
                   'ae_sample_variance_us2': (st[2]-st[1]**2/st[0])/(st[0]-1)}
            seed_metrics.append(met)
            seed_rows.append({**prefix,'seed':seed,'n_queries':13905,**met})
        row = {**prefix,'n_queries_per_seed':13905,'n_training_seeds':5}
        for field in seed_metrics[0]:
            values = np.array([r[field] for r in seed_metrics]); mean = float(values.mean()); sd = float(values.std(ddof=1))
            half = 2.7764451051977987*sd/math.sqrt(5)
            row.update({field+'_seed_mean':mean, field+'_seed_sd':sd,field+'_seed_variance':sd**2,
                        field+'_conditional_seed_t95_low':mean-half,field+'_conditional_seed_t95_high':mean+half})
        st = [np.mean([clusters[(c,seed)] for seed in protocol['seeds']],axis=0) for c in cids]
        row.update({'cluster_unit': 'source_model' if key[0]=='model_holdout' else 'heldout_runtime_stack',**bootstrap(st)})
        # Variance interval here is for the seed-mixture error distribution; report it separately from mean seed variance.
        row['aggregate_variance_interval_estimand'] = 'seed-mixture query absolute-error variance; cluster resampling seed-average first and second moments'
        agg_rows.append(row)
    csvout(results/'aggregate-panel.csv',agg_rows)
    csvout(results/'pooled-per-seed.csv',seed_rows)
    audit = {'complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),
             'protocol_sha256':sha(OUT/'absolute-error-protocol.json'),'code_sha256':sha(__file__),
             'verified_prediction_rows':total_rows,'unique_measured_queries':len(truths),'fits':len(seen_fits),
             'panel_rows':len(detailed),'fold_rows':len(fold_rows),'sufficient_rows':len(suff_rows),
             'aggregate_rows':len(agg_rows),'pooled_seed_rows':len(seed_rows),
             'max_truth_vs_physical_samples_difference_ms':max_true_diff,
             'max_mre_vs_frozen_summary_difference_percentage_points':max_mre_diff,
             'all_test_support_hashes_match':True,'all_saved_prediction_input_weight_identities_match':True,
             'no_original_four_test_identity_claim':True,'no_new_measurements':True,
             'outputs':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(results.glob('*.csv'))]}
    dump(results/'execution-audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else main()
