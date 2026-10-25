#!/usr/bin/env python3
"""Independent arithmetic and inference-unit audit of archived NPU paired blocks."""
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_relation_drops import HERE, ROOT, sha, write_json, write_csv, holm

SOURCE = ROOT / 'v2-实证增强/methods/paired-method-results-with-ppo.json'
PROTOCOL = HERE / 'copa-paired-scope-protocol.json'
NAMES = ['cross_dag', 'cross_plain_sum', 'target_operator_latency', 'uniform_at_most2', 'uniform_exact2']
T4 = 2.7764451051977987


def vector_summary(v):
    v = np.asarray(v)
    m, sd = float(v.mean()), float(v.std(ddof=1))
    half = T4 * sd / np.sqrt(len(v))
    signs = np.array(list(itertools.product([-1, 1], repeat=len(v))))
    p = float(np.mean(np.abs((signs * v).mean(axis=1)) >= abs(m)-1e-12))
    return {'values': v.tolist(), 'n_independent_process_blocks': len(v), 'mean': m,
            'sample_sd': sd, 'descriptive_t95': [m-half, m+half],
            'exact_two_sided_signflip_diagnostic_p': p}


def main():
    frozen = json.loads(SOURCE.read_text())
    source_paths = [PROTOCOL, Path(__file__), SOURCE]
    raw_hashes = {Path(x['path']).name: x['sha256'] for x in frozen['raw_artifacts']}
    output, sufficient, differences = [], [], []
    for context in frozen['contexts']:
        model, batch = context['model_id'], context['input_shape'][0]
        plan_masks = {name: context['plans'][name]['frozen_mask'] for name in NAMES[:3]}
        gains = {name: [] for name in NAMES}
        controls = []
        process_ids = []
        for run in range(101, 106):
            path = ROOT / f'v2-实证增强/evidence/paired_raw/npu-{model}-b{batch}-r{run}.json'
            assert sha(path) == raw_hashes[path.name]
            source_paths.append(path)
            d = json.loads(path.read_text())
            assert d['status'] == 'completed' and d['provenance'] == 'physical_measurement'
            assert d['model_id'] == model and d['input_shape'][0] == batch
            assert d['protocol']['reselection'] is False
            assert d['protocol']['independence_unit'] == 'fresh process block, not individual timing pair'
            process_ids.append(d['runtime']['pid'])
            bymask = {}
            for c in d['coalitions']:
                pairs = c['pairs']
                assert len(pairs) == 20 and c['final_candidate_feasible']
                orders = [p['order'] for p in pairs]
                assert orders.count('AB') == orders.count('BA') == 10
                for p in pairs:
                    assert p['status'] == 'ok' and p['output_equivalent']
                    assert abs(p['A']['latency_ms']-p['baseline_ms']) < 1e-9
                    assert abs(p['B']['latency_ms']-p['candidate_ms']) < 1e-9
                    assert abs(p['paired_delta_ms']-(p['baseline_ms']-p['candidate_ms'])) < 1e-9
                    a, b = p[p['order'][0]], p[p['order'][1]]
                    assert a['end_monotonic_ns'] < b['start_monotonic_ns']
                am = np.mean([p['baseline_ms'] for p in pairs])
                bm = np.mean([p['candidate_ms'] for p in pairs])
                gain = float(100*(am-bm)/am)
                bymask[c['mask']] = gain
                sufficient.append({'model': model, 'batch': batch, 'run_index': run,
                                   'pid': process_ids[-1], 'mask': c['mask'], 'n_pairs': 20,
                                   'sum_A_ms': float(am*20), 'sum_B_ms': float(bm*20),
                                   'raw_ratio_of_means_gain_percent': gain,
                                   'in_sample_accepted_gain_percent': max(0.0,gain)})
            assert set(bymask) == {0,1,2,3,4,5,6,8,9,10,12}
            controls.append(bymask[0])
            operational = {k: (0.0 if k == 0 else v) for k,v in bymask.items()}
            for name, mask in plan_masks.items():
                gains[name].append(operational[mask])
                old = context['plans'][name]['raw_gain_percent']['values'][run-101]
                assert abs(old-operational[mask]) < 1e-8
            gains['uniform_at_most2'].append(float(np.mean(list(operational.values()))))
            gains['uniform_exact2'].append(float(np.mean([v for k,v in operational.items() if bin(k).count('1') == 2])))
            assert abs(gains['uniform_at_most2'][-1] - context['random_expected_controls'][run-101]['uniform_at_most_k_raw_gain_percent']) < 1e-8
        assert len(set(process_ids)) == 5
        contrasts = {}
        for name in NAMES[1:]:
            rec = vector_summary(np.array(gains['cross_dag'])-np.array(gains[name]))
            rec.update({'model': model, 'batch': batch, 'comparison': 'cross_dag minus '+name})
            contrasts[name] = rec
            differences.append(rec)
        output.append({'model': model, 'batch': batch, 'frozen_masks': plan_masks,
                       'raw_gains_percent': {name: vector_summary(v) for name,v in gains.items()},
                       'in_sample_accepted_gains_percent': {name: vector_summary(np.maximum(v,0)) for name,v in gains.items() if name in NAMES[:3]},
                       'mask0_sham_raw_gain_percent': vector_summary(controls),
                       'paired_raw_contrasts_pp': contrasts})
    for rec, p in zip(differences,holm([x['exact_two_sided_signflip_diagnostic_p'] for x in differences])):
        rec['exploratory_holm16_diagnostic_p'] = p
    write_csv(HERE/'copa-paired-process-sufficient-statistics.csv',sufficient)
    write_json(HERE/'copa-paired-scope-results.json',{
        'kind': 'independent-secondary-physical-COPA-inference-scope-audit',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'all_checks_passed':True,
        'code_sha256':sha(Path(__file__)), 'protocol_sha256':sha(PROTOCOL),
        'raw_physical_blocks':20, 'raw_mask_records':220, 'timing_pairs':4400,
        'inferential_unit':'five fresh process blocks per fixed context',
        'contexts':output,
        'limits':[
            'Twenty adjacent pairs reduce timing noise but do not enlarge the independent sample size from5to100.',
            'Minimum possible two-sided sign-flip p with5nonzero blocks is2/32=0.0625; these data cannot establish p<0.05 by this exact diagnostic in any single context.',
            'Uniform random is an exact expectation across the measured small action domain; not independent randomized planner trials.',
            'Clipping raw negative gains at0uses the same measurements to accept/reject and estimate utility; it is not a fresh estimate of postdeployment speedup.',
            'Two relatedResNetarchitectures,batch1/4,oneNPUconfiguration,andfixedweights/images do not provide model/hardware population inference.',
            'No conclusion here validates original0.98physicalattributionerror,69/175operatorcausalranking,originalprivateFITAS15.6%,or66%costsaving.'
        ],
        'sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in source_paths]
    })
    print(json.dumps({'all_checks_passed':True,'contexts':len(output),'comparisons':len(differences)},indent=2))


if __name__ == '__main__':
    main()
