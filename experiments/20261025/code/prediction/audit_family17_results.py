"""Independent frozen-parameter, probe-byte, cost and derived-metric audit."""
import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(c,m):
    if not c:raise ValueError(m)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result-dir',required=True,type=Path)
    ap.add_argument('--old-summary',required=True,type=Path);ap.add_argument('--family-manifest',required=True,type=Path)
    ap.add_argument('--derived-metrics',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();require(not args.output.exists(),'do not overwrite audit')
    d=json.loads((args.result_dir/'summary.json').read_text());freeze=json.loads((args.result_dir/'extension-freeze.json').read_text())
    old=json.loads(args.old_summary.read_text());manifest=json.loads(args.family_manifest.read_text());derived=json.loads(args.derived_metrics.read_text())
    require(d.get('completed_utc') and len(d['results'])==33 and len(d['aggregate_results'])==6,'incomplete33-entry calibration matrix')
    require(d['original_evidence_summary_sha256']==sha(args.old_summary),'old evidence summary differs')
    require(freeze['family_manifest_sha256']==sha(args.family_manifest),'family14 manifest differs')
    require(d['extension_freeze_sha256']==sha(args.result_dir/'extension-freeze.json'),'extension freeze differs')
    require(d['code_sha256']==sha(args.result_dir.parent/'evaluate_family17_extension.py'),'adapter source differs')
    original={(r['method'],r['split_id']):r for r in old['results'] if r['feature_mode']=='probe'}
    controls_checked=0
    for entry in d['results']:
        method=entry['method'].removesuffix('_family17');control=original[(method,entry['split_id'])]
        require(entry['source_three_probe_validation_selected_configuration']==control['selected_validation_configuration'],'parameter or grid reselected')
        require(entry['configuration_reselected'] is False,'configuration reselection flag invalid')
        require(entry['metrics']['evaluation_record_ids_sha256']==control['test_metrics']['evaluation_record_ids_sha256'],'query support changed')
        controls_checked+=1
    expected={p['probe_id']:p for p in manifest['probes']};probe_checks=0;cost_checks=[]
    cost={r['device_id']:r for r in d['calibration_cost']}
    for source in freeze['calibration_sources']:
        path=Path(source['path']);require(sha(path)==source['sha256'],'auxiliary raw bytes changed')
        p=json.loads(path.read_text());require(p['manifest_sha256']==sha(args.family_manifest),'probe manifest binding differs')
        require(p['collector_sha256']==manifest['collector_sha256'],'probe source code binding differs')
        require(len(p['records'])==14 and p['successful_probes']==14,'not14 successful explicit outcomes')
        timed=0.
        for r in p['records']:
            require(r['status']=='measured' and r['output_all_finite'],'failed/invalid output not accounted for')
            require(r['input_weights_sha256']==expected[r['probe_id']]['input_weights_sha256'],'portable input/weight mismatch')
            ts=r['latency_samples_ms'];require(len(ts)==20 and all(math.isfinite(x) and x>0 for x in ts),'invalid calibration samples')
            require(math.isclose(statistics.mean(ts),r['latency_mean_ms'],rel_tol=1e-10,abs_tol=1e-12),'calibration timing mean mismatch')
            timed+=sum(ts)*8;probe_checks+=1
        oldtimed=sum(sum(r['latency_samples_ms'])*r.get('inner_repeats',8) for r in p['hardware']['probes'])
        c=cost[source['device_id']]
        require(math.isclose(c['total_recorded_timed_blocks_ms'],oldtimed+timed,rel_tol=1e-10,abs_tol=1e-10),'17-query cost arithmetic mismatch')
        require(c['total_query_types']==17 and c['all17_maximum_timed_call_budget']==2720 and c['all17_warmup_call_budget']==170,'auxiliary workload/call budget mismatch')
        require(c['new_fourteen_wall_seconds_including_runtime_build_warmup_output_check']==p['wall_seconds_including_runtime_validation'],'measured auxiliary wall differs')
        cost_checks.append(source['device_id'])
    require(probe_checks==70 and not d['calibration_failures'] and not d['all_stack_original3_fallback_families'],'unexpected failure/fallback in this all-success run')
    index={(r['method'],r['protocol']):r for r in derived['groups']}
    for a in d['aggregate_results']:
        other=index[(a['method'],a['protocol'])]
        require(other['records']==13905 and a['metrics']['records']==13905,'aggregate support differs')
        require(math.isclose(other['micro_MRE_percent'],a['metrics']['micro_mre_percent'],rel_tol=1e-10,abs_tol=1e-8),'independent per-record MRE differs')
    out={'kind':'independent_family17_result_integrity_audit','created_utc':datetime.now(timezone.utc).isoformat(),
        'audit_code_sha256':sha(__file__),'summary_sha256':sha(args.result_dir/'summary.json'),
        'all_checks_passed':True,'fixed_parameter_and_same_support_entries':controls_checked,
        'actual_probe_input_weight_and_timing_checks':probe_checks,'runtime_cost_checks':cost_checks,
        'independent_full_support_aggregate_metrics_checked':6,'all_original_results_retained':True,
        'scope':'No new parameter choices, training or physical measurement. Raw probes, exact prior parameters, same13905 record supports,17-type cost and independent per-record arithmetic are checked.'}
    args.output.write_text(json.dumps(out,indent=2));print(json.dumps(out))


if __name__=='__main__':main()
