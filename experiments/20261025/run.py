"""Portable entry point to the exact measured-validation implementations."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
COMMANDS={
 'profile':'prediction/collect_profiles.py',
 'profile-npu':'prediction/collect_profiles_npu.py',
 'predict':'prediction/evaluate_profiles.py',
 'evidence-memory':'prediction/evaluate_evidence_memory.py',
 'family17':'prediction/evaluate_family17_extension.py',
 'full-model':'interventions/full_model_coalitions.py',
 'assemble':'interventions/assemble_tasks.py',
 'paired-confirmation':'interventions/paired_confirmation.py',
 'quality-rejection':'interventions/quality_stress.py',
 'family-probes':'methods/family_calibration_probes.py',
 'attention-challenge':'prediction/attention_ood_challenge.py',
 'planner':'methods/analyze_measured.py',
 'migration':'methods/migration_bridge.py',
 'audit-traceability':'interventions/audit_full128_traceability.py',
 'audit-pairs':'interventions/audit_paired_confirmation.py',
}
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=COMMANDS);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
    args=a.args[1:] if a.args and a.args[0]=='--' else a.args
    script=ROOT/'code'/COMMANDS[a.command]
    if not script.is_file():raise FileNotFoundError(script)
    env=os.environ.copy();env.setdefault('HET_NPU_RECEIPTS',str(ROOT/'run-data/new-launch-receipts'))
    command=[sys.executable]
    if a.command=='profile-npu' or any(args[i:i+2]==['--device','npu'] for i in range(len(args)-1)):
        command.append(str(ROOT/'code/npu_launcher.py'))
    command += [str(script)]+args
    raise SystemExit(subprocess.run(command,env=env).returncode)
if __name__=='__main__':main()
