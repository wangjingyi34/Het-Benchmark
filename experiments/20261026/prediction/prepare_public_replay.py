"""Build an isolated replay layout from the already published 20261025 archive.

The prior release and restored data are read-only. Existing destinations are
refused. This preserves frozen path strings instead of rewriting provenance.
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prior-release',type=Path,required=True)
    ap.add_argument('--restored-root',type=Path);ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--mode',choices=['verify','recompute','retrain'],default='verify');args=ap.parse_args()
    prior=args.prior_release.resolve();restored=(args.restored_root or prior/'run-data').resolve();work=args.workspace.resolve()
    if work.exists():raise FileExistsError('Use a fresh workspace path: '+str(work))
    if not (restored/'prediction/formal-five-stack/predictions.jsonl.gz').exists():
        raise FileNotFoundError('First run the prior release restore_archives.py --extract, or pass its restored root explicitly.')
    if not (prior/'code/prediction/evaluate_profiles.py').exists():raise FileNotFoundError('Prior release evaluator source missing')
    manifest=json.loads((HERE/'publication-allowlist.json').read_text())
    destination=work/'v7-剩余条件实证闭环/prediction';destination.mkdir(parents=True)
    copied=[]
    for f in manifest['files']:
        rel=Path(f['path']);source=HERE/rel
        assert not rel.is_absolute() and '..' not in rel.parts
        assert sha(source)==f['sha256'],str(source)
        if args.mode!='verify' and rel.parts[0]=='absolute-error-results':continue
        if args.mode!='verify' and str(rel).startswith('modern-baseline/results/'):
            if args.mode=='retrain' or not str(rel).startswith('modern-baseline/results/checkpoints/'):continue
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);copied.append(str(rel))
    shutil.copy2(HERE/'publication-allowlist.json',destination/'publication-allowlist.json')
    v2=work/'v2-实证增强';v2.mkdir()
    (v2/'public-reproduction-check').symlink_to(restored,target_is_directory=True)
    (v2/'prediction').mkdir()
    (v2/'prediction/evaluate_profiles.py').symlink_to(prior/'code/prediction/evaluate_profiles.py')
    checked=0
    for protocol in [destination/'absolute-error-protocol.json',destination/'modern-baseline/protocol.json']:
        for src in json.loads(protocol.read_text())['sources']:
            assert sha(work/src['path'])==src['sha256'],src['path'];checked+=1
    receipt={'complete':True,'mode':args.mode,'workspace':str(work),'prior_release':str(prior),'restored_root':str(restored),
             'copied_files':len(copied),'source_hash_checks':checked,'original_data_mutated':False,
             'prediction_directory':str(destination),'python':sys.executable}
    (work/'replay-layout-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
