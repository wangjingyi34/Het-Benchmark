"""Replay scientific follow-up exports in a separate, task-owned workspace.

Requires the hash-restored 20261025 run-data directory. No confidential reviews,
SSH configuration, infrastructure privileges, or accelerator execution needed.
"""
import argparse,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;PREVIOUS=HERE.parent/'20261025'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(source,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frozen-data',type=Path,required=True)
    ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--studies',nargs='+',choices=['layout','prediction','statistics','economics'],default=['layout','prediction','statistics','economics'])
    a=ap.parse_args();dest=a.workspace.resolve();data=a.frozen_data.resolve()
    if dest.exists():raise FileExistsError('Use a new workspace; do not overwrite a previous replay')
    dest.mkdir(parents=True);v2=dest/'v2-实证增强';v4=dest/'v4-提分条件补强';v2.mkdir();v4.mkdir()
    (v2/'public-reproduction-check').symlink_to(data,target_is_directory=True)
    (v2/'evidence').symlink_to(data/'evidence',target_is_directory=True)
    for source in (PREVIOUS/'code/prediction').glob('*.py'):copy(source,v2/'prediction'/source.name)
    (v2/'prediction/formal-archive-restored').mkdir()
    (v2/'prediction/formal-archive-restored/formal-five-stack').symlink_to(data/'prediction/formal-five-stack',target_is_directory=True)
    copy(PREVIOUS/'results/market-cost-scenarios.json',v2/'methods/market-cost-scenarios.json')
    copy(HERE.parent/'20261025-analysis/results/official-market-cost-recheck.json',dest/'v3-逐问重构/evidence/official-market-cost-recheck.json')
    for part in ['prediction','statistics','economics','physical']:
        shutil.copytree(HERE/part,v4/part)
    # The collector deliberately rejects an existing completed output. Remove
    # only the task-owned copy; the published expected result stays immutable.
    (v4/'prediction/results.json').unlink()
    # These two paths were provenance-only private review hashes, not numerical
    # inputs. Remove only those list entries from the portable executed copy.
    sp=v4/'statistics/analyze_relation_drops.py';original=sp.read_text()
    old="ROOT / 'v2-实证增强/prediction/evaluate_profiles.py',\n                    ROOT / 'v4-提分条件补强/original-reviews.txt',\n                    ROOT / 'v3-逐问重构/question-crosswalk-v3.json']"
    new="ROOT / 'v2-实证增强/prediction/evaluate_profiles.py']"
    assert old in original;sp.write_text(original.replace(old,new))
    patches={'statistics':{'before_sha256':hashlib.sha256(original.encode()).hexdigest(),'executed_sha256':sha(sp),'change':'Remove two confidential review-only provenance paths; no numeric analysis changes.'}}
    (dest/'replay-adaptations.json').write_text(json.dumps(patches,indent=2))
    receipts=[]
    def run(name,args):
        log=dest/(name+'.log')
        with log.open('w') as f:r=subprocess.run([sys.executable]+[str(x) for x in args],stdout=f,stderr=subprocess.STDOUT)
        receipts.append({'study':name,'exit_code':r.returncode,'log':log.name});print(name,r.returncode,flush=True)
        if r.returncode:raise RuntimeError(log.read_text()[-5000:])
    if 'layout' in a.studies:
        path=v4/'physical/collected';path.mkdir()
        with zipfile.ZipFile(v4/'physical/layout-physical-v4.zip') as z:
            for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
            z.extractall(path)
        run('layout',[v4/'physical/summarize_layout.py','--raw',path/'layout-raw','--plan',path/'frozen-layout-plans.json','--output',v4/'physical/replayed-summary.json'])
        assert json.loads((v4/'physical/replayed-summary.json').read_text())==json.loads((HERE/'physical/layout-summary.json').read_text())
    if 'prediction' in a.studies:
        run('prediction',[v4/'prediction/relative_loss_kernel.py'])
        expected=json.loads((HERE/'prediction/results.json').read_text())['aggregates'];actual=json.loads((v4/'prediction/results.json').read_text())['aggregates']
        assert len(actual)==len(expected)
        # Arrays preserve the same deterministic component/mode/protocol order.
        def numeric(obj,path=''):
            if isinstance(obj,dict):
                for k,v in obj.items():yield from numeric(v,path+'/'+k)
            elif isinstance(obj,list):
                for i,v in enumerate(obj):yield from numeric(v,path+'/'+str(i))
            elif isinstance(obj,(int,float)) and not isinstance(obj,bool):yield path,float(obj)
        e=dict(numeric(expected));q=dict(numeric(actual));assert e.keys()==q.keys()
        assert all(abs(e[k]-q[k])<=1e-8*max(1,abs(e[k])) for k in e)
    if 'statistics' in a.studies:
        run('statistics',[sp])
        e=json.loads((HERE/'statistics/relation-drop-model-cluster-results.json').read_text());q=json.loads((v4/'statistics/relation-drop-model-cluster-results.json').read_text())
        assert e['results']==q['results']
    if 'economics' in a.studies:
        run('economics',[v4/'economics/recompute_service_envelope.py'])
        for name in ['capacity-normalized.csv','service-grid.csv','representative-cases.csv','results-summary.json']:
            if name.endswith('.json'):
                expected=json.loads((HERE/'economics'/name).read_text());actual=json.loads((v4/'economics'/name).read_text())
                # Wall-clock audit creation time is not a scientific output.
                expected.pop('created_utc',None);actual.pop('created_utc',None)
                assert expected==actual,name
            else:assert (HERE/'economics'/name).read_bytes()==(v4/'economics'/name).read_bytes(),name
    (dest/'replay-receipt.json').write_text(json.dumps({'all_requested_studies_passed':True,'studies':receipts,'new_hardware_runs':0,'confidential_review_inputs_required':False},indent=2)+'\n')
if __name__=='__main__':main()
