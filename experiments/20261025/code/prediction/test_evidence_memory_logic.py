"""Synthetic logic fixtures, not physical performance observations."""
import copy
import json
from pathlib import Path

import numpy as np

import evaluate_evidence_memory as e


def row(model,device,label,probe):
    return {'record_id':device+'/'+model,'model_id':model,'device_id':device,'latency_ms':label,
            'family':'Conv2D','operator':'call_module|Conv2d','exact_signature':'same-semantic-shape',
            'strict_features':[0.]*68,'probe_features':[0.]*71,'probe_latencies_ms':[probe]*3}


def main():
    rows=[row('train_a','s1',1.,1.),row('train_b','s2',100.,2.),row('query','s2',999.,2.)]
    train=np.array([0,1]);query=np.array([2]);checks={}
    cache=e.neighbors(rows,train,query,'semantic_evidence_memory',True)
    checks['device_validation_masks_query_stack']=cache[0]['anchor_indices'].tolist()==[0]
    checks['actual_probe_ratio_scales_training_evidence']=np.allclose(e.memory_predict(rows,cache,1,1.,1.),[2.])
    before=e.memory_predict(rows,cache,1,1.,0.)
    altered=copy.deepcopy(rows);altered[2]['latency_ms']=.00001
    after=e.memory_predict(altered,e.neighbors(altered,train,query,'semantic_evidence_memory',True),1,1.,0.)
    checks['query_target_cannot_change_prediction']=np.array_equal(before,after)
    altered=copy.deepcopy(rows);altered[0]['latency_ms']=4.
    checks['training_measurement_is_true_prediction_evidence']=np.allclose(e.memory_predict(altered,cache,1,1.,0.),[4.])
    altered=copy.deepcopy(rows);altered[2]['probe_latencies_ms']=[1e100]*3
    changed=e.neighbors(altered,train,query,'semantic_evidence_memory',True)
    checks['no_probe_arm_invariant_to_probe_values']=np.array_equal(before,e.memory_predict(altered,changed,1,1.,0.))
    ondevice=e.neighbors(rows,train,query,'semantic_evidence_memory',False)
    checks['model_holdout_prefers_available_ondevice_evidence']=ondevice[0]['anchor_indices'].tolist()==[1]
    a={'module_path':'','fx_op':'call_function','target':'<function example at 0x123abc>'}
    b={**a,'target':'<function example at 0xdeadbeef>'}
    checks['function_addresses_normalize']=e.canonical_operator(a)==e.canonical_operator(b)
    feature_train,feature_query=e.ridge_features(rows,np.array([0]),query,'strict')
    altered=copy.deepcopy(rows);altered[2]['operator']='unseen_class'
    tr2,qu2=e.ridge_features(altered,np.array([0]),query,'strict')
    checks['ridge_vocabulary_train_only']=np.array_equal(feature_train,tr2) and feature_query.shape==qu2.shape
    out={'kind':'synthetic_evidence_memory_logic_tests','physical_observations':0,'checks':checks,
         'passed':all(checks.values()),'tested_code_sha256':e.sha(Path(e.__file__))}
    Path(__file__).with_name('evidence-memory-logic-tests.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
    if not out['passed']:raise SystemExit(1)


if __name__=='__main__':main()
