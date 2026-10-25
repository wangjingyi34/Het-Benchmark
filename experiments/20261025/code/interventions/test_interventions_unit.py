"""Real architecture + random-weight implementation smoke; no task-accuracy claim."""
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval
from torchvision.models import resnet18
from full_model_coalitions import ACTIONS,assign,fx_graph,state_hash,disable_inplace_modules

torch.set_num_threads(4);torch.manual_seed(42)
m=resnet18(weights=None).eval();disabled=disable_inplace_modules(m);x=torch.randn(1,3,64,64)
with torch.inference_mode():ref=m(x)
cases=[]
for mask in range(16):
    c=copy.deepcopy(m)
    for i,(_,p,q) in enumerate(ACTIONS):
        if mask&(1<<i):assign(c,p,fuse_conv_bn_eval(c.get_submodule(p),c.get_submodule(q)));assign(c,q,nn.Identity())
    with torch.inference_mode():y=c(x)
    torch.testing.assert_close(y,ref,atol=1e-4,rtol=1e-4)
    cases.append({'mask':mask,'max_abs_error':float((y-ref).abs().max()),'passed':True})
g=fx_graph(m,x)
incoming={n['id']:0 for n in g['nodes']}
for e in g['edges']:incoming[e['target']]+=1
assert any(v>1 for v in incoming.values())
out={'kind':'implementation_unit_test','pretrained':False,'task_quality_evaluations':0,
     'model':'torchvision resnet18 random weights','coalition_equivalence':cases,
     'graph_nodes':len(g['nodes']),'graph_edges':len(g['edges']),'multi_input_nodes':sum(v>1 for v in incoming.values()),
     'inplace_disabled_modules':disabled,
     'state_sha256':state_hash(m),'all_passed':True}
Path('unit-test-results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
