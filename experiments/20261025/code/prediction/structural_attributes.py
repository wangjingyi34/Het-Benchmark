"""Label-free module/shape attributes, frozen before predictor fitting."""
import math
import torch
from torch import nn

NUMERIC_ATTRIBUTES=['in_channels','out_channels','groups','kernel_h','kernel_w',
    'stride_h','stride_w','padding_h','padding_w','dilation_h','dilation_w',
    'in_features','out_features','num_embeddings','embedding_dim','normalized_dim',
    'eps','attention_heads','attention_head_dim','is_causal','dropout_p',
    'bias_present','training_flag','input_channels_last','composite_leaf']

def pair(value,default=0):
    if isinstance(value,(tuple,list)):
        if not value:return [default,default]
        return [value[0],value[1] if len(value)>1 else value[0]]
    return [value,value]

def simple_module_attrs(m):
    out={'module_class':m.__class__.__module__+'.'+m.__class__.__name__,
         'training_flag':int(m.training),'bias_present':int(getattr(m,'bias',None) is not None)}
    for key in ['in_channels','out_channels','groups','in_features','out_features','num_embeddings','embedding_dim','eps']:
        value=getattr(m,key,None)
        if isinstance(value,(int,float,bool)):out[key]=float(value)
    for key,prefix in [('kernel_size','kernel'),('stride','stride'),('padding','padding'),('dilation','dilation')]:
        value=getattr(m,key,None)
        if value is not None:
            pp=pair(value)
            if all(isinstance(t,(int,float)) for t in pp):out[prefix+'_h']=float(pp[0]);out[prefix+'_w']=float(pp[1])
            else:out[key]=str(value)
    normalized=getattr(m,'normalized_shape',None)
    if normalized is not None:out['normalized_shape']=list(normalized);out['normalized_dim']=math.prod(normalized)
    if hasattr(m,'num_heads'):out['attention_heads']=int(m.num_heads)
    if isinstance(m,nn.Dropout):out['dropout_p']=float(m.p)
    return out

def attributes_for_node(model,node,model_id):
    out={}
    if node['module_path']:
        m=model.get_submodule(node['module_path']);out=simple_module_attrs(m)
        if m.__class__.__name__=='_DenseLayer':
            out['composite_leaf']=1
            out['child_modules']={name:simple_module_attrs(child) for name,child in m.named_modules() if name}
    shapes=node.get('input_shapes',[])
    if shapes:
        shape=shapes[0]['shape'];stride=shapes[0]['stride']
        if len(shape)==4 and len(stride)==4:
            out['input_channels_last']=int(stride[1]==1 and stride[3]==shape[1] and stride[2]==shape[1]*shape[3])
    if node['family']=='Attention' and shapes:
        qshape=shapes[0]['shape']
        if len(qshape)==4:
            out['attention_heads']=int(qshape[1]);out['attention_head_dim']=int(qshape[-1])
        if not node['module_path']:
            # The only functional SDPA calls in the frozen reference catalog
            # use dropout_p=0 and model-specific static causal flags.
            out['is_causal']=int(model_id=='causal_lm_ref');out['dropout_p']=0.
            out['functional_configuration_source']='frozen collect_profiles.Attention.forward'
    out['attribute_source']='reconstructed executable architecture plus recorded tensor shapes; no latency labels'
    return out
