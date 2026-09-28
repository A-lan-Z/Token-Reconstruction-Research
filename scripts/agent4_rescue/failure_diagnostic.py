"""Post-freeze evaluator diagnosis only; never used to choose/tune settings."""
from shared import *
from safetensors.torch import load_file
torch.set_num_threads(2);prefix=load_prefix(torch.bfloat16);guard()
source=(ROOT/'scripts/agent4_rescue/search.py').read_text()
source=source.replace("tokens=[128000];traces=[];ex=CurrentToken(prefix)","tokens=[128000];traces=[];ex=CurrentToken(prefix);debug=[]")
source=source.replace("old=float(residual.square().sum())","old=float(residual.square().sum())\n                    entry={'round':outer,'loss_before':old,'damping':damp,'gradient_norm':float(g.norm()),'delta_norm':float(delta.norm()),'z_norm':float(z.norm()),'trials':[]}\n                    debug.append(entry)")
source=source.replace("continuous+=1\n                        if new<old", "continuous+=1\n                        entry['trials'].append({'factor':factor,'loss':new})\n                        if new<old")
source=source.replace("return {'tokens':tokens,'trace':traces","return {'debug':debug,'tokens':tokens,'trace':traces")
namespace={};exec(compile(source,'instrumented_inverse_diagnostic','exec'),namespace)
obs=load_file(str(OUT/'development/observations.safetensors'))
result=namespace['reconstruct'](prefix,obs['dev_0'][:2],'inverse')
original=json.loads((E/'development_inverse/dev_0.json').read_text())
same=[c['token'] for c in result['trace'][0]['checks']]==[c['token'] for c in original['trace'][0]['checks']]
assert same
root=OLD/'gpu_retrospective/bfloat16';truth=json.loads((root/'evaluator_truth.json').read_text());oldobs=load_file(str(root/'observations.safetensors'))
ids=torch.tensor([truth['final_stress_0']],device='cuda')
with torch.no_grad():genuine=prefix.forward_full(ids)[0,-1].float()
mse=float((genuine-oldobs['final_stress_0'][-1].to('cuda').float()).square().mean())
write(E/'post_freeze_failure_diagnostic.json',{'environment':environment(),'inverse_first_development_position':result['debug'],'inverse_candidate_order_reproduced':same,'historical_failed_token_genuine_full_prefix_mse':mse,'scope':'Evaluator diagnosis after both study freeze and all final prediction freezes. No settings or outputs changed.'})
print(json.dumps({'inverse':result['debug'],'genuine_mse':mse},indent=2))
