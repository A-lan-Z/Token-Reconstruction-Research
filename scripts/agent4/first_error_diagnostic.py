"""After-freeze evaluator check: did a genuine match exist at first error?"""
from common import *
# No method output may be revised based on this diagnostic.
scored=json.loads((EVID/'final_cpu_score.json').read_text())['results']['final_cpu_adam_r2']
assert digest(EVID/'final_cpu_adam_r2/freeze.json')==scored['freeze_sha256']
truth=json.loads((OUT/'final/bfloat16/evaluator_truth.json').read_text())
obs=load_file(str(OUT/'final/bfloat16/observations.safetensors'))
torch.set_num_threads(2);p=load_prefix(torch.bfloat16,device='cpu');rows=[]
for row in scored['records']:
    pos=row['first_error']
    if pos is None:continue
    # At the first error, the source prefix equals the already reconstructed
    # prefix by definition. This evaluator-only check adds the untried truth token.
    ids=torch.tensor([truth[row['record_id']][:pos+1]])
    actual=p.forward_full(ids)[0,-1].float();target=obs[row['record_id']][pos].float()
    trace=next(x for x in row['positions'] if x['position']==pos)
    rows.append({'record_id':row['record_id'],'position':pos,'genuine_token_mse':float((actual-target).square().mean()),'selected_token_mse':trace['residual'],'genuine_token_was_discovered':trace['discovered'],'max_abs_genuine_residual':float((actual-target).abs().max())})
write(EVID/'first_error_diagnostic.json',{'environment':environment(),'rows':rows,'scope':'evaluator-only public counterfactual after both methods froze; not deployed search, no output revision','extra_forward_evaluations':len(rows)})
print('Post-freeze first-error genuine-match diagnostic completed')
