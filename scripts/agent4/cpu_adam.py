from common import *
from solver import *
torch.set_num_threads(2)
p=load_prefix(torch.bfloat16,device='cpu')
# Public development diagnostic only: neither source nor other-method output
# is passed to reconstruct. Source is only used here for capture and later score.
ids=torch.tensor([[128000,2028]])
obs=p.forward_full(ids)[0].clone()
result=reconstruct(p,obs,Settings(token_seconds=60, optimizer="adam", learning_rate=.01))
write(EVID/'cpu_adam_prediction.json',result)
score={'correct':result['tokens'][1]==int(ids[0,1]),'checks':len(result['trace'][0]['checks']),'best_mse':result['trace'][0]['best_mse'],'reason':result['trace'][0]['reason'],'seconds':result['seconds'],'initial_loss':result['trace'][0]['continuous_losses'][0],'final_loss':result['trace'][0]['continuous_losses'][-1],'first_gradient':result['trace'][0]['gradient_norms'][0],'prediction_sha256':digest(EVID/'cpu_adam_prediction.json'),'environment':environment(),'scope':'single public development token; CPU-only, not final cost comparison'}
write(EVID/'cpu_adam_score.json',score);print(json.dumps(score))
