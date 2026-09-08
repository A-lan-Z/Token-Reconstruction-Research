from common import *
from comparator import prepare,decode
from token_reconstruction.a1a2_configuration_search import decode_policy,resolved_policy_from_dict
from token_reconstruction.component_crossover import propose_public_a1
from solver import sync

torch.set_num_threads(2);prefix=load_prefix(torch.bfloat16);guard()
ids=torch.tensor([[128000,2028,374,264,1296]],device='cuda');obs=prefix.forward_full(ids).cpu()
lens,embeddings=prepare(prefix);mask=torch.ones(obs.shape[:2],dtype=torch.long);positions=torch.arange(obs.shape[1]).view(1,-1)
policy=resolved_policy_from_dict(json.loads((ROOT/'experiments/TRR-0002/configuration-search/causal-selection/winner.json').read_text())['policy'])
proposals=propose_public_a1(observations=obs,attention_mask=mask,lens=lens,normalized_embeddings=embeddings)
native=decode_policy(observations=obs,attention_mask=mask,position_ids=positions,candidates=proposals.candidates,a1_confidence=proposals.top1_confidence,precut=prefix,device=torch.device('cuda'),policy=policy,record_batch_size=1)
port=decode(prefix,obs[0],lens,embeddings,guard=guard)
assert native.predictions[0].tolist()==port['tokens']
assert all(t['checks'][i]['token']==proposals.candidates[0,t['position'],i].item() for t in port['trace'] for i in range(256))
write(EVID/'comparator_equivalence.json',{'native_predictions':native.predictions[0].tolist(),'port_predictions':port['tokens'],'same_candidates_k256':True,'same_predictions':True,'native_policy_id':policy.policy_id,'environment':environment(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
print('Native A1+A2 equivalence passed')
