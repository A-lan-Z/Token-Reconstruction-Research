from common import *
from solver import continuous_forward
import resource
torch.set_num_threads(2);p=load_prefix(torch.bfloat16,device='cpu')
ids=torch.tensor([[128000,2028,374,264,1296,13,220,16]*2])
full=p.forward_full(ids).clone();cache=p.new_cache()
cached=torch.cat([p.run_cached(ids[:,i:i+1],cache,i).clone() for i in range(16)],1)
iterative=torch.stack([p.forward_full(ids[:,:i+1])[0,-1] for i in range(16)]).unsqueeze(0)
cont=continuous_forward(p,p.embed_tokens(ids));assert torch.equal(full,cont)
assert cache.length==16
write(EVID/'cpu_cache_qualification.json',{'environment':environment(),'positions':16,'raw_continuous_discrete_exact':True,'full_vs_cached_mse_by_position':(full.float()-cached.float()).square().mean(-1).tolist(),'full_vs_iterative_mse_by_position':(full.float()-iterative.float()).square().mean(-1).tolist(),'committed_cache_length':cache.length,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
print('Real public16-position cache qualification completed')
