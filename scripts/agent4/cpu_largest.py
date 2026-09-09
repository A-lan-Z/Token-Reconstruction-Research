from common import *
from solver import *
import resource,psutil
torch.set_num_threads(2);p=load_prefix(torch.bfloat16,device='cpu');start=time.perf_counter()
# Public synthetic geometry qualification, not an evaluation source.
ids=torch.tensor([[128000]+[2028]*15]);obs=p.forward_full(ids).clone()
z=p.embed_tokens(ids).detach().clone().requires_grad_()
loss=(continuous_forward(p,z)[:,-1].float()+.001).square().mean();loss.backward()
assert torch.isfinite(z.grad).all()
score=p.embed_tokens.weight.float()@z[0,-1].float();assert torch.isfinite(score).all()
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert rss<10*2**30 and psutil.virtual_memory().available>=8*2**30
write(EVID/'cpu_largest_qualification.json',{'positions':16,'vocabulary':len(score),'peak_rss_bytes':rss,'host_available_bytes':psutil.virtual_memory().available,'seconds':time.perf_counter()-start,'finite_backward':True,'finite_full_vocabulary_scoring':True,'environment':environment(),'estimate':'CPU live peak on8position development ~2.9GB;16position activations add <50MB conservatively; cap10GiB, hostfree>=8GiB','deviation':'Initial final prediction launch preceded this explicit16position qualification. It was terminated before scoring and is excluded as an orchestration error; all6records are rerun under identical frozen settings, none dropped. The aborted attempt cost is retained separately.'})
print('16-position CPU qualification passed',rss)
