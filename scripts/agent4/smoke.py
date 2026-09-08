from common import *
from solver import *
from transformers import LlamaConfig,LlamaForCausalLM
from token_reconstruction.public_prefix import ContiguousPublicPrefix
torch.set_num_threads(2);torch.manual_seed(44)
config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=5,num_attention_heads=4,num_key_value_heads=2)
config._attn_implementation='eager'
prefix=ContiguousPublicPrefix(LlamaForCausalLM(config),4).eval().requires_grad_(False)
ids=torch.tensor([[1,4,17,8]])
obs=prefix.forward_full(ids)[0].clone()
result=reconstruct(prefix,obs,Settings(steps=32,learning_rate=1),bos=1)
assert result['tokens']==ids[0].tolist(), result['tokens']
z=prefix.embed_tokens(ids).detach().requires_grad_()
cont=continuous_forward(prefix,z)
assert torch.equal(cont,obs.unsqueeze(0))
cont.square().mean().backward();assert z.grad.abs().sum()>0
cache=prefix.new_cache();parts=[prefix.run_cached(ids[:,i:i+1],cache,i) for i in range(4)]
assert torch.allclose(torch.cat(parts,1),obs.unsqueeze(0),atol=1e-6)
# Independent trials do not mutate a pre-existing committed cache.
before=cache.length
reconstruct(prefix,obs,Settings(steps=32),bos=1)
assert cache.length==before
write(EVID/'synthetic_smoke.json',{'result':result,'continuous_discrete_exact':True,'cached_full_close':True,'cache_unchanged':True,'input_gradient_norm':float(z.grad.norm()),'cpu_threads':2})
print('Synthetic end-to-end smoke passed')
