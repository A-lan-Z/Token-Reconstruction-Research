from native import *
torch.set_num_threads(2)
start=time.perf_counter();env=environment();prefix=load_prefix();guard()
cache=new_context(prefix);ids=torch.arange(256,device='cuda')
with torch.no_grad():
    candidates(prefix,cache,ids,1)
sync();torch.cuda.reset_peak_memory_stats()
times=[]
for j in range(4):
    sync();t=time.perf_counter();y=candidates(prefix,cache,ids+256*j,1);sync();times.append(time.perf_counter()-t)
# Long-context qualification uses arbitrary public tokens, not source observations.
longcache=prefix.new_cache()
prefix.run_cached(torch.tensor([[128000]+list(range(126))],device='cuda'),longcache,0)
y=candidates(prefix,longcache,ids,127)
# Include largest stored table in resource qualification.
table=torch.empty((128256,2048),device='cuda',dtype=torch.float32)
table.zero_();sync();guard()
mem={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
del table
# Outputs can change under row batching: measure and retain, never claim identity silently.
many=candidates(prefix,cache,ids[:16],1).float()
one=torch.stack([candidates(prefix,cache,ids[i:i+1],1)[0] for i in range(16)]).float()
split=torch.cat([candidates(prefix,cache,ids[i:i+8],1) for i in [0,8]]).float()
write(X/'preflight.json',{'environment':env,'times_256':times,'estimated_table_seconds':sum(times)*128256/1024,'peak':mem,'batch16_vs_singleton_maxabs':float((many-one).abs().max()),'batch16_vs_singleton_relative_l2':float((many-one).norm()/one.norm()),'batch16_vs_batch8_maxabs':float((many-split).abs().max()),'largest_context':127,'largest_candidates':256,'fingerprint_bytes':128256*2048*4,'wall':time.perf_counter()-start,'prefix_sha256':digest(ASSETS/'backup/prefix.safetensors')})
print(json.dumps(json.loads((X/'preflight.json').read_text()),indent=2),flush=True)
