"""Public-checkpoint loader qualification, without evaluation source access."""
from common import *
from transformers import AutoModelForCausalLM
import gc,resource,argparse
parser=argparse.ArgumentParser();parser.add_argument("--output",default="native_loader_probe.json");args=parser.parse_args()
torch.set_num_threads(2)
# Two retained four-block BF16 prefixes plus transient full public model <5GiB;
# live host available was28GiB. No GPU allocation in this probe.
source=Path(json.loads((EVID/'backup.json').read_text())['source'])
full=AutoModelForCausalLM.from_pretrained(source,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().requires_grad_(False)
native=ContiguousPublicPrefix(full,4);del full;gc.collect()
legacy=load_prefix(torch.bfloat16,device='cpu')
ids=torch.tensor([[128000,2028,374,264,1296,13,220,16]*2])
a=native.forward_full(ids).float();b=legacy.forward_full(ids).float()
assert all(torch.equal(p,q) for p,q in zip(native.parameters(),legacy.parameters()))
write(EVID/args.output,{'environment':environment(),'weights_exact':True,'native_rotary_dtype':str(native.rotary_emb.inv_freq.dtype),'legacy_rotary_dtype':str(legacy.rotary_emb.inv_freq.dtype),'max_rotary_frequency_difference':float((native.rotary_emb.inv_freq.float()-legacy.rotary_emb.inv_freq.float()).abs().max()),'boundary_equal':torch.equal(a,b),'boundary_mse':float((a-b).square().mean()),'peak_cpu_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'source':'independently loaded pinned public checkpoint; public synthetic16token fixture only'})
print((EVID/args.output).read_text())
