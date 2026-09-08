"""Evaluator-only fixed descendant capture; target assets never passed to solver."""
from common import *
import argparse,shutil,gc,resource
from transformers import AutoTokenizer
from safetensors import safe_open
from safetensors.torch import save_file
p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');args=p.parse_args()
torch.set_num_threads(2);start=time.perf_counter();env=environment()
revision='7fa9d06a59246629244cdd3b6b92e4fc756baa0f'
snapshot=Path('/home/alanz/.cache/huggingface/hub/models--Vikhrmodels--Vikhr-Llama-3.2-1B-Instruct/snapshots')/revision
assets=OUT/'evaluator_only/static_backup';assets.mkdir(parents=True,exist_ok=False)
state={}
with safe_open(str(snapshot/'model.safetensors'),framework='pt',device='cpu') as f:
    for key in f.keys():
        if key=='model.embed_tokens.weight' or any(key.startswith(f'model.layers.{i}.') for i in range(4)):
            state[key.removeprefix('model.')]=f.get_tensor(key)
save_file(state,str(assets/'prefix.safetensors'));del state
for name in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','README.md']:shutil.copyfile(snapshot/name,assets/name)
restore=assets.parent/'static_restore';shutil.copytree(assets,restore)
files=[{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p),'restore_sha256':digest(restore/p.name)} for p in assets.iterdir()]
assert all(f['sha256']==f['restore_sha256'] for f in files)
public_tokenizer=AutoTokenizer.from_pretrained(OUT/'backup',local_files_only=True)
target_tokenizer=AutoTokenizer.from_pretrained(assets,local_files_only=True)
assert public_tokenizer.get_vocab()==target_tokenizer.get_vocab()
prefix=load_prefix(torch.bfloat16,device=args.device,asset_root=assets)
fixture=torch.tensor([[128000]+[2028]*15],device=args.device)
with torch.no_grad():restore_reference=prefix.forward_full(fixture).cpu()
del prefix;gc.collect()
prefix=load_prefix(torch.bfloat16,device=args.device,asset_root=restore)
assert torch.equal(restore_reference,prefix.forward_full(fixture).cpu())
# Exactly one fixed ordinary and one fixed stress row from final source panel.
records=json.loads((ROOT/'experiments/agent4-prefix-only-inversion/final_sources.json').read_text())
chosen=[next(r for r in records if r['group']==g) for g in ['ordinary','stress']]
out=OUT/'static/bfloat16';out.mkdir(parents=True,exist_ok=False)
obs={};truth={};meta=[]
for r in chosen:
    ids=[128000]+public_tokenizer.encode(r['text'],add_special_tokens=False)[:15]
    with torch.no_grad():obs[r['id']]=prefix.forward_full(torch.tensor([ids],device=args.device))[0].cpu().contiguous()
    truth[r['id']]=ids;meta.append({k:r[k] for k in ['id','group','source']})
save_file(obs,str(out/'observations.safetensors'));write(out/'evaluator_truth.json',truth);write(out/'metadata.json',meta)
write(EVID/'static_capture.json',{'environment':env,'end_utc':environment()['utc'],'target_model':'Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct','revision':revision,'target_weights':'evaluator only; no recovery model constructed from target','observation_dtype':'bfloat16 cast of published fp16 checkpoint','tokenizer_vocabulary_identical':True,'restore_output_exact':True,'largest_capture_positions':16,'backup_files':files,'observations_sha256':digest(out/'observations.safetensors'),'truth_sha256':digest(out/'evaluator_truth.json'),'seconds':time.perf_counter()-start,'peak_cpu_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'recovered_prefix_available':False,'scope':'static-surrogate component only; no Agent2 trajectory duplicated'})
print('Static descendant capture completed for2records; no answers displayed')
