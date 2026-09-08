import json,hashlib,time,datetime,subprocess,gc
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM
from scripts.trr_p12 import qualify_capture as q
from token_reconstruction.target_update import TargetLoRAConfig,install_target_lora,load_target_lora
root=Path.cwd();out=root/'outputs/TRR-P12/fullmodel-capture-review-r1';out.mkdir(exist_ok=False)
def bind(p):
 p=Path(p);return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic();fixture=root/'outputs/TRR-P12/qualification-fixture-r2/source-bundle.json';batch,fb=q._validate_fixture_fields(fixture);snapshot=Path('/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6');ad=root/'experiments/TRR-P12/target-qualification-r1/target-lora-qualification-0004.safetensors';device=torch.device('cuda:0');torch.cuda.init();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();free,total=torch.cuda.mem_get_info();assert free>=8*2**30
model=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval();model.requires_grad_(False);results=[]
for condition in ['baseline','qualification']:
 if condition=='qualification':
  installed=install_target_lora(model,TargetLoRAConfig(layers=(0,1,2,3),modules=('q_proj','v_proj'),rank=8,alpha=16,seed=6012));load_target_lora(installed,ad);model.requires_grad_(False)
 torch.cuda.synchronize();st=time.monotonic()
 with torch.inference_mode():
  y=model.model(input_ids=batch.token_ids.to(device=device,dtype=torch.long),attention_mask=batch.attention_mask.to(device=device,dtype=torch.long),position_ids=batch.position_ids.to(device),output_hidden_states=True,use_cache=False,return_dict=True).hidden_states[4][:,:128].cpu().contiguous()
 torch.cuda.synchronize();elapsed=time.monotonic()-st
 expected_path=root/f'outputs/TRR-P12/qualification-capture-r1/{condition}.safetensors';expected=load_file(str(expected_path))['activations'];assert tuple(y.shape)==(8,128,2048) and y.dtype==torch.bfloat16;delta=(y.float()-expected.float()).abs();different=y.ne(expected);new=out/f'{condition}.safetensors';save_file({'activations':y,'attention_mask':batch.attention_mask[:,:128].contiguous(),'position_ids':batch.position_ids[:,:128].contiguous()},str(new));results.append({'condition':condition,'first128_h_exact_equal':bool(torch.equal(y,expected)),'different_elements':int(different.sum()),'elements':y.numel(),'maximum_absolute_difference':float(delta.max()),'full_model_forward_seconds':elapsed,'qualified_prefix_observation':bind(expected_path),'full_model_observation':bind(new)})
 print(json.dumps(results[-1]),flush=True)
receipt={'schema':'token-reconstruction.trr-p12-independent-capture-review.v1','task_id':'TRR-P12','status':'EXACT_H_EQUIVALENCE_PASS' if all(r['first128_h_exact_equal'] for r in results) else 'NON_EQUIVALENT_PRESERVED_NO_PRODUCTION_RELEASE','scope':'already-opened public8row qualification fixture baseline and four-step adapter; no scientific evaluation sources','started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-t,'code':bind(__file__),'fixture':bind(fixture),'adapter':bind(ad),'configuration':{'dtype':'bfloat16','attention':'sdpa','batch':8,'padded_tokens':192,'first128_active':True,'cut':4,'full_model_call':'model.model(...,output_hidden_states=True,use_cache=False).hidden_states[4][:,:128]'},'resources':{'initial_free_bytes':free,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()},'conditions':results,'fresh_evaluation_truth_opened':False}
p=root/'experiments/TRR-P12/capture-equivalence-review-r1.json'
with p.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(receipt['status'],flush=True)
