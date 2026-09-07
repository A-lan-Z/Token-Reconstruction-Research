import argparse, json, time, gc
from pathlib import Path
import torch
from trr0010_production_provider import build_inputs
from trr0010_directional_fit_cli import _load_diagnostic_binding
p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--role',required=True);p.add_argument('--diagnostic',type=Path,required=True);a=p.parse_args()
b=json.loads(a.binding.read_text());d=_load_diagnostic_binding(a.diagnostic)
seen=[]
t=time.perf_counter()
value=build_inputs(b, {'device':'cpu'}, torch.device('cpu'), lambda stage: seen.append(stage), arm_name=a.arm, bank_role=a.role, checkpoint_steps=(0,1000,2000,4000,8000,12000,13000), output_root=Path('/tmp/trr0010_prod_smoke/out'), diagnostic_binding=d)
print(json.dumps({'arm':a.arm,'role':a.role,'status':'PASS_PROVIDER_ASSEMBLY_CPU','elapsed_seconds':time.perf_counter()-t,'stages':seen,'selected_start_sha256':value['base_state']['sha256'],'selected_start_step':value['base_state']['selected_step'],'schedule_count':len(value['schedule_steps']),'full_bank_bound':value['provider_receipt']['full_bank_row_count'],'diagnostic_callback':callable(value['diagnostic_callback']),'runtime_type':type(value['runtime']).__name__,'truth_opened':False},sort_keys=True))
# Deliberately do not invoke runner, validation, or diagnostic callback; release all payloads now.
del value
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
