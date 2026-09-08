"""Run from the restored package alone; no original package imports or labels."""
import argparse,json,sys,hashlib,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--observations',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
root=Path(a.package);sys.path.insert(0,str(root/'code'))
from trr0010_p09_fixed_loader import load_p09_fixed_state
from predict import _predict_row_package
m=json.loads((root/'package.json').read_text())
def checked(binding,path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 assert h.hexdigest()==binding['sha256'] and path.stat().st_size==binding['bytes']
for b in m['code']:checked(b,root/'code'/str(b['path']).split('/code/',1)[1])
checked(m['readout'],root/'readout.safetensors');E=load_file(str(root/'readout.safetensors'))['embeddings'].to('cuda')
t=load_file(a.observations);outputs={}
for name,d in m['methods'].items():
 path=root/(name+'.safetensors');checked(d['state'],path);model=load_p09_fixed_state(path,**d['loader_kwargs']).to('cuda').eval()
 outputs[name]=torch.stack([_predict_row_package(model,E,t['activations'][i],t['attention_mask'][i],t['position_ids'][i],device=torch.device('cuda')) for i in range(4)])
 del model
save_file(outputs,a.output)
print(json.dumps({'status':'CLEAN_RESTORED_PACKAGE_PREDICTIONS_COMPLETE','methods':list(outputs),'source_directory':str(root.resolve()),'observations_only':True}))
