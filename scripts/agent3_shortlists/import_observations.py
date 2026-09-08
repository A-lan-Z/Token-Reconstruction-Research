"""Lossless import of an evaluator-released observation cell (no truth)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from scripts.agent3_shortlists.core import binding,verify,write_json


def main(a):
    manifest=json.loads(Path(a.release).read_text())
    # A release consists of one domain/stage and the full opaque source order.
    required={'domain','stage','record_ids','observations'}
    if not required.issubset(manifest):raise ValueError('missing evaluator release fields')
    if manifest['domain'] not in ('pile','finance') or manifest['stage'] not in (0,64,128,256):raise ValueError('unregistered cell')
    with safe_open(verify(manifest['observations']),framework='pt',device='cpu') as f:
        if set(f.keys())!={'activations','attention_mask','position_ids'}:raise ValueError('only sanitized H/mask/positions accepted')
        tensors={k:f.get_tensor(k) for k in f.keys()}
    n=len(manifest['record_ids'])
    if n<32 or any(len(t)!=n for t in tensors.values()):raise ValueError('release source order/shape mismatch')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    selected={k:v[:32].contiguous() for k,v in tensors.items()}
    if selected['activations'].dtype!=torch.bfloat16 or selected['activations'].shape!=(32,128,2048):raise ValueError('wrong H geometry/dtype')
    if not (selected['attention_mask']==1).all():raise ValueError('full-length clips required')
    save_file(selected,str(out/'observations.safetensors'))
    write_json(out/'contract.json',{'schema':'agent3-sanitized-cell-v1','domain':manifest['domain'],'stage':manifest['stage'],
                'record_ids':manifest['record_ids'][:32],'observations':binding(out/'observations.safetensors')})
    write_json(out/'import-receipt.json',{'release':binding(a.release),'source_observations':manifest['observations'],'selection':'first32 fixed opaque order',
                'contract':binding(out/'contract.json'),'tensor_values_changed':False,'truth_opened':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--output',required=True);main(p.parse_args())
