"""Truth-free fixed B1/A1 shortlist generation. No evaluator or target imports."""
from __future__ import annotations
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import platform
import importlib.metadata
import resource
import subprocess
import sys
import time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from scripts.agent3_shortlists.core import binding, rank_scores, sha, verify, write_json

STATE_SHA = '088be6a6b2842d526f3dab39789728d9cfc23f2fb1fb892d107d46ade2382706'
READOUT_SHA = 'ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1'
PACKAGE_SHA = '5023a9e7cb19611952c0ce16a440ec9fde6b6444fd74aaf74a7efbb5c671c13b'
PACKAGE_FILES_SHA = '0003ab30ee60e95fb75d015a1abfd038e76e2e2d117d4b470db195ef2ec35f46'
LENS_SHA = '33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742'
REFERENCE_SHA = '10532a746cb8c30eb2caf338e206e1fa9d85e708d4db43a0d8fd4a2ff1a6f8bd'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Pinned asset changed: {path}')


def load_models(package_root, lens_path, reference_path, device):
    root = Path(package_root).resolve()
    file_manifest=Path(__file__).resolve().parents[2]/'experiments/agent3-b1-small-budget-a2/package-files.json'
    check_hash(file_manifest,PACKAGE_FILES_SHA)
    for rel,b in json.loads(file_manifest.read_text())['files'].items():
        check_hash(root/rel,b['sha256'])
        if (root/rel).stat().st_size!=b['bytes']:raise ValueError('package file size changed')
    check_hash(root / 'code/trr0012_package.py', PACKAGE_SHA)
    package = module(root / 'code/trr0012_package.py', '_agent3_preserved_package')
    descriptor, _ = package._load_package_descriptor(root)
    method = descriptor['methods']['expanded_fixed']
    if method['selected_step'] != 13000 or method['state_sha256'] != STATE_SHA:
        raise ValueError('wrong B1 selection')
    check_hash(root / method['state_path'], STATE_SHA)
    readout, readout_path = package._load_readout(root, descriptor)
    check_hash(readout_path, READOUT_SHA)
    readout = readout.to(device).contiguous()
    model, state_sha, _ = package._load_package_method(root, descriptor, 'expanded_fixed', device=device, readout=readout)
    assert state_sha == STATE_SHA
    check_hash(reference_path, REFERENCE_SHA)
    check_hash(lens_path, LENS_SHA)
    reference = module(reference_path, '_agent3_historical_reference')
    lens = reference.load_frozen_lens(Path(lens_path), device=device)
    model.requires_grad_(False)
    return package, model, readout, lens


def b1_logits(model, readout, activation, mask, positions):
    h = activation.to(readout.device, dtype=torch.float32).unsqueeze(0)
    m = mask.to(readout.device, dtype=torch.bool).unsqueeze(0)
    p = positions.to(readout.device, dtype=torch.long)
    projected = model.projected_hidden(h, m)
    return model.logits_from_rows(projected, torch.zeros_like(p[1:]), p[1:], readout)


def configure(device):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def guard(start, device):
    mem = dict((a.rstrip(':'), int(b) * 1024) for a,b,*_ in (l.split() for l in Path('/proc/meminfo').read_text().splitlines()))
    if mem['MemAvailable'] < 8 * 2**30:
        raise RuntimeError('host available memory below8GiB')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4 * 2**30:
        raise RuntimeError('process peak RSS exceeds4GiB')
    if time.monotonic() - start > 1800:
        raise RuntimeError('cell exceeded30minutes')
    if device.type == 'cuda' and torch.cuda.mem_get_info()[0] < 4 * 2**30:
        raise RuntimeError('GPU free memory below4GiB')


def run(args):
    start = time.monotonic()
    start_utc = utc()
    execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    code_bindings=[binding(Path(__file__)),binding(Path(__file__).with_name('core.py'))]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    configure(device)
    guard(start, device)
    contract = json.loads(Path(args.contract).read_text())
    # Contract is sanitized: no source payload or target weights accepted.
    if set(contract) != {'schema', 'domain', 'stage', 'record_ids', 'observations'}:
        raise ValueError('unexpected sanitized cell contract fields')
    obs_path = verify(contract['observations'])
    with safe_open(obs_path, framework='pt', device='cpu') as f:
        if set(f.keys()) != {'activations', 'attention_mask', 'position_ids'}:
            raise ValueError('forbidden or missing observation tensor')
        h, mask, pos = (f.get_tensor(k) for k in ('activations','attention_mask','position_ids'))
    n = len(contract['record_ids'])
    if n < 1 or len(set(contract['record_ids'])) != n or tuple(h.shape) != (n,128,2048):
        raise ValueError('observation/record geometry changed')
    if mask.shape != (n,128) or not (mask == 1).all() or not torch.equal(pos, torch.arange(128).expand(n,-1)):
        raise ValueError('requires full128-position clips')
    if h.dtype != torch.bfloat16 or not torch.isfinite(h).all():
        raise ValueError('BF16 finite observations required')
    load_start = time.monotonic()
    package, model, readout, lens = load_models(args.package, args.lens, args.reference, device)
    sync(device)
    load_seconds = time.monotonic() - load_start
    receipts = []
    for method in ('b1','a1'):
        ids_out, scores_out, scores_hash, phases = [], [], [], []
        for i in range(n):
            guard(start, device)
            sync(device)
            t = time.monotonic()
            with torch.inference_mode():
                logits = b1_logits(model, readout, h[i], mask[i], pos[i]) if method == 'b1' else lens(h[i,1:].to(device), readout).float()
                sync(device)
                t1 = time.monotonic()
                ids, scores = rank_scores(logits)
                sync(device)
                t2 = time.monotonic()
                ids_out.append(ids.cpu())
                scores_out.append(scores.cpu())
                scores_hash.append(package.tensor_digest(logits))
                t3 = time.monotonic()
                del logits, ids, scores
            phases.append({'record': i, 'scoring_seconds':t1-t,'ranking_seconds':t2-t1,'transfer_hash_seconds':t3-t2})
        p = out / f'{method}.safetensors'
        t = time.monotonic()
        save_file({'candidates':torch.stack(ids_out), 'scores':torch.stack(scores_out),
                   'predictions':torch.cat([torch.full((n,1),128000,dtype=torch.int64),torch.stack(ids_out)[:,:,0]],1)}, str(p))
        receipts.append({'method':method,'artifact':binding(p),'record_timings':phases,'full_score_tensor_sha256':scores_hash,'output_io_seconds':time.monotonic()-t})
    guard(start, device)
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=execution_commit:
        raise RuntimeError('code commit changed during cell')
    for b in code_bindings:verify(b)
    receipt = {'schema':'agent3-shortlist-prediction-v1','status':'FROZEN_NO_TRUTH','start_utc':start_utc,'end_utc':utc(),
               'contract':binding(args.contract),'domain':contract['domain'],'stage':contract['stage'],'record_ids':contract['record_ids'],
               'code_commit':execution_commit,
               'code_files':code_bindings,'package_files_sha256':PACKAGE_FILES_SHA,
               'package_manifest':binding(Path(args.package)/'package_manifest.json'),'state_sha256':STATE_SHA,'readout_sha256':READOUT_SHA,
               'lens':binding(args.lens),'reference':binding(args.reference),'methods':receipts,
               'command':sys.argv,'environment':{'numpy':importlib.metadata.version('numpy'),'safetensors':importlib.metadata.version('safetensors'),'python':platform.python_version(),'torch':torch.__version__,'device':str(device),'machine':platform.platform(),'cpu_threads':2,
                              'matmul_precision':torch.get_float32_matmul_precision(),'tf32':False,'deterministic_algorithms':True,'record_batch_size':1},
               'load_seconds':load_seconds,'total_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
               'candidate_simulations':0,'model_evaluations':{'b1_record_forwards':n,'a1_record_forwards':n},'truth_opened':False,'target_weights_loaded':False}
    write_json(out/'receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'output':str(out),'seconds':receipt['total_seconds']}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for key in ('contract','package','lens','reference','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    a=p.parse_args()
    try:
        run(a)
    except Exception as e:
        dest=Path(a.output)/'failure.json'
        if dest.parent.exists() and not dest.exists():
            write_json(dest,{'status':'FAILED_EXCLUDED','utc':utc(),'error':repr(e),'truth_opened':False})
        raise
