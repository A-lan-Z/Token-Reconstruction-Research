"""Bounded focused-practice study; reuses the native fixed decoder and train step."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace
import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from scripts.trr0013_vendor import fixed_control_runner as native
from token_reconstruction.trr_p09_fixed_control_adapter import FixedPublicReadoutHook
from token_reconstruction.trr0007_positionwise import build_residual_mlp512
from scripts.trr0010_p09_fixed_loader import load_p09_fixed_state
from scripts.trr0012_package import _predict_row_package


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def verify(b):
    p = Path(b['path'])
    if p.stat().st_size != b['bytes'] or sha(p) != b['sha256']:
        raise ValueError(f'Changed bound artifact: {p}')
    return p


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def save(path, tensors, metadata=None):
    p = Path(path)
    if p.exists():
        raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(temporary), metadata=metadata)
    os.replace(temporary, p)
    return artifact(p)


def provenance():
    return dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source=artifact(__file__), argv=sys.argv, torch=torch.__version__, python=sys.version,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')


class Guard:
    def __init__(self, seconds=2400, rss_gib=20, gpu_gib=8):
        self.start = time.perf_counter()
        self.seconds, self.rss, self.gpu = seconds, rss_gib * 2**30, gpu_gib * 2**30

    def check(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if time.perf_counter() - self.start > self.seconds or rss > self.rss:
            raise RuntimeError('Resource deadline/RSS guard failed')
        if torch.cuda.is_initialized():
            free, _ = torch.cuda.mem_get_info()
            if torch.cuda.memory_reserved() > self.gpu or free < 2 * 2**30:
                raise RuntimeError('GPU reserved/free memory guard failed')
        return dict(peak_rss_bytes=rss,
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0)


def load_start(inputs, device='cuda'):
    descriptor = json.loads(verify(inputs['package_manifest']).read_text())
    state = verify(inputs['starting_state'])
    model = load_p09_fixed_state(state, **descriptor['methods']['expanded_fixed']['loader_kwargs']).to(device).eval()
    table = load_file(str(verify(inputs['readout'])))['embeddings'].to(device)
    return model, table


def bank_parts(inputs):
    yield 0, inputs['b0_payload']
    p = verify(inputs['b1_manifest'])
    m = json.loads(p.read_text())
    for s in m['sharding']['shards']:
        b = dict(s['payload'])
        b['path'] = str(p.parent / b['path'])
        yield s['row_range']['start'], b


def normalize_payload(t):
    t = {k: t[k] for k in ('activations', 'token_ids', 'attention_mask', 'position_ids')}
    t['attention_mask'] = t['attention_mask'].bool()
    t['position_ids'] = torch.where(t['attention_mask'], t['position_ids'], 0)
    return t


def batch(t, rows):
    ix = torch.as_tensor(rows, dtype=torch.long)
    return SimpleNamespace(**{k: v[ix] for k, v in t.items()}, global_rows=tuple(rows))


def sampler(valid, losses, rng, draws=512, hard=False):
    """Uniform reference; hard arm mixes 50% uniform and 50% top-quintile draws.

    Difficulty is frozen starting-model fitting loss within the SAME record batch.
    No importance correction: the hard arm deliberately changes training weights.
    """
    eligible = np.argwhere(valid)
    if len(eligible) == 0:
        raise ValueError('No eligible post-BOS fitting positions')
    if not hard:
        picks = rng.choice(len(eligible), draws, replace=len(eligible) < draws)
    else:
        values = losses[eligible[:, 0], eligible[:, 1]]
        if not np.isfinite(values).all():
            raise ValueError('Invalid difficulty values')
        # Stable tie-break by record/position; fixed top20%, no tuned thresholds.
        difficult = np.argsort(-values, kind='stable')[:max(1, math.ceil(len(values) / 5))]
        n = draws // 2
        ordinary = rng.choice(len(eligible), n, replace=len(eligible) < n)
        focused = rng.choice(difficult, draws - n, replace=True)
        picks = np.concatenate((ordinary, focused))
        rng.shuffle(picks)
    chosen = eligible[picks]
    return chosen, len(np.unique(picks)) != len(picks)


def validate_frozen(freeze_path):
    """All bound code/inputs/states/predictions checked before truth is opened."""
    freeze = json.loads(Path(freeze_path).read_text())
    if freeze['status'] != 'PREDICTIONS_FROZEN_BEFORE_TRUTH':
        raise ValueError('Unfrozen predictions')
    if set(freeze['predictions']) != set(freeze['methods']):
        raise ValueError('Incomplete method predictions')
    for b in freeze['dependencies']:
        verify(b)
    predictions = {}
    for name, b in freeze['predictions'].items():
        t = load_file(str(verify(b)))
        if set(t) != {'predictions'}:
            raise ValueError('Prediction payload key mismatch')
        p = t['predictions']
        if p.dtype != torch.int64 or list(p.shape) != freeze['shape']:
            raise ValueError('Prediction geometry/dtype mismatch')
        if (p < 0).any() or (p >= freeze['vocabulary']).any() or not (p[:, 0] == freeze['bos']).all():
            raise ValueError('Prediction integrity failure')
        predictions[name] = p
    return freeze, predictions


def score_frozen(freeze_path, truth_binding):
    freeze, predictions = validate_frozen(freeze_path)
    # This line is deliberately AFTER every completeness and integrity check.
    truth = load_file(str(verify(truth_binding)))['truth']
    if list(truth.shape) != freeze['shape'] or truth.dtype != torch.int64:
        raise ValueError('Truth geometry mismatch')
    return {name: dict(correct_tokens=int((p[:, 1:] == truth[:, 1:]).sum()),
                       exact_records=int((p[:, 1:] == truth[:, 1:]).all(1).sum()))
            for name, p in predictions.items()}


def synthetic_smoke(output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(5013)
    model = build_residual_mlp512(hidden_size=16, vocabulary_size=32, context_width=4, bottleneck_size=8, seed=5013)
    E = F.normalize(torch.randn(32, 16), dim=1)
    t = dict(activations=torch.randn(8, 8, 16).bfloat16(), token_ids=torch.randint(0, 32, (8, 8)),
             attention_mask=torch.ones(8, 8, dtype=torch.bool), position_ids=torch.arange(8).expand(8, -1).clone())
    t['token_ids'][:, 0] = 0
    valid=t['attention_mask'].numpy().copy();valid[:,0]=False
    draws, replacement = sampler(valid, np.arange(64).reshape(8,8), np.random.default_rng(5013), 32, True)
    cfg=native.RunnerConfig(2,8,32,1,'token_accuracy',5013,1e-4,0,1,8,16,'torch.bfloat16')
    source=SimpleNamespace(batch_for_global_rows=lambda rows:batch(t,rows))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0,foreach=False)
    hook=FixedPublicReadoutHook('continued_fixed_readout','0'*64)
    for i in range(2):
        step=native.ScheduleStep(i,tuple(range(8)),tuple(draws[:,0].tolist()),tuple(draws[:,1].tolist()),replacement)
        native.train_one_step(model,hook,source,step,optimizer=opt,embedding=E,config=cfg,activation_dtype=torch.bfloat16)
    state=save(out/'state.safetensors',model.state_dict())
    restored=build_residual_mlp512(hidden_size=16,vocabulary_size=32,context_width=4,bottleneck_size=8,seed=0)
    restored.load_state_dict(load_file(str(verify(state))),strict=True)
    with torch.inference_mode():
        a=model(t['activations'].float(),t['attention_mask'],E)
        b=restored(t['activations'].float(),t['attention_mask'],E)
        assert torch.equal(a,b)
        pred=a.argmax(-1);pred[:,0]=0
    pb=save(out/'predictions.safetensors',{'predictions':pred})
    tb=save(out/'truth.safetensors',{'truth':t['token_ids']})
    freeze=dict(status='PREDICTIONS_FROZEN_BEFORE_TRUTH',methods=['synthetic'],predictions={'synthetic':pb},
                dependencies=[state,artifact(__file__)],shape=[8,8],vocabulary=32,bos=0)
    write_json(out/'freeze.json',freeze)
    result=score_frozen(out/'freeze.json',tb)
    bad=dict(freeze,predictions={});write_json(out/'incomplete.json',bad)
    try:
        score_frozen(out/'incomplete.json',{'path':'NONEXISTENT_TRUTH','sha256':'0'*64,'bytes':0})
    except ValueError as e:
        assert 'Incomplete' in str(e)
    else:
        raise AssertionError('Truth gate did not fail closed')
    write_json(out/'receipt.json',dict(status='PASS',native_steps=2,restored_logits_exact=True,
               incomplete_before_truth=True,score=result,provenance=provenance()))


def bank_check(inputs_path, output, limit=12000):
    inputs=json.loads(Path(inputs_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
    guard=Guard();started=utc();clock=time.perf_counter()
    model,E=load_start(inputs);load_seconds=time.perf_counter()-clock
    torch.cuda.reset_peak_memory_stats()
    losses=torch.full((limit,192),float('nan'));correct=torch.zeros((limit,192),dtype=torch.bool)
    mask_all=torch.zeros((limit,192),dtype=torch.bool);labels=torch.zeros((limit,192),dtype=torch.long)
    checked=[];qualifier=None;rows_seen=0
    with torch.inference_mode():
        for origin,b in bank_parts(inputs):
            if origin>=limit:break
            tensors=normalize_payload(load_file(str(verify(b))));checked.append(b)
            count=min(tensors['token_ids'].shape[0],limit-origin)
            for j in range(0,count,8):
                n=min(8,count-j);H=tensors['activations'][j:j+n].to('cuda',dtype=torch.float32)
                M=tensors['attention_mask'][j:j+n].to('cuda');Y=tensors['token_ids'][j:j+n].to('cuda')
                z=model.projected_hidden(H,M);valid=M.clone();valid[:,0]=False
                ix=valid.nonzero()
                for chunk in ix.split(256):
                    logits=model.logits_from_rows(z,chunk[:,0],chunk[:,1],E)
                    ce=F.cross_entropy(logits,Y[chunk[:,0],chunk[:,1]],reduction='none')
                    target_rows=origin+j+chunk[:,0].cpu();pos=chunk[:,1].cpu()
                    losses[target_rows,pos]=ce.cpu();correct[target_rows,pos]=(logits.argmax(1)==Y[chunk[:,0],chunk[:,1]]).cpu()
                mask_all[origin+j:origin+j+n]=valid.cpu();labels[origin+j:origin+j+n]=Y.cpu()
                if qualifier is None:
                    # Same public fixture through the deployed single-record path.
                    m=M[0,:128].cpu();h=H[0,:128].cpu();positions=torch.arange(128)
                    if bool(m.all()):
                        deployed=_predict_row_package(model,E,h,m,positions,device=torch.device('cuda'))
                        direct=model.logits_from_rows(z,torch.zeros(127,dtype=torch.long,device='cuda'),torch.arange(1,128,device='cuda'),E).argmax(1).cpu()
                        assert torch.equal(deployed[1:],direct),'Batch geometry changes deployed predictions'
                        qualifier=dict(record=origin+j,deployed_predictions_exact=True,training_batch_geometry=[n,192,2048])
                rows_seen+=n;guard.check()
                if rows_seen%512==0:print(json.dumps(dict(records=rows_seen,elapsed=time.perf_counter()-clock)),flush=True)
            del tensors
    validloss=losses[mask_all];errs=int((~correct&mask_all).sum());assert torch.isfinite(validloss).all() and rows_seen==limit
    freq=torch.bincount(labels[mask_all],minlength=128256)
    result=dict(status='PUBLIC_FITTING_DIFFICULTY_SCORED',started_utc=started,ended_utc=utc(),
                records=rows_seen,positions=int(mask_all.sum()),errors=errs,accuracy=1-errs/int(mask_all.sum()),
                mean_loss=float(validloss.mean()),loss_quantiles={str(q):float(torch.quantile(validloss,q)) for q in [0.,.5,.9,.95,.99,1.]},
                counts_loss_above={str(q):int((validloss>q).sum()) for q in [.01,.05,.1,.5,1.]},
                model_load_seconds=load_seconds,external_seconds=time.perf_counter()-clock,
                resources=guard.check(),geometry_qualifier=qualifier,inputs=artifact(inputs_path),
                scored_bank_payloads=checked,provenance=provenance())
    result['difficulty']=save(out/'difficulty.safetensors',dict(losses=losses,correct=correct,valid=mask_all,labels=labels,frequency=freq))
    write_json(out/'receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['scored_bank_payloads','provenance']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['smoke','bank-check']);p.add_argument('--inputs',default='experiments/TRR-0013/inputs.json');p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=12000)
    a=p.parse_args();torch.set_num_threads(1)
    if a.command=='smoke':synthetic_smoke(a.output)
    else:bank_check(a.inputs,a.output,a.limit)


if __name__=='__main__':
    main()
