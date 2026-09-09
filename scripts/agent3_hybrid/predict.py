"""One isolated sanitized observation cell, five fixed methods and warmed repeats."""
import argparse,json,subprocess,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from scripts.agent3_hybrid.common import *
from scripts.agent3_hybrid.runtime import configure,Resources,warmed,guard,resources,STATE_SHA,READOUT_SHA

def main(a):
    check_implementation();configure();start=time.monotonic();code=commit();c=json.loads(Path(a.contract).read_text())
    if set(c)!={'schema','domain','stage','record_ids','observations'} or c['schema']!='agent3-static-hybrid-observation-v1':raise ValueError('unsanitized contract')
    t=load_file(verify(c['observations']))
    if set(t)!={'activations','attention_mask','position_ids'}:raise ValueError('forbidden observation fields')
    h,m,p=[t[k] for k in ('activations','attention_mask','position_ids')]
    if h.shape!=(32,128,2048) or h.dtype!=torch.bfloat16 or not (m==1).all() or not torch.equal(p,torch.arange(128).expand(32,-1)):raise ValueError('observation geometry changed')
    if len(set(c['record_ids']))!=32:raise ValueError('source IDs not unique')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);r=Resources();rows={method:[] for method in METHODS};phases={method:[] for method in METHODS};times={method:[] for method in METHODS};peaks={};schedule=[]
    started=utc()
    for i in range(32):
        offset=i%len(METHODS);order=METHODS[offset:]+METHODS[:offset];schedule.append(list(order))
        for method in order:
            guard();torch.cuda.reset_peak_memory_stats()
            _,trace,timing,measurements=warmed(r,method,h[i:i+1],m[i:i+1],p[i:i+1])
            rows[method].append(trace);times[method].append(timing['records'][0]|{'record_index':i});phases[method].append(measurements)
            peak=resources();peaks[method]={k:max(peaks.get(method,{}).get(k,0),peak[k]) for k in ('peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','rss_peak_bytes')}
        print(json.dumps({'record_complete':i+1,'domain':c['domain'],'stage':c['stage'],'elapsed_seconds':time.monotonic()-start}),flush=True)
    methods=[];io_start=time.monotonic()
    for method in METHODS:
        artifact=out/(method+'.safetensors');tensors={k:torch.stack([row[k] for row in rows[method]]) for k in rows[method][0]};save_file(tensors,str(artifact))
        methods.append({'method':method,'artifact':binding(artifact),'timing_records':times[method],'phase_records':phases[method],'peak_resources':peaks[method]})
    if commit()!=code:raise ValueError('commit changed during scientific cell')
    check_implementation()
    receipt={'status':'FROZEN_CELL_NO_TRUTH','schema':'agent3-static-hybrid-cell-v1','code_commit':code,'implementation':binding(EV/'implementation-freeze.json'),'contract':binding(a.contract),'domain':c['domain'],'stage':c['stage'],'record_ids':c['record_ids'],'methods':methods,'arm_order_by_record':schedule,'start_utc':started,'end_utc':utc(),'wall_seconds':time.monotonic()-start,'public_resource_load_seconds':r.load_seconds,'output_io_seconds':time.monotonic()-io_start,'native_resource_evidence':r.native_evidence,'package_modules':r.package_modules,'state_sha256':STATE_SHA,'readout_sha256':READOUT_SHA,'target_weights_loaded':False,'source_truth_loaded':False,'static_public_prefix':True}
    write_json(out/'receipt.json',receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--output',required=True);main(p.parse_args())
