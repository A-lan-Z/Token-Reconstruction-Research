"""Development-only qualification on four already-opened PR25 clips."""
import argparse,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from scripts.agent3_shortlists.core import binding,verify,write_json
from scripts.agent3_hybrid.runtime import *

def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);configure();start=time.perf_counter()
    r=Resources();old=ROOT.parent/'agent3-b1-small-budget-a2/outputs/agent3-b1-small-budget-a2'
    results=[]
    for domain in ('pile','finance'):
        for stage in (0,256):
            cpath=old/'inputs'/f'{domain}-{stage}'/'contract.json';c=json.loads(cpath.read_text());t=load_file(verify(c['observations']))
            h,m,p=[t[k][:1] for k in ('activations','attention_mask','position_ids')]
            records=[]
            for method in METHODS:
                prediction,trace,times,calls=warmed(r,method,h,m,p)
                native_adapter=Adapter(r,method,instrument=False)
                other=native_adapter(h[0].cuda(),m[0].cuda(),p[0].cuda()).cpu()
                assert torch.equal(prediction[0],other),'instrumentation changed native output'
                if method=='a1_k256':
                    anchor=legacy._A2Adapter(precut=r.prefix,lens=r.lens,embeddings=r.readout,device=r.device,policy=policy(256))
                    expected=anchor(h[0].cuda(),m[0].cuda(),p[0].cuda()).cpu()
                    assert torch.equal(prediction[0],expected),'historical anchor mismatch'
                artifact=out/f'{domain}-{stage}-{method}.safetensors';save_file(trace,str(artifact))
                record={'method':method,'artifact':binding(artifact),'timing':times,'phases':calls,'instrumented_equals_native':True,'historical_anchor_equal':True if method=='a1_k256' else None,'peak_resources':resources()}
                if method=='b1_k256':
                    cpu=load_file(old/'predictions'/f'{domain}-{stage}'/'b1.safetensors')
                    record['pr25_cpu_gpu_top1_differing_positions']=int((cpu['predictions'][0]!=prediction[0]).sum())
                    record['pr25_cpu_gpu_ranked_candidate_differing_entries']=int((cpu['candidates'][0]!=trace['candidates']).sum())
                    record['cpu_gpu_equivalence_assumed']=False
                records.append(record);print(json.dumps({'domain':domain,'stage':stage,'method':method,'status':'PASS','warmed_seconds':times['records'][0]['measured_seconds']}),flush=True)
            results.append({'domain':domain,'stage':stage,'source_contract':binding(cpath),'record_id':c['record_ids'][0],'methods':records})
    evidence={'status':'PASS_DEVELOPMENT_INTEGRATION_QUALIFICATION','task_id':'agent3-static-prefix-hybrid','created_utc':datetime.now(timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cells':results,'native_resources':r.native_evidence,'load_seconds':r.load_seconds,'wall_seconds':time.perf_counter()-start,'peak_resources':resources(),'source_truth_read':False,'source_scope':'first already-opened PR25 record/domain at base and256; not fresh confirmation','static_public_prefix':True}
    write_json(out/'qualification.json',evidence)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args())
