"""Bounded CPU verification of preserved loader, scores, and proposer port."""
from pathlib import Path
import argparse
import json
import resource
import time
import torch
from scripts.agent3_shortlists.core import binding, rank_scores, write_json
from scripts.agent3_shortlists.predict import b1_logits, configure, guard, load_models, utc


def main(a):
    t=time.monotonic(); start=utc(); device=torch.device('cpu'); configure(device);guard(t,device)
    package,model,readout,lens=load_models(a.package,a.lens,a.reference,device)
    h,mask,pos,slots=package._observation_batch(Path(a.package)/'smoke/public_base_first2.safetensors')
    comparisons=[]
    with torch.inference_mode():
        for i in range(len(h)):
            scores=b1_logits(model,readout,h[i],mask[i],pos[i])
            ids,_=rank_scores(scores)
            native=package._predict_row_package(model,readout,h[i],mask[i],pos[i],device=device)
            assert torch.equal(ids[:,0],native[1:])
            # Check batched vs prescribed record1; only record1 is used in production.
            comparisons.append({'slot':slots[i],'package_row_top1_equal':True,'logits_sha256':package.tensor_digest(scores)})
        # Full native historical proposal flattens one record and processes chunk256.
        flat=h[0][mask[0].bool()][1:]
        native_a1=lens(flat[:256],readout).float()
        port_a1=lens(h[0,1:],readout).float()
        assert torch.equal(native_a1,port_a1)
        expected_a1=torch.topk(native_a1,512,dim=-1,sorted=True).indices[:,:256]
        ranked_a1,_=rank_scores(port_a1)
        # Ties may differ from topk; report rather than silently substitute.
        a1_rank_diffs=int((expected_a1!=ranked_a1).sum())
        staged=h[:2].float()
        projected=model.projected_hidden(staged,mask[:2].bool())
        row=torch.arange(2).repeat_interleave(127)
        positions=pos[:2,1:].reshape(-1)
        batched=model.logits_from_rows(projected,row,positions,readout).reshape(2,127,-1)
        row_scores=torch.stack([b1_logits(model,readout,h[i],mask[i],pos[i]) for i in range(2)])
        batched_equal=torch.equal(batched,row_scores)
        top1_equal=torch.equal(batched.argmax(-1),row_scores.argmax(-1))
        rank_equal=all(torch.equal(rank_scores(batched[i])[0],rank_scores(row_scores[i])[0]) for i in range(2))
        repeat=b1_logits(model,readout,h[0],mask[0],pos[0])
        assert torch.equal(repeat,row_scores[0])
        del batched,row_scores,repeat,scores,native_a1,port_a1
    guard(t,device)
    result={'status':'PASS_PRESERVED_RECORD1_PATH','start_utc':start,'end_utc':utc(),'seconds':time.monotonic()-t,
            'records':comparisons,'a1_native_one_record_scores_equal':True,'a1_top256_order_differences_from_topk512':a1_rank_diffs,
            'record2_vs_record1':{'full_scores_equal':batched_equal,'top1_equal':top1_equal,'top256_equal':rank_equal,
                                  'production_uses':'preserved one-record path regardless; batch2 probe not used as scientific output'},
            'repeat_scores_exact':True,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'truth_opened':False,'gpu_used':False,'package_manifest':binding(Path(a.package)/'package_manifest.json'),
            'numeric_settings':{'dtype':'float32','record_batch':1,'threads':2,'tf32':False,'matmul_precision':'highest'}}
    write_json(a.output,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('package','lens','reference','output'):p.add_argument('--'+k,required=True)
    main(p.parse_args())
