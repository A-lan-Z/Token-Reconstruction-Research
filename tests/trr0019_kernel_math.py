"""Independent dense-math check of shared-context attention; no model or labels."""
from pathlib import Path
import sys,json,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/trr0019'))
import torch
from shared_attention import shared_attention

def main():
    torch.set_num_threads(2);torch.manual_seed(1919064)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cases=[];start=time.time()
    for length in [1,39,127]:
        q=torch.randn(256,32,1,64,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(256,8,1,64,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        pastk=torch.randn(1,8,length,64,device='cuda',dtype=torch.bfloat16)
        pastv=torch.randn_like(pastk);savedk=pastk.clone();savedv=pastv.clone()
        actual=shared_attention(q,k,v,pastk,pastv)
        keys=torch.cat([pastk.expand(256,-1,-1,-1),k],2).repeat_interleave(4,1).float()
        values=torch.cat([pastv.expand(256,-1,-1,-1),v],2).repeat_interleave(4,1).float()
        logits=(q.float()@keys.transpose(-1,-2))/8
        reference=(logits.softmax(-1)@values).transpose(1,2).to(torch.bfloat16)
        torch.testing.assert_close(actual.float(),reference.float(),rtol=0.0078125,atol=0.0001)
        assert torch.equal(pastk,savedk) and torch.equal(pastv,savedv)
        k[0].add_(1);v[0].add_(2)
        changed=shared_attention(q,k,v,pastk,pastv)
        assert torch.equal(changed[1:],actual[1:])
        assert not torch.equal(changed[0],actual[0])
        cases.append({'past_positions':length,'candidate_count':256,
          'fp32_reference_max_abs':float((actual.float()-reference.float()).abs().max()),
          'bf16_reference_different_values':int((actual!=reference).sum()),
          'reference_comparison_rtol':0.0078125,'reference_comparison_atol':0.0001,
          'past_unchanged':True,'candidate_isolation':True})
        del q,k,v,pastk,pastv,savedk,savedv,actual,keys,values,logits,reference,changed
    torch.cuda.synchronize()
    result={'status':'PASS','seed':1919064,'cases':cases,'torch':torch.__version__,'command':sys.argv,
      'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'seconds':time.time()-start,'peak_allocated':torch.cuda.max_memory_allocated(),
      'peak_reserved':torch.cuda.max_memory_reserved(),'truth_or_model_loaded':False}
    path=ROOT/'experiments/TRR-0019/kernel_math_test.json'
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()