"""A1 top256 + direct-cosine A2. Full-prefix, scalar candidate geometry port."""
import time
import torch
from torch.nn import functional as F
from token_reconstruction.historical_inputlens_bridge import load_historical_lens_checkpoint
from common import OUT
from solver import sync

def prepare(prefix):
    lens=load_historical_lens_checkpoint(OUT/'backup/lens_alpaca.pt',device=prefix.embed_tokens.weight.device)
    # Historical reference uses normalized FP32 public embeddings.
    embeddings=F.normalize(prefix.embed_tokens.weight.detach().float(),dim=-1)
    return lens,embeddings

@torch.no_grad()
def decode(prefix,observation,lens,embeddings,guard=None):
    device=embeddings.device;tokens=[128000];trace=[];sync(device);start=time.perf_counter()
    for pos in range(1,len(observation)):
        if guard:guard()
        sync(device);begin=time.perf_counter();target=observation[pos].to(device).float()
        candidates=lens.topk(target.unsqueeze(0),embeddings,k=256)[0].tolist()
        checks=[]
        for candidate in candidates:
            predicted=prefix.forward_full(torch.tensor([tokens+[candidate]],device=device))[0,-1].float()
            cosine=float(F.cosine_similarity(predicted,target,dim=0))
            checks.append({'token':candidate,'cosine':cosine,'mse':float((predicted-target).square().mean())})
        best=max(checks,key=lambda c:c['cosine']);tokens.append(best['token']);sync(device)
        trace.append({'position':pos,'token':best['token'],'checks':checks,'best_mse':best['mse'],'seconds':time.perf_counter()-begin,'gradient_steps':0,'vocabulary_scans':1,'vocabulary_rows_scored':len(embeddings),'restarts':0,'reason':'fixed_k256'})
    sync(device)
    return {'tokens':tokens,'trace':trace,'seconds':time.perf_counter()-start,'method':'a1_a2_exhaustive_configuration_winner','port':'batch1; unpadded short records; full prefix recomputation; same K256 and direct cosine decision','preparation':'retained public Alpaca lens; historical fitting cost not remeasured'}
