"""A1 top256 + direct-cosine A2. Native cached candidate helper; short-record geometry port."""
import time
import torch
from torch.nn import functional as F
from token_reconstruction.historical_inputlens_bridge import load_historical_lens_checkpoint
from common import OUT
from solver import sync
from token_reconstruction.a1a2_configuration_search import _candidate_hidden

def prepare(prefix):
    lens=load_historical_lens_checkpoint(OUT/'backup/lens_alpaca.pt',device=prefix.embed_tokens.weight.device)
    # Historical reference uses normalized FP32 public embeddings.
    embeddings=F.normalize(prefix.embed_tokens.weight.detach().float(),dim=-1)
    return lens,embeddings

@torch.no_grad()
def decode(prefix,observation,lens,embeddings,guard=None):
    device=embeddings.device;tokens=[128000];trace=[];sync(device);start=time.perf_counter()
    cache=prefix.new_cache();prefix.run_cached(torch.tensor([[128000]],device=device),cache,0)
    for pos in range(1,len(observation)):
        if guard:guard()
        sync(device);begin=time.perf_counter();target=observation[pos].to(device).float()
        candidates=lens.topk(target.unsqueeze(0),embeddings,k=256)[0].tolist()
        candidates_tensor=torch.tensor([candidates],device=device)
        predicted=_candidate_hidden(prefix,cache=cache,parent_indices=torch.tensor([0],device=device),candidate_ids=candidates_tensor,position=pos)[0]
        cosines=F.cosine_similarity(predicted,target.unsqueeze(0),dim=-1)
        residuals=(predicted-target).square().mean(-1)
        checks=[{'token':c,'cosine':float(cosines[i]),'mse':float(residuals[i])} for i,c in enumerate(candidates)]
        best=checks[int(cosines.argmax())];tokens.append(best['token'])
        prefix.run_cached(torch.tensor([[best['token']]],device=device),cache,pos);sync(device)
        trace.append({'position':pos,'token':best['token'],'checks':checks,'best_mse':best['mse'],'seconds':time.perf_counter()-begin,'gradient_steps':0,'vocabulary_scans':1,'vocabulary_rows_scored':len(embeddings),'restarts':0,'reason':'fixed_k256'})
    sync(device)
    return {'tokens':tokens,'trace':trace,'seconds':time.perf_counter()-start,'method':'a1_a2_exhaustive_configuration_winner','port':'batch1; unpadded short records; native cached candidate execution; same K256 and direct cosine decision','preparation':'retained public Alpaca lens; historical fitting cost not remeasured'}
