"""Candidate-free sequence inversion; no separately learned parameters."""
import time
import torch
import torch.nn.functional as F

class SequenceInverse:
    def __init__(self, prefix):
        if any(p.requires_grad for p in prefix.parameters()):
            raise ValueError("prefix must be frozen")
        self.prefix = prefix
        # FP32 is an explicit approximate inverse, not native BF16 equivalence.
        self.prefix.float()
        self.weight = prefix.embed_tokens.weight.detach()
        self.norms = self.weight.square().sum(-1)
        self.median_rms = self.norms.median().sqrt() / self.weight.shape[1]**.5
        self.normalized = F.normalize(self.weight, dim=-1)
        self.bos = self.weight[128000].view(1,1,-1)
        self.signature = tuple((id(p), p._version) for p in prefix.parameters())

    def check(self):
        if self.signature != tuple((id(p),p._version) for p in self.prefix.parameters()):
            raise ValueError("prefix changed: reconstruct engine")

    def inputs(self, h):
        pos = torch.arange(h.shape[1],device=h.device).view(1,-1)
        return self.prefix.rotary_emb(h,pos), self.prefix._causal_mask(h,start_pos=0,total_tokens=h.shape[1])

    def forward(self, h, pe, mask):
        for layer in self.prefix.layers:
            h = self.prefix._hidden(layer(h,position_embeddings=pe,attention_mask=mask,use_cache=False))
        return h

    @torch.no_grad()
    def project(self, z, metric="euclidean"):
        z = z.reshape(-1,z.shape[-1])
        if metric == "cosine":
            ids = (F.normalize(z,dim=-1) @ self.normalized.T).argmax(-1)
        else:
            ids = (2*z@self.weight.T-self.norms).argmax(-1)
        ids[0] = 128000
        return ids

    def lbfgs(self, target, known, fn, steps):
        z = target[:,1:].clone().detach().requires_grad_()
        opt = torch.optim.LBFGS([z],lr=1.,max_iter=steps,max_eval=steps*2,
                               tolerance_grad=1e-8,tolerance_change=1e-10,
                               history_size=8,line_search_fn="strong_wolfe")
        losses=[]; best=[float("inf"),z.detach().clone()]
        def closure():
            opt.zero_grad(set_to_none=True)
            y = fn(torch.cat([known,z],dim=1))
            loss=(y[:,1:]-target[:,1:]).square().sum()/z.shape[1]
            if not torch.isfinite(loss): raise RuntimeError("nonfinite inverse loss")
            loss.backward()
            if not torch.isfinite(z.grad).all(): raise RuntimeError("nonfinite inverse gradient")
            v=float(loss.detach())
            losses.append(v)
            if v<best[0]: best[:]=[v,z.detach().clone()]
            return loss
        opt.step(closure)
        return torch.cat([known,best[1]],dim=1).detach(),losses

    def reverse(self, h, steps):
        pe,mask=self.inputs(h)
        with torch.no_grad():
            b=self.bos; bosstates=[]
            bpe,_=self.inputs(b)
            for layer in self.prefix.layers:
                before=b
                middle=b+layer.self_attn(layer.input_layernorm(b),position_embeddings=bpe)[0]
                b=middle+layer.mlp(layer.post_attention_layernorm(middle))
                bosstates.append((before,middle,b))
        y=h;tr=[]
        for i in reversed(range(len(self.prefix.layers))):
            layer=self.prefix.layers[i]
            for part,known,fn in [
                ("mlp",bosstates[i][1],lambda x,ly=layer:x+ly.mlp(ly.post_attention_layernorm(x))),
                ("attention",bosstates[i][0],lambda x,ly=layer:x+ly.self_attn(ly.input_layernorm(x),position_embeddings=pe,attention_mask=mask)[0])
            ]:
                y,losses=self.lbfgs(y,known,fn,steps)
                tr.append({"layer":i,"part":part,"loss":losses,"forward_backward_evaluations":len(losses)})
        return y,tr

    def joint(self,h,steps,init,optimizer="adam",snap_every=0):
        pe,mask=self.inputs(h)
        if init is None:
            init=h*self.median_rms/h.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
            init[:,0]=self.bos[:,0]
        if optimizer=="lbfgs":
            # LBFGS function separates target from starting vector.
            z=init[:,1:].clone().detach().requires_grad_()
            opt=torch.optim.LBFGS([z],lr=1,max_iter=steps,max_eval=steps*2,
                tolerance_grad=1e-8,tolerance_change=1e-10,history_size=8,line_search_fn="strong_wolfe")
        else:
            z=init[:,1:].clone().detach().requires_grad_()
            opt=torch.optim.Adam([z],lr=.01)
        trace=[];best=[float("inf"),z.detach().clone()];projections=0
        def closure():
            opt.zero_grad(set_to_none=True)
            y=self.forward(torch.cat([self.bos,z],dim=1),pe,mask)
            loss=(y[:,1:]-h[:,1:]).square().sum()/z.shape[1]
            if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
            loss.backward()
            if not torch.isfinite(z.grad).all():raise RuntimeError("nonfinite gradient")
            val=float(loss.detach());trace.append(val)
            if val<best[0]:best[:]=[val,z.detach().clone()]
            return loss
        if optimizer=="lbfgs":
            opt.step(closure)
        else:
            for i in range(steps):
                closure();opt.step()
                if snap_every and (i+1)%snap_every==0:
                    with torch.no_grad():
                        ids=self.project(torch.cat([self.bos,z],dim=1))
                        z.copy_(self.weight[ids[1:]].unsqueeze(0))
                        opt.state.clear();projections+=1
        return torch.cat([self.bos,best[1]],dim=1).detach(),{"loss":trace,
                "forward_backward_evaluations":len(trace),"intermediate_full_vocab_projections":projections}

    def decode(self, observation, method):
        self.check();torch.cuda.synchronize();start=time.perf_counter()
        h=observation.to(self.weight.device).float().unsqueeze(0)
        prep=time.perf_counter()-start
        if method.startswith("reverse"):
            steps=int(method.removeprefix("reverse"));z,tr=self.reverse(h,steps)
            evaluations=sum(x["forward_backward_evaluations"] for x in tr)/8
        else:
            optimizer,steps=method.split("_");steps=int(steps)
            z,tr=self.joint(h,steps,None,optimizer)
            evaluations=tr["forward_backward_evaluations"]
        torch.cuda.synchronize();solved=time.perf_counter()
        out={metric:self.project(z,metric).cpu() for metric in ("euclidean","cosine")}
        torch.cuda.synchronize();end=time.perf_counter()
        return out,{"total":end-start,"input":prep,"solve":solved-start-prep,
                    "projection_and_output":end-solved,
                    "continuous_sequence_forward_backward_equivalents":evaluations,
                    "discrete_candidate_simulations":0,"trace":tr}

