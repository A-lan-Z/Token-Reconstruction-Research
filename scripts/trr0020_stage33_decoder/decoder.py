"""Greedy causal reconstruction over full-vocabulary probability states."""
from pathlib import Path
import sys,time,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage33_replay"))
from engine import CurrentVocabulary,n,torch,F,forward,commit
CONFIGS=[(f"causal_factor{factor}_steps{steps}",factor,steps) for factor in [2.,1.,.5] for steps in [8,4,2,1]]
AUX_KEYS={"soft_error","confidence","update_trace"}
class CausalDecoder(CurrentVocabulary):
    @torch.no_grad()
    def prepare_all(self,guard):
        start=time.perf_counter()
        for pos in [127]+list(range(127)):self.ensure(pos);guard()
        return time.perf_counter()-start
    @torch.no_grad()
    def decode(self,observations,factor,steps,replay=True):
        if not 2<=len(observations)<=128 or not 1<=steps<=8:raise ValueError("unqualified geometry")
        self.metric._check()
        if len(self.graphs)!=128:raise RuntimeError("prepare every native graph first")
        n.sync();start=time.perf_counter()
        h=observations.to("cuda").float();length=len(h);self.factor.fill_(factor)
        pe=self.prefix.rotary_emb(h[None],torch.arange(length,device="cuda")[None]);cos,sin=pe[0][0],pe[1][0]
        for key,value in self.past:key.zero_();value.zero_()
        tokens=torch.full((length,),128000,device="cuda",dtype=torch.long)
        errors=torch.zeros(length,device="cuda");confidence=torch.ones(length,device="cuda")
        trace=torch.zeros(length-1,8,5,device="cuda");finite_logits=torch.ones((),dtype=torch.bool,device="cuda")
        self.co.copy_(cos[:1]);self.si.copy_(sin[:1]);self.token.fill_(128000)
        if replay:self.graphs[0][2].replay()
        else:self.commit_token(0)
        for pos in range(1,length):
            self.h.copy_(h[pos:pos+1]);self.co.copy_(cos[pos:pos+1]);self.si.copy_(sin[pos:pos+1])
            if replay:self.init_graph.replay()
            else:self.initialize()
            training,evaluating,committing=self.graphs[pos]
            for _ in range(steps):
                if replay:training.replay()
                else:self.train(pos)
            if replay:evaluating.replay()
            else:self.evaluate(pos)
            finite_logits.logical_and_(torch.isfinite(self.z).all())
            tokens[pos:pos+1].copy_(self.token);errors[pos:pos+1].copy_(self.final_error)
            confidence[pos:pos+1].copy_(self.confidence);trace[pos-1].copy_(self.trace)
            if pos<length-1:
                if replay:committing.replay()
                else:self.commit_token(pos)
        output={"tokens":tokens.cpu(),"soft_error":errors.cpu(),"confidence":confidence.cpu(),"update_trace":trace.cpu()}
        numeric=bool(finite_logits) and all(bool(torch.isfinite(v).all()) for v in output.values())
        n.sync();seconds=time.perf_counter()-start
        return output,{"total_seconds":seconds,"factor":factor,"steps":steps,"positions":length,"replay":replay,
          "prefix_forwards":(length-1)*(steps+1),"prefix_vjps":(length-1)*steps,"history_commit_forwards":length-1,
          "mixture_products":(length-1)*(steps+1),"probability_gradient_products":(length-1)*steps,
          "initialization_vocabulary_products":length-1,"vocabulary":self.vocab,"shortlist_size":None,
          "separate_candidate_verification_calls":0,"model_parameter_updates":0,"numeric_valid":numeric,
          "prefix_arithmetic":"FP32 native current-token map; TF32 disabled",
          "budget_rule":"clamp(factor * current observed cosine error,0,1),four scalar iterations",
          "timing_scope":"input transfer+full-vocabulary initialization+all causal updates/evaluations+emitted-history commits+outputs+sync; prior asset/table/capture preparation separate",
          "last_token_commit_skipped":True}
    @torch.no_grad()
    def history_reference(self,tokens):
        length=len(tokens);ids=tokens.to("cuda");base=self.E.index_select(0,ids)
        pe=self.prefix.rotary_emb(base[None],torch.arange(length,device="cuda")[None]);cos,sin=pe[0][0],pe[1][0]
        past=[(torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda"),torch.empty(p["kv_heads"],0,p["head_dim"],device="cuda")) for p in self.layers]
        for pos in range(length-1):
            _,_,current=forward(base[pos:pos+1],self.layers,past,cos[pos:pos+1],sin[pos:pos+1]);past=commit(past,current)
        rows=[]
        for layer,((ak,av),(rk,rv)) in enumerate(zip(self.past,past)):
            actual=[ak[:,:length-1],av[:,:length-1]];reference=[rk,rv]
            equal=all(torch.equal(a,b) for a,b in zip(actual,reference))
            rows.append({"layer":layer,"exact":equal,
              "actual_sha256":[hashlib.sha256(a.contiguous().cpu().numpy().tobytes()).hexdigest() for a in actual],
              "reference_sha256":[hashlib.sha256(a.contiguous().cpu().numpy().tobytes()).hexdigest() for a in reference]})
        return {"passed":all(v["exact"] for v in rows),"layers":rows,"scope":"independent sequential commit from emitted IDs only; no source labels"}
