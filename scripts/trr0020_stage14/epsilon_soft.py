"""Full-vocabulary Adam inversion with an explicit denominator floor."""
import torch,triton
import triton.language as tl
from discrete_soft import DiscreteSoftVocabulary

CONFIGS=[(f"{mode}_lr{str(lr).replace('.','p')}",mode,lr)
 for mode in ["fixed1e12","fixed1e16","fixed1e20","length_scaled"] for lr in [.15,.3,.6]]

@triton.jit
def epsilon_update_kernel(Z,M,V,G,E,LR,DECAY,N:tl.constexpr,VOCAB:tl.constexpr,EPS:tl.constexpr,GRAD_SCALE:tl.constexpr,BLOCK:tl.constexpr):
    ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=ix<N
    z=tl.load(Z+ix,mask,0);m=tl.load(M+ix,mask,0);v=tl.load(V+ix,mask,0);g=tl.load(G+ix,mask,0)*GRAD_SCALE
    lr=tl.load(LR);decay=tl.load(DECAY);error=tl.load(E+ix//VOCAB,mask,0)
    lr=lr*tl.sqrt(tl.minimum(tl.maximum(error/.05,.01),1.))
    m=m+.1*(g-m);v=v+.005*(g*g-v)
    z=(z-lr*tl.div_rn(m,tl.sqrt(v)+EPS*GRAD_SCALE))*decay
    tl.store(Z+ix,z,mask);tl.store(M+ix,m,mask);tl.store(V+ix,v,mask)

def reference_tests():
    g=torch.tensor([1e-5,1e-10,1e-14,1e-18],dtype=torch.float64)
    m=.1*g;v=.005*g.square();lr=.6
    steps={str(eps):(lr*m/(v.sqrt()+eps)).tolist() for eps in [1e-12,1e-16,1e-20]}
    # Scaling a mean loss to a sum is equivalent to scaling Adam epsilon down,
    # when moments start at zero and all gradients share that constant scale.
    size=127.;eps=1e-12
    summed=lr*(size*m)/((size*size*v).sqrt()+eps)
    normalized=lr*m/(v.sqrt()+eps/size)
    torch.testing.assert_close(summed,normalized,rtol=1e-12,atol=1e-12)
    assert steps["1e-16"][2]>100*steps["1e-12"][2]
    return {"gradient_values":g.tolist(),"first_step_by_epsilon":steps,
      "loss_scaling_identity_max_error":float((summed-normalized).abs().max()),
      "scope":"CPU float64 optimizer identity only; no measured claim about model gradients or accuracy"}

class EpsilonVocabulary(DiscreteSoftVocabulary):
    def __init__(self,prefix,mode,lr):
        super().__init__(prefix,"gini003",.003,0.,32)
        self.epsilon_mode=mode;self.requested_lr=lr;self.steps=128
    def epsilon(self,length):
        return {"fixed1e12":1e-12,"fixed1e16":1e-16,"fixed1e20":1e-20,"length_scaled":1e-12/(length-1)}[self.epsilon_mode]
    @torch.no_grad()
    def reset(self,h):
        super().reset(h);self.lr_tensor.fill_(self.requested_lr)
    def train_step(self,length):
        self.gradient.zero_();self.evaluate(length).backward()
        count=(length-1)*self.vocab
        epsilon_update_kernel[(triton.cdiv(count,1024),)](self.logits,self.moment,self.variance,self.gradient,
          self.position_error,self.lr_tensor,self.decay_tensor,count,self.vocab,self.epsilon(length),1. if self.epsilon_mode=="fixed1e12" else 1e6,1024,
          num_warps=4,enable_fp_fusion=False)
    def decode(self,observations):
        outputs,stats=super().decode(observations)
        stats.update(epsilon_mode=self.epsilon_mode,adam_epsilon=self.epsilon(len(observations)),base_learning_rate=self.requested_lr,optimizer_gradient_scale=1. if self.epsilon_mode=="fixed1e12" else 1e6,optimizer_numerics="FUSED_FP32_NO_FMA_SCALED_STATES_WHEN_EPS_REDUCED")
        return outputs,stats
