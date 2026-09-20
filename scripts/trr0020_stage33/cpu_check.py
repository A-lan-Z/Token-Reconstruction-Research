"""Independent autograd checks for the causal full-vocabulary chain rule."""
import torch,json,traceback
from probability_gradient import gradient
def reference():
 torch.set_num_threads(2);torch.manual_seed(200073);checks=[];failures=[]
 for dtype in [torch.float64,torch.float32]:
  for vocab,width in [(17,7),(257,13)]:
   for kind in ["linear","residual_tanh","zero","near_zero"]:
    try:
     E=torch.randn(vocab,width,dtype=dtype)
     if kind=="zero":E.zero_()
     if kind=="near_zero":E*=1e-14
     z=torch.randn(1,vocab,dtype=dtype);p=z.softmax(-1).detach().requires_grad_()
     W=torch.randn(width,width,dtype=dtype)/width**.5
     U=torch.randn(width,width,dtype=dtype)/width**.5
     past=torch.randn(1,width,dtype=dtype)*.3
     x=p@E
     if kind=="residual_tanh":
      act=torch.tanh(x@W.T+past);out=x+act@U.T
      vjp=lambda d:d+((d@U)*(1-act.detach().square()))@W
     else:
      out=x@W.T;vjp=lambda d:d@W
     target=torch.randn_like(out)
     loss=1-(torch.nn.functional.normalize(out,dim=-1)*torch.nn.functional.normalize(target,dim=-1)).sum()
     expected=torch.autograd.grad(loss,p)[0]
     actual,info=gradient(out.detach(),target,E,vjp)
     tol=2e-11 if dtype==torch.float64 else 2e-5
     torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
     torch.testing.assert_close(info["loss"][0],loss.detach(),rtol=tol,atol=tol)
     logits=z.detach().requires_grad_();pp=logits.softmax(-1);xx=pp@E
     oo=xx+torch.tanh(xx@W.T+past)@U.T if kind=="residual_tanh" else xx@W.T
     objective=1-(torch.nn.functional.normalize(oo,dim=-1)*torch.nn.functional.normalize(target,dim=-1)).sum()
     exact=torch.autograd.grad(objective,logits)[0]
     chained=pp.detach()*(actual-(actual*pp.detach()).sum(-1,keepdim=True))
     torch.testing.assert_close(chained,exact,rtol=tol,atol=tol)
     if not torch.isfinite(actual).all():raise RuntimeError("nonfinite")
     checks.append({"dtype":str(dtype),"vocabulary":vocab,"width":width,"map":kind,
       "probability_gradient_max_error":float((actual-expected).abs().max()),
       "logit_gradient_max_error":float((chained-exact).abs().max()),"fixed_history":True})
    except Exception:failures.append({"dtype":str(dtype),"vocabulary":vocab,"map":kind,"failure":traceback.format_exc()})
 return {"passed":not failures,"seed":200073,"checks":checks,"failures":failures,
  "scope":"tiny synthetic linear/nonlinear chain-rule qualification; no real-prefix or reconstruction claim"}
if __name__=="__main__":
 r=reference();print(json.dumps(r));raise SystemExit(0 if r["passed"] else 1)
