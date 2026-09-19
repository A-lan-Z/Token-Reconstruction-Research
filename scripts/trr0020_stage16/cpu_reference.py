"""Exact small CPU checks for target-independent normalized-prefix curvature."""
import itertools,json
import torch
torch.set_num_threads(2)

def reference():
    torch.manual_seed(200042)
    dtype=torch.float64
    a=torch.randn(4,5,dtype=dtype);b=torch.randn(4,dtype=dtype)
    x=torch.randn(5,dtype=dtype,requires_grad=True)
    r=torch.randn(5,5,dtype=dtype)+3*torch.eye(5,dtype=dtype)
    ri=torch.linalg.inv(r)
    def function(z):
        y=torch.tanh(a@z+b)
        return y/y.norm()
    j=torch.autograd.functional.jacobian(function,x)
    signs=torch.tensor(list(itertools.product([-1.,1.],repeat=4)),dtype=dtype)
    rows=[]
    for name,transform in [("raw",torch.eye(5,dtype=dtype)),("white",ri.T)]:
        gradients=[]
        for s in signs:
            g=torch.autograd.grad((function(x)*s).sum(),x)[0]
            gradients.append(g@transform)
        gradients=torch.stack(gradients)
        estimate=gradients.square().sum(-1).mean()/5
        exact=(j@transform).square().sum()/5
        torch.testing.assert_close(estimate,exact,rtol=1e-12,atol=1e-12)
        rows.append({"metric":name,"all_sign_expectation":float(estimate),
          "explicit_normalized_jacobian_trace_per_dimension":float(exact),
          "absolute_error":float((estimate-exact).abs())})
    h=torch.randn(4,dtype=dtype);h=h/h.norm()
    loss=1-(function(x)*h).sum()
    equivalent=.5*(function(x)-h).square().sum()
    torch.testing.assert_close(loss,equivalent,rtol=1e-12,atol=1e-12)
    e=torch.randn(23,5,dtype=dtype);gradient=torch.randn(5,dtype=dtype)
    curvature=rows[1]["all_sign_expectation"]
    score=(curvature*(x@r@r.T)-gradient)@e.T-.5*curvature*(e@r).square().sum(-1)
    delta=e-x
    dense=-(delta*gradient).sum(-1)-.5*curvature*(delta@r).square().sum(-1)
    torch.testing.assert_close(score-score[0],dense-dense[0],rtol=1e-12,atol=1e-12)
    return {"seed":200042,"device":"cpu","dtype":"float64","sign_vectors":16,"tests":rows,
      "cosine_squared_normalized_residual_identity":True,"full_vocabulary_quadratic_identity":True,
      "target_not_used_by_curvature_estimator":True,"passed":True}
if __name__=="__main__":print(json.dumps(reference(),indent=2))
