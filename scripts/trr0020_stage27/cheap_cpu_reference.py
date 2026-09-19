"""Dense formula checks and exact one-CGLS-step output equivalence."""
import json,torch
from cheap_update import direction,RULES
from stable_cgls import least_squares
def reference():
    torch.set_num_threads(2);torch.manual_seed(200066);checks=[]
    for dtype in [torch.float64,torch.float32]:
        a=torch.randn(4,8,8,dtype=dtype)*.2+torch.eye(8,dtype=dtype)
        b=torch.randn(4,8,dtype=dtype);b[0]=0
        mv=lambda v:(a@v[...,None])[...,0]
        rmv=lambda v:(a.transpose(-1,-2)@v[...,None])[...,0]
        for rule in RULES:
            value,work=direction(rule,b,mv,rmv)
            if rule=="cg1":
                expected,_=least_squares(mv,rmv,b,1,.0001)
                if not torch.equal(value,expected):raise RuntimeError("one-step execution differs")
                equal=True
            else:
                g=(a.transpose(-1,-2)@b[...,None])[...,0]
                alpha=float(rule.removeprefix("polyak"))*b.square().sum(-1)/g.square().sum(-1).clamp_min(torch.finfo(dtype).tiny)
                alpha[0]=0;expected=alpha[:,None]*g;equal=torch.equal(value,expected)
                torch.testing.assert_close(value,expected,rtol=1e-6,atol=1e-7)
            assert torch.equal(value[0],torch.zeros_like(value[0]))
            checks.append({"dtype":str(dtype),"rule":rule,"max_error":float((value-expected).abs().max()),"exact":equal,"zero_rhs_preserved":True,**work})
    return {"passed":True,"seed":200066,"checks":checks,"scope":"small algebra only; no reconstruction evidence"}
if __name__=="__main__":print(json.dumps(reference()))
