"""Independent integration checks using a full causal sequence computation."""
from support import *
from linearized_prefix import forward as full_forward
@torch.no_grad()
def causal_reference(engine,data):
    ids=data["tokens"].to("cuda");length=len(ids);checks=[]
    for pos in [1,length//2,length-1]:
        x=engine.weight[ids[:pos+1]].clone()
        x[pos].copy_(data["initial_embedding"][pos-1].to("cuda"))
        cos,sin=engine.prefix.rotary_emb(x[None],torch.arange(pos+1,device="cuda")[None])
        prediction,_=full_forward(x,engine.layers,cos[0],sin[0])
        actual=data["first_prediction"][pos-1].to("cuda")
        reference=prediction[pos]
        close=bool(torch.allclose(actual,reference,rtol=5e-4,atol=1e-4))
        scores=engine.score_tables(data["embedding_final"][pos-1:pos].to("cuda"))["white_cosine"]
        readout_equal=bool(torch.equal(scores.argmax(-1),ids[pos:pos+1]))
        checks.append({"position":pos,"max_absolute_error":float((actual-reference).abs().max()),
          "relative_l2_error":float((actual-reference).norm()/reference.norm()),"close":close,"readout_equal":readout_equal,
          "reference_selected_values":reference[:16].cpu().tolist(),"actual_selected_values":actual[:16].cpu().tolist()})
    commits=bool(torch.equal(data["committed_tokens"],data["tokens"][:-1]))
    lengths=bool(torch.equal(data["cache_lengths"],torch.arange(1,length)))
    return {"checks":checks,"committed_ids_equal":commits,"cache_lengths_equal":lengths,
      "passed":commits and lengths and all(v["close"] and v["readout_equal"] for v in checks),
      "rtol":5e-4,"atol":1e-4,"scope":"full-sequence forward over emitted context plus initial continuous current row; not a truth-prefix reference"}
