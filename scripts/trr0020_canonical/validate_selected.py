from support import *
import os,gc
from discrete_soft import DiscreteSoftVocabulary
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("wrong deterministic environment")
    bound=binding();env=n.environment();prefix=n.load_prefix()
    old=json.loads((ROOT/"experiments/TRR-0020/dev7_freeze.json").read_text())
    data_path=ROOT.parent/"TRR-0014/outputs/TRR-0014/fresh_r1/observations.safetensors"
    if n.digest(data_path)!=old["binding"]["observations_sha256"]:raise RuntimeError("changed anchor observations")
    data=n.load_file(str(data_path));rows=[]
    (X/"qualifications").mkdir(exist_ok=True)
    for method,old_method,selected in [("gini128","gini003","step128"),("halfhard256","hardhalf64","best_objective")]:
        engine=DiscreteSoftVocabulary(prefix,"gini003",.003,0.,32) if method=="gini128" else DiscreteSoftVocabulary(prefix,"hardhalf64",0.,.5,64)
        if method=="gini128":engine.steps=128
        for anchor in old["entries"]:
            if anchor["method"]!=old_method:continue
            if n.digest(ROOT/anchor["path"])!=anchor["sha256"]:raise RuntimeError("changed anchor prediction")
            out,stats=engine.decode(data[anchor["id"]]);n.guard()
            reference=n.load_file(str(ROOT/anchor["path"]))[selected]
            actual=out[selected]
            path=X/"qualifications"/f'selected_{method}_{anchor["id"]}.safetensors';n.save_file({"tokens":actual},str(path))
            equal=torch.equal(actual,reference)
            loss_equal=stats["loss_trace"]==anchor["stats"]["loss_trace"][:engine.steps+1]
            row={"method":method,"id":anchor["id"],"path":str(path.relative_to(ROOT)),"sha256":n.digest(path),
              "anchor_path":anchor["path"],"anchor_sha256":anchor["sha256"],"selected_field":selected,"tokens_equal":equal,"loss_prefix_equal":loss_equal,"seconds":stats["total_seconds"]}
            rows.append(row);print("SELECTED_ANCHOR",method,anchor["id"],equal,loss_equal,flush=True)
            if not equal or not loss_equal:
                n.write(X/"selected_anchor_failure.json",{"binding":bound,"environment":env,"rows":rows});raise RuntimeError("selected rule differs from frozen development")
        del engine;gc.collect();torch.cuda.empty_cache()
    n.write(X/"selected_anchor_validation.json",{"binding":bound,"environment":env,"rows":rows,"passed":True,
      "peak_reserved":torch.cuda.max_memory_reserved(),"peak_allocated":torch.cuda.max_memory_allocated(),"observations_sha256":n.digest(data_path),"truth_read":False})
if __name__=="__main__":main()
