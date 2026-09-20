"""Freeze all public history-mixing cases before calculating identity curves."""
from pathlib import Path
import json,hashlib,time,torch,statistics,subprocess,sys
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
MODES=["control","aa_logit_clip1","aa_probability_clip1","aa_probability_clip2"]
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def sources(binding):
    for k,v in binding.items():
        if k=="sources":
            for name,digest in v.items():
                if sha(ROOT/name)!=digest:raise RuntimeError("source changed")
        elif isinstance(v,dict):sources(v)
def main():
    phases=[];rows=[];b=None
    for mode in MODES:
        path=X/f"dev47_public_{mode}.json";q=json.loads(path.read_text())
        if not q["passed"] or len(q["runs"])!=6 or len(q["eager"])!=2 or len(q["gradients"])!=2:raise RuntimeError("incomplete public mode")
        if not all(v["passed"] for v in q["gradients"]):raise RuntimeError("gradient check")
        if b is None:b=q["binding"]
        if q["binding"]!=b:raise RuntimeError("binding differs")
        sources(q["binding"])
        if {(v["length"],v["rep"]) for v in q["runs"]}!={(n,r) for n in [128,40] for r in range(3)}:raise RuntimeError("coverage")
        for r in q["runs"]:
            if not r["repeat_exact"] or not r["anchor_exact"] or not r["stats"]["numeric_valid"]:raise RuntimeError("invalid output")
        for r in q["eager"]:
            if not r["bytes_equal"]:raise RuntimeError("eager changed")
        for r in q["runs"]+q["eager"]:
            if sha(ROOT/r["path"])!=r["sha256"]:raise RuntimeError("raw changed")
        phases.append({"path":str(path.relative_to(ROOT)),"sha256":sha(path),"data":q})
    freeze={"task_id":"TRR-0020","phases":[{k:v for k,v in q.items() if k!="data"} for q in phases],"binding":b,"public_cells":8,"public_repeated_decodes":24,"eager_checks":8,"frozen_unix":time.time()}
    with (X/"dev47_public_freeze.json").open("x") as f:json.dump(freeze,f,indent=2)
    ids=torch.randint(256,128000,(128,),generator=torch.Generator().manual_seed(200051));ids[0]=128000
    for phase in phases:
        q=phase["data"]
        for length in [128,40]:
            runs=[r for r in q["runs"] if r["length"]==length];first=runs[0];out=load_file(str(ROOT/first["path"]))
            curve={}
            for key,value in out.items():
                if key.startswith("step"):
                    ok=value[1:]==ids[1:length];curve[key]={"correct":int(ok.sum()),"scored":len(ok)}
            rows.append({"mode":q["mode"],"length":length,"curve":curve,"median64_seconds":statistics.median(r["stats"]["total_seconds"] for r in runs),
              "observed_error":first["stats"]["mirror_traces"]["loss"][-1],
              "history_coefficients":out["aa_trace"][:,0].tolist() if "aa_trace" in out else None})
    result={"task_id":"TRR-0020","scope":"known public numerical fixture identity curves; not benchmark accuracy",
      "rows":rows,"peak_reserved_GiB":max(p["data"]["peak_reserved"] for p in phases)/2**30,
      "freeze_sha256":sha(X/"dev47_public_freeze.json"),"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"created_unix":time.time()}
    with (X/"dev47_public_score.json").open("x") as f:json.dump(result,f,indent=2)
    for row in rows:print(json.dumps({k:v for k,v in row.items() if k!="history_coefficients"}))
if __name__=="__main__":main()
