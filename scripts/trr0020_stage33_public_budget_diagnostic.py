"""Public-only descriptive audit of four-iteration KL budget utilization."""
from pathlib import Path
import hashlib,json,statistics,sys,time,subprocess
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    started=time.time();probe=X/"dev33_public_probe.json";public=json.loads(probe.read_text())
    if not public["passed"] or public["truth_read"]:raise RuntimeError("invalid public study")
    rows=[]
    for cell in public["cells"]:
        if cell["steps"]!=8:continue
        ref=cell["repetitions"][0];p=ROOT/ref["path"]
        if sha(p)!=ref["sha256"]:raise RuntimeError("changed public artifact")
        with safe_open(p,framework="pt",device="cpu") as f:
            direct=f.get_tensor("direct_kl").tolist()
            budget=f.get_tensor("requested_budget").tolist()
            trace=f.get_tensor("trace").tolist()
        if len(direct)!=8 or len(budget)!=8 or len(trace)!=8:raise RuntimeError("incomplete trace")
        ratios=[a/b if b>0 else None for a,b in zip(direct,budget)]
        rows.append({k:cell[k] for k in ["length","position","factor","steps"]}|{
          "source":ref["path"],"source_sha256":ref["sha256"],"direct_kl":direct,"requested_budget":budget,
          "budget_used_ratio":ratios,"trace":trace})
    if len(rows)!=18:raise RuntimeError("wrong public grid")
    ratios=[v for row in rows for v in row["budget_used_ratio"] if v is not None]
    factors=[]
    for factor in [2.,1.,.5]:
        subset=[v for row in rows if row["factor"]==factor for v in row["budget_used_ratio"] if v is not None]
        factors.append({"factor":factor,"steps":len(subset),"minimum":min(subset),"median":statistics.median(subset),"maximum":max(subset)})
    result={"task_id":"TRR-0020","scope":"descriptive CPU analysis of already frozen public synthetic traces; no reconstruction labels opened",
      "public_probe_sha256":sha(probe),"public_cases":len(rows),"updates":len(ratios),
      "minimum_budget_utilization":min(ratios),"median_budget_utilization":statistics.median(ratios),
      "maximum_budget_utilization":max(ratios),"updates_below_95_percent":sum(v<.95 for v in ratios),
      "factors":factors,"rows":rows,"truth_read":False,"reconstruction_outputs_changed":False,
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "script_sha256":sha(Path(__file__)),"command":[sys.executable,*sys.argv],
      "start_unix":started,"end_unix":time.time()}
    with (X/"dev33_public_budget_diagnostic.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
if __name__=="__main__":main()
