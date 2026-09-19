from support import *
import subprocess,os
def main():
    bound=binding();phases=[];entries=[]
    # The configuration that hit the previous accumulated peak is qualified first.
    for index in [11,*range(11)]:
        subprocess.run([sys.executable,str(ROOT/"scripts/trr0020_stage13r1/worker.py"),"--index",str(index)],cwd=ROOT,check=True)
        name=CONFIGS[index][0];path=X/f"dev13_r1_phase_{name}.json";f=json.loads(path.read_text())
        if f["binding"]!=bound:raise RuntimeError("phase binding changed")
        for e in f["entries"]:verify(e,bound)
        entries.extend(f["entries"]);phases.append({"path":str(path.relative_to(ROOT)),"sha256":n.digest(path)})
    rows=selected_rows();expected={(r["id"],c[0]) for r in rows for c in CONFIGS}
    if len(entries)!=96 or {(e["id"],e["method"]) for e in entries}!=expected:raise RuntimeError("incomplete matrix")
    if binding()!=bound:raise RuntimeError("source changed")
    anchors=[e["original_anchor"] for e in entries if e["original_anchor"] is not None]
    if len(anchors)!=88 or not all(a["all_outputs_and_traces_equal"] for a in anchors):raise RuntimeError("original cell anchor coverage")
    n.write(X/"dev13_r1_freeze.json",{"task_id":"TRR-0020","binding":bound,"configurations":CONFIGS,"selected_records":rows,
      "entries":entries,"phases":phases,"original_cell_anchors":88,"public_config_anchors":12,
      "truth_read":False,"frozen_unix":time.time(),"resource_change":"one fresh process per configuration; geometry and method unchanged"})
    print("COMPLETE96CELLFREEZE_IDENTICAL88ANCHORS",flush=True)
if __name__=="__main__":main()
