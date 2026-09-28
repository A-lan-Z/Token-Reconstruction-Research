from support import *
import subprocess
def main():
    bound=binding();phases=[];entries=[]
    for index in [2,0,1,*range(3,12)]:
        subprocess.run([sys.executable,str(ROOT/"scripts/trr0020_stage14/worker.py"),"--index",str(index)],cwd=ROOT,check=True)
        path=X/f"dev14_phase_{CONFIGS[index][0]}.json";f=json.loads(path.read_text())
        if f["binding"]!=bound:raise RuntimeError("phase binding changed")
        for e in f["entries"]:verify(e,bound)
        entries.extend(f["entries"]);phases.append({"path":str(path.relative_to(ROOT)),"sha256":n.digest(path)})
    expected={(r["id"],c[0]) for r in rows() for c in CONFIGS}
    if len(entries)!=96 or {(e["id"],e["method"]) for e in entries}!=expected:raise RuntimeError("incomplete matrix")
    if binding()!=bound:raise RuntimeError("source changed")
    anchors=[e["original_anchor"] for e in entries if e["original_anchor"] is not None]
    if len(anchors)!=8 or not all(a["snapshots_and_trace_prefixes_equal"] for a in anchors):raise RuntimeError("control coverage")
    n.write(X/"dev14_freeze.json",{"task_id":"TRR-0020","binding":bound,"configurations":CONFIGS,"selected_records":rows(),
      "entries":entries,"phases":phases,"control_anchors":8,"truth_read":False,"frozen_unix":time.time()})
    print("COMPLETE96CELLFREEZE_CONTROL_IDENTICAL",flush=True)
if __name__=="__main__":main()
