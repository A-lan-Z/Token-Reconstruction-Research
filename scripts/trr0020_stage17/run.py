from support import *
import subprocess
def main():
    bound=binding();phases=[];entries=[]
    for index in [18,*range(18),19]:
        subprocess.run([sys.executable,str(ROOT/"scripts/trr0020_stage17/worker.py"),"--index",str(index)],cwd=ROOT,check=True)
        path=X/f"dev17_phase_{CONFIGS[index][0]}.json";f=json.loads(path.read_text())
        if f["binding"]!=bound:raise RuntimeError("phase binding changed")
        for e in f["entries"]:verify(e,bound)
        entries.extend(f["entries"]);phases.append({"path":str(path.relative_to(ROOT)),"sha256":n.digest(path)})
    expected={(r["id"],c[0]) for r in rows() for c in CONFIGS}
    if len(entries)!=160 or {(e["id"],e["method"]) for e in entries}!=expected:raise RuntimeError("incomplete matrix")
    if binding()!=bound:raise RuntimeError("source changed")
    if not all(e["original_anchor"]["snapshots_and_trace_prefixes_equal"] for e in entries):raise RuntimeError("control coverage")
    n.write(X/"dev17_freeze.json",{"task_id":"TRR-0020","binding":bound,"configurations":CONFIGS,"selected_records":rows(),
      "entries":entries,"phases":phases,"control_anchors":160,"truth_read":False,"frozen_unix":time.time()})
    print("COMPLETE160CELLFREEZE_ALL_WARM_ANCHORS_IDENTICAL",flush=True)
if __name__=="__main__":main()
