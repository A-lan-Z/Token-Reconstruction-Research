from support import *
def main():
    bound=binding();phases=[];entries=[]
    for index in [len(CONFIGS)-1]+list(range(len(CONFIGS)-1)):
        subprocess.run([sys.executable,str(ROOT/"scripts/trr0020_stage29/worker.py"),"--index",str(index)],cwd=ROOT,check=True)
        path=X/f"dev29_phase_{CONFIGS[index][0]}.json";phase=json.loads(path.read_text())
        if phase["binding"]!=bound:raise RuntimeError("changed phase")
        for entry in phase["entries"]:verify(entry,bound)
        entries.extend(phase["entries"]);phases.append({"path":str(path.relative_to(ROOT)),"sha256":n.digest(path)})
    expected={(row["id"],cfg[0]) for row in rows() for cfg in CONFIGS}
    if len(entries)!=48 or {(e["id"],e["method"]) for e in entries}!=expected:raise RuntimeError("incomplete matrix")
    if binding()!=bound:raise RuntimeError("source changed")
    n.write(X/"dev29_freeze.json",{"task_id":"TRR-0020","binding":bound,"configurations":CONFIGS,
      "selected_records":rows(),"entries":entries,"phases":phases,"warm_anchors":48,"mean_anchors":48,
      "truth_read":False,"frozen_unix":time.time()})
    print("COMPLETE48CELLFREEZE",flush=True)
if __name__=="__main__":main()
