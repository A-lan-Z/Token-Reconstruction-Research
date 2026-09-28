from support import *
def main():
    bound=binding();entries=[];phases=[]
    for mode in ["locked","control"]:
        subprocess.run([sys.executable,str(ROOT/"scripts/trr0020_stage38_grid/worker.py"),"--mode",mode],cwd=ROOT,check=True)
        path=X/f"dev38_grid_phase_{mode}.json";phase=json.loads(path.read_text())
        if phase["binding"]!=bound:raise RuntimeError("phase binding")
        for entry in phase["entries"]:verify(entry,bound)
        entries.extend(phase["entries"]);phases.append({"path":str(path.relative_to(ROOT)),"sha256":n.digest(path)})
    expected={(row["id"],name) for row in rows() for name,_,_ in CONFIGS}
    if len(entries)!=48 or {(v["id"],v["method"]) for v in entries}!=expected:raise RuntimeError("incomplete48cellmatrix")
    if binding()!=bound:raise RuntimeError("binding changed")
    n.write(X/"dev38_grid_freeze.json",{"task_id":"TRR-0020","binding":bound,"configurations":CONFIGS,"entries":entries,
      "phases":phases,"selected_records":rows(),"truth_read":False,"frozen_unix":time.time()})
    print("FIRST_DECODER_COMPLETE48CELLFREEZE",flush=True)
if __name__=="__main__":main()
