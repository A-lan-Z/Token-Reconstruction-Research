from grid48_support import *
def main():
    bound=binding();entries=[];phases=[]
    for mode in ["aa_logit_clip1_scaled","control","aa_probability_clip2","aa_probability_clip1"]:
        subprocess.run([sys.executable,str(ROOT/"scripts/trr0020_stage48_r1/worker.py"),"--mode",mode],cwd=ROOT,check=True)
        p=X/f"dev48_r1_phase_{mode}.json";phase=json.loads(p.read_text())
        if phase["binding"]!=bound:raise RuntimeError("phase binding")
        for entry in phase["entries"]:verify(entry,bound)
        entries.extend(phase["entries"]);phases.append({"path":str(p.relative_to(ROOT)),"sha256":n.digest(p)})
    expected={(row["id"],name) for row in rows() for name,_,_ in CONFIGS}
    if len(entries)!=96 or {(v["id"],v["method"]) for v in entries}!=expected:raise RuntimeError("incomplete96cellmatrix")
    if binding()!=bound:raise RuntimeError("binding changed")
    n.write(X/"dev48_r1_freeze.json",{"task_id":"TRR-0020","binding":bound,"configurations":CONFIGS,"entries":entries,
      "phases":phases,"selected_records":rows(),"truth_read":False,"frozen_unix":time.time()})
    print("HISTORY_DECODER_COMPLETE96CELLFREEZE",flush=True)
if __name__=="__main__":main()
