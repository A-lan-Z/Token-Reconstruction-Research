from support import *
def main():
    bound=binding();entries=[];phases=[]
    for factor in [2.,1.,.5]:
        subprocess.run([sys.executable,str(ROOT/"scripts/trr0020_stage33_decoder/worker.py"),"--factor",str(factor)],cwd=ROOT,check=True)
        p=X/f"dev33_decoder_phase_factor{factor}.json";phase=json.loads(p.read_text())
        if phase["binding"]!=bound:raise RuntimeError("changed phase")
        for entry in phase["entries"]:verify(entry,bound)
        entries.extend(phase["entries"]);phases.append({"path":str(p.relative_to(ROOT)),"sha256":n.digest(p)})
    expected={(row["id"],cfg[0]) for row in rows() for cfg in CONFIGS}
    if len(entries)!=96 or {(e["id"],e["method"]) for e in entries}!=expected or binding()!=bound:raise RuntimeError("incomplete or changed matrix")
    n.write(X/"dev33_decoder_freeze.json",{"task_id":"TRR-0020","binding":bound,"configurations":CONFIGS,"selected_records":rows(),
      "entries":entries,"phases":phases,"truth_read":False,"frozen_unix":time.time()})
    print("COMPLETE96CAUSALFREEZE",flush=True)
if __name__=="__main__":main()
