from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
for mode in ["aa_probability_clip2","control","aa_logit_clip1","aa_probability_clip1"]:
    subprocess.run([sys.executable,str(Path(__file__).parent/"public_worker.py"),"--mode",mode],cwd=ROOT,check=True)
