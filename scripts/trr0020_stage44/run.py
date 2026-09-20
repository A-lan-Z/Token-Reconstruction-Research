"""Sequential isolated original-fixture preparation followed by kernel checks."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
for length in [128,40]:
    subprocess.run([sys.executable,str(Path(__file__).parent/"collect.py"),"--length",str(length)],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(Path(__file__).parent/"gpu_check.py")],cwd=ROOT,check=True)
