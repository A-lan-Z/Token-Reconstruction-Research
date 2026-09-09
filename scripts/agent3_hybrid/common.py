from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
from scripts.agent3_shortlists.core import binding,verify,write_json
ROOT=Path(__file__).resolve().parents[2]
EV=ROOT/'experiments/agent3-static-prefix-hybrid'
OUT=ROOT/'outputs/agent3-static-prefix-hybrid'
SHARED=ROOT.parent/'TRR-P12'
METHODS=('a1_k256','b1_k256','b1_k16','a1_k16','b1_alone')
STAGES=(0,64,128,256)
DOMAINS=('pile','finance')
def utc():return datetime.now(timezone.utc).isoformat()
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def check_implementation(path=None):
    path=Path(path or EV/'implementation-freeze.json');r=json.loads(path.read_text())
    if r['status']!='FROZEN_AFTER_DEVELOPMENT_BEFORE_CONFIRMATION':raise ValueError('implementation not frozen')
    for b in r['code']:verify(b)
    for b in r.get('assets',[]):verify(b)
    verify(r['qualification']);verify(r['plan']);return r
