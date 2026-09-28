"""Independent explicit-vs-factored vocabulary scoring check (CPU)."""
import json,sys,time,platform
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/trr0020"))
from full_vocabulary import FullVocabularyA2
g=torch.Generator().manual_seed(200020)
engine=FullVocabularyA2.__new__(FullVocabularyA2)
engine.raw=torch.randn(31,11,generator=g)
engine.norms=engine.raw.square().sum(-1)
h=torch.randn(7,11,generator=g)
context=torch.randn(7,11,generator=g)
actual=engine.shifted_select(h,context)
explicit=F.cosine_similarity(engine.raw[None]+context[:,None],h[:,None],dim=-1).argmax(-1)
assert torch.equal(actual,explicit)
# Vocabulary-wide winner may be the final entry: verify no hidden top-K cap.
engine.raw=torch.eye(257);engine.norms=engine.raw.square().sum(-1)
h=engine.raw[-1:].clone();context=torch.zeros_like(h)
assert engine.shifted_select(h,context).item()==256
result={"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
    "factorized_score_matches_explicit":True,"final_vocabulary_entry_can_win":True,
    "device":"cpu","torch":torch.__version__,"python":platform.python_version()}
destination=ROOT/"experiments/TRR-0020/factorization_test.json"
with destination.open("x") as f:json.dump(result,f,indent=2)
print(json.dumps(result))
