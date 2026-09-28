"""Isolated execution of the unchanged development13 reconstruction rule."""
from pathlib import Path
import sys,json,time,hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0020_stage13"))
spec=importlib.util.spec_from_file_location("trr0020_dev13_original_predict",ROOT/"scripts/trr0020_stage13/dev13_predict.py")
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
n=original.n;torch=n.torch;CONFIGS=original.CONFIGS;Engine=original.ConstrainedEmbedding
X=ROOT/"experiments/TRR-0020";OUT=ROOT/"outputs/TRR-0020/dev13_r1"
INPUT=original.INPUT
def binding():
    bound=original.binding()
    extras=list((ROOT/"scripts/trr0020_stage13r1").glob("*.py"))+[X/"DEV13_R1_PLAN.md"]
    bound["sources"].update({str(p.relative_to(ROOT)):n.digest(p) for p in extras})
    return bound
def selected_rows():
    rows=[];counts={}
    for row in json.loads((INPUT/"metadata.json").read_text()):
        key=row["condition"],row["group"];counts.setdefault(key,0)
        if counts[key]<2:rows.append(row);counts[key]+=1
    assert len(rows)==8
    return rows
def same_numerics(a,sa,b,sb):
    return set(a)==set(b) and all(torch.equal(v,b[k]) for k,v in a.items()) and all(
      sa[key]==sb[key] for key in ["loss_trace","observed_error_trace","nearest_distance_trace"])
def verify(entry,bound):
    if entry["binding"]!=bound:raise ValueError("changed cell binding")
    if n.digest(ROOT/entry["path"])!=entry["sha256"]:raise ValueError("changed output")
    data=n.load_file(str(ROOT/entry["path"]))
    keys={"step0","step16","step32","step64","step128","best_objective","best_position_error"}
    if set(data)!=keys:raise ValueError("wrong output variants")
    for ids in data.values():
        if tuple(ids.shape)!=(entry["positions"],) or ids.dtype!=torch.int64 or int(ids[0])!=128000:
            raise ValueError("wrong output geometry/BOS")
    return data
