"""Consolidate existing TRR-0020 evidence. This script runs no reconstruction."""
from pathlib import Path
import json,re,hashlib,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0020"
TOPICS=[
"Residual cancellation","Full-vocabulary soft Adam","Reset-moment optimization",
"Diagonal gradient approximation","GPU graph execution and loss variants",
"Deterministic mixed precision","Discreteness penalties and hardening",
"Softmax gradient preconditioning","Hard-context soft optimization",
"Direct quadratic updates and smaller penalties","Continuous convex updates",
"Lower-dimensional query parameterization","Embedding penalties and isolated memory-safe rerun",
"Denominator floors and learning rates","Warm start plus direct quadratic refinement",
"Isotropic sensitivity","Cached directional curvature","Current-residual rank-one curvature",
"Fitted-embedding readout","Higher-power residual weighting",
"Short warm-start refinement","Layer-factor inversion","CGLS inverse correction",
"Normalized-output correction","Full causal derivatives and global solve",
"Full causal scaling","Per-position current-token updates","Alternative metrics",
"Full-vocabulary mirror updates","Observed-error scaling","Probability-change calibration",
"Per-position KL budgets","Causal full-vocabulary direction",
"First-error audit and backtracking","Norm-preserving probability mixtures",
"Public interpolation paths and loss barriers","Direct full-table context correction",
"Direct first-token constraint","Coupled residual continuation",
"Linear-solver stability and ridge audit","Runtime profiling",
"Fused KL scalar evaluation","Full decoder with approximate fused reduction",
"Exact pointwise fusion","Full reconstruction with exact pointwise fusion",
"Reusing forward probabilities","Bounded history mixing qualification",
"Bounded history mixing reconstruction and stable revision",
"Moment-bound update qualification","Moment-bound full reconstruction"]
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def describe(p):return {"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha(p)}
def main():
    started=time.time()
    matrix=json.loads((X/"canonical_moment_r1/canonical_matrix.json").read_text())
    score=json.loads((X/"canonical_moment_r1/score.json").read_text())
    assert matrix["complete"] and matrix["required_cells"]==70
    state=json.loads((ROOT/"coordination/STATE.json").read_text())
    manifest=json.loads((X/"manifest.json").read_text())
    counts=manifest["completed_development"]
    assert len(counts)==36 and sum(counts.values())==2816
    stages=[];files=[p for p in X.iterdir() if p.is_file()]
    for number,topic in enumerate(TOPICS,1):
        pattern=re.compile(r"^(?:soft_)?dev"+str(number)+r"(?:_|[.])",re.I)
        evidence=[p for p in files if pattern.match(p.name) and p.suffix in [".json",".md",".txt"]]
        evidence +=[p for p in files if re.match(r"^DEV"+str(number)+r"(?:_|[.])",p.name)]
        records={k:v for k,v in manifest.items() if re.match(r"^dev"+str(number)+r"(?:_|$)",k)}
        conclusions={k:{kk:vv for kk,vv in v.items() if kk in ["status","conclusion","scope","plan","summary","archive","execution_commit","cells","reconstruction_cells"]} for k,v in records.items() if isinstance(v,dict)}
        statuses={k:v for k,v in state.items() if re.match(r"^dev"+str(number)+r"(?:_|$)",k)}
        stages.append({"stage":number,"topic":topic,"completed_matrix_cells":{k:v for k,v in counts.items() if re.match(r"^dev"+str(number)+r"(?:_|$)",k)},
          "recorded_statuses":statuses,"recorded_conclusions":conclusions,
          "metadata_artifacts":[describe(p) for p in sorted(set(evidence))]})
    failures=[]
    for p in files:
        if p.suffix==".json" and any(s in p.stem.lower() for s in ["failure","failed","repair","numerics_audit"]):
            failures.append(describe(p))
    for p in sorted((X/"canonical_moment").glob("*failure*.json")):failures.append(describe(p))
    code=[]
    for p in sorted((ROOT/"scripts").glob("trr0020*")):
        code += list(p.rglob("*.py")) if p.is_dir() else ([p] if p.suffix==".py" else [])
    canonical=[]
    for folder in ["canonical","canonical_refine","canonical_moment","canonical_moment_r1","objective_audit"]:
        folderpath=X/folder
        principal=[p for p in folderpath.glob("*.json") if any(s in p.name for s in ["matrix","score","summary","freeze","gate","archive","registry","failure","launch","validation","preflight"])]
        canonical.append({"folder":str(folderpath.relative_to(ROOT)),"metadata":[describe(p) for p in sorted(principal)]})
    ledger={"task_id":"TRR-0020","status":"OWNER_REQUESTED_STOP_AFTER_CURRENT_BENCHMARK",
      "owner_request":"coordination/requests/TRR-0020-STOP-AFTER-CURRENT-BENCHMARK.md",
      "original_scientific_goal_claimed_complete":False,"new_experiments_authorized":False,
      "research_stages":50,"completed_development_matrices":36,"completed_development_method_input_cells":2816,
      "development_sampling_limit":"These are repeated method/input evaluations on the same eight opened development observations, not2816independent test inputs. Public numerical fixtures and excluded attempts are separate.",
      "canonical_methods":35,"canonical_cells":70,"current_benchmark_replicate_cells":1728,
      "stages":stages,"failure_and_repair_records":failures,"canonical_and_objective_audit_records":canonical,
      "source_files":[describe(p) for p in sorted(set(code))],
      "archive_note":"Raw members, part hashes and previous roundtrip checks remain in each archive manifest. This index hashes metadata and source files; it does not rerun experiments or claim a new verification of every historical binary object.",
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"start_unix":started,"end_unix":time.time()}
    with (X/"RESEARCH_LEDGER.json").open("x") as f:json.dump(ledger,f,indent=2)
    recent=[]
    for p in files:
        if re.match(r"^(?:DEV|dev)(?:49|50)(?:_|[.])",p.name) and p.suffix in [".json",".md",".txt"]:recent.append(p)
    for folder in [X/"canonical_moment",X/"canonical_moment_r1"]:
        recent+=list(folder.glob("*.json"))+list(folder.glob("*.md"))+list(folder.glob("*.txt"))
    recent=sorted(set(recent));launches=[]
    for p in recent:
        if p.suffix!=".json":continue
        d=json.loads(p.read_text())
        if "command" in d and any(k in d for k in ["execution_commit","code_commit"]):
            launches.append({"receipt":str(p.relative_to(ROOT)),"sha256":sha(p),**{k:d[k] for k in ["command","execution_commit","code_commit","start_unix","end_unix","returncode"] if k in d}})
    provenance={"task_id":"TRR-0020","request":"coordination/requests/TRR-0020-CONTINUE-22.md","superseding_stop_request":ledger["owner_request"],
      "execution_records":launches,"artifacts":[describe(p) for p in recent],
      "archive_parts":"Each archive manifest binds its ordered part files and logical raw members.",
      "scope":"Dev49qualification, Dev50development, and complete canonical moment comparison; retrospective supplied-public-prefix evidence.",
      "failed_attempts":["Dev49initial import failed before numerical work; preserved separately, R1passed.",
        "Canonical moment initial matrix import failed before reconstruction; original52qualification checks retained, R1full requalification and matrix completed."],
      "benchmark_source_commit":json.loads((X/"canonical_moment_r1/matrix_launch.json").read_text())["execution_commit"],
      "goal_completed":False,"stop_requested":True,"created_unix":time.time()}
    with (X/"CONTINUATION22_PROVENANCE.json").open("x") as f:json.dump(provenance,f,indent=2)
    print(json.dumps({"stages":len(stages),"matrices":36,"cells":2816,"source_files":len(ledger["source_files"]),
      "failure_records":len(failures),"recent_artifacts":len(recent),"execution_records":len(launches)},indent=2))
if __name__=="__main__":main()
