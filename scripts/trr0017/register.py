from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];X=ROOT/"experiments/TRR-0017"
old=json.loads((ROOT/"experiments/TRR-0015/registry.json").read_text())
new=[
 {"id":"prefix_parallel_continuous96","decision_rule":"FP32 same prefix; initial H rescaled to median public embedding RMS; whole-sequence L-BFGS96, history8 strong-Wolfe maxeval192; lowest observed MSE continuous iterate; final full-vocabulary cosine argmax; BOS only; no A1/shortlist","candidate_budget":None,"can_abstain":False},
 {"id":"prefix_parallel_discrete64","decision_rule":"prefix-weight-metric argmax initialization;64 whole-sequence Adam steps lr0.003; actual vocabulary embeddings each forward, straight-through proxy gradient; cosine objective; other-position KV gradients detached; eager BF16 attention; lowest observed mean-cosine-loss full sequence; BOS only; no A1/shortlist","candidate_budget":None,"can_abstain":False}]
methods=old["methods"]+new;setups=old["setups"]
with (X/"registry.json").open("x") as f:json.dump({"task_id":"TRR-0017","registered_before_execution":True,
     "methods":methods,"setups":setups,"required_cells":[{"setup_id":s,"method_id":m["id"]} for s in setups for m in methods],
     "inherited_cells":52,"new_cells":4,"total_cells":56,"current_invocations":1360,
     "shared_context":"byte-identical execution optimization of prefix_native_fragment256; not a new decision rule",
     "selection":"continuous96 preserves best development identifier result; discrete64 is strongest lower-latency prose family; no claim either is near baseline"},f,indent=2)

