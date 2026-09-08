# B1 shortlists under target drift — work in progress

The correct-token top8/16/32 rates and whole-clip coverage have not yet been measured. The study is waiting for the shared TRR-P12 target trajectory and sanitized observations. This is a preparation status, not a scientific result.

The frozen design uses32 paired sources per domain (Pile-like and Finance), four target stages0/64/128/256, and identical128-position clips. Stage64 is the first fitted LoRA descendant of the matched public base;128/256 test continued prefix-changing adaptation. B1 remains the preserved step13000 TRR-0012/expanded_fixed_replication_1 with state SHA256088be6a6b2842d526f3dab39789728d9cfc23f2fb1fb892d107d46ade2382706.

Completed preparation: actual B1/readout/loader byte verification and task copy, independent Windows backup verification, native one-record B1/A1 CPU smoke, deterministic nested full-vocabulary shortlists, separate full-matrix freeze and evaluator, and six focused tests. The full32-record CPU qualification passed in64.06seconds with3.27GiB peak RSS; this synthetic repeated-smoke cell is not a scientific evaluation. No new fitting, target generation, P03 opening, global-state mutation, or GPU work has been performed by this task.

The small-budget hybrid has not been tested. The validated A2 path uses untouched public model-prefix weights and its own causal reconstructed-token cache; no validated maintained recovered-model-prefix path is available in the inspected/reused implementation. Stage1 will be completed independently of this integration gap. No hybrid quality, speedup, or end-to-end tracking claim is made.

Prospective limits, source/compute coordination, access separation, stage schedule and uncertainty are in `experiments/agent3-b1-small-budget-a2/plan.md`. Structured handoff is `experiments/agent3-b1-small-budget-a2/manifest.json`. This auxiliary shortlist study does not complete or replace the dual-canonical reconstruction matrix.
