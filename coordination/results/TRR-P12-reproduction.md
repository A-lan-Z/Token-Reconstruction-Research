# TRR-P12 reproduction record

Metadata-only handoff assembled 2026-09-09 from existing receipts; no experiment was rerun. Full historical argv arrays are linked below.

## Immutable bindings

- Plan: [manifest](../../experiments/TRR-P12/manifest.json), SHA-256 302cef73349927615e39b116e985f796a9c70b6f9c97932a119c0acb1d3f0a8d, base commit 81299e07d4213cd75f455496d623a8371516688f.
- Amendment: [plan-amendment-r1.json](../../experiments/TRR-P12/plan-amendment-r1.json), SHA-256 e960951e52922bd65dfeb48d14780f68c3e3c3a79e041f7234af378a3b2fad42; final ranges pile [0,10000), finance [50000,52000).
- Exclusion union: [identity_union_extension_r2.json](../../experiments/TRR-P12/exclusions/identity_union_extension_r2.json), SHA-256 bd2e641f5f10d249595b89f48aeeba2d97a2b71688190177f4ebe4f4ac022a0b. Inherited P11 union/audit/release/confirmation/transfer SHAs: 125275eab38c66117d45f5e0df4085058dae8a5c939fff1554cab7e608eb1fe1, 43416d0d1945821812278df8dfc3ee7639f872e9c8407abdf245fb281bda9d65, 0e1816711965b0601a2e00ce4c187bded9f0c0fca4493e71ef9478a772659f95, fe7129e9d7230100d1900facea98b15b8000f382a0bd7035b2039b4d9429bf68, 8674ea5c824c5f3bc4aa10c44439422654572af546684567046f4a5ffb717c0f.
- Agent1 reservation SHAs: dee8b7e1d70342c154a4217c982d47d3917d4e11c9cad33b6ebdc87c79a6cf33 and a6a47eb11731fca900b5a5f91e5512b1193ffcbe812a6e42ec678a10cec8c6a1.
- Frozen panel: outputs/TRR-P12/sources-r2/panel.json, SHA-256 f48a126fbae6a5e88e1292caf0675d8973f67ab415b706b7fd35e966659ad08e; 128 paired records/domain, stages 0/64/128/256.
- Target bundle: outputs/TRR-P12/target-sources-r1/bundle.json, SHA-256 5f717f0b1ba9195d98566b681c378de30a8397d5cefa2b337b521d99c0437b6d. Alpaca revision dce01c9b08f87459cf36a430d809084718273017, Arrow SHA-256 f45103036ed651f4c06d0a3c3e0fb7d53acb3074ed5c8e804a69c1efc1cea794.
- Fixed target: Llama-3.2-1B-Instruct revision 9213176726f574b556790deb65791e0c5aa438b6; seed 6012; batch 2; sequence 128; 256 train/64 validation; LoRA rank 8, alpha 16, layers 0–3 q_proj/v_proj; stages 0/64/128/256.

## Runtime and fresh-output rule

Recorded pins: Python 3.12.3; torch 2.10.0+cu128; transformers 5.3.0; safetensors 0.7.0; datasets 4.8.3; tokenizers 0.22.2; numpy 1.26.4; accelerate 1.13.0; huggingface-hub 1.7.2. P11 actual-r2 readout/state/tensor-state SHAs are ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1, 088be6a6b2842d526f3dab39789728d9cfc23f2fb1fb892d107d46ade2382706, and 209155048df936359870a1419803800d7bad4ab72a46e09bf3287648079964da.

Historical r1 outputs are create-only evidence. For a fresh run, choose new roots and fail closed:
~~~bash
set -euo pipefail
RUN=outputs/TRR-P12/reproduction-r1
BACKUP=/mnt/c/Users/alanz/Token-Reconstruction-Backups/TRR-P12/reproduction-r1
test ! -e "$RUN" && test ! -e "$BACKUP"
test -d /mnt/c/Users/alanz/Token-Reconstruction-Backups
mkdir "$RUN" "$BACKUP"
df -h /mnt/c/Users/alanz/Token-Reconstruction-Backups
~~~
The external backup must have at least 10 GiB free; copy and independently hash each adapter before advancing stages. The historical target command and backup binding are in [target-execution-r1.json](../../experiments/TRR-P12/target-execution-r1.json).

The commands below are the historical execution record, not a copy-and-paste rerun in the existing namespace. A new reproduction must replace every output path and bind its newly generated paths/hashes in the corresponding manifests and release checks. Do not overwrite retained evidence or relax a hash gate to reuse it. Reusing the preserved artifacts for verification does not authorize a new target sweep.

## Ordered interfaces

1. Source selection, exact commands: [source-preparation-execution-r2.json](../../experiments/TRR-P12/source-preparation-execution-r2.json).
~~~bash
python3 scripts/trr_p12/select_panel.py --execute --root . --amendment experiments/TRR-P12/plan-amendment-r1.json --output outputs/TRR-P12/sources-r2/panel.json
python3 scripts/trr_p12/target_sources.py --execute --root . --panel outputs/TRR-P12/sources-r2/panel.json --output outputs/TRR-P12/target-sources-r1/bundle.json
~~~
2. Target training: exact argv is in [target-execution-r1.json](../../experiments/TRR-P12/target-execution-r1.json); it is target_train.py full with the bundle, pinned model snapshot, target-r1 output, external backup root, CUDA, and seed 6012.
3. Capture: exact eight argv arrays are in [capture-execution-r1.json](../../experiments/TRR-P12/capture-execution-r1.json); capture.py uses the panel, bundle, target run receipt, domain, stage, optional matching adapter, CUDA, 120-second timeout, and create-only cell root.
4. B1: exact eight argv arrays and watchdogs are in [b1-execution-r1.json](../../experiments/TRR-P12/b1-execution-r1.json); run_b1.py consumes each observation with the read-only P11 package and writes prediction/projected/geometry/receipt artifacts.
5. A1+A2: sanitized input SHA-256 is 9a5b5ee4f416d12ff64ddc7ea3431d7937bb3b1ffefbe63695f8b14c65ce4170; CPU and qualification/resume commands are in [a1-stage-input-cpu-check-r1.json](../../experiments/TRR-P12/a1-stage-input-cpu-check-r1.json); completed receipt SHA-256 is 468424af2829838cc993d6a4c659f6e0277bc767cf35306856827707fb4d8052.
6. Geometry: exact finance and pile argv are in [geometry-finance-execution-r1.json](../../experiments/TRR-P12/geometry-finance-execution-r1.json) and [geometry-pile-execution-r1.json](../../experiments/TRR-P12/geometry-pile-execution-r1.json); both use the P11 readout, logit scale 71.9100112915039, CUDA, 8 GiB free-GPU guard, and four stages.
7. Descriptor and freeze:
~~~bash
/usr/bin/python3 scripts/trr_p12/build_evaluation_descriptor.py --panel outputs/TRR-P12/sources-r2/panel.json --capture-root outputs/TRR-P12/capture-r1 --b1-root outputs/TRR-P12/b1-r1 --geometry-dir pile=outputs/TRR-P12/geometry-r1/pile --geometry-dir finance=outputs/TRR-P12/geometry-r1/finance --a1-receipt outputs/TRR-P12/a1-a2-r2/run.resume.receipt.json --output experiments/TRR-P12/evaluation-freeze-descriptor-r1.json --execute
/usr/bin/python3 -m scripts.trr_p12.evaluation freeze --descriptor experiments/TRR-P12/evaluation-freeze-descriptor-r1.json --output experiments/TRR-P12/evaluation-freeze-r1.json
~~~
Descriptor SHA-256 2e3f26b07178221ca15aae697aee9c5034d373c4a871d6a0c4d78756869f3b1f; freeze SHA-256 3436de3394c0113d91303a0271d637371362cbe95aafa8638c0746e5920edf7a.
8. After [root-truth-release-r1.json](../../experiments/TRR-P12/root-truth-release-r1.json) only, run the dry-run then execute materializer and fixed score:
~~~bash
/usr/bin/python3 -m scripts.trr_p12.materialize_truth --panel outputs/TRR-P12/sources-r2/panel.json --freeze-receipt experiments/TRR-P12/evaluation-freeze-r1.json --source-inputs experiments/TRR-P11/selector/public_source_inputs_r1.json --output-root outputs/TRR-P12/private-evaluation/truth-r1 --dry-run
/usr/bin/python3 -m scripts.trr_p12.materialize_truth --panel outputs/TRR-P12/sources-r2/panel.json --freeze-receipt experiments/TRR-P12/evaluation-freeze-r1.json --source-inputs experiments/TRR-P11/selector/public_source_inputs_r1.json --output-root outputs/TRR-P12/private-evaluation/truth-r1 --execute
/usr/bin/python3 -m scripts.trr_p12.evaluation score --freeze-receipt experiments/TRR-P12/evaluation-freeze-r1.json --truth-descriptor outputs/TRR-P12/private-evaluation/truth-r1/truth.json --output experiments/TRR-P12/score-r1.json
~~~
Truth descriptor SHA-256 d3ebba74e58bf4fa31fd7897bae6266c6d5b3227ad6ad2e088564cee9c5a4ca6; score SHA-256 f517995d0370501af2ed476882cd272cec40bd0c8a20994d5e4cecab59f9b9f6; bootstrap seed 9012, draws 10000.

## Truth boundary

Truth was opened only after complete freeze and explicit release; source text was transiently materialized for evaluator-only truth and scoring then consumed it. P03 was never opened. Do not describe labels as unopened after scoring: authorized evaluator truth was opened after freeze. No source text, true IDs, or token payloads are included here. The compact chain is [truth-score-execution-r1.json](../../experiments/TRR-P12/truth-score-execution-r1.json), SHA-256 fb832bf5aeff98ca54a7ebcbcbefa593e62b900d5ed40e3a40a33aaa54a23732.
