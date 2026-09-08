TRR-0013 is a single-owner, exploratory comparison of focused versus ordinary public continuation from the rebuilt TRR-0012 B1. It does not restore the lost PR20 weights.

The prospective contract is `contract.json`. `inputs.json` binds the actual starting state and fixed readout. The source ledgers contain opaque identities only; the curator's token payloads are excluded from version control and must not be read by prediction commands. The final scoring command validates every method/cell and all bindings before opening those payloads.

Run from this task worktree with the published local assets at the manifest paths, Python3/PyTorch/safetensors/transformers/datasets/scipy/numpy installed, and the RTX5080 lease held. Every output directory is create-only; use a new attempt name if reproducing. No network data download or paid compute is used.

Execution sequence (exact attempt commands are also recorded in receipts/logs):

```bash
python3 scripts/trr0013.py smoke --output outputs/TRR-0013/synthetic_smoke_r1
python3 scripts/trr0013.py bank-check --output outputs/TRR-0013/original_bank_difficulty_r1
python3 scripts/trr0013_curator.py select --phase correction --output outputs/TRR-0013/correction_selection_r1
python3 scripts/trr0013_curator.py capture --phase correction --selection outputs/TRR-0013/correction_selection_r1/selection.json --output outputs/TRR-0013/correction_capture_r1
python3 scripts/trr0013_fit.py prepare --correction outputs/TRR-0013/correction_capture_r1/manifest.json --output outputs/TRR-0013/fit_preparation_r1
python3 scripts/trr0013_fit.py qualify --preparation outputs/TRR-0013/fit_preparation_r1/manifest.json --output outputs/TRR-0013/fit_qualifier_r1
python3 scripts/trr0013_fit.py fit --arm ordinary --preparation outputs/TRR-0013/fit_preparation_r1/manifest.json --output outputs/TRR-0013/fit_ordinary_r1
python3 scripts/trr0013_fit.py fit --arm focused --preparation outputs/TRR-0013/fit_preparation_r1/manifest.json --output outputs/TRR-0013/fit_focused_r1
python3 scripts/trr0013_curator.py select --phase evaluation --additional outputs/TRR-0013/correction_selection_r1/selection.json --output outputs/TRR-0013/evaluation_selection_r1
python3 scripts/trr0013_curator.py capture --phase evaluation --selection outputs/TRR-0013/evaluation_selection_r1/selection.json --output outputs/TRR-0013/evaluation_capture_r1
```

The original-bank diagnostic is reusable for these exact B1 bytes and bank hashes. `fit_preparation_r1` reuses its scores and computes only the new correction-pool losses. The reference schedule is the unchanged native CPU PyTorch sampler. Focused schedules retain the first256 reference draws in every identical eight-record batch and replace the other256 with uniform replacement draws from the initial-loss top quintile. There is no importance correction; this deliberately changes empirical CE weighting. Neither arm receives additional parameters, readout learning, or A2 inference.

All grid model states and optimizer states are written before selection, verified through the strict native loader, and retained. The selected package must pass a separate clean restore before final evaluation. Full-vocabulary validation CE and prediction curves are public selection data; final natural-panel truth is not a selection input.

The fitting qualifier uses20 discarded updates. Its native training geometry is unchanged at8×192×2048 with512 sampled positions. Cache values were compared exactly against every original payload boundary; no numerical microbatch workaround is introduced. The generic watchdog may wrap long commands with a2400-second deadline,20GiB RSS cap and6GiB free-host floor; the in-process GPU guard enforces8GiB reserved and2GiB free.

Historical failed attempt: the first8-record bank classifier received int32 labels and failed before completing any classification. The preserved log documents the error; labels were converted to int64 as in the unchanged native fitting step. No failed model state was selected. Long-source tokenizer warnings during selection are expected because source identity is computed from the full public record; the model receives only the declared192-token clip.
