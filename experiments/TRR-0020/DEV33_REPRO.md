# Development33 reproduction

This is retrospective development evidence, not a new canonical comparison.
Use an isolated copy of the task worktree and its referenced sibling assets.
Some record writers and archives are create-only; do not overwrite historical
receipts or rerun over the archived outputs.

Runtime: Python3.12.3, torch2.10.0+cu128, Transformers5.3.0, RTX5080.
TF32 disabled; deterministic algorithms enabled. Before each command set
`PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1`.
The public and replay guard ceilings below match their saved preflights.
The public fixtures, prefix/config/observation hashes, all frozen source hashes,
hardware, full commits and exact commands are in the phase receipts and
DEV33_PROVENANCE.json.

1. At d4351873d721c4f1adc95a78c705a833b924deb6, execute the public study:
   `python3 scripts/trr0020_resource_guard.py --receipt experiments/TRR-0020/dev33_guard.json --timeout 1200 -- python3 scripts/trr0020_stage33/probe.py`.
   At a159fabb82ac3fc39963cb523a1855a91529b819, archive and summarize with `python3 scripts/trr0020_stage33_archive.py`
   and `python3 scripts/trr0020_stage33_summary.py`.
2. At89a7943185b14eab4a86cfd68771c542b7188e48, qualify exact replay:
   `python3 scripts/trr0020_resource_guard.py --receipt experiments/TRR-0020/dev33_replay_guard.json --timeout 1500 -- python3 scripts/trr0020_stage33_replay/qualify.py`.
   At7c8b3f2235a391cade8c4b68802804ce2e7b4007, archive with
   `python3 scripts/trr0020_stage33_replay_archive.py`.
3. At7c8b3f2235a391cade8c4b68802804ce2e7b4007, execute the96-cell decoder:
   `python3 scripts/trr0020_resource_guard.py --receipt experiments/TRR-0020/dev33_decoder_guard.json --timeout 1800 -- python3 scripts/trr0020_stage33_decoder/run.py`.
   This runs three isolated workers with factors2,1,.5. Their repeated public,
   eager and committed-history checks must pass before their real inputs.
4. Require terminal guard exit0 and the complete96-cell freeze. At
   a2427b21f69beec4cb6384531c2fb457af4beb46, run
   `python3 scripts/trr0020_stage33_decoder/score.py`.
   Its complete matrix, source/hash and three negative gates precede labels.
   Exact scoring stdout/stderr and execution timestamps are in
   dev33_decoder_scoring_run.json and its two referenced text files.
5. Run `python3 scripts/trr0020_stage33_decoder_archive.py`, then
   `python3 scripts/trr0020_stage33_decoder_summary.py`.
   The public-only CPU diagnostic is
   `python3 scripts/trr0020_stage33_public_budget_diagnostic.py`.

Raw archives store content-addressed ZIP members. Concatenate the listed parts
in order, verify the archive SHA256, then use each archive JSON's logical_members
mapping to recover original paths. All raw members were read back and hashed.
Public and replay artifact receipts are separate from reconstruction receipts.

The supplied prefix is a public checkpoint, not the unavailable live target
prefix or an adaptively recovered prefix. Reconstruction labels are used only
after all selected outputs are frozen. All twelve configurations are reported;
no active canonical method has been added.
