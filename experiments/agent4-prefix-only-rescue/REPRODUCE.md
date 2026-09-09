# Reproduction

Use exact commits in phase freeze receipts; Python3.12.3, torch2.10.0+cu128, transformers5.3.0.
From root set OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src.
All output names are create-only; rerun in a fresh output root, never overwrite.

Read PLAN.md and freeze.json. Run in order:
1. run.py --method original --name replay_original; then profile/replay_profile and a1a2/replay_a1a2.
2. qualify.py; inverse_preflight.py.
3. prepare.py --panel development; matrix.py --panel development.
4. restore.py creates actual backup/restore copies and freshly extracts dependency archives.
5. Freeze the retained settings; prepare.py --panel final originally selected source bytes after freeze.
6. matrix.py --panel final --methods original cached discrete a1a2.
7. Only after every final output is frozen: failure_diagnostic.py.
8. PYTHONPATH=src python3 -m pytest -q tests/test_agent4_rescue.py.

These scripts are under scripts/agent4_rescue. Wrap GPU matrices with the inherited
scripts/agent4/watchdog.py --receipt NEW_PATH --timeout 1800 -- python3 SCRIPT...
Coordinate a GPU lease; exact guard commands and full commits are in receipts.
For exact replay use retained final_sources.json rather than selecting new sources.
Keep BF16 observations and the numerical executor classification unchanged.

The shared loader initially resolves the inherited pilot outputs through a read-only
symlink. New independent backups and restored dependencies live in
outputs/agent4-prefix-only-rescue; backup_restore.json indexes actual assets/hashes.
When relocating, recreate the documented input layout or point the loader at these
backups. No private target weights or fitted initialization is required; only the
A1+A2 comparator loads the supplied historical public lens.
