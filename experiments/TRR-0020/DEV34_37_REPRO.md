# Development34-37 reproduction

Use the full execution commit from DEV34_37_PROVENANCE.json for each phase.
Paths are relative to TRR-0020. Use the recorded WSL Python3 runtime with
PYTHONPATH=src, CUBLAS_WORKSPACE_CONFIG=:4096:8 and PYTHONDONTWRITEBYTECODE=1.
Workers disableTF32 and enable deterministic algorithms.

Use a separate reproduction checkout/output root. Writers are create-only;
never overwrite the original evidence. Restore prerequisite raw paths from the
hash-indexed archives. Exact guarded GPU commands, environment, working directory
and commits are in dev34_launch.json, dev34_stress_launch.json, dev35_launch.json,
dev36_launch.json and dev37_launch.json.

CPU qualification commands:
- python3 scripts/trr0020_stage34/cpu_check.py
- python3 scripts/trr0020_stage35/cpu_check.py
- python3 scripts/trr0020_stage37/cpu_check.py

Development36 depends on dev34stress and dev35public raw gradient fixtures.
Development37 uses the public IDs from dev34stress and the existing TRR-0014
nativeBOS response table. Its exact external path, SHA256, native256-row geometry,
prefix hash and build cost are in dev37_public_probe.json.table_source.
This is a derived public asset, not a fitted predictor. Generation code is
scripts/trr0014/lookup_probe.py; that historical file also runs unrelated
reconstruction probes, so do not invoke the entire file merely to rebuild it.

The archive and summary commands are python3 followed by the corresponding
scripts/trr0020_stage34_archive.py / stage34_summary.py,
scripts/trr0020_stage34_stress_archive.py / stage34_stress_summary.py,
scripts/trr0020_stage35_archive.py / stage35_summary.py,
scripts/trr0020_stage36_archive.py / stage36_summary.py, and
scripts/trr0020_stage37_archive.py / stage37_summary.py.
Each abbreviated summary filename has the same trr0020_ prefix and scripts/
directory. The exact executed command and commit are in its archive receipt.

Concatenate listed archive parts in order, verify the completeSHA256, then
restore logical_members paths from the indexed ZIP objects. Every member was
round-trip verified. The development37 response table remains an external
pre-existing dependency with its full hash and reproduction provenance.

All completed GPU jobs exited0. No research GPU job remains live. None of these
public diagnostics is a new active canonical benchmark method; the goal remains
unmet. Publication stays local under the previously recorded approval restriction.
