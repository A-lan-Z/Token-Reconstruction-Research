# TRR-0003 comparison timing-source archive

This directory contains byte-identical tracked copies of the raw JSON files used for `experiments/TRR-0003/comparison_summary_v1.json` and `.md`. The executed originals remain under `outputs/TRR-0003/`; the comparison generator verifies each archive copy's byte count and SHA-256 before using it as a source binding.

The archive includes the four Track A per-cell `evidence.json` files, the Track B selected-panel `prediction_evidence.json`, the comparator `run_evidence.json`, and the twelve comparator per-cell `*.evidence.json` files. Tensor/model artifacts remain at their recorded paths and are bound by the per-row retained-state metadata where applicable.

The summary retains both paths: `timing_source_path` identifies the executed source and `timing_source_archived_path` identifies this tracked copy. The raw score file was not rewritten; its obsolete merged-root `timing_path` values were not used.
