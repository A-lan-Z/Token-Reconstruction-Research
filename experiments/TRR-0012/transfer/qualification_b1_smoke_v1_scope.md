# Qualification v1 scope

`qualification_b1_smoke_v1.json` is a CPU-only check of the ordinary frozen package alias `scripts/trr0012_package.py` (`5023a9e7…`) on the four bundled smoke records. It validates the package's vendored B1 loader, projected-feature geometry, full-vocabulary logits, and stored smoke predictions.

It does not exercise the production `scripts/trr0010_eval_runner.py` registration/loader path or CUDA execution. The changed-path comparison is recorded separately in `qualification_b1_bridge_cuda_v1.json`, which loads the same four records through both package and production paths and compares features, full logits, and predictions.
