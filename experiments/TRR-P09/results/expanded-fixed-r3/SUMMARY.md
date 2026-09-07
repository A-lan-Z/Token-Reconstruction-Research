# TRR-P09 frozen B1 audit

Status: **PASS** (93/93 checks).

Selected step `13000`; checkpoint file SHA `5757506003f0a4b82cb1cd6de2a82f11345f2dbcbce13c8cf18c4f515de3ec3a`; logical state SHA `209155048df936359870a1419803800d7bad4ab72a46e09bf3287648079964da`. The selected state strict-loaded on CPU into residual_mlp512 (hidden 2048, vocabulary 128256, context 128, bottleneck 512); no forward or scoring was performed.

Exact schedule: `6656000` valid draws, `1237403` unique valid record-position pairs, `5418597` repeats, `11996` records actually sampled, `11996` records present in batches, `104000` record-batch exposures. Available post-BOS positions: `1243710`; separate public support: `45631` token IDs.

Raw receipt remains runtime-bound at `/tmp/trr-p09-runtime/fixed-control-b1-r3/receipt.json` SHA `bc75d1c6837cd27bc2980cb448d4c309cecb279ab729b70db13367bac3eed47a`. Full raw learning curve is exported separately. Watchdog evidence remains at `/tmp/trr-p09-runtime/watchdog-fixed-control-b1-r3`.

Audit command: `/usr/bin/python3 scripts/trr_p09/audit_fixed_control_run.py --bank B1 --receipt /tmp/trr-p09-runtime/fixed-control-b1-r3/receipt.json --schedule /tmp/trr-p09-runtime/common-schedules-r1/schedule-b1-seed4010.safetensors --watchdog-dir /tmp/trr-p09-runtime/watchdog-fixed-control-b1-r3 --b0-binding /tmp/trr-p09/experiments/TRR-P09/setup/b0-immutable-loader-binding-r1.json --b1-manifest /tmp/trr-p09-runtime/stage1-capture-r2/bank_manifest-qualified-r1.json --support-binding /tmp/trr-p09/experiments/TRR-P09/setup/public-support-binding-r1.json --output-root /tmp/trr-p09-runtime/fixed-control-b1-r3-audit-r1 --repo-output /tmp/trr-p09/experiments/TRR-P09/results/expanded-fixed-r3 --repository-root /tmp/trr-p09`
Audit script SHA: `33ab14b31a737c57f5dfb2b537fb871c43374a2a6b2ed1ba4a73dacfad5d3c64`

Public payload access was limited to attention-mask/position sidecars and full-file hashing; no activation or token-ID tensor values, private/final truth, embedding, model forward, or source text persistence occurred during this audit.
