# Continuation21 reproduction and provenance

All work is local on task/TRR-0020. The prior public-publication restriction persists.
The full-vocabulary no-proposer objective remains active and unmet.

Use PYTHONPATH=src, CUBLAS_WORKSPACE_CONFIG=:4096:8 and PYTHONDONTWRITEBYTECODE=1.
Exact software/hardware and hashes are indexed in CONTINUATION21_PROVENANCE.json.
Receipts and output writers are create-only. Reproduce in an isolated checkout/output namespace and preserve the original evidence.
GPU runs must repeat their resource preflight and use the recorded guard; do not launch overlapping workers.

## Recorded commands

- experiments/TRR-0020/dev46_cpu_reference.json: commit 5330bc497d241d8583e63b18efb1d389f53b66e4; command ["/usr/bin/python3", "scripts/trr0020_stage46/cpu_check.py"].
- experiments/TRR-0020/dev46_launch.json: commit c853866de02e2411e74872d44ea48b57f302bb44; command ["python3", "scripts/trr0020_resource_guard.py", "--receipt", "experiments/TRR-0020/dev46_guard.json", "--timeout", "900", "--", "python3", "scripts/trr0020_stage46/run.py"].
- experiments/TRR-0020/dev46_score_launch.json: commit c853866de02e2411e74872d44ea48b57f302bb44; command ["python3", "scripts/trr0020_stage46/score.py"].
- experiments/TRR-0020/dev46_score_r1_launch.json: commit b0b6387aee4fac4b0624c2a4702b562f724d558e; command ["python3", "scripts/trr0020_stage46_score_r1.py"].
- experiments/TRR-0020/dev46_gpu_archive.json: commit b0b6387aee4fac4b0624c2a4702b562f724d558e; command ["/usr/bin/python3", "scripts/trr0020_stage46_gpu_archive.py"].
- experiments/TRR-0020/dev46_archive.json: commit b0b6387aee4fac4b0624c2a4702b562f724d558e; command ["/usr/bin/python3", "scripts/trr0020_stage46_archive.py"].
- experiments/TRR-0020/dev46_summary.json: commit b0b6387aee4fac4b0624c2a4702b562f724d558e; command ["/usr/bin/python3", "scripts/trr0020_stage46_summary.py"].
- experiments/TRR-0020/dev47_cpu_reference.json: commit b0352d8dbf35342976eb502286ad9989d8882014; command ["/usr/bin/python3", "scripts/trr0020_stage47/cpu_check.py"].
- experiments/TRR-0020/dev47_launch.json: commit 77cf29e4ba98df8cde36599ff1a2b7bf9c4db751; command ["python3", "scripts/trr0020_resource_guard.py", "--receipt", "experiments/TRR-0020/dev47_guard.json", "--timeout", "900", "--", "python3", "scripts/trr0020_stage47/run.py"].
- experiments/TRR-0020/dev47_failed_archive.json: commit 449f76b51b0529cea21d872c42549aa7f6f1199d; command ["/usr/bin/python3", "scripts/trr0020_stage47_archive.py", "--attempt", "failed"].
- experiments/TRR-0020/dev47_r1_cpu_reference.json: commit e2010d6293af59c0f535d54f83d63dc96359cb21; command ["/usr/bin/python3", "scripts/trr0020_stage47_r1/cpu_check.py"].
- experiments/TRR-0020/dev47_r1_launch.json: commit 33105073d0a5020d2edd52b210d20f9d6151d215; command ["python3", "scripts/trr0020_resource_guard.py", "--receipt", "experiments/TRR-0020/dev47_r1_guard.json", "--timeout", "900", "--", "python3", "scripts/trr0020_stage47_r1/run.py"].
- experiments/TRR-0020/dev47_r1_score_launch.json: commit 449f76b51b0529cea21d872c42549aa7f6f1199d; command ["python3", "scripts/trr0020_stage47_r1_score.py"].
- experiments/TRR-0020/dev47_r1_archive.json: commit 449f76b51b0529cea21d872c42549aa7f6f1199d; command ["/usr/bin/python3", "scripts/trr0020_stage47_archive.py", "--attempt", "r1"].
- experiments/TRR-0020/dev48_launch.json: commit e72d83e2d66644ab1083606c63591d6540eba38b; command ["python3", "scripts/trr0020_resource_guard.py", "--receipt", "experiments/TRR-0020/dev48_guard.json", "--timeout", "1200", "--", "python3", "scripts/trr0020_stage48/run.py"].
- experiments/TRR-0020/dev48_score_launch.json: commit e72d83e2d66644ab1083606c63591d6540eba38b; command ["python3", "scripts/trr0020_stage48/score.py"].
- experiments/TRR-0020/dev48_failed_archive.json: commit 42100af8122fbf70313aba63d23d4702070ffd45; command ["/usr/bin/python3", "scripts/trr0020_stage48_failed_archive.py"].
- experiments/TRR-0020/dev48_r1_cpu_reference.json: commit 061b31d2695895e8735a0faccc4812856205accb; command ["/usr/bin/python3", "scripts/trr0020_stage48_r1/cpu_check.py"].
- experiments/TRR-0020/dev48_r1_launch.json: commit ef9272203de8557c1aee77734f6432fe92f7c440; command ["python3", "scripts/trr0020_resource_guard.py", "--receipt", "experiments/TRR-0020/dev48_r1_guard.json", "--timeout", "1200", "--", "python3", "scripts/trr0020_stage48_r1/run.py"].
- experiments/TRR-0020/dev48_r1_score_launch.json: commit ef9272203de8557c1aee77734f6432fe92f7c440; command ["python3", "scripts/trr0020_stage48_r1/score.py"].
- experiments/TRR-0020/dev48_r1_archive_launch.json: commit b0ab1b7e80945ace2d8fb7321d6e6eff030eef9a; command ["python3", "scripts/trr0020_stage48_r1_archive.py"].
- experiments/TRR-0020/dev48_r1_summary_launch.json: commit b0ab1b7e80945ace2d8fb7321d6e6eff030eef9a; command ["python3", "scripts/trr0020_stage48_r1_summary.py"].

All reconstruction outputs freeze before retrospective truth scoring. The initial development48 attempt must remain excluded and unscored. The repaired rule uses distinct names, bindings, paths and CPU qualification.

Accepted reconstruction matrices this continuation: development46 (32cells) and development48 revision1 (96cells). Development47 is public numerical qualification only. The rejected initial development48 does not count toward accepted totals.

Archives deduplicate exact bytes: concatenate numbered parts in listed order, verify archive_sha256, and map logical_members to objects. Every archived member was round-trip verified.

Existing34method68cell canonical comparison remains complete. No method from this continuation is promoted; do not borrow canonical scores from a different rule. All jobs are terminal and no GPU job remains.
