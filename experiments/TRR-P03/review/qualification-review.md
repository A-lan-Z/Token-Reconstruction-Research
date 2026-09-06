# TRR-P03 Stage 1 qualification review

## Disposition

**PASS for release of the planned full Stage 1 matrix, subject to the existing strict joint pre-score receipt and watchdog.** This is a qualification decision only: it does not open truth or authorize scoring of the 10-record qualifier. The full run must still complete the 24-record, both-arm matrix and obtain a VALIDATED strict joint receipt before truth scoring.

## Scope and checks

I audited only receipt and watchdog metadata for the four sequential qualifier commands. No observations, truth files, prediction tensors/rows, or score outputs were opened. The qualifier used the predeclared 10-record subset (six length-128 records plus anchors p03-s1-r0007, p03-s1-r0009, p03-s1-r0011, p03-s1-r0012), with lengths 39 and 128 and 924 scored tokens. It is qualification evidence, not a Stage 1 result.

The audited evidence is in runtime/qualifier-observations-bundle-{a,b}/generation_evidence.json, runtime/qualifier-reconstruction-bundle-{a,b}/{preflight.json,reconstructor_evidence.json,freeze_receipt.json}, and the corresponding runtime/watchdog/qualifier-{generation-a,generation-b,reconstruction-a,reconstruction-b}/ receipts.

Generation for bundles A and B completed with exit status 0, no watchdog errors or termination, the frozen source commit de0fcae6b9e17b1a0d41017a77e662c3bb0f06ed, deterministic CPU settings (Torch threads 8/inter-op 1), and batch size 4. Bundle A used the pinned public base Llama snapshot; bundle B used the evaluator-only Vikhr snapshot. Reconstruction for both arms used the pinned public base Llama snapshot, full methods (raw_boundary.cosine, projected_boundary.cosine, historical_a1.cosine, and historical_a1_a2_anchor.cosine), all four exact anchors, and canonical query/prototype chunks 256/8192. The A2 metadata records the fixed four-anchor policy and published torch.topk proposal-order tie rule.

Both reconstruction arms have matching plan hash f9c4d4bfa0fd8649891154326b32f5ad1e29d53d33af606fd4d53534d6debcb5, source commit, shared prepared projected artifact (SHA-256 8fa4e65ca5ae0c4492c16290403f38126894f5d41383bd2e2b178fbb85003ba7), prototype artifact (SHA-256 51abc304d51134777d55347b219fe659817b9f0319add99756eeac6e9b6dd9a3), lens artifact (SHA-256 33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742), record geometry, method coverage, and freeze status. Watchdog finish receipts and their referenced artifact hashes were internally consistent for all four commands.

## Resource qualification

All four watchdog guards passed with an 8 GiB RSS cap, 10 GiB minimum host-available-memory threshold, and 900 s timeout. The generation sampled peaks were 3,383,255,040 bytes (bundle A) and 5,019,549,696 bytes (bundle B); minimum sampled host availability was 22,417,666,048 and 19,958,067,200 bytes. Reconstruction sampled peaks were 7,189,983,232 and 7,898,628,096 bytes, with minimum sampled host availability 18,219,745,280 and 17,531,809,792 bytes.

The conservative reconstruction evidence peak is bundle B's internal process_max_rss_kib=7,803,792, or 7,991,083,008 bytes (7.442 GiB). Against the 8 GiB cap (8,589,934,592 bytes), that leaves 598,851,584 bytes (0.558 GiB) of measured margin. This is tight enough to retain the watchdog and stop on a guard failure.

The full matrix grows from 10 records/924 scored tokens to 24 records/1,482 scored tokens: 14 records and 558 scored positions, or 572 positions including one BOS per added record. With hidden size 2,048, the added BF16 observation/query storage is about 2.23 MiB; a float32 query copy upper bound is about 4.47 MiB. At fixed query chunk 256 and prototype chunk 8,192, score scratch is bounded independently of record count. The A2 candidate tensor increase at the maximum 129-position record and k=256 is about 1.76 MiB; the four-anchor computation remains fixed. Thus the direct geometry increase is approximately 8.46 MiB before allocator overhead, small relative to the measured margin but not a proof of peak equivalence.

## Release conditions

Run the unchanged canonical full-matrix commands with the same source, public-base reconstruction path for both arms, methods, tie policies, chunks, CPU/thread settings, and watchdog. Do not introduce a batching workaround. After both full prediction roots are frozen, run the strict validator bound to the frozen plan and both roots; only then open truth and score. A failed guard or any source/geometry/arm mismatch is a stop condition requiring review.
